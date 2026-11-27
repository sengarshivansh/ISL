"""
Train and evaluate a FaST-ISL model from a YAML config.

    python -m islt.train --config configs/fast.yaml
    python -m islt.train --config configs/fast.yaml --eval-only runs/fast/best.pt

Writes runs/<name>/{config.yaml, log.jsonl, best.pt, results.json, predictions_*.tsv}.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import AugConfig, SignDataset, load_isign, make_collate
from .metrics import subset_scores
from .model import FaSTISL, ModelConfig, load_tokenizer

DEFAULTS = {
    "name": "run",
    "out_dir": "runs",
    "seed": 42,
    "data": {"type": "isign", "shard_dir": None, "csv": None, "limit": None,
             "val_pct": 5, "test_pct": 5, "max_frames": 256},
    "model": {},
    "aug": {"speed_prob": 0.0, "speed_range": [0.5, 2.0], "jitter": 0.0, "frame_drop": 0.0},
    "train": {"epochs": 30, "batch_size": 32, "lr_encoder": 3e-4, "lr_t5": 1e-4,
              "weight_decay": 0.01, "warmup_steps": 1000, "grad_clip": 1.0,
              "num_workers": 2, "patience": 6, "val_max": 2000, "label_smoothing": 0.1,
              "max_hours": None,     # stop cleanly before a session limit (Kaggle: 12 h) and resume later
              "word_dropout": 0.0},  # share of decoder input words hidden during training
    "eval": {"speeds": [1.0, 1.5, 2.0], "num_beams": 4, "max_new_tokens": 48, "batch_size": 64,
             "no_repeat_ngram": 3},
}


def deep_update(base, new):
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path, overrides=None):
    import copy
    cfg = copy.deepcopy(DEFAULTS)
    deep_update(cfg, yaml.safe_load(Path(path).read_text()) if path else {})
    deep_update(cfg, overrides)
    return cfg


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def build_data(cfg):
    d = cfg["data"]
    if d["type"] == "synthetic":
        from .synthetic import make_dataset
        tr, ctr = make_dataset(d.get("n_train", 2000), seed=1)
        va, cva = make_dataset(d.get("n_val", 200), seed=2)
        te, cte = make_dataset(d.get("n_test", 300), seed=3)
        clips = {**ctr, **cva, **cte}
        return {"train": tr, "val": va, "test": te}, clips.__getitem__
    splits, store = load_isign(d["shard_dir"], d["csv"], d["val_pct"], d["test_pct"], d["limit"])
    return splits, store.get


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict(model, items, get_raw, tok, cfg, device, speed=1.0):
    ds = SignDataset(items, get_raw, speed=speed, max_frames=cfg["data"]["max_frames"])
    dl = DataLoader(ds, batch_size=cfg["eval"]["batch_size"], shuffle=False,
                    collate_fn=make_collate(tok), num_workers=0)
    model.eval()
    hyps, refs, uids = [], [], []
    for batch in dl:
        b = to_device(batch, device)
        out = model.generate(b, max_new_tokens=cfg["eval"]["max_new_tokens"],
                             num_beams=cfg["eval"]["num_beams"],
                             no_repeat_ngram_size=cfg["eval"].get("no_repeat_ngram", 0))
        hyps += [h.strip() for h in tok.batch_decode(out, skip_special_tokens=True)]
        refs += batch["text"]
        uids += batch["uid"]
    return uids, hyps, refs


def evaluate(model, splits, get_raw, tok, cfg, device, run_dir):
    results = {}
    for speed in cfg["eval"]["speeds"]:
        uids, hyps, refs = predict(model, splits["test"], get_raw, tok, cfg, device, speed)
        results[f"speed_{speed}"] = subset_scores(hyps, refs)
        with open(run_dir / f"predictions_speed{speed}.tsv", "w") as f:
            f.write("uid\treference\tprediction\n")
            for u, r, h in zip(uids, refs, hyps):
                f.write(f"{u}\t{r}\t{h}\n")
        s = results[f"speed_{speed}"]["all"]
        print(f"[test x{speed}] BLEU-4 {s['bleu4']:.2f} chrF {s['chrf']:.2f} ROUGE-L {s['rougeL']:.2f}",
              flush=True)
    (run_dir / "results.json").write_text(json.dumps(results, indent=1))
    return results


def train(cfg):
    set_seed(cfg["seed"])
    run_dir = Path(cfg["out_dir"]) / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    device = pick_device()
    tok = load_tokenizer()
    splits, get_raw = build_data(cfg)
    print({k: len(v) for k, v in splits.items()}, "device:", device, flush=True)

    mcfg = ModelConfig(**cfg["model"])
    model = FaSTISL(mcfg).to(device)
    print(f"params: {model.n_params() / 1e6:.2f}M total, "
          f"{sum(p.numel() for p in model.encoder.parameters()) / 1e6:.2f}M pose encoder", flush=True)

    t = cfg["train"]
    aug = AugConfig(speed_prob=cfg["aug"]["speed_prob"], speed_range=tuple(cfg["aug"]["speed_range"]),
                    jitter=cfg["aug"]["jitter"], frame_drop=cfg["aug"]["frame_drop"])
    train_ds = SignDataset(splits["train"], get_raw, aug=aug, max_frames=cfg["data"]["max_frames"],
                           seed=cfg["seed"])
    dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, collate_fn=make_collate(tok),
                    num_workers=t["num_workers"], drop_last=True,
                    persistent_workers=t["num_workers"] > 0)

    t5_params = [p for p in model.t5.parameters() if p.requires_grad]
    t5_ids = {id(p) for p in t5_params}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in t5_ids]
    opt = torch.optim.AdamW([{"params": other, "lr": t["lr_encoder"]},
                             {"params": t5_params, "lr": t["lr_t5"]}], weight_decay=t["weight_decay"])
    total = t["epochs"] * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / t["warmup_steps"]) *
                                              0.5 * (1 + math.cos(math.pi * min(1, s / max(1, total)))))
    # bf16 only where the GPU has it in hardware (A100/L4 and newer); a T4 would only emulate it,
    # which is slower than fp32. fp16 is not used: T5 is known to overflow in fp16.
    use_amp = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
    scaler = torch.amp.GradScaler(enabled=False)
    ls = t["label_smoothing"]
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=ls)

    val_items = splits["val"][: t["val_max"]]
    best, bad, step, start_epoch = -1.0, 0, 0, 0
    last = run_dir / "last.pt"
    if last.exists():  # resume after a Colab disconnect
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        best, bad, step, start_epoch = st["best"], st["bad"], st["step"], st["epoch"] + 1
        print(f"resumed from epoch {st['epoch']} (best {best:.2f})", flush=True)
    log = open(run_dir / "log.jsonl", "a")
    t_start, epoch_sec = time.time(), 0.0
    for epoch in range(start_epoch, t["epochs"]):
        if bad >= t["patience"]:
            break
        # another epoch plus the final test evaluation must fit in the time budget, else stop here;
        # last.pt is already saved, so the next session resumes from this epoch
        if t.get("max_hours") and time.time() - t_start + 1.5 * epoch_sec > t["max_hours"] * 3600:
            print(f"time budget of {t['max_hours']} h reached after epoch {epoch - 1}: "
                  "run again to resume", flush=True)
            log.close()
            return None
        model.train()
        t0, run_loss, run_aux = time.time(), 0.0, 0.0
        for batch in dl:
            b = to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(b, word_dropout=t.get("word_dropout", 0.0))
                ce = loss_fn(out.logits.flatten(0, 1).float(), b["labels"].flatten())
                loss = ce + mcfg.contrastive * out.aux_loss.float()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += ce.item()
            run_aux += out.aux_loss.detach().item()
            step += 1
        _, hyps, refs = predict(model, val_items, get_raw, tok,
                                {**cfg, "eval": {**cfg["eval"], "num_beams": 1}}, device)
        val = subset_scores(hyps, refs)
        score = val["all"]["bleu4"] + 0.01 * val["all"]["chrf"]  # BLEU first, chrF breaks ties
        rec = {"epoch": epoch, "step": step, "loss": run_loss / max(1, len(dl)),
               "aux_loss": run_aux / max(1, len(dl)),
               "val_bleu4": val["all"]["bleu4"], "val_chrf": val["all"]["chrf"],
               "val_question_acc": val["nonmanual"]["question_acc"],
               "val_negation_acc": val["nonmanual"]["negation_acc"],
               "sec": round(time.time() - t0)}
        epoch_sec = time.time() - t0
        if mcfg.use_face and mcfg.fusion == "gate" and model.encoder.last_gate is not None:
            rec["mean_gate"] = float(model.encoder.last_gate.mean())
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if score > best:
            best, bad = score, 0
            torch.save({"model": model.state_dict(), "model_cfg": mcfg.to_dict(), "cfg": cfg},
                       run_dir / "best.pt")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "best": best, "bad": bad, "step": step, "epoch": epoch}, last)
        if bad >= t["patience"]:
            print("early stop", flush=True)
            break
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    results = evaluate(model, splits, get_raw, tok, cfg, device, run_dir)
    last.unlink(missing_ok=True)  # finished: the optimizer state is no longer needed
    return results


def load_model(path, device=None):
    device = device or pick_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mc = ModelConfig(**ckpt["model_cfg"])
    model = FaSTISL(mc)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["cfg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-only", help="checkpoint to evaluate instead of training")
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.epochs=2 model.face_region=none")
    args = ap.parse_args()
    over = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        node = over
        *parents, leaf = k.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(v)
    cfg = load_config(args.config, over)
    if args.eval_only:
        device = pick_device()
        model, _ = load_model(args.eval_only, device)
        splits, get_raw = build_data(cfg)
        run_dir = Path(cfg["out_dir"]) / cfg["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        evaluate(model, splits, get_raw, load_tokenizer(), cfg, device, run_dir)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
