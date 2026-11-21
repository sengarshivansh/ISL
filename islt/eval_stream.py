"""
Continuous (streaming) evaluation.

Builds long unsegmented streams by joining consecutive test sentences from the
same source video (iSign clips are cut from longer videos, so this restores
continuous signing). Between sentences we insert a hold of `pause` seconds
(the last frame repeated); pause=0 joins them back to back, the hardest case.
Optional `speed` replays the whole stream faster.

The segmenter cuts each stream, every segment is translated, and we report
  * boundary F1 (sentence ends within ±0.5 s)
  * stream BLEU-4 / chrF: all hypotheses of a stream joined vs. all references joined
  * model latency per sentence (ms)

    python -m islt.eval_stream --checkpoint runs/FaST/best.pt --config configs/isign_base.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import sacrebleu

from .data import source_video
from .features import resample_speed
from .stream import SegmenterConfig, Translator, boundary_f1, segment_stream


def build_streams(items, get_raw, per_stream=5, pause_s=0.8, fps=10.0, seed=0, max_streams=200):
    by_video = defaultdict(list)
    for uid, text in items:
        by_video[source_video(uid)].append((uid, text))
    rng = random.Random(seed)
    streams = []
    for vid, clips in by_video.items():
        clips.sort(key=lambda x: int(x[0].rsplit("-", 1)[1]) if x[0].rsplit("-", 1)[1].isdigit() else 0)
        for k in range(0, len(clips) - per_stream + 1, per_stream):
            group = clips[k:k + per_stream]
            parts, texts, bounds, t = [], [], [], 0
            for uid, text in group:
                x = np.asarray(get_raw(uid), np.float32)
                parts.append(x)
                bounds.append((t, t + len(x)))
                texts.append(text)
                t += len(x)
                hold = int(round(pause_s * fps * rng.uniform(0.75, 1.25)))
                if hold:
                    parts.append(np.repeat(x[-1:], hold, axis=0))
                    t += hold
            streams.append((np.concatenate(parts), texts, bounds))
    rng.shuffle(streams)
    return streams[:max_streams]


def evaluate_streams(streams, translator: Translator | None, seg_cfg=None, speed=1.0, tol_s=0.5,
                     segmenter=None):
    seg_cfg = seg_cfg or SegmenterConfig()
    f1s, hyps, refs, lat = [], [], [], []
    for stream, texts, bounds in streams:
        if speed != 1.0:
            stream = resample_speed(stream, speed)
            bounds = [(int(a / speed), int(b / speed)) for a, b in bounds]
        segs = segment_stream(stream, seg_cfg, segmenter)
        f1s.append(boundary_f1(segs, bounds, tol=int(tol_s * seg_cfg.fps)))
        if translator is not None:
            out = []
            for s in segs:
                text, ms = translator.translate(s.clip)
                out.append(text)
                lat.append(ms)
            hyps.append(" ".join(out))
            refs.append(" ".join(texts))
    res = {"boundary_f1": float(np.mean([f["f1"] for f in f1s])),
           "boundary_precision": float(np.mean([f["precision"] for f in f1s])),
           "boundary_recall": float(np.mean([f["recall"] for f in f1s])),
           "n_streams": len(streams), "speed": speed}
    if translator is not None:
        res.update({"stream_bleu4": sacrebleu.corpus_bleu(hyps, [refs]).score,
                    "stream_chrf": sacrebleu.corpus_chrf(hyps, [refs]).score,
                    "latency_ms_mean": float(np.mean(lat)) if lat else 0.0,
                    "latency_ms_p95": float(np.percentile(lat, 95)) if lat else 0.0})
    return res


def main():
    from .train import build_data, load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--out")
    ap.add_argument("--pauses", nargs="*", type=float, default=[0.8, 0.4, 0.0])
    ap.add_argument("--speeds", nargs="*", type=float, default=[1.0, 1.5, 2.0])
    ap.add_argument("--max-streams", type=int, default=200)
    ap.add_argument("--per-stream", type=int, default=5)
    ap.add_argument("--boundary", help="learned boundary checkpoint (default: pause-based segmenter)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    splits, get_raw = build_data(cfg)
    tr = Translator(args.checkpoint) if args.checkpoint else None
    segmenter = None
    if args.boundary:
        from .boundary import LearnedSegmenter
        segmenter = LearnedSegmenter(args.boundary)
    results = []
    for pause in args.pauses:
        streams = build_streams(splits["test"], get_raw, per_stream=args.per_stream, pause_s=pause,
                                max_streams=args.max_streams)
        for sp in args.speeds:
            r = {"pause_s": pause, "segmenter": "learned" if args.boundary else "pause",
                 **evaluate_streams(streams, tr, speed=sp, segmenter=segmenter)}
            print(json.dumps(r), flush=True)
            results.append(r)
    name = "stream_results_learned.json" if args.boundary else "stream_results.json"
    out = Path(args.out or (Path(args.checkpoint).parent / name if args.checkpoint else Path("runs") / name))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
