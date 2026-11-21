"""
Data loading for iSign keypoint shards (and the synthetic toy language).

Shards come from colab/isign_pose_convert.py: shard_XXXXX.npz = {uid: (T, 714) float16}.
Labels come from iSign_v1.1.csv (uid, text).

Split: iSign uids look like "<video_id>-<n>". All clips cut from the same source
video go to the same split, so the test set never contains a sentence from a
video seen in training (prevents near-duplicate leakage).
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import make_streams, resample_speed

NEGATION_WORDS = {"not", "no", "never", "don't", "dont", "doesn't", "didn't", "can't",
                  "cannot", "won't", "isn't", "aren't", "wasn't", "nothing", "nobody", "none"}


def source_video(uid: str) -> str:
    return uid.rsplit("-", 1)[0]


def split_of(uid: str, val_pct: int = 5, test_pct: int = 5) -> str:
    h = int(hashlib.md5(source_video(uid).encode()).hexdigest(), 16) % 100
    if h < test_pct:
        return "test"
    if h < test_pct + val_pct:
        return "val"
    return "train"


def nonmanual_tags(text: str) -> dict[str, bool]:
    """Sentence-level tags used for the non-manual evaluation subset."""
    t = text.strip().lower()
    words = {w.strip(".,!?\"'") for w in t.replace("n't", " n't").split()} | set(t.split())
    return {"question": t.endswith("?"), "negation": bool(words & NEGATION_WORDS)}


class ShardStore:
    """Lazy access to clips spread over many .npz shards."""

    def __init__(self, shard_dir: str | Path):
        self.files = sorted(Path(shard_dir).glob("shard_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no shard_*.npz files in {shard_dir}")
        self._open: dict[int, np.lib.npyio.NpzFile] = {}
        self.index: dict[str, int] = {}
        for i, f in enumerate(self.files):
            with np.load(f) as z:
                for uid in z.files:
                    self.index[uid] = i

    def __contains__(self, uid):
        return uid in self.index

    def get(self, uid: str) -> np.ndarray:
        i = self.index[uid]
        if i not in self._open:
            if len(self._open) > 64:  # keep file handles bounded
                self._open.pop(next(iter(self._open))).close()
            self._open[i] = np.load(self.files[i])
        return self._open[i][uid]

    def __getstate__(self):  # DataLoader workers re-open their own handles
        s = self.__dict__.copy()
        s["_open"] = {}
        return s


@dataclass
class AugConfig:
    speed_prob: float = 0.0          # chance a training clip is resampled
    speed_range: tuple = (0.5, 2.0)  # log-uniform speed factor
    jitter: float = 0.0              # gaussian noise on coordinates (image units)
    frame_drop: float = 0.0          # chance each frame is dropped


def augment(raw: np.ndarray, cfg: AugConfig, rng: random.Random) -> np.ndarray:
    x = raw.astype(np.float32)
    if cfg.speed_prob and rng.random() < cfg.speed_prob:
        lo, hi = np.log(cfg.speed_range[0]), np.log(cfg.speed_range[1])
        x = resample_speed(x, float(np.exp(rng.uniform(lo, hi))))
    if cfg.frame_drop and len(x) > 8:
        keep = np.array([rng.random() >= cfg.frame_drop for _ in range(len(x))])
        if keep.sum() >= 4:
            x = x[keep]
    if cfg.jitter:
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, cfg.jitter, x.shape)
        x = np.where(x != 0, x + noise.astype(np.float32), 0)
    return x


class SignDataset(Dataset):
    def __init__(self, items: Sequence[tuple[str, str]], get_raw: Callable[[str], np.ndarray],
                 face_region="all", aug: AugConfig | None = None, speed: float = 1.0,
                 max_frames: int = 256, seed: int = 0):
        self.items = list(items)
        self.get_raw = get_raw
        self.face_region = face_region
        self.aug = aug
        self.speed = speed           # fixed replay speed for the fast-signing test sets
        self.max_frames = max_frames
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        uid, text = self.items[i]
        raw = np.asarray(self.get_raw(uid), dtype=np.float32)
        if self.speed != 1.0:
            raw = resample_speed(raw, self.speed)
        if self.aug is not None:
            raw = augment(raw, self.aug, self.rng)
        if len(raw) > self.max_frames:  # uniform subsample keeps the whole sentence
            raw = raw[np.linspace(0, len(raw) - 1, self.max_frames).round().astype(int)]
        streams = make_streams(raw, self.face_region)
        return {"uid": uid, "text": text, **{k: torch.from_numpy(v) for k, v in streams.items()}}


def make_collate(tokenizer, max_target_len: int = 64):
    stream_keys = ("body", "hands", "face", "head", "present")

    def collate(batch):
        lengths = torch.tensor([len(b["body"]) for b in batch])
        t = int(lengths.max())
        out = {"uid": [b["uid"] for b in batch], "text": [b["text"] for b in batch]}
        for k in stream_keys:
            d = batch[0][k].shape[1]
            pad = torch.zeros(len(batch), t, d)
            for i, b in enumerate(batch):
                pad[i, :len(b[k])] = b[k]
            out[k] = pad
        out["mask"] = torch.arange(t)[None] < lengths[:, None]
        enc = tokenizer(out["text"], padding=True, truncation=True,
                        max_length=max_target_len, return_tensors="pt")
        labels = enc.input_ids.clone()
        labels[enc.attention_mask == 0] = -100
        out["labels"] = labels
        return out

    return collate


def load_isign(shard_dir, csv_path, val_pct=5, test_pct=5, limit=None):
    """Returns {'train': [(uid, text)], 'val': [...], 'test': [...]}, store."""
    import pandas as pd
    store = ShardStore(shard_dir)
    df = pd.read_csv(csv_path).dropna(subset=["text"])
    df = df[df.uid.isin(store.index.keys())]
    if limit:
        df = df.iloc[:limit]
    splits = {"train": [], "val": [], "test": []}
    for uid, text in zip(df.uid, df.text):
        splits[split_of(uid, val_pct, test_pct)].append((uid, str(text).strip()))
    return splits, store
