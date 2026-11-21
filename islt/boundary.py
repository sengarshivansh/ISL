"""
Learned sentence-boundary detector for continuous signing.

A pause threshold fails when people sign fast or barely pause (the pause
shrinks below the threshold). This small causal network reads the same
keypoint streams as the translator and predicts, frame by frame, whether a
sentence ended `delay` frames ago. The short look-ahead (0.3 s at 10 fps)
lets it see that a new sentence has started.

Training data = stitched streams: consecutive sentences joined with a random
pause (0 to 1.2 s, often none) and replayed at a random speed (0.5-2x), so the
detector cannot rely on pause length.

    python -m islt.boundary --config configs/isign_base.yaml --out runs/boundary
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import source_video
from .features import STREAM_DIMS, make_streams, resample_speed
from .model import Proj, with_velocity

DELAY = 3  # frames of look-ahead (0.3 s at 10 fps)
IN_KEYS = ("body", "hands", "head", "present")  # face expression is not needed to find sentence ends


class BoundaryNet(nn.Module):
    def __init__(self, d=128, layers=2):
        super().__init__()
        self.body = Proj(STREAM_DIMS["body"] * 2, d, 0.1)
        self.hands = Proj(STREAM_DIMS["hands"] * 2, d, 0.1)
        self.head = Proj(STREAM_DIMS["head"] * 2, d, 0.1)
        self.present = nn.Linear(STREAM_DIMS["present"], d)
        self.conv = nn.Conv1d(d, d, 5, padding=4)  # causal after trimming
        self.gru = nn.GRU(d, d, layers, batch_first=True)
        self.out = nn.Linear(d, 1)

    def features(self, s, mask):
        h = (self.body(with_velocity(s["body"], mask)) + self.hands(with_velocity(s["hands"], mask))
             + self.head(with_velocity(s["head"], mask)) + self.present(s["present"]))
        c = self.conv(h.transpose(1, 2))[:, :, : h.shape[1]].transpose(1, 2)  # causal
        return torch.relu(c) + h

    def forward(self, s, mask, state=None):
        y, state = self.gru(self.features(s, mask), state)
        return self.out(y).squeeze(-1), state


def make_training_stream(clips, rng: random.Random, speed_range=(0.5, 2.0), max_pause_s=1.2, fps=10):
    """clips: list of (T, 714). Returns (stream, end_frames)."""
    speed = float(np.exp(rng.uniform(np.log(speed_range[0]), np.log(speed_range[1]))))
    parts, ends, t = [], [], 0
    for x in clips:
        x = resample_speed(np.asarray(x, np.float32), speed)
        parts.append(x)
        t += len(x)
        ends.append(t)
        pause = 0 if rng.random() < 0.35 else int(rng.uniform(0, max_pause_s) * fps)
        if pause:
            parts.append(np.repeat(x[-1:], pause, axis=0))
            t += pause
    return np.concatenate(parts), ends[:-1]  # the stream end itself is not a boundary to detect


def labels_for(n, ends, delay=DELAY, width=1):
    """Target at frame t = 1 if a sentence ended near frame t - delay."""
    y = np.zeros(n, np.float32)
    for e in ends:
        for k in range(e - width, e + width + 1):
            if 0 <= k + delay < n:
                y[k + delay] = 1.0
    return y


def group_by_video(items):
    groups = {}
    for uid, _ in items:
        groups.setdefault(source_video(uid), []).append(uid)
    for v in groups.values():
        v.sort(key=lambda u: int(u.rsplit("-", 1)[1]) if u.rsplit("-", 1)[1].isdigit() else 0)
    return [g for g in groups.values() if len(g) >= 2]


def sample_batch(groups, get_raw, rng, batch=16, n_sent=(2, 5), max_len=600):
    streams, targets = [], []
    for _ in range(batch):
        g = rng.choice(groups)
        k = rng.randint(n_sent[0], min(n_sent[1], len(g)))
        i = rng.randint(0, len(g) - k)
        stream, ends = make_training_stream([get_raw(u) for u in g[i:i + k]], rng)
        stream = stream[:max_len]
        streams.append(make_streams(stream))
        targets.append(labels_for(len(stream), [e for e in ends if e < len(stream)]))
    t = max(len(y) for y in targets)
    out = {k: torch.zeros(batch, t, streams[0][k].shape[1]) for k in IN_KEYS}
    y = torch.zeros(batch, t)
    mask = torch.zeros(batch, t, dtype=torch.bool)
    for b, (s, tg) in enumerate(zip(streams, targets)):
        n = len(tg)
        for k in IN_KEYS:
            out[k][b, :n] = torch.from_numpy(s[k])
        y[b, :n] = torch.from_numpy(tg)
        mask[b, :n] = True
    return out, mask, y


def train_boundary(groups, get_raw, out_dir, steps=3000, lr=1e-3, device="cpu", seed=0, log_every=200):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    net = BoundaryNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pos_weight = torch.tensor(8.0, device=device)  # boundaries are rare
    for step in range(1, steps + 1):
        s, m, y = sample_batch(groups, get_raw, rng)
        s = {k: v.to(device) for k, v in s.items()}
        m, y = m.to(device), y.to(device)
        logits, _ = net(s, m)
        loss = nn.functional.binary_cross_entropy_with_logits(logits[m], y[m], pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0:
            print(json.dumps({"step": step, "loss": round(loss.item(), 4)}), flush=True)
    torch.save({"model": net.state_dict()}, out_dir / "boundary.pt")
    return net


def trim_static(clip: np.ndarray, start: int, thr: float = 0.02, keep: int = 2):
    """Remove still frames (pause holds) from both ends of a segment, keeping `keep`
    frames of context. Returns (clip, start) or (None, start) if nothing moves."""
    from .stream import frame_energy
    e = np.array([0.0] + [frame_energy(clip[i - 1], clip[i])[0] for i in range(1, len(clip))])
    moving = np.where(e > thr)[0]
    if len(moving) == 0:
        return None, start
    a = max(0, moving[0] - 1 - keep)
    b = min(len(clip), moving[-1] + 1 + keep)
    return clip[a:b], start + a


class LearnedSegmenter:
    """Frame-by-frame segmentation with BoundaryNet. Same interface as stream.Segmenter."""

    def __init__(self, checkpoint, fps=10.0, threshold=0.5, min_s=0.8, max_s=12.0, refractory_s=0.6,
                 device="cpu"):
        from .stream import Segment  # noqa: F401
        self.net = BoundaryNet()
        self.net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
        self.net.eval().to(device)
        self.device, self.fps, self.thr = device, fps, threshold
        self.min_len, self.max_len = int(min_s * fps), int(max_s * fps)
        self.refractory = int(refractory_s * fps)
        self.reset()

    def reset(self):
        self.frames, self.probs, self.state = [], [], None
        self.buf_start, self.n, self.last_cut, self.last_t = 0, 0, -10 ** 9, None
        self.window = []
        self.active = True
        self.ema = 0.0

    def _step_prob(self, frame):
        """Probability for the newest frame. The conv needs the last 5 frames and the
        velocity the one before, so features are computed over a short rolling window;
        the GRU state carries everything older."""
        self.window = (self.window + [frame])[-8:]
        st = make_streams(np.stack(self.window))
        x = {k: torch.from_numpy(st[k])[None].to(self.device) for k in IN_KEYS}
        m = torch.ones(1, len(self.window), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            feats = self.net.features(x, m)[:, -1:]
            y, self.state = self.net.gru(feats, self.state)
            return float(torch.sigmoid(self.net.out(y))[0, 0])

    def push(self, frame, t=None):
        from .stream import Segment
        if t is not None:
            if self.last_t is not None and t - self.last_t < 1.0 / self.fps - 1e-3:
                return []
            self.last_t = t
        frame = np.asarray(frame, np.float32)
        p = self._step_prob(frame)
        self.frames.append(frame)
        self.probs.append(p)
        self.ema = p
        self.n += 1
        out = []
        # a boundary DELAY frames ago is confirmed once its probability is a local peak above threshold
        if len(self.probs) >= 3:
            c = self.probs[-2]
            if c >= self.thr and c >= self.probs[-3] and c >= self.probs[-1]:
                end = self.n - 2 - DELAY  # stream index just after the detected sentence end
                cut = end - self.buf_start
                if cut >= self.min_len and self.n - self.last_cut >= self.refractory:
                    out += self._emit(self.frames[:cut], self.buf_start)
                    self.frames, self.probs = self.frames[cut:], self.probs[cut:]
                    self.buf_start, self.last_cut = end, self.n
        if len(self.frames) >= self.max_len:
            cut = len(self.frames)
            out += self._emit(self.frames, self.buf_start)
            self.frames, self.probs = [], []
            self.buf_start += cut
        return out

    def _emit(self, frames, start):
        from .stream import Segment
        clip, start = trim_static(np.stack(frames), start)
        if clip is None or len(clip) < self.min_len:
            return []
        return [Segment(clip, start, start + len(clip))]

    def flush(self):
        out = self._emit(self.frames, self.buf_start) if self.frames else []
        self.frames, self.probs = [], []
        return out


def main():
    from .train import build_data, load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/boundary")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = load_config(args.config)
    splits, get_raw = build_data(cfg)
    groups = group_by_video(splits["train"])
    print(len(groups), "training videos with >= 2 sentences", flush=True)
    train_boundary(groups, get_raw, args.out, steps=args.steps, device=args.device)


if __name__ == "__main__":
    main()
