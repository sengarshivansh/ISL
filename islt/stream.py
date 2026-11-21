"""
Streaming (continuous) translation: unsegmented keypoint stream -> sentences -> English.

Real signers do not press a button between sentences. `Segmenter` watches the
motion energy of the hands and closes a sentence when the signer pauses
(hands still or dropped for `pause_s` seconds), or when a sentence gets too
long (`max_s`; it then cuts at the calmest recent frame). Each closed segment is
translated immediately, so the delay after a sentence ends is one pause plus
one model call.

    seg = Segmenter()
    for t, frame in camera:                 # frame = (714,) raw MediaPipe vector
        for clip in seg.push(frame, t):     # yields finished sentence clips (T, 714)
            print(translator.translate(clip))
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .features import L_SHOULDER, R_SHOULDER, split_raw


@dataclass
class SegmenterConfig:
    fps: float = 10.0          # model frame rate; faster input is decimated to this
    energy_on: float = 0.06    # wrist motion (shoulder widths / frame) that counts as signing
    energy_off: float = 0.03   # below this the signer is pausing (hysteresis)
    rest_y: float = 1.6        # wrists lower than this (shoulder widths below shoulders) = hands down
    pause_s: float = 0.6       # pause length that ends a sentence
    min_s: float = 0.8         # shorter bursts are ignored (scratching, adjusting)
    max_s: float = 12.0        # force a cut so latency stays bounded
    pre_s: float = 0.3         # context kept before the detected start
    smooth: float = 0.5        # EMA factor for the energy


def frame_energy(prev: np.ndarray | None, cur: np.ndarray, rest_y: float = 1.6) -> tuple[float, bool]:
    """Motion energy of the two hands between frames, in shoulder widths, and
    whether any hand is up in the signing space."""
    pose, _, lh, rh = split_raw(cur[None])
    sh = pose[0, [L_SHOULDER, R_SHOULDER], :2]
    width = max(float(np.linalg.norm(sh[0] - sh[1])), 1e-3)
    centre = sh.mean(0)
    hands = []
    for h in (lh[0], rh[0]):
        if np.abs(h).sum() > 0:
            hands.append(h[:, :2].mean(0))
    up = any((p[1] - centre[1]) / width < rest_y for p in hands)
    if prev is None:
        return 0.0, up
    _, _, plh, prh = split_raw(prev[None])
    e = 0.0
    for a, b in ((plh[0], lh[0]), (prh[0], rh[0])):
        if np.abs(a).sum() > 0 and np.abs(b).sum() > 0:
            e = max(e, float(np.linalg.norm(b[:, :2].mean(0) - a[:, :2].mean(0))) / width)
    return e, up


@dataclass
class Segment:
    clip: np.ndarray   # (T, 714) raw frames of one sentence
    start: int         # index of its first frame in the decimated stream
    end: int           # one past its last frame


@dataclass
class Segmenter:
    cfg: SegmenterConfig = field(default_factory=SegmenterConfig)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.buf: list[np.ndarray] = []      # frames of the open sentence
        self.energies: list[float] = []
        self.buf_start = 0                    # stream index of buf[0]
        self.pre: list[np.ndarray] = []
        self.active = False
        self.quiet = 0
        self.ema = 0.0
        self.prev = None
        self.last_t = None
        self.n = 0                            # frames accepted so far

    def push(self, frame: np.ndarray, t: float | None = None) -> list[Segment]:
        """Feed one raw frame (714,). `t` in seconds; frames closer than 1/fps
        are skipped (decimation). Returns finished sentences (usually none)."""
        c = self.cfg
        if t is not None:
            if self.last_t is not None and t - self.last_t < 1.0 / c.fps - 1e-3:
                return []
            self.last_t = t
        frame = np.asarray(frame, np.float32)
        e, up = frame_energy(self.prev, frame, c.rest_y)
        self.prev = frame
        self.ema = c.smooth * self.ema + (1 - c.smooth) * e
        idx = self.n
        self.n += 1
        signing = up and self.ema > (c.energy_off if self.active else c.energy_on)
        if not self.active:
            self.pre = (self.pre + [frame])[-max(1, int(c.pre_s * c.fps)):]
            if signing:
                self.active, self.quiet = True, 0
                self.buf = list(self.pre)
                self.energies = [0.0] * len(self.buf)
                self.buf_start = idx - len(self.buf) + 1
            return []
        self.buf.append(frame)
        self.energies.append(self.ema)
        self.quiet = 0 if signing else self.quiet + 1
        out = []
        if self.quiet >= int(c.pause_s * c.fps):
            keep = len(self.buf) - self.quiet + 1          # sentence + one pause frame
            if keep >= int(c.min_s * c.fps):
                out.append(Segment(np.stack(self.buf[:keep]), self.buf_start, self.buf_start + keep))
            self.active, self.quiet = False, 0
            self.pre = self.buf[-max(1, int(c.pre_s * c.fps)):]
            self.buf, self.energies = [], []
        elif len(self.buf) >= int(c.max_s * c.fps):
            half = len(self.energies) // 2
            cut = half + int(np.argmin(self.energies[half:])) + 1
            out.append(Segment(np.stack(self.buf[:cut]), self.buf_start, self.buf_start + cut))
            self.buf, self.energies = self.buf[cut:], self.energies[cut:]
            self.buf_start += cut
        return out

    def flush(self) -> list[Segment]:
        """End of stream: return the open sentence, if long enough."""
        out = []
        if self.active and len(self.buf) >= int(self.cfg.min_s * self.cfg.fps):
            out.append(Segment(np.stack(self.buf), self.buf_start, self.buf_start + len(self.buf)))
        self.active, self.buf, self.energies, self.quiet = False, [], [], 0
        return out


class Translator:
    """Wraps a trained checkpoint for single-clip translation."""

    def __init__(self, checkpoint: str | None = None, device=None, num_beams: int = 4):
        import torch
        from .model import FaSTISL, ModelConfig, load_tokenizer
        from .train import load_model, pick_device
        self.torch = torch
        self.device = device or pick_device()
        self.tok = load_tokenizer()
        if checkpoint:
            self.model, _ = load_model(checkpoint, self.device)
        else:  # untrained model: lets the app run before training is done
            self.model = FaSTISL(ModelConfig()).to(self.device).eval()
        self.num_beams = num_beams
        self.trained = checkpoint is not None

    def translate(self, clip: np.ndarray) -> tuple[str, float]:
        from .features import make_streams
        t0 = time.perf_counter()
        s = make_streams(clip)
        b = {k: self.torch.from_numpy(v)[None].to(self.device) for k, v in s.items()}
        b["mask"] = self.torch.ones(1, len(clip), dtype=self.torch.bool, device=self.device)
        out = self.model.generate(b, num_beams=self.num_beams)
        text = self.tok.decode(out[0], skip_special_tokens=True).strip()
        return text, (time.perf_counter() - t0) * 1000


def segment_stream(stream: np.ndarray, cfg: SegmenterConfig | None = None, segmenter=None) -> list[Segment]:
    """Offline helper: run a segmenter (default: pause-based) over a whole (T, 714) stream at model fps."""
    seg = segmenter or Segmenter(cfg or SegmenterConfig())
    if segmenter is not None:
        seg.reset()
    out = []
    for f in stream:
        out += seg.push(f)
    return out + seg.flush()


def boundary_f1(pred, gold, tol: int = 5) -> dict[str, float]:
    """Sentence-boundary F1: a predicted sentence end matches a gold end within `tol` frames."""
    p_ends = [p.end if isinstance(p, Segment) else p[1] for p in pred]
    g_ends = [e for _, e in gold]
    used, tp = set(), 0
    for pe in p_ends:
        best = min(((abs(pe - ge), j) for j, ge in enumerate(g_ends) if j not in used), default=None)
        if best and best[0] <= tol:
            used.add(best[1])
            tp += 1
    prec = tp / max(1, len(p_ends))
    rec = tp / max(1, len(g_ends))
    return {"precision": 100 * prec, "recall": 100 * rec,
            "f1": 100 * (2 * prec * rec / max(1e-9, prec + rec)), "n_pred": len(p_ends), "n_gold": len(g_ends)}
