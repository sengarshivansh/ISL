"""
A synthetic "toy sign language" in the exact 714-value iSign keypoint layout.

It exists to test the whole pipeline end to end without the real dataset, and
it is built so that some meaning is carried ONLY by the face, as in ISL:
  * each word = a word-specific right-hand path + hand shape (manual channel)
  * yes/no question  -> eyebrows raised for the whole sentence, text ends in "?"
  * negation         -> head shakes left-right, text gets "not" (no hand sign for it)
A body+hands-only model cannot tell "you go home" from "you go home ?" or
"you not go home"; a face-aware model can. Results on this data are a sanity
check of the code, never a paper result.
"""
from __future__ import annotations

import hashlib
import random

import numpy as np

from .features import FACE_REGIONS, FEAT_DIM, FACE_START, LH_START, N_FACE, RH_START

VOCAB = ["i", "you", "we", "go", "come", "eat", "want", "like", "home", "school",
         "water", "food", "friend", "today", "tomorrow", "help", "work", "book", "doctor", "happy"]
SUBJECTS, VERBS = VOCAB[:3], VOCAB[3:8]
OBJECTS = VOCAB[8:14] + VOCAB[17:19]
EXTRAS = ["today", "tomorrow", "help", "work", "happy"]


def _word_params(word: str):
    r = np.random.default_rng(int(hashlib.md5(word.encode()).hexdigest()[:8], 16))
    return {"centre": r.uniform([0.42, 0.45], [0.62, 0.70]),
            "radius": r.uniform(0.02, 0.07, 2),
            "freq": r.integers(1, 4),
            "phase": r.uniform(0, 2 * np.pi),
            "spread": r.uniform(0.3, 1.0, 5)}


_FACE_RNG = np.random.default_rng(7)
_FACE_TEMPLATE = _FACE_RNG.uniform(-1, 1, (N_FACE, 2)) * np.array([0.04, 0.05])
_BROWS = np.array(FACE_REGIONS["eyebrows"])
_HAND_TEMPLATE = np.concatenate([[[0, 0]], *[
    np.stack([np.full(4, (f - 2) * 0.012), -np.arange(1, 5) * 0.012], 1) for f in range(5)]])


def _hand(centre, spread):
    pts = _HAND_TEMPLATE.copy()
    for f in range(5):
        pts[1 + 4 * f: 5 + 4 * f] *= np.array([1.0, spread[f]])
    return pts + centre


def make_sentence(rng: random.Random):
    words = [rng.choice(SUBJECTS), rng.choice(VERBS), rng.choice(OBJECTS)]
    if rng.random() < 0.4:
        words.append(rng.choice(EXTRAS))
    kind = rng.choices(["statement", "question", "negation"], [0.4, 0.3, 0.3])[0]
    return words, kind


def sentence_text(words, kind):
    if kind == "negation":
        return " ".join(words[:1] + ["not"] + words[1:])
    if kind == "question":
        return " ".join(words) + " ?"
    return " ".join(words)


def render(words, kind, rng: random.Random, frames_per_word=8, noise=0.002) -> np.ndarray:
    nr = np.random.default_rng(rng.randrange(1 << 30))
    pieces = []
    for w in words:
        p = _word_params(w)
        n = max(3, int(round(frames_per_word * rng.uniform(0.8, 1.25))))
        t = np.linspace(0, 1, n)
        ang = 2 * np.pi * p["freq"] * t + p["phase"]
        path = p["centre"] + np.stack([np.cos(ang), np.sin(ang)], 1) * p["radius"]
        pieces += [(xy, p["spread"]) for xy in path]
    rest = (np.array([0.62, 0.85]), np.ones(5))
    seq = [rest] * 3 + pieces + [rest] * 3
    t_total = len(seq)
    x = np.zeros((t_total, FEAT_DIM), np.float32)
    face_c = np.array([0.5, 0.28])
    for i, (rh_xy, spread) in enumerate(seq):
        pose = np.zeros((33, 4))
        pose[:, :2] = [0.5, 0.6]
        pose[0, :2] = face_c
        pose[11, :2], pose[12, :2] = [0.62, 0.48], [0.38, 0.48]
        pose[13, :2], pose[14, :2] = [0.66, 0.62], [0.34, 0.62]
        pose[16, :2] = rh_xy
        pose[15, :2] = [0.38, 0.85]
        pose[:, 3] = 1.0
        fc = face_c.copy()
        if kind == "negation":
            fc[0] += 0.012 * np.sin(2 * np.pi * i / 6)
        face = _FACE_TEMPLATE + fc
        if kind == "question":
            face[_BROWS, 1] -= 0.012
        lh = _hand(np.array([0.38, 0.85]), np.ones(5))
        rh = _hand(rh_xy, spread)
        x[i, :132] = pose.reshape(-1)
        x[i, FACE_START:LH_START] = np.concatenate([face, np.zeros((N_FACE, 1))], 1).reshape(-1)
        x[i, LH_START:RH_START] = np.concatenate([lh, np.zeros((21, 1))], 1).reshape(-1)
        x[i, RH_START:] = np.concatenate([rh, np.zeros((21, 1))], 1).reshape(-1)
    mask = x != 0
    x = x + nr.normal(0, noise, x.shape).astype(np.float32) * mask
    return x


def make_dataset(n: int, seed: int = 0):
    """Returns (items [(uid, text)], clips {uid: (T, 714)})."""
    rng = random.Random(seed)
    items, clips = [], {}
    for i in range(n):
        words, kind = make_sentence(rng)
        uid = f"syn{seed}_{i // 4}-{i % 4}"
        clips[uid] = render(words, kind, rng)
        items.append((uid, sentence_text(words, kind)))
    return items, clips


def make_stream(sentences, rng: random.Random, pause_frames=(8, 14)):
    """Concatenate several sentences into one continuous stream with pauses,
    like a person signing several sentences in a row. Returns (stream, texts, boundaries)."""
    parts, texts, bounds, t = [], [], [], 0
    for words, kind in sentences:
        clip = render(words, kind, rng)
        parts.append(clip)
        texts.append(sentence_text(words, kind))
        bounds.append((t, t + len(clip)))
        t += len(clip)
        rest = np.repeat(clip[-1:], rng.randint(*pause_frames), axis=0)
        parts.append(rest)
        t += len(rest)
    return np.concatenate(parts), texts, bounds
