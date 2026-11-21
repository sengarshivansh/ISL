"""
Per-frame feature layout and normalisation, shared by training, evaluation and
the live demo, so the model always sees the same numbers.

Raw frame vector (714 values, written by colab/isign_pose_convert.py):
    pose 33 x (x, y, z, visibility) | face 152 x (x, y, z) | left hand 21 x 3 | right hand 21 x 3
x, y are 0..1 image coordinates; a part that MediaPipe did not detect is all zeros.

`make_streams` turns a (T, 714) clip into separate, normalised streams:
    body   (T, 25*2)  upper body, centred on the shoulder midpoint, scaled by shoulder width
    hands  (T, 2*21*2*2) each hand twice: in body coordinates and relative to its own wrist
    face   (T, 152*2) face points relative to the face centre, scaled by face width
    head   (T, 6)     head pose proxies: face centre (x, y) and scale vs. shoulders,
                      roll (eye line angle), yaw and pitch proxies (nose offset)
    present (T, 4)    1/0 flags: body, left hand, right hand, face detected
Face regions (mouth, eyes, eyebrows, oval, nose) select subsets of `face` for ablations.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

N_POSE, N_FACE, N_HAND = 33, 152, 21
POSE_DIM = N_POSE * 4
FACE_START = POSE_DIM
LH_START = FACE_START + N_FACE * 3
RH_START = LH_START + N_HAND * 3
FEAT_DIM = RH_START + N_HAND * 3  # 714

UPPER_BODY = list(range(25))  # nose .. hips; legs are rarely in frame for signers
L_SHOULDER, R_SHOULDER = 11, 12

_REGIONS = json.loads((Path(__file__).with_name("face_regions.json")).read_text())
FACE_MP_INDICES: list[int] = _REGIONS["face_mediapipe_indices"]
FACE_REGIONS: dict[str, list[int]] = _REGIONS["regions"]  # positions inside the 152-point subset
# "head" = rigid face outline + nose: carries head pose (nods, shakes, tilts)
FACE_REGIONS["head"] = sorted(FACE_REGIONS["oval"] + FACE_REGIONS["nose"])
_NOSE_TIP = FACE_MP_INDICES.index(1)
_L_EYE_OUT, _R_EYE_OUT = FACE_MP_INDICES.index(263), FACE_MP_INDICES.index(33)

STREAM_DIMS = {
    "body": len(UPPER_BODY) * 2,
    "hands": 2 * N_HAND * 2 * 2,
    "head": 6,
    "present": 4,
}


def face_points(region: str | list[str] = "all") -> list[int]:
    """Indices (0..151) of the face points to keep. region: 'all', 'none', a region name, or a list."""
    if region in ("all", None):
        return list(range(N_FACE))
    if region == "none":
        return []
    names = [region] if isinstance(region, str) else list(region)
    idx: set[int] = set()
    for n in names:
        idx.update(FACE_REGIONS[n])
    return sorted(idx)


def split_raw(x: np.ndarray):
    """(T, 714) -> pose (T,33,4), face (T,152,3), lh (T,21,3), rh (T,21,3)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != FEAT_DIM:
        raise ValueError(f"expected (T, {FEAT_DIM}) keypoints, got {x.shape}")
    t = len(x)
    return (x[:, :POSE_DIM].reshape(t, N_POSE, 4),
            x[:, FACE_START:LH_START].reshape(t, N_FACE, 3),
            x[:, LH_START:RH_START].reshape(t, N_HAND, 3),
            x[:, RH_START:].reshape(t, N_HAND, 3))


def _present(part: np.ndarray) -> np.ndarray:
    return (np.abs(part).sum(axis=(1, 2)) > 0).astype(np.float32)


def _fill_missing(arr: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Carry the last seen value forward (and the first seen value backward) for
    frames where a part was not detected, so normalisation never divides by zero."""
    if present.all() or not present.any():
        return arr
    out = arr.copy()
    idx = np.where(present > 0, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    first = int(np.argmax(present > 0))
    idx[:first] = first
    return out[idx]


def make_streams(raw: np.ndarray, face_region: str | list[str] = "all") -> dict[str, np.ndarray]:
    pose, face, lh, rh = split_raw(raw)
    t = len(pose)
    pres_body = _present(pose[:, :, :3])
    pres_lh, pres_rh, pres_face = _present(lh), _present(rh), _present(face)

    pose_xy = _fill_missing(pose[:, :, :2], pres_body)
    centre = (pose_xy[:, L_SHOULDER] + pose_xy[:, R_SHOULDER]) / 2            # (T, 2)
    width = np.linalg.norm(pose_xy[:, L_SHOULDER] - pose_xy[:, R_SHOULDER], axis=-1)
    width = np.where(width > 1e-4, width, 1.0)[:, None, None]                  # (T,1,1)

    def to_body(p):
        return (p - centre[:, None, :]) / width

    body = to_body(pose_xy[:, UPPER_BODY]).reshape(t, -1)

    hand_feats = []
    for hand, pres in ((lh, pres_lh), (rh, pres_rh)):
        xy = hand[:, :, :2]
        glob = to_body(xy)
        wrist = xy[:, :1]
        size = np.linalg.norm(xy[:, 9] - xy[:, 0], axis=-1)                    # wrist -> middle MCP
        size = np.where(size > 1e-4, size, 1.0)[:, None, None]
        local = (xy - wrist) / size
        f = np.concatenate([glob, local], axis=-1) * pres[:, None, None]
        hand_feats.append(f.reshape(t, -1))
    hands = np.concatenate(hand_feats, axis=-1)

    fxy = _fill_missing(face[:, :, :2], pres_face)
    fc = fxy.mean(axis=1)                                                     # face centre (T,2)
    eye_vec = fxy[:, _L_EYE_OUT] - fxy[:, _R_EYE_OUT]
    fw = np.linalg.norm(eye_vec, axis=-1)
    fw = np.where(fw > 1e-4, fw, 1.0)
    face_local = (fxy - fc[:, None]) / fw[:, None, None]
    keep = face_points(face_region)
    face_feat = (face_local[:, keep] * pres_face[:, None, None]).reshape(t, -1)

    nose_off = (fxy[:, _NOSE_TIP] - fc) / fw[:, None]
    head = np.stack([
        (fc[:, 0] - centre[:, 0]) / width[:, 0, 0],
        (fc[:, 1] - centre[:, 1]) / width[:, 0, 0],
        fw / width[:, 0, 0],
        np.arctan2(eye_vec[:, 1], eye_vec[:, 0]),   # roll
        nose_off[:, 0],                              # yaw proxy
        nose_off[:, 1],                              # pitch proxy
    ], axis=-1) * pres_face[:, None]

    present = np.stack([pres_body, pres_lh, pres_rh, pres_face], axis=-1)
    out = {"body": body, "hands": hands, "face": face_feat, "head": head, "present": present}
    return {k: np.nan_to_num(np.clip(v, -10, 10)).astype(np.float32) for k, v in out.items()}


def resample_speed(raw: np.ndarray, speed: float) -> np.ndarray:
    """Replay a clip `speed` times faster (>1) or slower (<1) by linear
    interpolation between frames. Used both for augmentation and for the
    fast-signing test sets. Missing parts stay zero instead of being blended."""
    raw = np.asarray(raw, dtype=np.float32)
    t = len(raw)
    new_t = max(2, int(round(t / speed)))
    if t < 2 or new_t == t:
        return raw.copy()
    pos = np.linspace(0, t - 1, new_t)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (pos - lo)[:, None].astype(np.float32)
    a, b = raw[lo], raw[hi]
    out = a * (1 - w) + b * w
    # a part missing in either neighbour frame -> take the nearer frame as-is
    near = np.where(w[:, 0] < 0.5, lo, hi)
    zero_mix = ((a == 0) | (b == 0))
    out[zero_mix] = raw[near][zero_mix]
    return out
