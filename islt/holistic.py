"""
Live MediaPipe Holistic -> the same 714-value frame vector as the iSign shards.

Frames must NOT be mirrored before detection: iSign's left/right hand labels come
from un-mirrored video. Mirror only what you display.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from .features import FACE_MP_INDICES, FEAT_DIM

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
             "holistic_landmarker/float16/latest/holistic_landmarker.task")
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "holistic_landmarker.task"


def ensure_model(path: Path = DEFAULT_MODEL) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading MediaPipe Holistic model to {path} ...", flush=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def result_to_vector(result) -> np.ndarray:
    v = np.zeros(FEAT_DIM, np.float32)
    if result.pose_landmarks:
        v[:132] = np.array([[p.x, p.y, p.z, p.visibility or 0.0]
                            for p in result.pose_landmarks[:33]]).reshape(-1)
    if result.face_landmarks:
        f = result.face_landmarks
        v[132:132 + len(FACE_MP_INDICES) * 3] = np.array(
            [[f[i].x, f[i].y, f[i].z] for i in FACE_MP_INDICES]).reshape(-1)
    start = 132 + len(FACE_MP_INDICES) * 3
    for k, hand in enumerate((result.left_hand_landmarks, result.right_hand_landmarks)):
        if hand:
            v[start + 63 * k: start + 63 * (k + 1)] = np.array([[p.x, p.y, p.z] for p in hand]).reshape(-1)
    return v


def vector_to_draw(v: np.ndarray) -> dict[str, list]:
    """Compact x, y lists for drawing an overlay in the browser."""
    out = {}
    pose = v[:132].reshape(33, 4)[:, :2]
    if np.abs(pose).sum():
        out["pose"] = pose[:25].round(4).tolist()
    face = v[132:132 + 456].reshape(152, 3)[:, :2]
    if np.abs(face).sum():
        out["face"] = face.round(4).tolist()
    for k, name in enumerate(("left_hand", "right_hand")):
        h = v[588 + 63 * k: 588 + 63 * (k + 1)].reshape(21, 3)[:, :2]
        if np.abs(h).sum():
            out[name] = h.round(4).tolist()
    return out


class HolisticExtractor:
    def __init__(self, model_path: Path | None = None):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (HolisticLandmarker, HolisticLandmarkerOptions,
                                                   RunningMode)
        opts = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path or DEFAULT_MODEL))),
            running_mode=RunningMode.VIDEO)
        self.landmarker = HolisticLandmarker.create_from_options(opts)
        self.last_ts = -1

    def __call__(self, rgb: np.ndarray, ts_ms: int) -> np.ndarray:
        import mediapipe as mp
        ts_ms = max(int(ts_ms), self.last_ts + 1)  # VIDEO mode needs increasing timestamps
        self.last_ts = ts_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        return result_to_vector(self.landmarker.detect_for_video(image, ts_ms))

    def close(self):
        self.landmarker.close()


def video_to_keypoints(path: str, target_fps: float = 10.0, extractor: HolisticExtractor | None = None):
    """Read a video file and return (T, 714) keypoints at target_fps."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(fps / target_fps, 1.0)
    ext = extractor or HolisticExtractor()
    frames, i, next_pick = [], 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= next_pick:
            frames.append(ext(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(i * 1000 / fps)))
            next_pick += step
        i += 1
    cap.release()
    if extractor is None:
        ext.close()
    return np.stack(frames) if frames else np.zeros((0, FEAT_DIM), np.float32)
