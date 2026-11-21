"""
iSign keypoint extraction: reads videos straight out of the split zip parts
(no joining, no unzipping, so disk use stays at ~58 GB) and saves MediaPipe
Holistic keypoints in shards of SHARD_SIZE clips each.

Each shard is a compressed .npz: {uid: float16 array of shape (frames, FEAT_DIM)}.
Finished shards are skipped on the next run, so the job can be resumed after a
Colab disconnect.
"""
import bisect
import io
import json
import os
import shutil
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".avi", ".mov")


class ConcatFile(io.RawIOBase):
    """Treats several byte-split files (part_aa, part_ab, ...) as one seekable file."""

    def __init__(self, paths):
        self.files = [open(p, "rb") for p in paths]
        self.sizes = [os.path.getsize(p) for p in paths]
        self.offsets = [0]
        for s in self.sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)
        self.total = sum(self.sizes)
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.total}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, buf):
        if self.pos >= self.total:
            return 0
        i = bisect.bisect_right(self.offsets, self.pos) - 1
        local = self.pos - self.offsets[i]
        n = min(len(buf), self.sizes[i] - local)
        self.files[i].seek(local)
        data = self.files[i].read(n)
        buf[:len(data)] = data
        self.pos += len(data)
        return len(data)

    def close(self):
        for f in self.files:
            f.close()
        super().close()


def open_split_zip(part_paths):
    return zipfile.ZipFile(io.BufferedReader(ConcatFile(part_paths), buffer_size=1 << 20))


def list_videos(part_paths):
    with open_split_zip(part_paths) as zf:
        return sorted(n for n in zf.namelist() if n.lower().endswith(VIDEO_EXTS))


# ---------------------------------------------------------------- keypoints

def face_indices():
    """Face points that matter for ISL non-manual markers: lips, eyes, eyebrows,
    face oval, nose and irises (160 of MediaPipe's 478)."""
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as F
    idx = set()
    for group in (F.FACE_LANDMARKS_CONTOURS, F.FACE_LANDMARKS_LEFT_IRIS,
                  F.FACE_LANDMARKS_RIGHT_IRIS, F.FACE_LANDMARKS_NOSE):
        for c in group:
            idx.update((c.start, c.end))
    return sorted(idx)


def layout(face_idx):
    """Describes the per-frame feature vector, saved next to the shards."""
    return {
        "pose": {"points": 33, "values": ["x", "y", "z", "visibility"], "start": 0},
        "face": {"points": len(face_idx), "values": ["x", "y", "z"], "start": 33 * 4,
                 "mediapipe_indices": face_idx},
        "left_hand": {"points": 21, "values": ["x", "y", "z"], "start": 33 * 4 + len(face_idx) * 3},
        "right_hand": {"points": 21, "values": ["x", "y", "z"], "start": 33 * 4 + len(face_idx) * 3 + 63},
        "feat_dim": 33 * 4 + len(face_idx) * 3 + 126,
        "missing": "all zeros when a part is not detected",
    }


def frame_vector(result, face_idx):
    def xyz(lms, n, idx=None):
        if not lms:
            return np.zeros(n * 3, np.float32)
        pts = lms if idx is None else [lms[i] for i in idx]
        return np.array([[p.x, p.y, p.z] for p in pts], np.float32).ravel()

    pose = (np.array([[p.x, p.y, p.z, p.visibility or 0.0] for p in result.pose_landmarks], np.float32).ravel()
            if result.pose_landmarks else np.zeros(33 * 4, np.float32))
    return np.concatenate([
        pose,
        xyz(result.face_landmarks, len(face_idx), face_idx),
        xyz(result.left_hand_landmarks, 21),
        xyz(result.right_hand_landmarks, 21),
    ])


# ---------------------------------------------------------------- workers

_W = {}  # per-process state: zip handle, landmarker, config


def _init_worker(part_paths, model_path, target_fps, tmp_dir):
    import mediapipe  # noqa: F401  (import inside the worker process)
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions, RunningMode

    # IMAGE mode: every frame is handled independently, so one landmarker can be
    # reused across videos without tracking state leaking from one clip into the next.
    _W["landmarker"] = HolisticLandmarker.create_from_options(HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path), running_mode=RunningMode.IMAGE))
    _W["zip"] = open_split_zip(part_paths)
    _W["face_idx"] = face_indices()
    _W["target_fps"] = target_fps
    _W["tmp_dir"] = Path(tmp_dir)
    _W["tmp_dir"].mkdir(parents=True, exist_ok=True)


def extract_video(path, landmarker, face_idx, target_fps):
    import mediapipe as mp

    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(src_fps / target_fps, 1.0)
    frames, i, next_pick = [], 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= next_pick:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            frames.append(frame_vector(result, face_idx))
            next_pick += step
        i += 1
    cap.release()
    return np.stack(frames).astype(np.float16) if frames else None


def _process_shard(job):
    shard_id, names, out_dir = job
    out = Path(out_dir) / f"shard_{shard_id:05d}.npz"
    if out.exists():
        return shard_id, 0, 0, 0.0
    t0 = time.time()
    clips, failed = {}, 0
    for name in names:
        uid = Path(name).stem
        tmp = _W["tmp_dir"] / Path(name).name
        try:
            tmp.write_bytes(_W["zip"].read(name))
            arr = extract_video(tmp, _W["landmarker"], _W["face_idx"], _W["target_fps"])
            if arr is None:
                failed += 1
            else:
                clips[uid] = arr
        except Exception as e:  # a broken video should not kill the whole shard
            print(f"[shard {shard_id}] {name}: {e}")
            failed += 1
        finally:
            tmp.unlink(missing_ok=True)
    # write locally, copy to a .partial name, then rename: a disconnect never
    # leaves a half-written shard that would be skipped on resume
    local = _W["tmp_dir"] / out.name
    np.savez_compressed(local, **clips)
    partial = out.with_suffix(".partial")
    shutil.copyfile(local, partial)
    os.replace(partial, out)
    local.unlink()
    return shard_id, len(clips), failed, time.time() - t0


def run(part_paths, model_path, out_dir, target_fps=10, shard_size=500, limit=None,
        workers=None, tmp_dir="/tmp/isign_work"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list_videos(part_paths)
    print(f"{len(names)} videos in the zip")
    if limit:
        names = names[:limit]

    fidx = face_indices()
    (out_dir / "layout.json").write_text(json.dumps({**layout(fidx), "target_fps": target_fps}, indent=1))

    jobs = [(i // shard_size, names[i:i + shard_size], str(out_dir)) for i in range(0, len(names), shard_size)]
    todo = [j for j in jobs if not (out_dir / f"shard_{j[0]:05d}.npz").exists()]
    print(f"{len(jobs)} shards total, {len(jobs) - len(todo)} already done, {len(todo)} to go")
    if not todo:
        return

    workers = workers or os.cpu_count()
    t0, done_clips = time.time(), 0
    with Pool(workers, initializer=_init_worker,
              initargs=(list(map(str, part_paths)), str(model_path), target_fps, tmp_dir)) as pool:
        for k, (sid, n_ok, n_fail, secs) in enumerate(pool.imap_unordered(_process_shard, todo), 1):
            done_clips += n_ok + n_fail
            rate = done_clips / (time.time() - t0)
            left = (len(todo) - k) * shard_size / max(rate, 1e-9)
            print(f"shard {sid:05d}: {n_ok} ok, {n_fail} failed, {secs:.0f}s | "
                  f"{k}/{len(todo)} shards | {rate:.2f} clips/s | ~{left / 3600:.1f} h left", flush=True)
