"""
Convert iSign's ready-made MediaPipe Holistic pose files into compact keypoint shards.

The pose zip (≈170 GB, split into part_aa..part_ad) is streamed straight from
Hugging Face and unzipped on the fly, so nothing large is ever written to disk.
Each .pose file is reduced to the same per-frame layout as isign_pipeline.py:
    pose 33 x (x, y, z, visibility) + face subset x (x, y, z) + 2 hands x 21 x (x, y, z)
with x, y normalised to 0..1 (like a live MediaPipe webcam result).

Shards are .npz files of SHARD_SIZE clips: {uid: float16 array (frames, FEAT_DIM)}.
Clip order inside the zip never changes, so finished shards are skipped on resume.
"""
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import requests
from pose_format import Pose
from stream_unzip import stream_unzip

POSE_PARTS = ["iSign-poses_v1.1_part_aa", "iSign-poses_v1.1_part_ab",
              "iSign-poses_v1.1_part_ac", "iSign-poses_v1.1_part_ad"]
REPO = "Exploration-Lab/iSign"
N_FACE_MESH = 468  # iSign pose files have the 468-point face mesh (no iris points)


# ---------------------------------------------------------------- byte sources

def http_chunks(url, token, size, chunk=8 << 20, max_retries=50):
    """Yields the bytes of one remote file; on a dropped connection it resumes
    from the last byte received instead of starting over."""
    pos, retries = 0, 0
    while pos < size:
        try:
            headers = {"Authorization": f"Bearer {token}", "Range": f"bytes={pos}-"}
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                r.raise_for_status()
                for c in r.iter_content(chunk):
                    pos += len(c)
                    yield c
        except (requests.RequestException, ConnectionError) as e:
            retries += 1
            if retries > max_retries:
                raise
            print(f"  connection dropped at {pos / 1e9:.1f} GB ({e.__class__.__name__}), resuming...")
            time.sleep(5)


def hf_part_chunks(token):
    from huggingface_hub import HfApi, hf_hub_url
    info = HfApi(token=token).dataset_info(REPO, files_metadata=True)
    sizes = {s.rfilename: s.size for s in info.siblings}
    for name in POSE_PARTS:
        print(f"streaming {name} ({sizes[name] / 1e9:.1f} GB)", flush=True)
        yield from http_chunks(hf_hub_url(REPO, name, repo_type="dataset"), token, sizes[name])


def file_part_chunks(paths, chunk=8 << 20):
    """Same as hf_part_chunks but from local files (for testing)."""
    for p in paths:
        with open(p, "rb") as f:
            while c := f.read(chunk):
                yield c


# ---------------------------------------------------------------- conversion

def face_indices():
    """Lips, eyes, eyebrows, face oval and nose: the face points that carry ISL
    non-manual markers. Iris points (468+) are not in iSign's pose files."""
    try:
        from mediapipe.tasks.python.vision import FaceLandmarksConnections as F
    except ImportError:  # same list, saved in the repo (islt/face_regions.json)
        here = Path(__file__).resolve().parent
        for p in (here / "face_regions.json", here.parent / "islt" / "face_regions.json"):
            if p.exists():
                return json.loads(p.read_text())["face_mediapipe_indices"]
        raise
    idx = set()
    for group in (F.FACE_LANDMARKS_CONTOURS, F.FACE_LANDMARKS_NOSE):
        for c in group:
            idx.update((c.start, c.end))
    return sorted(i for i in idx if i < N_FACE_MESH)


def layout(face_idx, target_fps):
    face_start = 33 * 4
    lh_start = face_start + len(face_idx) * 3
    return {
        "pose": {"points": 33, "values": ["x", "y", "z", "visibility"], "start": 0},
        "face": {"points": len(face_idx), "values": ["x", "y", "z"], "start": face_start,
                 "mediapipe_indices": face_idx},
        "left_hand": {"points": 21, "values": ["x", "y", "z"], "start": lh_start},
        "right_hand": {"points": 21, "values": ["x", "y", "z"], "start": lh_start + 63},
        "feat_dim": lh_start + 126,
        "target_fps": target_fps,
        "coords": "x, y normalised to 0..1 by frame width/height; z as stored by MediaPipe",
        "missing": "all zeros when a part is not detected",
        "source": "iSign v1.1 pose files (MediaPipe Holistic)",
    }


def pose_to_array(raw, face_idx, target_fps):
    pose = Pose.read(raw)
    offsets, start = {}, 0
    for c in pose.header.components:
        offsets[c.name] = start
        start += len(c.points)

    w, h = pose.header.dimensions.width, pose.header.dimensions.height
    xyz = np.ma.filled(pose.body.data[:, 0], 0).astype(np.float32)   # (T, points, 3)
    conf = np.asarray(pose.body.confidence[:, 0], np.float32)        # (T, points)
    xyz[..., 0] /= w
    xyz[..., 1] /= h

    # keep frames at target_fps (same picking rule as the video pipeline)
    step = max(float(pose.body.fps) / target_fps, 1.0)
    picks = np.unique(np.ceil(np.arange(0, len(xyz), step)).astype(int))
    picks = picks[picks < len(xyz)]
    xyz, conf = xyz[picks], conf[picks]

    def part(name, idx):
        return xyz[:, offsets[name] + np.asarray(idx)].reshape(len(xyz), -1)

    o = offsets["POSE_LANDMARKS"]
    pose_vis = np.concatenate([xyz[:, o:o + 33], conf[:, o:o + 33, None]], axis=2).reshape(len(xyz), -1)
    return np.concatenate([
        pose_vis,
        part("FACE_LANDMARKS", face_idx),
        part("LEFT_HAND_LANDMARKS", range(21)),
        part("RIGHT_HAND_LANDMARKS", range(21)),
    ], axis=1).astype(np.float16)


def _save_shard(clips, out, tmp_dir):
    # write locally, copy to a .partial name, then rename: a disconnect never
    # leaves a half-written shard that would be skipped on resume
    local = Path(tmp_dir) / out.name
    np.savez_compressed(local, **clips)
    partial = out.with_suffix(".partial")
    shutil.copyfile(local, partial)
    os.replace(partial, out)
    local.unlink()


def run(chunks, out_dir, target_fps=10, shard_size=500, limit=None, tmp_dir="/tmp/isign_pose_work"):
    out_dir, tmp_dir = Path(out_dir), Path(tmp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    face_idx = face_indices()
    (out_dir / "layout.json").write_text(json.dumps(layout(face_idx, target_fps), indent=1))
    done = {int(p.stem.split("_")[1]) for p in out_dir.glob("shard_*.npz")}
    print(f"{len(done)} shards already done, they will be skipped", flush=True)

    t0, n, shard, clips, failed = time.time(), 0, 0, {}, 0
    for name, _, member in stream_unzip(chunks):
        name = name.decode()
        if not name.endswith(".pose"):
            for _ in member:  # every member must be read through to move the stream on
                pass
            continue
        shard = n // shard_size
        raw = b"".join(member)
        if shard not in done:
            try:
                clips[Path(name).stem] = pose_to_array(raw, face_idx, target_fps)
            except Exception as e:  # a broken file should not stop the whole run
                print(f"  {name}: {e}")
                failed += 1
        n += 1

        if n % shard_size == 0:
            if shard not in done:
                _save_shard(clips, out_dir / f"shard_{shard:05d}.npz", tmp_dir)
                print(f"shard {shard:05d} saved | {n} clips read | {failed} failed | "
                      f"{n / (time.time() - t0):.0f} clips/s", flush=True)
            clips = {}
        if limit and n >= limit:
            clips = {}  # never save a cut-off shard: it would be skipped on a later, bigger run
            break

    if clips:  # the last, smaller shard at the very end of the dataset
        _save_shard(clips, out_dir / f"shard_{shard:05d}.npz", tmp_dir)
    print(f"finished: {n} clips, {failed} failed, {(time.time() - t0) / 60:.0f} min", flush=True)
