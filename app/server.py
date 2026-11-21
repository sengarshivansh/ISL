"""
FaST-ISL demo server.

    .venv/bin/python -m app.server                      # untrained: shows segmentation only
    ISL_CHECKPOINT=runs/FaST/best.pt .venv/bin/python -m app.server
    open http://localhost:8000

WebSocket /ws protocol
  client -> server: binary = 8-byte little-endian float64 timestamp (s) + JPEG bytes
  server -> client: {"type": "frame", "landmarks": {...}, "signing": bool, "energy": float}
                    {"type": "caption", "text": str, "latency_ms": float, "duration_s": float}
POST /api/video (multipart file) -> {"sentences": [{"start_s", "end_s", "text", "latency_ms"}]}
"""
from __future__ import annotations

import asyncio
import os
import struct
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from islt.holistic import HolisticExtractor, vector_to_draw, video_to_keypoints
from islt.stream import Segmenter, SegmenterConfig, frame_energy, segment_stream


def make_segmenter():
    """Learned boundary detector if available (robust to fast signing and no pauses),
    otherwise the pause-based rule."""
    if BOUNDARY:
        from islt.boundary import LearnedSegmenter
        return LearnedSegmenter(BOUNDARY, fps=FPS)
    return Segmenter(SegmenterConfig(fps=FPS))

STATIC = Path(__file__).parent / "static"
CHECKPOINT = os.environ.get("ISL_CHECKPOINT")
BOUNDARY = os.environ.get("ISL_BOUNDARY")  # learned sentence-boundary model (islt/boundary.py)
FPS = 10.0

app = FastAPI(title="FaST-ISL")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
_translator = None
_model_lock = asyncio.Lock()


def translator():
    global _translator
    if _translator is None and CHECKPOINT:
        from islt.stream import Translator
        _translator = Translator(CHECKPOINT)
    return _translator


async def translate(clip: np.ndarray) -> tuple[str, float, bool]:
    tr = translator()
    if tr is None:
        return "", 0.0, False
    async with _model_lock:  # one model call at a time
        text, ms = await asyncio.to_thread(tr.translate, clip)
    return text, ms, True


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return {"model_loaded": bool(CHECKPOINT), "checkpoint": CHECKPOINT, "fps": FPS,
            "segmenter": "learned" if BOUNDARY else "pause"}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    extractor = await asyncio.to_thread(HolisticExtractor)
    seg = await asyncio.to_thread(make_segmenter)
    t0, prev, motion = None, None, 0.0
    try:
        while True:
            msg = await sock.receive_bytes()
            ts = struct.unpack("<d", msg[:8])[0]
            t0 = ts if t0 is None else t0
            img = cv2.imdecode(np.frombuffer(msg[8:], np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vec = await asyncio.to_thread(extractor, rgb, int((ts - t0) * 1000))
            done = await asyncio.to_thread(seg.push, vec, ts - t0)
            e, up = frame_energy(prev, vec)
            prev = vec
            motion = 0.6 * motion + 0.4 * e  # for the on-screen signing indicator only
            await sock.send_json({"type": "frame", "landmarks": vector_to_draw(vec),
                                  "signing": bool(up and motion > 0.03), "energy": round(float(motion), 4)})
            for s in done:
                await sock.send_json({"type": "translating"})
                text, ms, trained = await translate(s.clip)
                await sock.send_json({"type": "caption", "text": text, "latency_ms": round(ms),
                                      "duration_s": round(len(s.clip) / FPS, 1), "trained": trained})
    except WebSocketDisconnect:
        pass
    finally:
        extractor.close()


@app.post("/api/video")
async def video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        t0 = time.perf_counter()
        kps = await asyncio.to_thread(video_to_keypoints, path, FPS)
        segs = segment_stream(kps, SegmenterConfig(fps=FPS), make_segmenter() if BOUNDARY else None)
        hands_seen = (np.abs(kps[:, 588:]).sum(1) > 0).mean() if len(kps) else 0.0
        if not segs and len(kps) >= FPS and hands_seen > 0.2:  # one pre-cut sentence clip
            from islt.stream import Segment
            segs = [Segment(kps, 0, len(kps))]
        out = []
        for s in segs:
            text, ms, trained = await translate(s.clip)
            out.append({"start_s": round(s.start / FPS, 1), "end_s": round(s.end / FPS, 1),
                        "text": text, "latency_ms": round(ms), "trained": trained})
        return JSONResponse({"frames": len(kps), "sentences": out,
                             "total_s": round(time.perf_counter() - t0, 1)})
    finally:
        os.unlink(path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
