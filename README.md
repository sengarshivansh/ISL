# FaST-ISL

**Face-aware, Speed-robust, Streaming translation of continuous Indian Sign Language (ISL) into English text and speech.**

A person signs continuously in front of a camera. The system finds where each sentence ends and translates it to English. It then reads the sentence aloud.

Three things make it different from current ISL translators:

| | Current ISL systems (WSLP 2026, SignFlow, …) | FaST-ISL |
|---|---|---|
| Face and head | Dropped: body and hand keypoints only | Gated face stream (mouth, eyes, eyebrows) and a separate head-pose stream |
| Signing speed | One speed, never tested faster | Multi-rate encoder, 0.5–2× speed augmentation, tests at 1.5× and 2× |
| Continuous signing | Pre-cut clips | Learned sentence-boundary detector that works at 2× speed and with no pauses, plus a live web demo with speech |

The plan and current status are in [`ROADMAP.md`](ROADMAP.md). The literature review is in [`RESEARCH_GAP_ANALYSIS.md`](RESEARCH_GAP_ANALYSIS.md). The paper draft is in [`paper/`](paper/).

## Layout

```
islt/            the library
  features.py    keypoint layout, normalisation, face regions, speed resampling
  data.py        iSign shard loader, video-disjoint split, augmentation, batching
  model.py       FaST-ISL: streams -> gated fusion -> multi-rate conv -> Transformer -> T5
  train.py       training + evaluation (python -m islt.train --config ...)
  metrics.py     BLEU-4, chrF, ROUGE-L, question/negation accuracy
  stream.py      pause-based streaming segmenter + translator
  boundary.py    learned sentence-boundary detector (fast / no-pause signing)
  eval_stream.py continuous-signing evaluation (boundary F1, stream BLEU, latency)
  holistic.py    live MediaPipe Holistic -> model input
  synthetic.py   toy sign language for testing the pipeline without iSign
  report.py      results -> LaTeX/Markdown tables
app/             demo website (FastAPI + browser webcam)
configs/         experiment configs; configs/ablations/ = the paper's experiment matrix
colab/           data conversion and GPU training notebooks
paper/           ACL-format paper draft (cd paper && tectonic main.tex)
tests/           unit tests (python -m pytest)
```

## Quick start (Mac)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                                   # 17 tests
.venv/bin/python -m islt.train --config configs/synthetic.yaml   # pipeline check on toy data (~5 min)
.venv/bin/python -m app.server                               # demo at http://localhost:8000
```

The demo runs before any model is trained. In that mode it detects and times each sentence but does not translate it. After training, run it with the real model:

```bash
ISL_CHECKPOINT=runs/FaST/best.pt ISL_BOUNDARY=runs/boundary/boundary.pt .venv/bin/python -m app.server
```

## Real experiments (iSign)

1. Accept the iSign terms on Hugging Face (`Exploration-Lab/iSign`). Add your token as `HF_TOKEN` in Colab secrets.
2. Run `colab/isign_convert_poses.ipynb`. It streams the 170 GB of pose files and writes about 127k small keypoint clips to Google Drive.
3. Run `colab/train_fastisl.ipynb` on a GPU. It trains all 9 configurations. Runs resume after a disconnect.
4. Copy `runs/` back. Then run `python -m islt.report runs` to rebuild the paper's table.

iSign is released for research use only.
