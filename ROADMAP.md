# ISL Project Roadmap

**Goal:** continuous Indian Sign Language (ISL) from a camera → English text → speech, plus a research paper and a demo website.

**Working title:** *FaST-ISL: Face-aware, Speed-robust, Streaming Translation of Continuous Indian Sign Language*

Every step below ends with a git commit. If a step turns out wrong, `git log` shows the commit just before it, and `git revert <hash>` (or `git checkout <hash> -- <path>`) undoes only that step.

---

## Research gap (checked 21 Nov 2026)

Full analysis: `RESEARCH_GAP_ANALYSIS.md`. The latest check covered all 24 papers in the WSLP 2026 proceedings (aclanthology.org/volumes/2026.wslp-1).

| What current ISL translation systems do | What is missing |
|---|---|
| Use **body and hand keypoints only**: ISLiM (iSign BLEU-4 5.85), Shared-Encoder Contrastive Grounding, Dhamecha et al., SignFlow | **The face.** ISL marks questions, negation and emphasis with the face and head. No WSLP 2026 ISL paper uses face features. |
| Train and test at one natural signing speed | **Robustness to fast signing.** No ISL paper measures what happens when a person signs faster than the training data. |
| Translate pre-cut clips offline (ISLiM: 384 ms per clip) | **Streaming.** No continuous ISL→English system cuts a live, unsegmented stream into sentences and translates as you sign. |

**Our three contributions**

1. **Face-aware fusion.** A gated face stream (mouth, eyes, eyebrows, head pose) added to the body and hand streams, plus an ablation of which facial region matters most for ISL.
2. **Speed robustness.**
   - Training: random temporal resampling at 0.5×–2× speed, plus a speed-normalising motion input.
   - Testing: the iSign test set replayed at 1.5× and 2× speed.
   - The target is a +2 % relative or larger improvement over the body + hands baseline on the normal test set, with a much smaller drop on fast signing.
3. **Streaming system.** Motion-energy pause detection splits a live keypoint stream into sentences, and each sentence is translated as soon as it ends.
   - Output is English text plus speech.
   - We report latency and FPS on a laptop.

---

## Ordered plan

| # | Step | Output | Needs data? | Status |
|---|---|---|---|---|
| 0 | Gap check against latest literature | `ROADMAP.md`, gap table | No | ✅ done |
| 1 | Shared feature layout + normalisation + face regions | `islt/features.py` | No | ✅ done |
| 2 | Dataset loader (shards → tensors), signer-safe split, speed augmentation | `islt/data.py` | No (tested on synthetic) | ✅ done |
| 3 | Model: multi-stream encoder + gated face fusion + T5 decoder; a lighter from-scratch decoder option | `islt/model.py` | No | ✅ done |
| 4 | Training + evaluation (BLEU-4, chrF, ROUGE-L, speed-perturbed test sets) | `islt/train.py`, `islt/evaluate.py` | Runs on synthetic; real run needs data | ✅ done |
| 5 | Ablation configs (no-face / mouth / eyes / eyebrows / head / all; ± speed aug) | `configs/*.yaml`, `scripts/run_ablations.sh` | Real run needs data | ✅ done |
| 6 | Streaming segmenter + real-time translator + TTS | `islt/stream.py` | No | ✅ done |
| 7 | Demo web app: browser MediaPipe → WebSocket → model → captions + voice | `app/` | No (works with untrained model) | ✅ done |
| 8 | Colab GPU training notebook | `colab/train_fastisl.ipynb` | Needs iSign keypoints on Drive | ✅ done |
| 9 | Paper draft (ACL format): intro, related work, method, experimental setup; result tables left as TODO | `paper/` | No | ✅ done |
| 6b | Learned sentence-boundary detector (fast and no-pause signing) | `islt/boundary.py` | Tested on synthetic; iSign run in Colab | ✅ done |
| 10 | **Real experiments** → fill tables → final paper | results, figures | **Yes** | 🔒 blocked: see below |

### Blocker for step 10

The iSign dataset on Hugging Face is gated, and the HF token stored on this Mac is **invalid** (401). To unblock, do one of these:

- **On the Mac:** run `! .venv/bin/hf auth login --force` with a token from huggingface.co/settings/tokens. Your account must have accepted the iSign terms.
- **On Colab (recommended, no 170 GB download on home internet):** run `colab/isign_convert_poses.ipynb` with `HF_TOKEN` set in Colab secrets. Then run `colab/train_fastisl.ipynb`.

---

## Experiment matrix (step 10)

| ID | Body | Hands | Face | Speed aug | Purpose |
|---|---|---|---|---|---|
| B0 | ✓ | ✓ | – | – | Baseline (the WSLP-style setup) |
| B1 | ✓ | ✓ | – | ✓ | Effect of speed augmentation alone |
| F-mouth | ✓ | ✓ | mouth | ✓ | Region ablation |
| F-eyes | ✓ | ✓ | eyes | ✓ | Region ablation |
| F-brows | ✓ | ✓ | eyebrows | ✓ | Region ablation |
| F-head | ✓ | ✓ | oval + nose (head pose) | ✓ | Region ablation |
| **FaST** | ✓ | ✓ | all, gated | ✓ | Full model |
| FaST-nogate | ✓ | ✓ | all, concat | ✓ | Is the gate needed? |

Every model is evaluated on the iSign test set at **1.0×, 1.5× and 2.0×** speed, and on the question/negation subset (sentences ending in `?`, or containing *not / no / never / don't*).

---

## Results so far: synthetic sanity check (21 Nov 2026)

These numbers come from a toy sign language (`islt/synthetic.py`). In it, questions are marked only by the eyebrows and negation only by a head shake. They show the code does what it should. **They are not ISL results.**

| Model | BLEU-4 | BLEU-4 @2× speed | Question acc | Negation acc |
|---|---|---|---|---|
| Body + hands (baseline) | 68.5 | 16.1 | 65% | 69% |
| + speed aug, multi-rate | 68.7 | 53.7 | 68% | 69% |
| + gated face/head (no speed aug) | 99.7 | 28.1 | 100% | 100% |
| **FaST-ISL (full)** | **98.6** | **83.3** | **100%** | **100%** |

**Continuous signing.** Test sentences were joined 4 at a time into one stream.

| Pause between sentences | Speed | Boundary F1, pause rule | Boundary F1, learned detector |
|---|---|---|---|
| 0.8 s | 1× | 90 | 100 |
| 0.8 s | 2× | 40 | 100 |
| none | 2× | 40 | 100 |

The pause rule breaks as soon as signing gets faster. The learned detector does not.

## First real iSign run (Kaggle, 27 Nov 2026)

The baseline (`B0_body_hands`) was trained for 30 epochs, about 20 minutes per epoch on a T4.

| | BLEU-4 | chrF | ROUGE-L | BLEU-4 @1.5× |
|---|---|---|---|---|
| B0 v1 (test, video-disjoint split) | 0.84 | 14.8 | 9.6 | 1.00 |

**Diagnosis.** The predictions (`runs/v1_B0_body_hands/predictions_speed1.0.tsv`) show the model guesses the topic, not the content. A textbook clip gets "Page number 76"; a news clip gets generic news. The score at 1.5× speed is no worse than at normal speed, which also shows the model hardly reads the signing.

**v2 fix** (`configs/isign_base.yaml`):
- Word dropout 0.25 on the decoder input.
- Clip↔sentence contrastive loss (weight 0.3; identical sentences count as positives).
- No repeated 3-grams when decoding.
- 20 epochs.

The Kaggle notebook now trains B0 and FaST side by side on the two T4s. Results go to `runs_v2/`.

## Next (needs the iSign data)

1. Fix the Hugging Face login: `hf auth login`, or set `HF_TOKEN` in Colab secrets.
2. In Colab, run `isign_convert_poses.ipynb`, then `train_fastisl.ipynb`.
3. Copy `runs/` back, then run `python -m islt.report runs --out paper/results_table.tex`.
4. Fill in the paper's `\todo{}` items.
5. Check every bibliography entry marked VERIFY.
