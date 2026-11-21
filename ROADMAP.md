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
| 1 | Shared feature layout + normalisation + face regions | `islt/features.py` | No | ⏳ |
| 2 | Dataset loader (shards → tensors), signer-safe split, speed augmentation | `islt/data.py` | No (tested on synthetic) | ⏳ |
| 3 | Model: multi-stream encoder + gated face fusion + T5 decoder; a lighter from-scratch decoder option | `islt/model.py` | No | ⏳ |
| 4 | Training + evaluation (BLEU-4, chrF, ROUGE-L, speed-perturbed test sets) | `islt/train.py`, `islt/evaluate.py` | Runs on synthetic; real run needs data | ⏳ |
| 5 | Ablation configs (no-face / mouth / eyes / eyebrows / head / all; ± speed aug) | `configs/*.yaml`, `scripts/run_ablations.sh` | Real run needs data | ⏳ |
| 6 | Streaming segmenter + real-time translator + TTS | `islt/stream.py` | No | ⏳ |
| 7 | Demo web app: browser MediaPipe → WebSocket → model → captions + voice | `app/` | No (works with untrained model) | ⏳ |
| 8 | Colab GPU training notebook | `colab/train_fastisl.ipynb` | Needs iSign keypoints on Drive | ⏳ |
| 9 | Paper draft (ACL format): intro, related work, method, experimental setup; result tables left as TODO | `paper/` | No | ⏳ |
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
