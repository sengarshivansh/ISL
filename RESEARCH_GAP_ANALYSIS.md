# Research Gap Analysis: Indian Sign Language → Text / Voice

_Last updated: 17 Oct 2026_

## 1. Project goal

Build **Indian Sign Language (ISL) detection software**: a camera watches a signer, and the system outputs **English text** and **spoken voice** (text-to-speech).

- Direction: **sign → text/voice** (recognition and translation).
- Out of scope: text → sign (sign *generation/production*, avatars). Papers on generation are listed only where they support an argument.
- Second deliverable: a **research paper** built on a real, defensible gap.

### Key terms

| Term | Meaning | Output |
|---|---|---|
| ISLR (isolated recognition) | One video clip = one sign | A single word/gloss |
| CSLR (continuous recognition) | A full signed sentence | A sequence of glosses (ISL word order) |
| SLT (translation) | A full signed sentence | A fluent English sentence |

A useful product needs **SLT** (continuous signing → proper English). Isolated recognition alone cannot hold a conversation.

---

## 2. Papers reviewed (36 local + web check)

### 2.1 ISL recognition and translation (core)

| # | Paper | Task | Input | Uses face? | Key result | Main limitation |
|---|---|---|---|---|---|---|
| X_01 | **iSign** (Joshi et al., ACL Findings 2024) | SLT benchmark | Video / MediaPipe pose | Baselines: no | 118k video–sentence pairs. Best baseline BLEU-4 ≈ **1.47** | Translation quality very low. Paper says ISL non-manual markers carry grammar, but baselines don't model them |
| X_03 | **ISLTranslate** (Joshi et al., 2023) | SLT dataset | Video | No | 31k pairs | Small, low scores |
| — | **WSLP 2025**: Roy Chowdhury & Sanyal, *Motion-Aware Modeling* | SLT on iSign | Pose | Not stated | BLEU-4 **8.58** on iSign test (best reported 2025 number) | Still < 10 BLEU |
| — | **PoseStitch-SLT** (EMNLP 2025) | SLT on iSign | Pose | — | BLEU-4 0.55 → 3.43 | Low absolute quality |
| X_07 | **Dhamecha et al.**, WSLP 2026 shared task (arXiv 2609.12993) | SLT on iSign | **33 body + 2×21 hand keypoints only** | **No** | T5-small + motion features best. Ranked 5th on the 2026 leaderboard. Code is public | Face landmarks discarded; single run, single dataset |
| Toward_Real-Time | **SignFlow** (Geetha et al., IEEE Access 2025) | Real-time CSLR | RGB (3D-ResNet) + MediaPipe **Pose + Hand** | **No** | WER 19 on its continuous ISL set | Domain-limited (eRaktkosh FAQ vocabulary). Fails on out-of-vocabulary signs. Outputs glosses, not fluent English |
| s41598 | **ISL-CA** (Nedungadi et al., Sci. Reports 2025) | Real-time ISL chatbot for e-governance | RGB, CNN + ViT | Not explicitly modelled | 97.5% accuracy, 6.53 s mean response, SUS 71.8 | Closed domain. 6.5 s latency is too slow for conversation |
| X_04 | **Srivastava et al.** (Wireless Pers. Comm. 2024) | "Continuous" SLR | MediaPipe Holistic (face + hands + body) | **Yes** (raw landmarks) | 88.23% accuracy | Small self-recorded dataset, LSTM classifier over a fixed set of phrases. Not open translation |
| X_02 | **HWGAT + FDMSE-ISL** (Patra et al., 2024) | Isolated | Upper-body skeleton | No | 2,002 words, 40k videos, 20 deaf signers | Isolated only |
| X_05 | **Kinematic GCN + Elastic Transformer** (Garg et al., Oct 2026) | Isolated | Skeleton | No | 89.8%, 120 classes | Isolated only |
| X_06 | **EfficientSign** (2026) | Alphabet images | RGB static | No | 99.94% on 26 letters | Static fingerspelling only |
| — | **LiST** (Information 2023) | Continuous ISL translation | Full RGB frame (Inception V3 + LSTM) | Implicitly (whole frame) | 91.2% | Small INCLUDE-based sentence set. Face not modelled explicitly. Not real-time at scale |
| X_08 | ASL ↔ ISL with LLMs (2024) | ASL → ISL | Random Forest on gestures | No | — | Weak recognition front end |

### 2.2 General (non-ISL) recognition/translation methods

| # | Paper | Relevance |
|---|---|---|
| X_10 | **TwoStream-SLR/SLT** (NeurIPS 2022) | Strong RGB + keypoint fusion for SLR/SLT (PHOENIX, CSL-Daily) |
| X_11 | **ConSignformer** (2024) | Conformer for CSLR. Same Amrita group as SignFlow |
| X_12 | Signer-invariant Conformer (2025) | Unseen-signer CSLR from pose (Isharah): WER 13.07% |
| X_13 | Consistency + signer removal (Zuo & Mak) | Signer-independent CSLR. Keypoint-guided attention to **face and hands** |
| X_20 | **Signformer** (2024) | 0.57M-parameter SLT, edge-deployable. Not tested on ISL |
| X_22 | **Sign2GPT** (ICLR 2024) | Gloss-free SLT with LLM adapters |
| X_21 | SignGPT (2026) | Body + hand + **face** tokens into an LLM (ASL/DGS) |
| X_26 | **FEA-SLT** (2026) | **Facial-expression-aware gloss-free SLT**. SOTA gloss-free BLEU on PHOENIX14T and CSL-Daily, plus a facial-sensitive evaluation (questions). **Not on ISL** |
| X_30 | **Facial features in SLR** (Pham & Avramidis, 2025) | Mouth is the most useful facial region. Only 6% / 5% / 3% of 2015–2020 SLR work used head / mouth / eyes |
| X_14 | Low-resource SLR/SLT review (2026) | Lists "**non-manual feature neglect**" as one of 7 core low-resource challenges. Facial features cut confusion between look-alike signs from 37% → 11% |
| Surveys | Al-Qurishi 2021; Tao 2024 | No model generalises well enough for commercial deployment. Multimodal input beats single-stream |

### 2.3 Generation papers (out of scope, but supporting evidence)

X_09, X_15–X_19, X_23–X_25, X_27–X_29, Patra IJCAI 2026. These are all text → sign. Several (X_23, X_24, X_28 M3T) argue that **non-manual features are grammatically obligatory** and "cannot be recovered from manual articulators alone" (M3T). That is the same argument, made from the generation side.

---

## 3. Where the field stands for ISL

1. **Data now exists.** iSign (118k sentence pairs) and the WSLP 2025 and 2026 shared tasks make continuous ISL→English translation a benchmarked task.
2. **Translation quality is still very poor.** The best public iSign BLEU-4 is about 8.6, compared with ~25+ on PHOENIX14T. ISL→English is far from solved.
3. **Pose-based pipelines dominate.** These use MediaPipe and the `pose-format` library because they are light, private and real-time friendly. They keep **body and hands**, and most drop the face.
4. **Real-time systems exist but are closed-domain.** SignFlow and ISL-CA work for a fixed government-service vocabulary, not for open conversation.
5. **Face-aware translation is proven elsewhere.** FEA-SLT (CSL/DGS) and SignGPT (ASL/DGS) show gains from facial channels, but nobody has done this on **ISL**.

---

## 4. Research gaps

### Gap 1 (primary): Continuous ISL→English translation ignores the face

- Every real-data continuous ISL recognition/translation system reviewed uses **hands + body only**:
  - SignFlow: Pose + Hand
  - Dhamecha WSLP 2026: 33 body + 42 hand points
  - iSign baselines: no face modelling
- The iSign paper's own linguistic analysis says ISL **non-manual markers** carry grammar: questions, negation, emphasis, role shift.
- The face-aware SLT methods (FEA-SLT, SignGPT) have been evaluated only on Chinese, German and American sign languages.
- The only ISL work that feeds face landmarks (Srivastava et al., X_04) is a small closed-set LSTM classifier, not open translation.
- **Gap:** No study measures how much face/head information helps **continuous ISL→English translation on a real benchmark (iSign)**, or which facial region (mouth vs. eyebrows/eyes vs. head pose) matters most for ISL.

### Gap 2: No real-time, open-vocabulary ISL → text → voice system with measured latency

- SignFlow outputs glosses (not English) and fails on out-of-vocabulary signs.
- ISL-CA takes 6.5 s per response.
- iSign/WSLP systems are offline models with no latency reporting and no speech output.
- **Gap:** An end-to-end camera → English → speech pipeline for ISL, evaluated for both translation quality **and** latency (ms per sentence, FPS on a laptop or phone).

### Gap 3: No ISL evaluation set that tests non-manual grammar

- BLEU cannot tell whether a model got a yes/no question or a negation right.
- FEA-SLT built a facial-sensitive evaluation for CSL. Nothing similar exists for ISL.
- **Gap:** A small, annotated **ISL non-manual test subset** (e.g. questions, negation, emphasis), with targeted metrics such as question-type accuracy and negation accuracy.

### Gap 4: Lightweight / on-device ISL translation

- The efficient models either work on static alphabets only (EfficientSign) or were never tested on ISL (Signformer, 0.57M parameters).
- **Gap:** A compact model (≲ 10M parameters) that runs continuous ISL translation on a CPU or phone.

### Gap 5: Signer-independence for continuous ISL

- Isolated ISL work tests unseen signers (HWGAT, Garg et al.). Continuous ISL translation results rarely report a signer-disjoint split.
- **Gap:** Report ISL translation on unseen signers and diverse recording conditions.

### Gap 6 (practical): Data licensing for real software

- iSign is released for **research use only**. Shipping software needs its own consented data, or a separate licence.
- This shapes the product roadmap, not the paper.

---

## 5. Recommended paper direction

**Working title:** _Face-Aware, Real-Time Continuous Indian Sign Language Translation to Text and Speech_

### Contributions
1. **Model.** A lightweight pose-to-text translator for iSign that fuses **body + hands + face/head** landmarks. The face stream uses a small subset (mouth, eyebrows, eyes, head pose), not all 468 points, with gated or cross-attention fusion.
   - Starting point: Dhamecha et al.'s public T5 + pose encoder (WSLP 2026).
2. **Ablation.** Measure which facial region helps ISL translation: no face vs. mouth vs. eyebrows/eyes vs. head pose vs. all. This follows Pham & Avramidis, but for **continuous ISL translation**.
3. **Non-manual test subset.** A few hundred iSign sentences tagged for questions, negation and emphasis, with targeted accuracy metrics on top of BLEU/chrF.
4. **Real-time system.** Webcam → MediaPipe Holistic → model → English text → TTS. Report end-to-end latency and FPS on commodity hardware.

### Research questions
- **RQ1:** Does adding face/head landmarks improve ISL→English translation on iSign (BLEU-4, chrF, ROUGE-L)?
- **RQ2:** Which facial region contributes most?
- **RQ3:** Do the gains concentrate on sentences with non-manual grammar (questions, negation)?
- **RQ4:** Can the model run in real time on a laptop CPU or a phone?

### Baselines to beat
| Baseline | Number to compare |
|---|---|
| iSign paper baselines | BLEU-4 ≈ 1.47 |
| Roy Chowdhury & Sanyal, WSLP 2025 | BLEU-4 8.58 (iSign test) |
| Dhamecha et al., WSLP 2026 (reproduce with their code) | Same split; body + hands only |

### Metrics
- BLEU-4, chrF, ROUGE-L (shared-task standard)
- Non-manual subset accuracy
- Latency (ms), FPS, parameter count
- Optional: a small user study with Deaf ISL users

---

## 6. Software architecture (for the app)

```
Webcam ──► MediaPipe Holistic (body 33 + hands 2×21 + face subset)
       ──► normalise / sliding window
       ──► Pose encoder (+ face stream, fusion) ──► T5-small decoder
       ──► English text ──► TTS (e.g. offline engine) ──► Voice
```

---

## 7. Risks and things to verify before writing

- [ ] **Check that the iSign `.pose` files include face landmarks.** Load one file and print its header components. If they do, the face data is there and earlier work simply discarded it, which strengthens Gap 1.
- [ ] **Read the WSLP 2026 proceedings (29 Oct 2026)**, especially *Shared-Encoder Contrastive Grounding for Pose-Based ISL Translation* and *ISLiM*, to make sure neither already fuses the face.
- [ ] Get the final WSLP 2026 SLT leaderboard numbers (Codabench).
- [ ] Note the BLEU scale: Dhamecha et al. report BLEU of 0.19–0.30 on their held-out split. Confirm whether that is a 0–1 or 0–100 scale before quoting it.
- [ ] An earlier session noted a face-fusion ISL paper (Verma 2026, *Int. J. Special Education*) that used synthetic data. It was **not re-verified** today, so confirm it before citing.
- [ ] iSign has no gloss annotations, so the non-manual subset must be tagged by hand. Budget annotation time, ideally with an ISL interpreter.

---

## 8. References and links

- iSign: <https://aclanthology.org/2024.findings-acl.643.pdf> · dataset <https://huggingface.co/datasets/Exploration-Lab/iSign>
- WSLP 2025 proceedings: <https://aclanthology.org/volumes/2025.wslp-main/>
- WSLP 2026 workshop and shared task: <https://exploration-lab.github.io/WSLP-2026/> · <https://exploration-lab.github.io/WSLP-2026/task/>
- Dhamecha et al. 2026: <https://arxiv.org/abs/2609.12993> · code <https://github.com/manavdhamecha77/Sign-Lang-Trans>
- PoseStitch-SLT (EMNLP 2025): <https://aclanthology.org/2025.emnlp-main.698>
- LiST: <https://doi.org/10.3390/info14020079>
- FEA-SLT: <https://github.com/TuGuobin/FEA-SLT>
- Low-resource SLR/SLT review: <https://arxiv.org/pdf/2605.12096>
- Local full texts: `notes/*.txt`. PDFs: `papers_extra/`, `papers_web/`
