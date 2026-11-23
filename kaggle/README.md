# Training on Kaggle (free: ~30 GPU hours a week, runs in the background)

1. **Account.** Sign in at kaggle.com and verify your phone number (Settings). Without that, Kaggle turns off GPU and internet.
2. **Step 1: convert the data** (once, about 1 hour, CPU).
   - Create → New Notebook → File → Import Notebook. Use `kaggle/1_convert_isign.ipynb` from this repo (GitHub link or upload).
   - Right panel: Accelerator **None**, Internet **On**. Under Add-ons → Secrets, add `HF_TOKEN`.
   - Click **Save Version → Save & Run All (Commit)**.
   - When it finishes, the Output tab shows `isign_pose_keypoints/` with 255 shards.
3. **Step 2: train** (repeat until everything is done).
   - Import `kaggle/2_train_fastisl.ipynb` the same way.
   - Right panel: Accelerator **GPU T4 x2**, Internet **On**.
   - Add Input → Your Work → the step-1 notebook.
   - Click **Save Version → Save & Run All (Commit)**. It trains for up to about 11 hours, then stops cleanly.
   - For the next session, also add **this notebook's previous version** as an input, then commit again. Finished models are skipped, and the current model resumes from its last epoch.
4. **Results.** Download `runs/` from the last version's Output tab into `ISL/runs/` on the Mac. Then run `python -m islt.report runs`.

Order: baseline (`B0_body_hands`), then `FaST` (these two give the main result), then the 7 ablations, the boundary detector and the streaming evaluation.
