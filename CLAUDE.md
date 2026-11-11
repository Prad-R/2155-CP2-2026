# CLAUDE.md — 2.155/6 Challenge Problem 2 (City Zoning)

Repo: `git@github.com:Prad-R/2155-CP2-2026.git` (fork of `decode-mit/2155-CP2-2026`), cloned at `/home/pradr/2.156/cp_2`.

## Standing instructions from the user

- **Keep this file and `UPDATES.md` current.** `CLAUDE.md` = stable facts, conventions, current plan. `UPDATES.md` = dated log of what was done, results, and decisions (newest first). Update both whenever something material changes.
- **All compute runs on compute nodes via Slurm.** Never run training, feature extraction over the 500k grids, or optimisation on the login node (login slice is 4 CPUs / 10 GB, shared).
- **Never use the `pi_faez` partition/account.** Not for tests, not as a fallback.
- **Long tasks:** `mit_normal` (CPU, 12 h max) and `mit_normal_gpu` (GPU, 6 h max). Account: `mit_general`. `mit_preemptable` is also allowed (user, Nov 11).
- **Quick tests:** same partitions, short walltime (`-t 0:10:00`–`0:30:00`), small resources, tiny data subset. Verify a script end-to-end this way before submitting the long version.
- **Every job must be preemption-safe and requeueable** (see template below).
- Follow the cluster guard rails in `/etc/claude-code/CLAUDE.md` (bounded commands, no scans of `/orcd/scratch` or `/orcd/pool`, read `~/orcd/.quota` instead of `du`).

## The task

- A city layout is a 7x7 grid; each cell is one of 5 districts: 0 Residential, 1 Industrial, 2 Commercial, 3 Park, 4 Office.
- Four advisors (0 Wellness, 1 Tax, 2 Transportation, 3 Business) rate a layout in [0, 1]. Their scoring functions are unknown.
- Data: `datasets/grids_{0..4}.npy` = 500,000 grids (`load_grids()`); `datasets/scores.npy` = 500,000 x 4 ratings, **only 5,000 labelled per advisor**, rest NaN. Labelled scores are roughly uniform on [0, 1].
- **Deliverable:** one `.npy`, shape `(100, 7, 7)`, integer dtype, values 0–4, saved as `submissions/submission_<timestamp>.npy`. Grids do **not** have to come from the dataset; generating new ones is encouraged.
- **Validity:** a grid counts only if *every* advisor rates it >= 0.75 (rated by the true advisors after submission; we never see those scores).
- **Score (diversity):** over valid grids only, each grid contributes the mean normalised Hamming distance (fraction of the 49 cells that differ) to its 10 nearest *other valid* grids; sum of contributions / 100. See `diversity_score()` in `utils_public.py`.
  - Score <= fraction of valid grids. An invalid grid loses its whole contribution.
  - Duplicates / near-duplicates contribute ~0.
  - Scores above 0.75 earn nothing extra: only the threshold matters, so confidence in validity matters more than a high predicted mean.
- **Submission feedback (from the user, 2026-11-09):** 5 submissions per day; each returns only the final diversity score (no valid count, no per-grid or per-advisor information). Every submission and its score must be logged (array + score + date) so it can be used as data.
- **Leaderboard (screenshot 2026-11-09):** refreshes every 30 min, shows the score to 10 decimals. Scores are exact multiples of 1/49000 (checked on 3 entries), i.e. the leaderboard reveals the exact integer sum of 10-nearest-neighbour Hamming counts over valid grids. Top score 0.7749 (Shape Grapplers); top 10 are all >= 0.7566; rank 27 is 0.7446. Each entry carries a free-text message.
- **Deadline:** Nov 13 2026, end of day, and the report must be written before then. Working target: final submission locked by end of Nov 12, Nov 13 kept for the report. At most ~20 submissions in total.
- Baseline in the notebook: linear regression on raw cells + district counts, take the 100 grids with highest predicted min-advisor score. Predicts 0 valid grids and low diversity.

## What a good solution needs

1. **Accurate per-advisor surrogates with uncertainty** from 5,000 labels each (one-hot cells, counts, adjacency / neighbourhood / distance features, CNNs or GNNs, ensembles; labels overlap between advisors only partially, so check the overlap).
2. **Conservative validity estimate**: choose on a lower confidence bound of the min over advisors, not the mean prediction. Calibrate on held-out labels.
3. **Diverse generation/selection**: search beyond the dataset (e.g. evolutionary / simulated-annealing search on the surrogates, or generative models), then pick 100 that maximise the k=10 nearest-neighbour Hamming diversity subject to the validity margin.
4. Guard against surrogate exploitation: optimised grids drift out of distribution, where surrogates are least trustworthy.

## Repository layout

- `starter_notebook.ipynb` — problem statement and baseline (large because of embedded images; read cell sources only, do not dump it).
- `utils_public.py` — `load_grids`, `select_rated_subset`, `onehot_and_flatten`, `diversity_score`, plotting helpers.
- `datasets/`, `assets/`, `submissions/` (submission files are gitignored, as is any top-level `*.npy`).
- `requirements.txt` — numpy, pandas, scipy, scikit-learn, matplotlib, pillow, jupyter. README asks for Python 3.10 in a `uv` venv.

## Environment notes

- Conda env **`2.156_cp_2`** (Python 3.10.22; numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, pandas 2.3.3, matplotlib 3.10.9, torch 2.14.1+cu130, lightgbm 4.7.0, xgboost 3.2.0). Rebuild with `scripts/setup_env.sh` (idempotent, run it through `srun`).
- Activate in any script or job with:
  ```bash
  source /etc/profile.d/modules.sh
  module load miniforge/26.7.2-0
  conda activate 2.156_cp_2
  ```
  `module`/`conda` are not on PATH in Claude's non-interactive shell; use the lines above or `bash -lc`.
- GPU verified on `mit_normal_gpu` (L40S, driver 590.48.01, CUDA 13.1): torch sees the GPU and runs. The "NVIDIA driver too old" warning torch prints on `mit_normal` CPU nodes is harmless.
- Storage headroom is fine (HOME ~45/200 GB as of 2026-11-09). Put large intermediate artefacts on scratch, not in the repo.

## Slurm job template (preempt-safe, requeueable)

```bash
#!/bin/bash
#SBATCH -J cp2-<name>
#SBATCH -A mit_general
#SBATCH -p mit_normal            # or mit_normal_gpu with --gres=gpu:1; never pi_faez
#SBATCH -t 12:00:00              # mit_normal max 12h, mit_normal_gpu max 6h
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --open-mode=append       # keep logs across requeues
#SBATCH --signal=B:USR1@120      # warn the batch shell 120 s before kill
#SBATCH -o logs/%x-%j.out

# on preemption/timeout: forward to python so it checkpoints, then requeue
trap 'kill -USR1 "$PID"; wait "$PID"; scontrol requeue "$SLURM_JOB_ID"' USR1
python -u script.py --ckpt-dir ckpt/<name> --resume &
PID=$!
wait "$PID"
```

Rules for the Python side:
- Checkpoint periodically and on `SIGUSR1`/`SIGTERM`; write to a temp file then `os.replace` (atomic).
- Always start with `--resume`: load the latest checkpoint if present, otherwise start fresh. Resuming must be idempotent.
- Fixed seeds, saved with the checkpoint (including RNG state).
- Write a `DONE` marker on completion and exit early if it exists, so a requeued finished job is a no-op.

## Submission rules (user)

- **Use all 5 submissions every day, Nov 9–13.** A slot is never left unused; if the intended set is not ready, use the slot for an information probe.
- The user uploads the file and reports the score. Claude builds each file with `src/make_submission.py`, which records the exact all-valid score beforehand in `results/submission_log.jsonl`. Fill in `returned_score` when the user reports it.
- Returned score == recorded all-valid score means all 100 grids are valid. Each invalid grid lowers the score by roughly (all-valid score)/100, so the deficit gives the number of failures. Exact identification of which grids failed (`divscore.decode`) is only reliable for <= ~2 failures, because one score carries about 15 bits.
- Approved plan: `~/.claude/plans/hazy-moseying-perlis.md`.
- **Reminder the user asked for (Nov 11): targeted labelling.** A submission made of a certified all-valid set (S8) with ~10 grids replaced by new, riskier ones tells us exactly which of the 10 failed. Raise this whenever the day's 5 submissions are planned, especially if the optimiser runs do not beat 0.754.

## Key findings so far

- **Advisors are NOT rotation/flip symmetric (found Nov 11; the earlier claim was wrong).** Adjacency terms depend on direction. Exact integer rules are known for Tax and Wellness, and Transportation fits with zero residual on central labels; Business is still open. Formulas, conventions and what is left: `UPDATES.md` entry "advisors are NOT rotation/flip symmetric"; scripts in `explore/` (`tax_exact.py`, `well_exact.py`, `trans2.py`, `blocks.py`). Do not use D4 augmentation or invariant-only features; the b0/z0 surrogates were built that way.
- Wellness and Tax conflict (parks). No dataset grid has all four labels.
- Scores take few distinct values, so the advisor rules are likely functions of integer features (counts, distances, adjacency).
- Dataset grids predicted valid are tightly clustered (best 100-subset scores ~0.32), so a competitive score needs generated grids.
- Wellness is the weakest surrogate.
- With validity ignored the optimiser reaches 0.7752 (leaders: 0.7749); with our margin (P(valid) >= 0.95) it plateaus at ~0.754 from any start. The gap is the cost of the validity margin, mostly Wellness (tightest advisor for 60–80% of grids) against Tax (parks pinned to 7–9, Residential to 9–11).
- Submission evidence (S5–S10): at margin 0.95 about 1.2% of grids fail (model says 3.6%), at 0.90 about 4%. One failure costs ~0.0075. Best so far: S8 = 0.7540 (100/100 valid). The leaderboard keeps every submission as a separate row, so the best one stands. Certified all-valid sets: S2, S3, S5, S8.
- **Labels are quantile-transformed raw scores** (values sit on a k/999 grid, with exact ties): the score is the rank of a discrete raw statistic. Modelling z = inverse-normal(label) fits much better than modelling the label (Wellness linear model 0.83 -> 0.89; Tax is almost linear in pair counts, 0.97). Second-generation models work in z: `src/train_gbmz.py`, `src/train_cnn.py --ztarget`, `src/blend_z.py` -> `results/blend_z0.json`; load any generation with `surrogate.load_surrogates(tag)`.
- More model families on the plain label do not help Wellness (all stop at R2 ~0.90).
- Rotating/flipping grids in a finished set does not raise the score; cyclic shifts are not a symmetry.

## Code map

- `jobs/run.sbatch` — generic requeue-safe wrapper; pass the python command as arguments.
- `src/train_gbm.py` (exploration, features, LightGBM CV, dataset predictions), `src/train_cnn.py` (CNN CV with/without augmentation), `src/blend.py` (OOF blend, P(valid) for dataset grids), `src/search.py` (GPU local search for new grids), `src/make_submission.py` (diverse 100-subset + log), `src/divscore.py` (exact score, decode, subset selection), `src/surrogate.py`, `src/features.py`, `src/common.py`.
- `src/optimize_set.py` (set annealing under a hard validity margin), `src/optimize_soft.py` (same with a soft validity penalty that hardens; can start from an invalid set), `src/patterns.py` (pattern analysis -> `results/patterns/`), `jobs/multi.sbatch` (several runs on one GPU).
- `src/features2.py` (extra invariant features), `src/train_feat.py` (model-family comparison), `src/eval_subs.py <blend tags>` (check a model generation against the known failure counts of every submission).
- Outputs: `ckpt/<tag>/` (models, per-unit results), `cache/` (features, dataset predictions), `results/` (summaries, submission log), `logs/`.
- Light steps (blend, building a submission) run through `srun` on `mit_normal`, not on the login node.

## Queue notes

- Per-user limits: `mit_normal_gpu` 2 GPUs / 32 CPUs; `mit_normal` 96 CPUs. `mit_preemptable` (4 GPUs/user, 2-day limit, jobs can be preempted) **approved by the user on Nov 11**; submit with `-p mit_preemptable` (same requeue-safe wrappers). It can have a queue wait of hours.
- `mit_normal` can be badly congested in the evening (Nov 9, 22:00: ~1,900 pending jobs, estimated start 6.5 h away for a 16-CPU/30-min job). **If a job is pending, check `squeue -o "%r %S"` for the estimated start instead of waiting.** When a GPU slot is free, `mit_normal_gpu` starts within seconds, so run short CPU steps there.
- Nov 11, 00:00: `mit_normal` estimated start was 4 days away; treat it as unavailable. With both GPU slots busy, run light steps inside a running job: `srun --jobid=<id> --overlap -n 1 -c 2 ...` (the `-n 1` is required).
- Pack several small runs on one GPU with `jobs/multi.sbatch <file of commands>`.

## Current status

See `UPDATES.md` for the dated log.
