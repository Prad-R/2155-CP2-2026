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

- **Changed by the user on Nov 12 (evening): better sets are held back on purpose.** The user does not upload S20/S21 or anything later, so that other teams do not see the score, and uploads the best file shortly before the deadline on Nov 13. Do not ask for uploads or returned scores before then; keep building the best file and always say which single file is the one to upload. Nothing newer than commit `c962381` is on GitHub (the fork is public); do not push before the deadline unless asked.
- (Earlier rule, Nov 9–12, superseded by the line above: use all 5 submissions every day; a slot is never left unused; if the intended set is not ready, use the slot for an information probe.)
- The user uploads the file and reports the score. Claude builds each file with `src/make_exact_submission.py` (exact rules; `src/make_submission.py` is the old surrogate-based builder), which records the exact all-valid score beforehand in `results/submission_log.jsonl`. Fill in `returned_score` when the user reports it.
- Returned score == recorded all-valid score means all 100 grids are valid. Each invalid grid lowers the score by roughly (all-valid score)/100, so the deficit gives the number of failures. Exact identification of which grids failed (`divscore.decode`) is only reliable for <= ~2 failures, because one score carries about 15 bits.
- Approved plan: `~/.claude/plans/hazy-moseying-perlis.md`.
- Slot timing is uncertain: S19 was accepted at ~22:30 EST on Nov 11 after five uploads earlier that day, so the limit is not per EST calendar day (or is not enforced). Ask the user how many slots are left rather than assuming.
- **Starting a new long job needs the user's go-ahead in the same session.** On Nov 12 at 23:31 the session's permission check refused `jobs/submit_r5.sh` (96 cores, 12 h) because the user had been asked about an overnight run and had not answered. Prepare and test the job, then give the user the one command; do not look for another way to start it.
- **Reminder the user asked for (Nov 11): targeted labelling.** A submission made of a certified all-valid set (S8) with ~10 grids replaced by new, riskier ones tells us exactly which of the 10 failed. Raise this whenever the day's 5 submissions are planned, especially if the optimiser runs do not beat 0.754.

## Key findings so far

- **All four advisor rules are known exactly (Nov 11).** They are deterministic integer rules, NOT rotation/flip symmetric (adjacency terms depend on direction; the earlier symmetry claim was wrong). Code: `src/rules.py` (`raw_scores`, `valid`, `THRESH`). Formulas and how they were found: `UPDATES.md`, Nov 11 entries. They reproduce all 20,000 labels and every scored submission to the exact integer. A grid is valid iff raw >= [Wellness 70, Tax 17, Transportation -187, Business -1155] (Tax and Business one step inside the label threshold because one raw value can carry two adjacent labels there). Validity is therefore certain before submitting: never use surrogate models, D4 augmentation or invariant features for validity again.
- Wellness and Tax conflict (parks). No dataset grid has all four labels.
- Scores take few distinct values, so the advisor rules are likely functions of integer features (counts, distances, adjacency).
- Dataset grids predicted valid are tightly clustered (best 100-subset scores ~0.32), so a competitive score needs generated grids.
- **Best uploaded: S19 = 0.7753061224 (= 37990; Nov 11 evening, rank 1, 100/100 valid, exact match with the predicted score; S18 = 0.7733877551 before it). Best built, i.e. THE FILE TO UPLOAD unless a later one is listed in `UPDATES.md`: S22 = 0.7764897959 (= 38048; `submissions/S22_20261112_233848.npy`, sha b2cb6b36, Nov 12 23:38, all valid, held back; from the fixed-core scan chain `kt23`). Earlier held-back files: S21 = 37997 (`submissions/S21_20261112_154245.npy`), S20 = 37996.** Built with `src/anneal.py` under the exact rules. Next team: 0.7752244898 (= 37986) when last seen (Nov 11 22:30). 38000 means every 10-NN distance is 38; it was the ceiling for sets annealed from scratch (best 37997) and is passed only with a fixed core of code grids (next bullets), whose grids have fewer than 10 neighbours at distance <= 38 and so contribute more than 380.
- **What moves the score now (`cp2-optR3`, `UPDATES.md` Nov 12 12:45):** (1) New record sets come from warm chains (constant T 0.70–0.80) with `--focus 5 --lam 0.3 --dmin 39 --lam2 0.1 --dmin2 41`: up to +6 over a 37990 start, half of the gains within 100M steps. `--focus 2 --lam 0.5 --dmin 39` at constant T hardly improves a 37990 start (+0/+2 in 64 chains). (2) A cold chain (T 0.35–0.45, `--lam` 0.15–0.25, `--focus 2`) *started from a record set* stays near it and polishes it by +0 to +4; the gain depends on the set, not on the seed (R4 stage 1: `v073` 37996 got nothing from 22 islands, `v069` 37995 went to 37997). Cooling a warm state instead freezes in ~16 close pairs, so do not cool: start cold from a best set. `cp2-optR4` (`src/islands.py`, tags `y001..y096`) combines both.
- **Construction route (Nov 12, `UPDATES.md` 16:25–23:50):** the 125 affine functions over GF(5)^2 drawn on the 49 cells (24 sites as two cells, one as one cell, a district relabeling per cell) are pairwise 39, 40 or 49 apart, i.e. 0.7959 if 100 of them were valid. The design searches (`--crystal` mode of `src/anneal.cpp`, designs in `ckpt/crystal/`) made at most 49 of 125 valid (`d01`, `d02`, `d04`; `c11` 47), so a full design is out of reach.
- **Best recipe now: fixed core + heat (Nov 12, 23:30).** Start set from `explore/core_prep.py` (the K valid code grids first, then the other code grids; `cache/core_B.npy` = `d01`, K 49), core kept fixed with `--frozen K`, chain run hot: `--focus 5 --dmin 39` at constant T 1.75 with `--lam 0.3`, or T 2.0 with `--lam 0.6`. Such chains held 38031–38048 after 2–2.5M steps (two minutes) and were still climbing when the job ended; a 36M-step cooling chain from T 1.5 reached 38015. The same start at T <= 1.0 sticks at 37930–37970 (which is why the first fixed-core chains looked negative), T >= 2.5 destroys the set, a free core is ~50 lower, no pair penalty ~130 lower. Core grids then contribute ~383 each and the other grids ~377.5. The level these chains settle at over hours is not known: that is what `cp2-optR5` is for.
- (Superseded by the exact rules: the surrogate-era findings below about margins, P(valid) and Wellness error describe models b0/z0 only.)
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
- **Exact pipeline (use this):** `src/rules.py` (exact raw scores and validity), `src/optimize_exact.py` (set annealer under exact validity, ~550 steps/s per core, repairs invalid start sets), `src/anneal.py` + `src/anneal.cpp` (C++ port of the same annealer, 15–25x faster, pair penalty `--lam/--dmin/--lam2/--dmin2`; `python src/anneal.py selftest` cross-checks the C++ rules), `src/best_chains.py` (best sets across chains, `--verify`), `src/make_exact_submission.py` (refuses to write unless 100 distinct grids are all valid; reads `.npy`, `.npz` or a chain's `state.bin`; logs to `results/submission_log.jsonl`), `jobs/multi_cpu.sbatch` (several CPU runs, requeue-safe; optional skip guard `CP2_SKIP_IF`, e.g. `explore/reached.sh 38000`), `jobs/build_best.sbatch` (builds a submission as a batch job, so it can wait behind a `--dependency`). `src/islands.py` (one "island" = several `anneal.py` stages in a row, tags `<island>s<k>`; a later stage restarts from the island's own best or from the best set of the whole run; requeue-safe; quick test `explore/test_islands.sh`), `jobs/make_r4.sh` -> `jobs/opt_r4.txt` and `jobs/submit_r4.sh` (the island run `cp2-optR4` plus the build job `cp2-next` behind it). **One command to build a submission from the best verified chain: `explore/build_best.sh S<n> ["purpose"]`, or `explore/build_best.sh next`, which takes the next free S number and does nothing unless a chain beats every submission built so far.** Newest binary `bin/anneal6` (select with `ANNEAL_BIN`; R4 islands use `bin/anneal3`): adds `--special M --boost B` (stronger pair penalty for the first M grids), `--frozen M` (first M grids never change) and the `--crystal` design search (`--keep`, `--pairing`, `--cinit`, `--cstep`). Construction scripts: `explore/crystal_run.sh`, `explore/crystal_run2.sh`, `explore/crystal_check.py`, `explore/core_prep.py` (start set: valid code grids first), `explore/core_run.sh` (12 fixed-core chains, cold), `explore/core_cool.sh` (14 cooling chains), `explore/core_scan.sh` (26 constant-T chains), `explore/core_quick.sh` (build + one-minute check). **Overnight fixed-core run `cp2-optR5`: `jobs/make_r5.sh` -> `jobs/opt_r5.txt` (96 islands, groups `za`/`zb`/`zc`/`zd` = one per core; `HOT=n` sets the number of 20M-step hot stages, default 9 = 240M steps, ~9 h), `jobs/submit_r5.sh` (queues it plus the build job `cp2-nextR5`), `explore/test_r5.sh` (one island with scaled-down steps).** In `src/islands.py` a stage can name its own rule (`own:1x10000000`, `best:...`). `explore/` holds the rule-recovery scripts (`wb.sh` runs a command on a compute node: inside the `cp2-wb` workbench job if one is running, otherwise in a short `srun` on `mit_preemptable`; `verify3.py` re-checks labels and submissions; `tax_exact.py`, `well_exact.py`, `trans2.py`, `bus_exact.py`).
- Outputs: `ckpt/<tag>/` (models, per-unit results), `cache/` (features, dataset predictions), `results/` (summaries, submission log), `logs/`.
- Light steps (blend, building a submission) run through `srun` on `mit_normal`, not on the login node.

## Queue notes

- Per-user limits: `mit_normal_gpu` 2 GPUs / 32 CPUs; `mit_normal` 96 CPUs. `mit_preemptable` (4 GPUs/user, 2-day limit, jobs can be preempted) **approved by the user on Nov 11**; submit with `-p mit_preemptable` (same requeue-safe wrappers). It can have a queue wait of hours.
- `mit_normal` can be badly congested in the evening (Nov 9, 22:00: ~1,900 pending jobs, estimated start 6.5 h away for a 16-CPU/30-min job). **If a job is pending, check `squeue -o "%r %S"` for the estimated start instead of waiting.** When a GPU slot is free, `mit_normal_gpu` starts within seconds, so run short CPU steps there.
- Nov 11, 00:00: `mit_normal` estimated start was 4 days away; treat it as unavailable. With both GPU slots busy, run light steps inside a running job: `srun --jobid=<id> --overlap -n 1 -c 2 ...` (the `-n 1` is required).
- Pack several small runs on one GPU with `jobs/multi.sbatch <file of commands>`.
- Nov 12, 12:30: our fair-share factor is 0.03 after two days of 96-core runs and `mit_preemptable` had ~1,900 pending jobs, so new jobs wait even though nodes show idle cores (a 15-min, 3-CPU `srun` did not start in 10 min). `sbatch --test-only` gave "now + 18 h" for every size and time limit, so it says little about backfill; check `squeue -o "%r %S"` on the real job. For a light step while one of our jobs is running, use its allocation: `WBNAME=<job name> explore/wb.sh -c 2 <command>`. A step sees only the `-c` cores it asked for, and two overlapping steps can be given the same cores, so steps that run many processes at the same time must each ask for `-c 96` (the whole allocation). Every step dies when the job's batch script ends. **Never edit a script while a job or step is executing it:** the edit is made from the login node, the compute node's open file goes stale over NFS, and bash silently stops at that point with exit code 0 (Nov 12: `explore/crystal_run2.sh` was fixed while it waited on its searches, so its follow-on chains never started). Write a new file instead.
- Some `mit_preemptable` nodes count hardware threads as CPUs (e.g. node2004: 128 cores, `CPUTot=256`): there a plain `-c 96` is only 48 cores and 96 single-thread chains run at half speed (~4.3k steps/s per chain). Submit multi-chain CPU jobs with `--threads-per-core=1`, which makes Slurm hand out whole cores on such nodes. **Do not judge a node by threads per core (`sinfo %Z`) alone:** node4408 reports 2 threads per core but has 96 real cores counted as 96 CPUs (`scontrol show node <n>`: compare `CPUTot` with `Sockets` x `CoresPerSocket`) and ran the chains at ~9.7k steps/s, faster than the 192-core node19xx type (~8.3k). On Nov 12 `cp2-optR4` was moved off node4408 on that wrong reading, which cost ~8 min and ~15% speed. Measure the speed from the first checkpoint before moving a job. If a job really has to move: `scancel --signal=USR1 --batch <id>` (checkpoint and requeue), then `scontrol update JobId=<id> Features="<node>|<node>"` while it is pending (every node carries its own name as a feature).

## Current status

See `UPDATES.md` for the dated log.
