# UPDATES

Dated log, newest first. Stable conventions live in `CLAUDE.md`.

## 2026-11-12 23:50: a fixed core plus heat breaks 38000 (38048 after two minutes of scanning); overnight run `cp2-optR5` prepared but NOT started

- **Result.** With the valid code grids kept fixed and the chain started hot (T 1.5–2.0), the annealer goes well past every earlier set: 38015 from a 36M-step cooling chain, 38031–38048 from constant-T chains after only 2–2.5M steps, still climbing when the job ended. All verified with the Python rules. Best before tonight: 37997 (S21).
- `cp2-optR4` ended by itself at 23:30:00 (rc 0), still at 37997; every chain running inside its allocation ended with it.
- Evening tests in detail (all inside the R4 allocation):
  - `ka01..ka12` (core `c11`, 47 grids, constant T 0.35–0.75): flat at 37783–37933 after 62–90M steps. Too cold (see below).
  - Second design search `d01..d14`: 49 valid at best (`d01`, `d02`, `d04`), 48 (`d06`), 47 (all the others started from `c11`), the two fresh ones 35 and 39. `explore/crystal_run2.sh` stopped silently after the searches because it was edited while running, so its follow-on chains never ran (now a rule in `CLAUDE.md`).
  - `kb01..kb14` (`explore/core_cool.sh`; start `cache/core_B.npy` = the 49 valid grids of `d01`, pairs at 39/40/49: 905/242/29, plus 51 repaired code grids; core fixed; T falling over 36M steps, `--focus 5`). Best score by start temperature: 0.9 -> 37930–37947 (constant 0.9: 37960); 1.0 -> 37946–37968; 1.2 -> 37968 (boost 2), 37992 (boost 1); **1.5 -> 38015 (`kb09`: `--lam 0.3`, end T 0.5) and 38005 (`kb10`: `--lam 0.5`, end T 0.6)**. `--boost 2`, `--lam2`, `--nswap 400`, `--lam 0.15` and boost 0.5 made no difference or a small loss. `kb09` held 37986 at 2M steps (T 1.41), 38013 at 12M (T 1.04), 38015 at 18M (T 0.87) and nothing new below that; its core grids contribute 382.4 each (8.1 neighbours at <= 38 on average), the other grids 377.9.
  - Temperature scan `kt01..kt26` (`explore/core_scan.sh`: same start, constant T, 23:24–23:30, cut at 2–2.5M steps by the end of R4; two chains per setting, best score of each):

    | setting (core fixed, `--lam 0.3 --focus 5` unless stated) | best after 2–2.5M steps |
    |---|---|
    | T 1.3 | 38014, 37981 |
    | T 1.5 | 38012, 38036 |
    | T 1.75 | 38040, **38048** |
    | T 2.0 | 38019, 38025 |
    | T 2.0, `--lam 0.6` | **38048**, 38018 |
    | T 1.5, `--lam 0.6` | 37959, 37970 |
    | T 2.5 / 3.0 / 4.0 | 37898–37911 / 37793–37794 / 37563–37578 |
    | T 1.5, core free | 37955, 37979 |
    | T 1.5, no pair penalty | 37884, 37883 |
    | T 1.5, `--focus 2` / `--focus 10` | 37997, 38007 / 37998, 38013 |

    The running score rose by 10–15 per 500k steps to the end (e.g. `kt23`: 37992, 38002, 38030, 38034), so the level these chains settle at is not known. `kt23` at 38048: core grids 383.3 each (6.5 neighbours at <= 38, 41 of 49 below 10), other grids 377.5.
  - Reading: the earlier fixed-core chains failed because they were too cold, not because 47–49 core grids are too few. The repaired start set needs T >= 1.5 to rearrange; above T 2.5 the set falls apart; the core must stay fixed (-50 when free); the pair penalty is needed (-130 without) and a stronger one goes with a higher T.
- **S22 built at 23:38 by `cp2-next` (the build job queued behind R4): `submissions/S22_20261112_233848.npy`, sha b2cb6b36, 38048 = 0.7764897959, all 100 valid (chain `kt23`, verified; `kt06` also holds 38048, `kt05` 38040). This is the file to upload** unless a later entry names a better one. Held back like S20/S21.
- **Overnight run `cp2-optR5`: prepared and checked, not submitted.** The session's permission check refused the submission because the user had not answered the overnight question, so it waits for the user: `jobs/submit_r5.sh` queues it (96 cores, 12 h limit, `mit_preemptable`) with the build job `cp2-nextR5` behind it.
  - `jobs/make_r5.sh` -> `jobs/opt_r5.txt`: 96 islands, 240M steps each (~9 h): 9 hot stages of 20M steps at the island's own (T, `--lam`) with T 1.3–2.25, then 40M steps cooling to T 0.8, then 10M cold polish with the core fixed and 10M with the core free. 60 "winner" islands restart every stage from the best set of their group, 18 follow their own best only, 18 cool slowly from T 2.0 to 1.0 over 180M steps. Groups by core: `za` = `core_B` (60 islands, 8 of them starting from the scan chains `kt23`, `kt06`, `kt05`, `kt04`), `zb` = `core_d02`, `zc` = `core_d04`, `zd` = `core_c11` (12 each). `HOT=4 bash jobs/make_r5.sh` gives a 140M-step version (~5.2 h) if the run starts late.
  - `src/islands.py`: a stage can now name its own rule (`own:1x10000000`), used for the polish stages.
  - Check: `explore/test_r5.sh` (island 1 with the steps scaled down; 2 cores, 2.5 min in a short `cp2-light` allocation at 23:38) passed: 3 hot stages restarting from the group's best, the cooling stage and both polish stages ran in order, every stage was verified all-valid by the Python rules, and the core rows stayed unchanged. The script only scales the default list (`HOT=9`).
- User decision recorded (Nov 12 evening): S20, S21 and anything later are held back on purpose and uploaded shortly before the deadline, so that other teams do not see the score. Do not ask for uploads before then.

## 2026-11-12 20:50: probe below the pass mark (47 of 125); fixed-core test started on the freed cores

- Probe result (finished 17:37): best design `c11` (`--keep 60`, domino layout) has 47 valid grids of 125; the pass mark was 60. The others: `--keep 100` 19–29 valid, `--keep 80` 28–30, `c12` (`--keep 60`, random pairing) 27. `explore/crystal_check.py` gives the same valid counts with the Python rules and confirms minimum distance 39 among all 125 grids of every design (its per-advisor failure counts for `c05` are 4 higher than the C++ ones only because it counts the 4 grids with a missing district type as failing every advisor). `c11` was still improving in its last 500k moves (45 -> 47) and its 13 next grids are almost valid (deficits 0.01–0.24).
- Search for a full design (~100 valid) is stopped, as agreed. What is tested instead, on cores that were idle or wasted (session interrupted 17:50–20:40 by the usage limit, so only ~2.3 h of the job were left):
  - **Fixed-core test**: start set = the 47 valid `c11` grids (pairs at 39/40/49: 823/232/26) + the 53 invalid code grids closest to validity (`explore/core_prep.py` -> `cache/core_c11.npy`, `.k`). The 100 code grids score 39000 with validity ignored. The annealer's repair phase makes the 53 valid (~160 moves, score ~37260), then anneals with the first 47 grids fixed. New option `--frozen M` in `src/anneal.cpp` (binary `bin/anneal6`, built with `-mavx2`; self-test passed, 22200 grids identical to `rules.py`; a 400k-step check left the frozen rows unchanged and reached 37762). `explore/core_run.sh`: 12 chains `ka01..ka12`, constant T 0.35–0.75, `--special 47` with `--boost` 1–3, chain 10 without pair penalty, chains 11–12 with the core free. Logs `logs/core-ka*.out`; they run until the job ends and checkpoint on SIGTERM.
  - **Second design search** (`explore/crystal_run2.sh`): 14 searches `d01..d14` of ~10M moves (35 min), 12 continuing from `c11` (new options `--cinit <design file>` and `--cstep x` = fixed extra cost per invalid grid, `--keep` 60–80), 2 fresh. When they finish, the script starts 12 more fixed-core chains from the design with the most valid grids (`kb01..kb12`), or on the `c11` core with other seeds if none beats 47 (`kc01..kc12`). Five searches (d01, d02, d04, d06, d11) died at once on a typo in their arguments (a lost space) and were restarted two minutes later as a separate step.
  - Cores: the 12 idle ones plus 14 from R4 islands stopped at 20:46 (`own`-rule islands y001 y007 y010 y015 y027 y030 y032 y035 y037 y040 y042 y045 y047 y048, all at 37994–37996 with no gain since the morning; resumable). 70 R4 islands continue; R4 is still at 37997.
- Why the test is worth two hours although the mark was missed: a core grid has no neighbour at 38 inside the core, so it earns 390 minus the number of other grids at 38 (while that is below 10). The gain over 38000 is about K x (10 - (100 - K) x q), q = share of core-to-other pairs at 38: zero at q = 19% for K = 47, +96 at q = 15%. Annealed sets have q = 25%, the squeeze test reached 16–20%, a code grid repaired by one cell change has 15%. So K = 47 can pay only a little, and K matters more than anything else (hence the second search).
- Step note: a job step started with `explore/wb.sh -c N` sees only N cores (`nproc`), so two steps running at once must both ask for `-c 96` to float over the whole allocation.

## 2026-11-12 16:45: squeeze test negative (stopped early); construction probe started at the user's request

- Squeeze test stopped at 16:39 after 6–8M of 20M steps (`scancel` of the job step; chains `sq01..sq12` keep their checkpoints). Result: negative.
  - `--special`/`--boost`: the special grids' close-neighbour count stayed at 16–20 on average (start 19); no further grid fell below 10. Boost 5 reached 16.2 but with 24 pairs at 37 (score 37957); boost 1–2 kept the score at 37986–37993 with 18 neighbours.
  - Uniform `--lam` 1 / 2 / 4: 23 / 65–69 / 101 pairs below 38, scores 37957 / 37856–37866 / 37781. A stronger penalty trades 38-pairs for 37-pairs.
- Construction probe (user: "do the probe"; pass mark agreed beforehand: at least 60 of the 125 code grids valid). `src/anneal.cpp` gained a `--crystal` mode (`crystal_main`, binary `bin/crystal`): simulated annealing over the cell-to-site layout and the 49 per-cell district relabelings of the 125 affine-function grids; energy = sum of the `--keep` smallest validity deficits; uses the same `raw4`/`deficit` code as the annealer. A 60k-move trial (13 s) from a random design gave 11 valid of 125; `explore/crystal_check.py` confirmed with the Python rules: 11 valid, minimum distance among all 125 grids 39, parks per grid 9.8 on average (8.7 among the valid ones), failures Tax 87, Business 89, Wellness 32, Transportation 20.
- Probe run: `explore/crystal_run.sh`, 12 searches `c01..c12` of 16M moves (~1 h at 4.4k moves/s, T 2 -> 0.05), started 16:39 inside the R4 allocation on the 12 freed cores; `--keep` 100 (8 searches), 80 (2), 60 (2); half start from a domino layout, half from a random pairing. Logs `logs/crystal-c*.out`, designs and grids `ckpt/crystal/c*.bin`. The cycling islands y085–y096 stay stopped.

## 2026-11-12 16:25: why the score stops at ~38000, and a squeeze test (user: "theoretical limit 0.8081, someone got 0.7778 last year")

- 0.8081 = 39.6/49 is the mean pairwise distance when every cell holds each district equally often; it would be the score only if all 4,950 pairs were equally far apart. 0.7778 = 38112.
- Structure of S21 (`explore/set_stats.py`): pairs at distance 37: 2, 38: 1216, 39: 1613, 40: 1081, 41+: 1038; mean 39.515, which is already the maximum the per-cell frequencies allow (cells are balanced; parks sit at 17% because valid grids have 7–9 parks). Every grid has 16–33 neighbours at exactly 38 (mean 24.3; one grid has 9). The score takes the 10 nearest, so each grid scores 380 and only earns more with fewer than 10 neighbours at 38. 38112 would need about 11 per grid instead of 24, i.e. half of the 38-pairs gone.
- The count of 38-pairs hardly moves with the settings (`explore/chain_stats.py`, current states of finished chains): `--lam 0.3`: ~1315; `--lam 0.5`: 1150–1180 at any T from 0.35 to 0.90; `--lam 0.5 --dmin 40`: 1134; polish chains: ~1220. Minimum over grids 14–18. So the annealer sits in a jammed state at 23–26 neighbours per grid; more of the same can only remove the last 37-pairs (38000 plus a point or two).
- Without the validity rules a far better set exists: the 125 affine functions on a 5x5 array over GF(5), each site drawn as a pair of cells (a [49,3,39] code), are pairwise at distance 39, 40 or 49, i.e. score 0.7959. Making such grids valid means choosing 49 per-cell district relabelings and a cell layout so that ~100 of the patterns pass all four rules; the code forces 9.8 parks per grid on average (valid grids: 7–9) and partial success does not pay unless 60–80 grids come from the code. Judged a research project, not attempted.
- Squeeze test, running 16:20–~17:15 inside the R4 allocation on the 12 cores of the cycling islands y085–y096 (stopped with SIGTERM, resumable; they had gained nothing): `explore/sq_run.sh`, chains `sq01..sq12`, 20M steps from S21 sorted by number of close neighbours (`cache/sq_start_S21_sorted.npy`), logs `logs/sq-sq*.out`.
  - New annealer option (`bin/anneal5`, source `src/anneal.cpp`; self-test passed; running R4 chains still use `bin/anneal3`): `--special M --boost B` multiplies the pair penalties of the first M grids by 1+B, to thin the close neighbours of chosen grids below 10. sq01–sq06: M 8/16/32, B 1/2/5.
  - sq07–sq12: uniform strong penalty, `--lam` 1–4 at T 0.45–1.2.
  - A 200k-step trial (M 16, B 5, T 0.6) took the special grids from 19 to 16 close neighbours on average but created ~18 pairs at 37 (score 37961): about 2.7 neighbours removed per 37-pair created, and break-even needs about 4. Expectation low.

## 2026-11-12 15:45: `cp2-optR4` after its first stage: 37997; S21 built; duplicate "S20" file renamed

- Status at 15:40 EST: running 3 h on node1911, no errors, not preempted. The 60 polish/cycling islands finished stage 1 (50M cold steps) at ~14:25 and are at 32–36M steps of stage 2; the 36 warm islands are at ~86M of their 150M warm steps.
- Stage 1 by start set (gain over the start):
  - `v073` (37996): 16 polish + 6 cycling islands, every setting: no gain.
  - `v069` (37995): of 10 polish islands, 4 reached **37997** (three at T 0.45 `--lam 0.25`, one at T 0.35 `--lam 0.15`), 4 reached 37996 (T 0.45, T 0.40, T 0.35), 2 no gain (T 0.50 `--lam 0.3`; the `lam2` variant). The 3 cycling islands: no gain.
  - `v018` (37994): one +2 (T 0.50 `--lam 0.3`), four no gain. `v070 v072 v074 v090 v092`: no gain.
  - Thermal cycling (T 0.60 -> 0.35): no gain in any of the 12 islands.
- Stage 2: the 40 `best` islands restarted from the four 37997 sets (`y017s1 y018s1 y020s1 y024s1`); nothing above 37997 after 32–36M steps.
- Warm islands at 86M steps: 2 x 37996 (`y052` at T 0.70, `y067` at T 0.75), 3 x 37994, 10 x 37992, 21 still at 37990. Their polish stages start ~17:50.
- So far this confirms the picture from R3: a polish gain is a property of the start set (0 for `v073`, +2 for `v069`) and is small; the best final set is warm record + polish gain.
- **S21** = `submissions/S21_20261112_154245.npy` (sha 798e01f3): 37997 = **0.7754489796**, all 100 valid. Built inside the R4 allocation (`WBNAME=cp2-optR4 explore/build_best.sh next`). Not yet uploaded. `cp2-next` will now build S22 at the end of the run only if something beats 37997.
- Correction to the 02:10 entry: the automatic build `cp2-s20` had already run at 01:59 EST when I cancelled it and had left `submissions/S20_20261112_015905.npy`, byte-identical to S19 (checked with `cmp`). Renamed to `submissions/dup_of_S19_20261112_015905.npy`, log entry renamed to `S19dup` with a note, so that it cannot be uploaded as S20 by mistake. The real S20 is `S20_20261112_120759.npy` (37996).

## 2026-11-12 12:45: `cp2-optR3` finished at 37996; S20 built; island run `cp2-optR4` queued

- `cp2-optR3` ended ~12:30 EST, 96/96 chains done, never preempted. Final best **37996** (`v073`), then `v069` 37995, six at 37994 (`v018 v070 v072 v074 v090 v092`), 18 at 37992, 70 still at the 37990 they started from.
- **S20** = `submissions/S20_20261112_120759.npy` (sha 7569edfd): best set of `v073`, 37996 = **0.7754285714**, all 100 valid. Built at 12:08 while the last chains were finishing; nothing better appeared afterwards. Not yet uploaded.
- What R3 showed (bests relative to the 37990 start, per chain):
  - The main bet failed. `--focus 2 --lam 0.5 --dmin 39` at constant T 0.60–0.90 (64 chains x 300M steps): +0 or +2, one +4. The sawtooth 0.92 -> 0.60 (8 chains): nothing. A constant-setting chain is stationary, so more time at the same setting only raises its record very slowly; the 37990 starts were already extreme for this setting.
  - `--focus 5 --lam 0.3 --dmin 39 --lam2 0.1 --dmin2 41` did far better with 12 chains: T 0.70: +5 +4 0 +4; T 0.80: +6 +4 0 0; T 0.65: 0 +2 0 +2. Its running score is lower (about 37970–37980, 10–15 pairs at 37) but its records are higher. Half of the >= +4 records were there by 100M steps.
  - Cold chains started from a good set stay near it instead of freezing at ~16 close pairs (that only happens when cooling from a warm state): at T 0.45 `--lam 0.25` the running score is ~37991 with ~5 pairs at 37. They act as a local polish, and the result depends on the start set, not on the seed: the `r752` set went to +4 at T 0.45 (by 30M steps) and at T 0.35 `--lam 0.15` (by 170M) and to +2 in all four T 0.60 chains; the `r761` set got nothing from the same settings (+2 once, at T 0.25 `--lam 0.05`).
  - `v069` = 37995 is odd, so one grid there already has fewer than 10 neighbours at distance <= 38: the first step beyond "38000 − 2 x pairs at 37".
- New tool `src/islands.py`: runs several annealing stages in a row for one "island" (each stage is an ordinary `anneal.py` chain `ckpt/<tag>s<k>/`, so it is resumable and verified). A later stage restarts either from the island's own best (`own`) or from the best set any island of the run holds at that moment (`best`, ties spread). Tested end to end, including stop-and-resume (`explore/test_islands.sh`, run inside the R3 job because a 15-min `srun` sat in the queue for 10 min).
- **`cp2-optR4`** (job 28834865, `jobs/opt_r4.txt` written by `jobs/make_r4.sh`, launcher `jobs/submit_r4.sh`): 96 islands `y001..y096`, 300M steps each (~10.5 h), 96 cores on `mit_preemptable`.
  - y001–y048 polish: cold chains (T 0.35–0.50, `--lam` 0.15–0.30; mostly T 0.45 `--lam 0.25 --focus 2`) from `v073` (16), `v069` (10), the four warm-found 37994 sets (5 each) and `v090`/`v092` (1 each); 6 stages x 50M; 31 islands follow the run's best set, 17 keep their own.
  - y049–y084 new sets: 150M warm steps with the `lam2` setting at T 0.70/0.75/0.80 from the two 37990 sets, then 3 x 50M cold polish of the island's own record.
  - y085–y096 mild thermal cycling (T 0.60 -> 0.35, five cycles per 50M-step stage) from `v073`, `v069` and three 37994 sets. Untested idea, small share.
  - Expectation (reasoning from six cold chains and two start sets, so weak): polishing adds +0 to +4 depending on the set; 37998–38000 is plausible, more is possible but unproven.
- `cp2-next` (job 28834866) waits behind R4 and runs `explore/build_best.sh next`: it builds the next S number only if a chain beats every submission built so far. The same command works by hand at any time during the run.
- First start 12:32:54 on node4408: 96/96 islands started from the right sets, no errors. **I then moved the job, and that was a mistake.** `sinfo` showed node4408 with 96 CPUs and 2 threads per core, which I read as 48 physical cores (the node2004 situation that halved `cp2-optP`). In fact node4408 has 2 x 48 = 96 real cores and Slurm counts cores as CPUs there (`scontrol show node`: `CPUTot=96`, `Sockets=2`, `CoresPerSocket=48`), and the chains were running at ~9.7k steps/s (1.03M steps in 106 s), faster than node19xx (~8.3k). The move: `scancel --signal=USR1 --batch` at 12:34:47 (96/96 islands stopped cleanly, progress kept), then `scontrol update JobId=28834865 Features="node1903|...|node1916"`.
- **Running since 12:42:12 on node1911** (192 cores, one thread per core): 96/96 islands resumed from ~1.03M steps, no errors, ~8k steps/s. Cost of the move: 7.5 min in the queue plus ~15% speed, so the run ends ~22:55 EST instead of ~21:20. Not moved back: another requeue would risk more queue time than it saves. The job keeps the node19xx constraint if it is ever requeued. `--threads-per-core=1` was doing its job all along (on thread-counting nodes `sbatch --test-only` planned 192 processors for the 96 chains); `jobs/submit_r4.sh` uses it, without a node constraint.
- Stage times at ~8.2k steps/s: each 50M-step stage takes ~1.7 h, so the polish islands finish stage 1 at ~14:25, stage 2 ~16:05, stage 3 ~17:50; the warm stage of y049–y084 also ends ~17:50.
- Queue: `mit_preemptable` had 1,892 pending jobs at 12:30 and our fair-share factor is down to 0.03 after two days of 96-core runs; `sbatch --test-only` estimated a start on Nov 13 06:05 for every job size from 24 to 96 cores, although many nodes show idle cores. `mit_normal` estimate: Nov 18.

## 2026-11-12 02:10: `cp2-optR2` finished at 37990 (no gain after half-way); follow-up re-aimed at the warm temperature window

- `cp2-optR2` ended ~01:55 EST, 96/96 chains done. Final best 37990 (`r761`, `r752`), then 37987 and four at 37986: the same as at half-way. **No new submission: the best set is still S19.** The automatic S20 build (`cp2-s20`) was cancelled because it would only have rebuilt S19.
- Why, from the per-chain logs (10M-step resolution): both 37990 sets were found by step 70M at T ~0.84–0.88. Across the 96 chains the final best first appeared at T between 0.53 and 1.08 (bulk 0.76–0.98; the seven chains >= 37986 at T 0.59–0.88) and never after T < 0.53, i.e. not in the last 35% of the schedule. At the cold end (T 0.37) every setting sits at 15–17 pairs below 38 (score ~37966; `--dmin 40`: 21) for `--lam` 0.3 and 0.5 alike. So the best sets are downward fluctuations of the close-pair count at warm temperature, kept by best-so-far tracking, and cooling freezes in ~16 close pairs. Longer cooling is not the lever; time spent in the window T ~0.6–0.9 is.
- Mean of chain bests per block: `f2 lam0.5` 37982.2 (max 37990), `f5 lam0.3+lam2` 37978.6, `f5 lam0.5` 37977.8, `f2 lam0.3` 37975.9, `f5 lam0.3` 37974.5, `f5 lam0.5 dmin40` 37974.1.
- Decision on the follow-up (user asleep, asked me to decide): needed, because the best is below 38000, but not as queued. `cp2-optR3` (28762545) had started at 01:59 on node1904 with the cooling design (now `jobs/opt_r3_cooling.txt`; chains r801–r896 have ~7 min of progress and are abandoned). `jobs/opt_r3.txt` was rewritten and the job requeued at 02:06 through its own checkpoint-and-requeue trap (`scancel --signal=USR1 --batch`), so that it restarts with the new list:
  - 96 chains `v001..v096`, 300M steps each, all started from the two 37990 sets (odd tags from `ckpt/r761/state.bin`, even from `ckpt/r752/state.bin`), so no chain can end below 37990.
  - 64 x constant T in {0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90} with `--focus 2 --lam 0.5 --dmin 39`; 12 x `--focus 5 --lam 0.3 --dmin 39 --lam2 0.1 --dmin2 41` at T 0.65/0.70/0.80; 4 x `--focus 5 --lam 0.5 --dmin 39` at T 0.65; 8 x sawtooth T 0.92 -> 0.60 with `--cycles 6`; 8 x cold clean-up tests (T 0.25–0.45 with smaller `--lam`).
  - Rough expectation from R2's hit rates (about 17x more chain-steps inside the window than R2's best block had): 37994–37998, with 38000 possible. This is an estimate, not a measurement.
  - Restart confirmed: running again from 02:11:29 on node1907 (5 min after the requeue), 96/96 chains started from score 37990, no errors; due ~12:30 EST. Logs `logs/cp2-optR3-28762545-<n>.out` (appended after the few lines of the abandoned r8xx chains). A background check in the Claude session (every 10 min) ends when a chain reaches 38000 or the job ends.
  - To see the best so far without building anything: `explore/reached.sh 37992`. To build a submission from it: `explore/build_best.sh S20`.
- 38000 is not a hard ceiling: a grid with fewer than 10 neighbours at distance <= 38 contributes more than 380. Every set so far has >= 10 such neighbours per grid, so in practice score = 38000 − 2 x (pairs at 37).

## 2026-11-11 22:45: S19 returned 0.7753061224 — exact match, rank 1; follow-up run queued behind `cp2-optR2`

- User uploaded S19 (~22:30 EST) and sent the leaderboard row: **rank 1, "2.5 elbaf", 0.7753061224** = expected 37990, so all 100 grids are valid. Logged in `results/submission_log.jsonl`.
- The upload was accepted before midnight EST although five files had already gone in on Nov 11 EST. The slot rule in the earlier entries ("per EST calendar day") is therefore wrong or not enforced; how many slots remain is unknown. Ask the user rather than assume.
- User (going to sleep): start the follow-up once `cp2-optR2` finishes, and decide whether it is needed. Rule used: needed unless some chain already holds a 38000 set (38000 = every 10-NN distance is 38, the practical ceiling; even the unconstrained optimiser stopped at 37998). Set up so that it does not depend on a Claude session:
  - `cp2-optR3` (job 28762545) runs `jobs/opt_r3.txt`, queued with `--dependency=afterany:28733722`. `jobs/multi_cpu.sbatch` gained an optional guard `CP2_SKIP_IF`; R3 is submitted with `CP2_SKIP_IF="explore/reached.sh 38000"` and exits at once if the target is already reached (checked again after any requeue). Launcher: `jobs/submit_r3.sh`. Cancel with `scancel -n cp2-optR3`.
  - `cp2-s20` (job 28762547) runs `jobs/build_best.sbatch S20 ...` with the same dependency: it builds S20 from the best verified chain when R2 ends. Check `ls submissions/S20_*` and `logs/cp2-s20-*.out`.
- Why `cp2-optP` runs at half the speed of `cp2-optR2` (14–17M against 28–30M steps/h per chain): its node (node2004; 49 nodes carry the feature `numa_node`) has 2 hardware threads per core, so `-c 96` is only 48 cores. node1904 has 192 real cores. R3 is submitted with `--threads-per-core=1` so that each chain gets a physical core on any node type.

## 2026-11-11 22:15: overnight chains at 37990 (above the leaders' last known 37986); S19 built; idle workbench jobs released

- Status at 21:57 EST: `cp2-optR2` (r701–r796, 240M steps) at 120–134M steps, due ~02:15 EST; `cp2-optP` (p401–p496, 100M steps) at 60–72M, due ~00:15 EST. All `q5xx`, `d`, `dd`, `pol` chains finished.
- Best verified sets so far (`best_chains.py --verify`): `r761` and `r752` at **37990 = 0.7753061224**, five more at 37986. The leaders' last known score was 37986 (screenshot ~17:40 EST). The chains are only half-way down the temperature schedule.
- Penalty settings at half-way, by share of the top 25: `--focus 2 --lam 0.5 --dmin 39` (r749–r764) 9; `--focus 5 --lam 0.3 --dmin 39 --lam2 0.1 --dmin2 41` (r765–r780) 7; `--focus 5 --lam 0.5 --dmin 39` 4; `--focus 5 --lam 0.3 --dmin 39` 2; `--focus 2 --lam 0.3 --dmin 39` 1; `--focus 5 --lam 0.5 --dmin 40` 0. Read the final ranking when the chains finish.
- Chain length is what pays: 60M steps (q5xx) best 37974, 100M (p, unfinished) 37984, 240M (r, unfinished) 37990. Re-annealing a finished set at low temperature does not (pol1/pol2 from c7 = 37896: 37902 and 37896 after 10M steps).
- **S19** = `submissions/S19_20261111_220945.npy` (sha 94913ac2): best set of chain `r761` at 126M steps, 37990 = **0.7753061224**, all 100 valid under `rules.THRESH`. For the first slot of Nov 12 (slots open at midnight EST). Not yet uploaded.
- Tooling: `src/make_exact_submission.py` now reads a C++ `state.bin` directly (it failed on one before). New `explore/build_best.sh S<n> ["purpose"]` builds a submission from the best verified chain in one command. `explore/wb.sh` falls back to a short `srun` on `mit_preemptable` (job `cp2-light`, <= 4 CPUs, 15 min) when no workbench job is running.
- Both workbench jobs (`cp2-wb` 24 CPUs, `cp2-wbg` 12 CPUs + GPU) were idle and were cancelled to free the nodes.
- Prepared, **not submitted**: `jobs/opt_r3.txt` (r801–r896, 300M steps, T 1.2 -> 0.35: 32 x best setting, 32 x best setting plus `--lam2 0.1 --dmin2 41`, 16 x second-best setting, 16 x `--focus 2 --lam 0.7 --dmin 39`) and `jobs/submit_r3.sh`, which queues it to start when `cp2-optR2` ends. `sbatch --test-only` accepted the submission line. Left for the user to start.

## 2026-11-11 (evening): S18 returned 0.7733877551 — exact match, 100/100 valid, rank 3

- User uploaded S18 at 22:37:04Z (17:37 EST); leaderboard row "2.5 elbaf, 0.7733877551, rank 3". Expected 37896 = 0.7733877551: **exact match, all 100 grids valid.** First set built under the exact rules; previous best was S8 = 0.7540 (rank 25).
- This also confirms the two thresholds no earlier submission had tested at margin 0: S18 contains grids at Wellness = 70 and Transportation = −187 exactly, and they are valid.
- All 5 slots of Nov 11 are used (S11, S14, S13, S15, S18). Slots are per EST calendar day (S6–S10 at 23:45 EST Nov 10 and S11–S18 on Nov 11 EST all counted separately), so the next ones open at midnight EST.
- User asked to pause Claude work to save usage. Nothing new was launched; the cluster jobs from the 17:45 entry run unattended (`cp2-optP` due ~22:00, `cp2-optR2` overnight, workbench `q501..q523` due ~20:15). Next step when resumed: `explore/wb.sh python -u src/best_chains.py --top 10 --verify`, then `src/make_exact_submission.py` on the best chain's `state.bin`.

## 2026-11-11 17:45: S18 built (0.77339); "wall-thinning" penalty; big preemptable jobs; work pushed to branch `exact-rules`

- **S18** = `submissions/S18_20261111_173529.npy` (sha f3f62dcc): chain `c7` (10M steps, T 1.2 -> 0.3, no penalty), 37896 = **0.7733877551**, all 100 grids valid under `rules.THRESH`. Built for the 5th slot of Nov 11; not yet reported as submitted.
- Label thresholds vs safe thresholds: `c13_labelthr` (Tax 16, Business −1157, same seed and schedule as `c7`) ended at 37890 vs 37896. The safe thresholds cost nothing measurable; keep them.
- **Why plain annealing stalls:** at the end every grid has ~33 neighbours at exactly distance 38, so almost any valid cell change pulls ~6 of them down to 37. The 10-NN score gives no reward for turning a 38 into a 39.
- **Fix: pair penalty** added to the annealing objective (the reported score is still the true one): `--lam L --dmin D` subtracts `L·max(0, D − d)` for every pair (optional second tier `--lam2/--dmin2`, decay `--lam-end`). At equal steps it is worth +40–60 points: 2M-step readings 37906–37944 with penalty (best: `--lam 0.5 --dmin 40`, 27 pairs below 38) vs 37836–37880 without. `--lam 1.0 --dmin 39` is too strong early.
- Builds: `bin/anneal3` (penalty table, ~6,500–11,000 steps/s depending on node), `bin/anneal4` (adds `--lam-end`); `src/anneal.py` picks the newest unless `ANNEAL_BIN` is set. `src/best_chains.py --verify` lists the best verified set of every chain.
- Parallel tempering (`bin/pt`, 12 replicas, 30 min): 37902, no better per core than single chains. Dropped.
- Tag collision: `ckpt/d1..d4` already existed from the surrogate era (DONE files), so new runs with those tags exit immediately. Check `ls ckpt` before choosing tags.
- Jobs: `cp2-optP` 28733365 (96 chains x 100M steps, six penalty settings x 16 seeds, tags `p401..p496`, due ~22:00); `cp2-optR2` 28733722 (96 x 240M steps overnight, tags `r701..r796`, pending); workbench `q501..q523` (60M steps, due ~20:15); benchmark `d5..d12`, `dd1..dd4`, `pol1/2` on the second workbench. The no-penalty job `cp2-optN` was cancelled.
- Git: commit `2ceee15` on branch `exact-rules`, pushed to origin (user request). `main` unchanged. The working tree stays on `exact-rules`. `bin/` is ignored.

## 2026-11-11 17:30: C++ annealer, first exact-validity results (0.771–0.773), what limits the score

- **Python chains finished** (`g1..g16`, 1.5M steps): start temperature 1 -> 37782–37802 (0.7711–0.7715); start temperature 2–3 -> ~37838 (0.7722). Slow cooling from a hot start matters; the start set does not.
- **Unconstrained reference** (`g21_unc`, same optimiser, validity off): 37998 (0.77547). So exact validity costs ~0.003 at equal effort, and the leaders' 37986 is just under "every 10-NN distance = 38" (38000).
- **Why the constrained sets stop short** (`explore/tight.py`, `tight2.py` on `g15`): R and P counts are pinned by the true rules too (R 8–11, P 7–9; Tax needs R and few P, Business caps R at the number of jobs, Wellness needs parks). Tax is the tightest advisor (28 grids within 2 of the threshold). 1,622 of 4,950 pairs sit at exactly 38 and 96 pairs are below (93 at 37): score = 38000 − 2·(pairs at 37) − 4·(pairs at 36). The close pairs are spread over all grids (one connected cluster of 83 grids), so this is a global packing problem, not a few bad grids.
- **`src/anneal.cpp` -> `bin/anneal2`** (wrapper `src/anneal.py`): C++ port of the same annealer; rules cross-checked against `rules.py` on 21,700 grids (`python src/anneal.py selftest`); the wrapper re-verifies every final set in Python. ~6,500 steps/s per core (Python: ~400). Options: `--t0 --t1 --steps --nswap (-1 = all swaps) --focus --lam --dmin --cycles`. Resumable (`ckpt/<tag>/state.bin`).
- **`src/pt.cpp` -> `bin/pt`**: parallel tempering (threads). 12 replicas, T 0.5–1.3: no better than one annealing chain at equal steps per replica so far.
- **Ops lesson:** `srun --overlap -c N` steps inside one job are pinned to the *same first N cores* (chains ran at 50–85%). All chains now run in a single `-c 24` step (`explore/run_multi.sh jobs/opt_all.txt logs/optA`); `explore/wb.sh` passes `--cpu-bind=none`.
- Running: `c1..c25` in the workbench (10M–100M steps, schedules and option variants), `cp2-optN` (job 28731489, 96 chains x 150M steps, pending). First C++ readings: 37896 (0.7734) at 6M steps of a 10M-step run.

## 2026-11-11 16:10: all four advisor rules exact; every submission reproduced; exact-validity optimiser running

- **Business solved** (`explore/bus4.py` ... `bus_exact.py`): a labour-market rule. Exact on all 5,000 labels (746/746 levels single-valued, monotone):
  `−24·nI − 37·nC − 32·nP − 22·nO − 8·max(0, 10·nR − 3·nI − 3·nC − 4·nO) + 10·#(I–I adjacent) − 2·#(C not in column 0) − 5·#(C not in bottom row)`
  (each R supplies 10 workers; I and C employ 3, O employs 4; 8 per unemployed worker). Found by: one feature `min(3nR, nI+nC+nO)` cut the residual 73 -> 17 levels; inside each regime the score is linear; ACE on the two-regime family gave R2 = 1.000000 with coefficients that are integers after x10.
- **Transportation confirmed on all labels incl. tails** (218/218 levels). All four rules are in `src/rules.py` (`raw_scores`, `valid`).
- **Thresholds** (first raw score whose label >= 0.75): Wellness 70, Tax 16, Transportation −187, Business −1157. Tax (11 grids) and Business (~7% of grids) sometimes show two adjacent labels for one raw value, not tied to the dataset file; most likely float arithmetic in the organisers' code. Business −1157 straddles 0.75 (labels 0.74975 and 0.75125). `rules.THRESH` is therefore one step inside for those two: **[70, 17, −187, −1155]**; `rules.THRESH_LABEL` = [70, 16, −187, −1157].
- **Every scored submission (S1–S15, 13 files) is reproduced to the exact integer** by the four rules with `THRESH_LABEL` (`explore/verify3.py`): failures were Tax in S6, S7, S9, S10, S13, S14 (margins −1 to −8), Tax + Wellness in S4, S11, S15, Transportation in S1. Business never failed. This also fixes which grids failed in each set.
- **`src/optimize_exact.py`**: the set annealer with exact validity (no model in the loop, ~550 steps/s on one core; start sets are repaired first). Test from S8: 0.7540 -> 0.7576 in 3,000 steps. Production: 20 chains `g1..g20` inside the workbench job (`jobs/opt_g.txt`, logs `logs/optG-<n>.out`), 1.5M–4M steps, starts S8 / S13 / S12 / u1. After 10,000 steps: 0.7667 (S8 start), 0.7687 (S13 start), 0.7638 (u1 start), all 100 grids exactly valid.
- The unconstrained optimum `u1` is far from valid under the true rules (97 of 100 grids invalid), so the leaders' 0.7752 is a constrained optimum, not "validity for free".
- `src/bus_model.py` (un-augmented CNN for Business, 4 levels held-out error) was an interim guard and is no longer used.

## 2026-11-11 (early afternoon): advisors are NOT rotation/flip symmetric; exact rules for Tax, Wellness, Transportation

Second Claude session (scripts in `explore/`, nothing submitted, no existing file changed except this log and `CLAUDE.md`).

- **The symmetry assumption was wrong.** Splitting adjacency by direction (`explore/dir1.py`) drops the Tax residual from 6.3 to 0.9 levels. Every D4-augmented CNN and every invariant feature set averaged this away; that is the "noise floor" all model families hit. Un-augmented sum-pooled CNNs fit the training set to ~0 where augmented ones could not.
- Labels are a quantile transform of an integer raw score; in the central range consecutive label values are consecutive integers (`explore/chain.py`), so the level index is an exact regression target. `blocks.exact_check(f, y)` tests a candidate formula: every label value must map to a single score, monotonically.
- Conventions: `g[r, c]`, row 0 on top; "left of" = smaller column; "above" = smaller row; d = Manhattan distance to the nearest cell of that type; adjacent = 4-neighbour, each pair once. All formulas up to an additive constant.
- **Tax (exact, all 5,000 labels, 360/360 levels single-valued; `explore/tax_exact.py`):**
  `10·nR + 3·nC − 20·nP − 2·#(R–I adjacent) + 8·#(P immediately left of R) + 3·#(P immediately right of R) + 10·#(P directly below R) + 3·#(P directly above R)`
- **Wellness (exact on the 4,847 labelled grids that contain I, C and P: 457/457 levels, monotone; `explore/well_exact.py`):**
  `Σ_R [−3·d(R→C)² − d(R→P)²] + Σ_P 2·d(P→I) + 20·#(P immediately left of R) + 20·#(P–P adjacent) + 15·#(R directly above P) + 4·#(R directly above I)`
- **Transportation (zero residual, all-integer coefficients, on central-range labels 0.08–0.92 with R, I, C present; tails not yet checked; `explore/trans2.py`):**
  `−2·Σ_R d(R→C) − Σ_I d(I→R) − 5·Σ_I d(I→C) − 5·Σ_C d(C→I) − Σ_O d(O→R) + 3·#(I–I adjacent) + 3·#(C–C adjacent) + nR − #(R in column 0)`
- **Business: not solved.** Leads: also directional (un-augmented 3x3 sum-pooled CNN 4.5 levels held-out, R2 0.9986, vs 7.9 with augmentation); a free function of the 5 counts explains ~95.7% (GBM, z); count effects are kinked (rises with nR to ~10 then falls steeply; nC flat to ~7 then falls; nP falls beyond ~10; nI = 0 very bad); raw score is not consecutive integers, so the level index is not a linear target. ACE (`explore/bus3.py`) best central R2 0.973, not exact.
- Not done: (1) `exact_check` of Transportation on all labels, and the values used when a type is absent (profile fit suggests distance 10 in Transportation; unverified; Wellness unknown); (2) Business rule; (3) raw threshold for label >= 0.75 per advisor (smallest raw score whose label >= 0.75); (4) check S1–S15 against the exact rules (certified sets must pass); (5) re-run the set optimiser with exact validity. The b0/z0 surrogates rest on the wrong symmetry assumption.
- Ops: `mit_normal`/`mit_normal_gpu` estimated starts were 17–25 h away; `mit_preemptable` started within a minute (one preemption). Workbench pattern: `sbatch -J cp2-wb -A mit_general -p mit_preemptable -t 12:00:00 -c 24 --mem=64G --wrap "sleep infinity"`, then `explore/wb.sh python -u explore/x.py` (`WBNAME=cp2-wbg` for a GPU one). Both workbench jobs were cancelled when the session stopped.

## 2026-11-11 11:30 — two-model optimiser runs d1-d4 finished

Each grid must pass both model generations (z0 margin / b0 margin). Scores if all valid; expected failures under z0 / b0:
- d1 (0.95 / 0.90): 0.7581020408, 2.4 / 6.4
- d2 (0.95 / 0.80): 0.7596938776, 2.9 / 9.5
- d3 (0.95 / 0.95): 0.7542653061, 1.7 / 3.5
- d4 (0.90 / 0.90): 0.7585306122, 3.5 / 6.8
("NOT FEASIBLE" in the logs is the known bookkeeping quirk; final margins hold.) Same trade-off as z1-z4: every 0.004 of paper score costs about one more expected failure (0.0075 each). None submitted. `cp2-optR` still running.

## 2026-11-11 11:45 — sets rebuilt from confirmed-valid grids only (no gain)

- User idea: pick mutually far grids from each cluster of the atlas, valid grids only. `src/pick_valid.py` (pool = 375 confirmed-valid grids from S2, S3, S5, S8, S14 probes): farthest-point picks per character/layout cluster (proportional and equal quotas), global farthest-point, then exact swap search to a local optimum; also started from S2, S3, S5, S8.
- Farthest-point picks alone: 0.693-0.719. After swap search every start lands at 0.7525-0.7539; S8 itself (0.7540) cannot be improved by any single swap. Final sets contain only S5/S8 grids: no S2, S3 or probe grid survives.
- Same search over confirmed + near-certain grids (1,175): cluster picks (C and L) again end at 0.7539-0.7540 with S5/S8 grids only; starts from S10, S11, S13 stay exactly those sets (no single swap from any other set improves them). The search never mixes sets: each optimised set is a local optimum against the whole pool. Run stopped by its time cap at the S6 start. Log: `logs/pick_valid_near.out`.
- Conclusion: the confirmed pool holds no set better than S8. Outputs: `results/pick_valid_confirmed.json`, `results/pick_valid/*.npy`.

## 2026-11-11 11:10: design atlas (visualisation), probe decode corrected

- User: stop spending submissions on experiments; look at the valid designs for patterns. The 5th slot of Nov 11 is unused unless the user decides otherwise (S16/S17 are built but not submitted).
- **Bug fixed in `src/make_probe.py`**: the outcome table marked failures at positions 90+i although the 8 risky grids sit at 92..99. Tables recomputed (`scratch_tests/redecode.py`). **S14**: still exactly 1 failure, but it is risky #1 (z1 grid 33, b0 P 0.649), not #3. **S15**: still 6 failures; three patterns fit; risky #3, #4, #5 (b0 P 0.10, 0.20, 0.30) are invalid in all three, the most plausible pattern is the six lowest-P grids failing with the 0.45 and 0.60 grids valid. Conclusions unchanged. Log entries corrected.
- `src/viz_data.py` -> `results/viz/designs.json`: 1,184 distinct submitted grids (375 confirmed valid, 800 from sets with 1–4 unknown failures, 4 confirmed invalid, 5 unresolved), own-set and pooled contribution, z0 prediction, t-SNE positions and ward clusters for two similarities (character = counts + adjacency; layout = cell by cell).
- `src/viz_png.py` -> `results/viz/designs.html` (interactive page, published as a private artifact) and PNGs: `map_character.png`, `map_layout.png`, `clusters_character.png`, `clusters_layout.png`, `gallery_confirmed.png`, `gallery_top_bottom.png`.
- First observations: character clusters C1 (538 grids, Industrial-heavy: ~12 Ind, 9 Off), C2 (318, Office 12.5), C3 (220, Office 14), C4/C5 (99, all S1 dataset grids: Office ~18–19, Ind/Com ~5, low contribution). Layout clustering has silhouette < 0: no cell-level structure, as expected for diversity-optimised sets. Parks sit at 7.4–8.1 and Residential at 10–11 in every optimised cluster.

## 2026-11-11 10:40: S13 and S15 results — the loose-margin idea is refuted

- **S15 (tiered probe): 0.7060204082 vs 0.7425306122 -> 6 of the 8 low-confidence grids invalid** (unique decode). Invalid: b0 P 0.07, 0.10, 0.20, 0.30, 0.45, 0.60; valid: the two at 0.03. Models expected ~6.2 failures. The valid region does not extend far past the margin; the "label noise, margins too wide" idea is wrong.
- **S13 (z3): 0.7310408163 vs 0.7612857143 -> 4 failures** (z0 expected 4.9, b0 18.3). With S11 (2 vs 1.3) this says **z0 is roughly calibrated on z0-optimised sets; b0 is ~3x pessimistic.** Expected net score of any single set is therefore ~0.74–0.745 whatever the margin; a clean 0.754+ needs luck or repair.
- New route: **locate and replace failures.** `src/make_half.py` builds a split probe (part of a submitted set + certified S8 grids) and a score table. Built S17 (60 S11 grids + 40 S8; 170 candidate pairs for S11's two failures; 97% chance the score pins down the kept ones; all-valid 0.7133673469) and S16 (50 S13 grids + 50 S8; 101,193 candidate 4-sets, 45% chance; all-valid 0.7090204082). S13 needs 2–3 probes, S11 needs one.
- Submissions used today: 4 of 5 (S11, S14, S13, S15).
- Note: the D/R optimiser logs show "below-pmin N deficit 0.00": float residue in the deficit bookkeeping, the sets are feasible; "best-feasible" is not tracked in those runs, the final state is used.

## 2026-11-11 10:05: "think bigger" — what the leaders' score means, rule-recovery attempt, tiered probe

- **The leader's score is exactly our unconstrained optimum**: 0.7752244898 = integer 37986 = `ckpt/u1/final.npz` (validity ignored). So the top team has 100 valid grids at the diversity ceiling (10-NN distance ~38 of 49 cells; ours ~36.9): for them validity costs nothing. Either their validity model is far better, or the true valid region is much larger than our models' margins allow.
- **Rule recovery with wide bases failed again.** `src/features3.py` (1,060 columns: complete pair-distance basis, nearest-distance histograms, neighbour thresholds, components; `src/train_f3.py` -> `results/f3_summary.json`) and `src/features4.py` (2,900 columns: path distances with blocked district types, per-cell nonlinear density terms; `src/train_f4.py`, log `logs/cp2-f4-*.out`). Best held-out R2(y): Wellness 0.914 (v1+v2+v3) / 0.909 (v3+v4), Tax 0.977, Transportation 0.983, Business 0.986: no jump for any advisor. Wellness stops at ~0.91 for every family and its learning curve is flat, the signature of noise in the labels (the notebook says the advisors are "only human"). If the true advisors are noise-free, our margins are far too conservative; S14 (7 of 8 "risky" grids valid) points the same way.
- **Self-training (`mt1`) finished**: z0 + mt1 held-out R2(y) / sigma_z: Wellness 0.9171/0.277 (z0 0.9132/0.280), Tax 0.9800/0.100, Transportation 0.9861/0.135 (z0 0.9818/0.152), Business 0.9947/0.071. Small real gains, mostly Transportation.
- **S15 built (tiered probe)**: 92 S8 grids + 8 grids with b0 P(valid) 0.60, 0.45, 0.30, 0.20, 0.10, 0.07, 0.03, 0.03 (from w1/w4/w8 finals; `cache/probe_tier.npz`, table `results/probes/S15.json`). All-valid 0.7425306122. Tests directly how far beyond the model margin the valid region extends.
- Started `cp2-optR` (preemptable, 28690686): r1–r3 from z3 with loose two-model margins (z0/b0: 0.80/0.50, 0.60/0.30, 0.40/0.10), so aggressive sets exist if S15/S13 say the region is larger.

## 2026-11-11 09:35: S11 and S14 results

- **S11 (z2, z0 margin 0.98): 0.7427142857 vs 0.7578367347 -> 2 failures** (many candidate pairs, not identifiable). z0 expected 1.3, b0 expected 7.0. So a z0 margin of 0.98 on a z0-optimised set gives the same ~2 failures as b0 at 0.95 on b0-optimised sets: the higher all-valid score is cancelled.
- **S14 (probe): 0.7363469388 vs 0.7422040816 -> exactly 1 of the 8 risky grids failed** (unique decode: z1 grid 39, b0 P 0.703). The other 7, including the one with b0 P 0.432, are valid. b0 expected ~2.6 failures among these 8, z0 ~0.3. First exactly labelled generated grids: 7 valid, 1 invalid (`results/probes/S14.json`).
- Reading: truth sits between the generations (z0 optimistic ~1.5–3x on its own sets, b0 pessimistic ~2.6x). Each set at ~0.757 with 2 failures lands at ~0.742. To beat 0.7540 we need a set at >= 0.757 with zero failures, or identify and replace the failing grids.
- Submissions used today: 2 of 5 (S11, S14).

## 2026-11-11 09:15: cross-check of the z sets, S11–S14 built, two-model runs started

- Slurm back at ~09:00. `cp2-mt1` and `cp2-optS` had been cancelled at 08:12 (not by us); `mt1` resubmitted (28686046).
- **Cross-check (`src/xeval.py` -> `results/xeval.json`, per grid `results/xeval_<tag>.npz`): the first-generation models distrust the z sets.** Expected failures, z0 / b0: z1 3.4 / 13.4, z2 1.3 / 7.0, z3 4.9 / 18.3, z4 4.5 / 16.9. On S4–S10 b0 over-predicted failures about 2.6x and z0 about 1.65x, but those sets were not optimised against z0; here z0 is the exploited model. Realistic guess: z2 ~2–3 failures, z1 ~4–5, i.e. no better than 0.754 unless z0 is right.
- Built: **S11** = z2 (0.7578), **S12** = z1 (0.7594), **S13** = z3 (0.7613), **S14** = probe (`src/make_probe.py`): 92 S8 grids + the 8 z1 grids b0 trusts least (b0 P 0.43–0.73, z0 >= 0.95). All-valid 0.7422; outcome table `results/probes/S14.json` (250 distinct scores for 256 outcomes; unambiguous for <= 2 failures). 10 risky grids gave 300+ collisions, so 8 is the practical size.
- `optimize_soft.py` gained `--blend2/--pmin2` (a second model generation must also accept every grid) and `--lam-hard` (repair pressure in the hard phase; 100 was too weak, which is why z3/z4 stayed slightly below margin).
- Running (`mit_normal_gpu`, 45k steps, ~5 h): d1 z2 start, z0 0.95 + b0 0.90; d2 z2 start, z0 0.95 + b0 0.80; d3 S8 start, z0 0.95 + b0 0.95; d4 z1 start, z0 0.90 + b0 0.90 (jobs 28686044/45).

## 2026-11-11 05:50: optimiser runs finished

Final results (`ckpt/<tag>/final.npz`); "feasible" = every grid at or above the run's margin under the run's own models.

| Run | Models | Start | Setting | Result |
|---|---|---|---|---|
| z1 | z0 | b1 (S8) | hard 0.95, 200 swaps/step | **0.7594 feasible** (int 37211) |
| z2 | z0 | b1 (S8) | hard 0.98, 200 swaps/step | **0.7578 feasible** (int 37134) |
| z3 | z0 | b1 | soft lam0 5 | 0.7613, 43 grids below margin (min P 0.873) |
| z4 | z0 | u1 | soft lam0 0.02 | 0.7574, 30 below (min P 0.887) |
| w7 | b0 | b2 | soft lam0 0.5 | 0.7542 feasible (no gain over start) |
| w1–w6, w8 | b0 | u1 / b2 / x4 | soft | not feasible, 64–96 grids below margin, 0.746–0.756 |
| s1–s4 | b0 | b1 / x4 | swaps | preempted 01:28 at step 18k, never restarted (s1 0.7547, s4 margin 0.5: 0.7655) |

- The gain comes from the z0 models (+0.005 at the same margin), not from the soft-penalty path. Starting from the unconstrained optimum does not work: once hardened, grids below the margin cannot recover.
- `mt1` (self-training) never restarted after its second preemption; no summary.
- Slurm controller unreachable from 02:23 to at least 05:48, so nothing could be submitted, requeued or packaged (packaging runs through `srun`).
- Next, once Slurm answers: score z1–z4 under b0 as well (`eval`), package the day's submissions, requeue `mt1` and the s runs.

## 2026-11-11 02:25: shared network result, mid-run optimiser readings

- **`mt0` (shared four-advisor network) finished: no useful gain.** Held-out R2(y) / sigma_z near threshold, z0 blend -> z0 + mt: Wellness 0.9132/0.280 -> 0.9148/0.279, Tax 0.9789/0.104 -> 0.9798/0.099, Transportation 0.9818/0.152 -> 0.9821/0.151, Business 0.9941/0.074 -> 0.9942/0.074. Alone it beats the single-advisor CNN only on Wellness (0.876 vs 0.870). Not worth a new blend.
- **`mt1` (self-training) preempted twice**, 2 of 5 folds done (Wellness fold R2 0.918 / 0.929 vs 0.891 / 0.896 for mt0). It mostly copies its teacher (z0), so the fold numbers are not yet evidence of a gain; wait for the blend summary.
- Slurm controller unreachable from the login node at 02:23 (`squeue` connect failure); running jobs were still writing logs.
- Optimiser readings (not final): **z1 (z0 models, margin 0.95, swaps) 0.7577 at step 32k and rising; z2 (margin 0.98) 0.7572**; s1 (b0 models + swaps) 0.7547; s4 (margin 0.5) 0.7655; soft runs not yet hardened. `cp2-optS` preempted again at 01:28.

## 2026-11-10 (morning): overnight optimiser results

Both overnight jobs completed (`cp2-optB` 28533948 at 23:00, `cp2-optX` 28548860 at 02:49). Best all-valid score per run, from `ckpt/<tag>/final.npz`:

| Run | Start | Margin | Setting | Score |
|---|---|---|---|---|
| b1 | S5 set | 0.95 | 200k steps | 0.7540 |
| b2 | S5 set | 0.95 | 200k steps | 0.7542 |
| b3 | S5 set | 0.95 | 200k steps | 0.7531 |
| b4 | S5 set | 0.90 | 200k steps | 0.7550 |
| x1 | random pool | 0.95 | swaps 0.03, 4 cycles | 0.7534 |
| x2 | random pool | 0.95 | swaps 0.10, 4 cycles | 0.7529 |
| x3 | random pool | 0.95 | swaps 0.03, 8 cycles | 0.7529 |
| x4 | random pool | 0.95 | no swaps, 1 cycle | 0.7543 |

- Random starts (0.62) reach the same 0.753–0.754 as the runs continued from S5. The plateau is therefore not a local optimum of the search; it is the limit of the region the current surrogates accept at P(valid) >= 0.95.
- Whole-grid swaps and reheating did not help. Lowering the margin to 0.90 gives +0.001.
- Next: widen the accepted region (better Wellness model / recover the advisor rules), not more search.

## 2026-11-10 (late morning): pattern analysis

`src/patterns.py` (srun, ~2 min) writes `results/patterns/report.md` (all tables), `patterns.json`, `fig1_score_vs_count.png`, `fig2_composition.png`, `fig3_cell_maps.png`.

- **Where the score is lost:** our sets average 0.804 over all pairs (maximum possible for 100 grids is 0.808), but the 10 nearest neighbours average 0.754. Composition is not the bottleneck; the spread of pairwise distances is.
- **Counts are pinned by advisor conflicts:** every optimised grid has 7–9 Park and 9–11 Residential cells. Industrial (4–16), Office (4–19), Commercial (6–13) are free.
- **Wellness is the tightest advisor for 61–81 of 100 grids**, although its mean prediction is the highest but one (0.905): its error (sd 0.074 at 0.75–0.85) forces a large safety gap. Tax is next (13–22 grids, mean prediction 0.83).
- **What each advisor responds to (3-fold R2, counts only -> with distances/pairs):** Wellness 0.54 -> 0.86 (parks, and parks near Commercial/Residential); Tax 0.88 -> 0.97 (few parks, more Commercial/Residential); Transportation 0.38 -> 0.92 (Residential scattered and close to every other use); Business 0.94 from counts alone, non-linear (few, unclustered Commercial; more Industrial/Office).
- **Conflicts:** parks (Wellness + / Tax −), Commercial (Tax + / Business −).
- **Wellness model gap:** misses 22.5% of truly valid grids; residual correlates (−0.2) with park–Commercial proximity, so the rule saturates or is shaped differently from what the model learnt.
- Independent runs end ~30 cells apart from each other's nearest grid (36.5 within a set): many different sets reach the same score.
- Follow-up: `cp2-optU` (tag `u1`) anneals with the validity check off, to measure how much of the gap to 0.808 the validity constraint costs.

## 2026-11-11 (01:00): shared network and self-training (user request), running

- `src/train_mt.py`: one CNN trunk with four advisor heads, z targets, all ~20,000 labels, D4 augmentation; same folds per advisor.
- `src/make_pseudo.py`: z0-blend predictions for 100,000 unlabelled dataset grids, kept per fold so the fold-k student only sees teachers that never saw fold k (`cache/pseudo_z0.npz`).
- Jobs on `mit_preemptable`: `cp2-mt0` (28669406, shared network only), `cp2-mt1` (28669407, pseudo-label pre-training then real labels + 0.3 x pseudo loss). Results -> `results/mt0_summary.json`, `results/mt1_summary.json` (also blends with gbmz/cnnz).
- `cp2-optS` was preempted once at ~00:57 and requeued (resumes from checkpoints).
- Reminder saved (CLAUDE.md + memory): targeted-labelling submissions.

## 2026-11-11 (01:00): search for the missing Wellness rule — nothing found

`scratch_tests/wellness_resid.py` (residuals of blend z0, out-of-fold) and `scratch_tests/levels.py`:
- ~190 new candidate features (Euclidean and gravity-style proximity, distance to clusters >= 3, "at least two within radius 2", perimeters, count products and ratios): largest correlation with the residual 0.07; a model predicting the residual from old, new or all features gets R2 < 0. Refit with them added: 0.920 -> 0.923 (z space).
- Learning curve (LightGBM, z space): 1000 grids 0.879, 2000 0.904, 3000 0.915, 4000 0.919. Still rising slowly, so partly data-limited.
- Worst misses are extreme compositions: under-predicted grids are Office-dominated with zero Industrial; over-predicted ones combine many Industrial with many Park cells. Outside the region our sets occupy (Industrial 4–16).
- Level structure: z is linear in the rank of the distinct label values (R2 0.990–0.998), consistent with an integer-spaced raw score. Levels in the middle 84%: Wellness 218, Tax 156, Transportation 90, Business 446. No linear combination of our features reproduces the levels exactly (best: Tax, residual sd 6.4 levels).
- Conclusion: treat the Wellness error (~0.075 near the threshold) as fixed for the rest of the project; gains must come from the optimiser and the margin trade-off.

## 2026-11-11 (00:45): second-generation models (blend `z0`) checked against submissions

`ckpt/gbmz` + `ckpt/cnnz` -> `results/blend_z0.json` (`src/blend_z.py`). OOF, label space, old blend b0 -> new blend z0:

| Advisor | R2 | error sd where prediction >= 0.65 |
|---|---|---|
| Wellness | 0.906 -> 0.913 | 0.076 -> 0.075 |
| Tax | 0.977 -> 0.979 | 0.030 -> 0.029 |
| Transportation | 0.966 -> 0.982 | 0.049 -> 0.035 |
| Business | 0.990 -> 0.994 | 0.027 -> 0.018 |

Check against known failure counts (`src/eval_subs.py b0 z0`): expected failures summed over S4–S10 are 31.7 (b0) and 19.8 (z0) against 12 observed. z0 is better calibrated but cannot tell the clean sets (S5, S8) from the two-failure sets (S6, S7, S9). Wellness is still the weakest advisor for ~70% of grids; Tax's share rises to ~25–30% under z0.

`src/optimize_soft.py` now loads any model generation (`--blend z0`) and records the start state as best when it is feasible. Job `cp2-optZ` (28668543, `mit_preemptable`): z1 hard + swaps at 0.95, z2 at 0.98, z3 soft + swaps, z4 from the unconstrained set; all under z0, 100k steps.

## 2026-11-11 (after midnight): labels are quantile-transformed; z-space models; swap moves

- **User approved `mit_preemptable`.**
- **Label structure** (`scratch_tests/vals.py`): label values lie on a k/999 grid with many exact ties (e.g. 0.437437 x44), i.e. the output of a 1000-point quantile transform of a discrete raw score. 0.75 is the top quartile of the raw score.
- **Modelling z = inverse-normal(label)** (`scratch_tests/ztarget.py`, R2 measured back in label space):

| Advisor | ridge on label | ridge on z | ridge on z, counts + pair counts only | LightGBM on z | old GBM / CNN |
|---|---|---|---|---|---|
| Wellness | 0.826 | 0.893 | 0.752 | 0.909 | 0.898 / 0.858 |
| Tax | 0.909 | 0.974 | 0.968 | 0.972 | 0.966 / 0.973 |
| Transportation | 0.879 | 0.953 | 0.627 | 0.967 | 0.943 / 0.951 |
| Business | 0.721 | 0.789 | 0.677 | 0.979 | 0.964 / 0.988 |

  Tax's raw score is close to a linear function of counts and neighbour-pair counts. Business is strongly non-linear.
- **Second-generation models** in z: `ckpt/gbmz` (done), `ckpt/cnnz` (training), blend `z0`, check against submissions with `src/eval_subs.py b0 z0`.
- **Optimiser gap hypothesis:** with Park pinned to 7–9 and Residential to 9–11, single-cell changes alter two counts at once and are mostly forbidden; two-cell swaps keep counts. Added `--nswap` to `src/optimize_soft.py`. Job `cp2-optS` (28667165, `mit_preemptable`, pending): s1/s3 hard constraint + swaps from b1/x4, s2 soft + swaps, s4 margin 0.5 (how large is the region where the models merely predict valid?).

## 2026-11-11 (00:30): non-CNN predictors (experiment `feat2`, running)

`src/features2.py` (longer-range pair counts, neighbourhood densities, distance histograms, cluster detail, "access to every use" features; all rotation/flip invariant) and `src/train_feat.py` (LightGBM and an MLP on features, same 5 folds). Runs inside job 28665056 (`srun --overlap -n 1 -c 4`), log `logs/cp2-feat2.out`, results `results/feat2_summary.json`.

Wellness (R2 / error sd where prediction >= 0.65): old GBM 0.898 / 0.078; old CNN 0.858 / 0.093; LightGBM v1+v2 features 0.896 / 0.079; v2 only 0.867 / 0.089; MLP 0.876 / 0.095; blend of everything 0.912 / 0.072 (old blend 0.906 / 0.076).
- Four model families and two feature sets all stop at ~0.90 for Wellness. More models of the same kind will not fix it; either the rule uses something none of the features express, or the labels carry noise.
- Other advisors still running.

## 2026-11-10 (23:45 EST): results of S6–S10

Uploaded 23:45–23:48 EST (04:45Z Nov 11). Decoded with `decode_cli.py`, all in `results/submission_log.jsonl`.

| Sub | Set | Margin | All-valid score | Returned | Failed grids |
|---|---|---|---|---|---|
| S6 | b2 | 0.95 | 0.7541836735 | 0.7391224490 | 2 (deficit 738; 1,413 candidate pairs) |
| S7 | x4 | 0.95 | 0.7543061224 | 0.7392040816 | 2 (deficit 740; 1,675 candidate pairs) |
| S8 | b1 | 0.95 | 0.7540000000 | **0.7540000000** | 0, all 100 certified valid; new best, rank 22 |
| S9 | x1 | 0.95 | 0.7534285714 | 0.7384081633 | 2 (deficit 736) |
| S10 | b4 | 0.90 | 0.7550000000 | 0.7247959184 | 4 (deficit 1480) |

- Margin 0.95: 6 failures in 500 submitted grids (S5–S9) = 1.2% per grid; the model predicts 3.6%. Margin 0.90: 4% against a predicted 7%. The models are conservative by about 2–3x, but validity at 0.95 is **not** guaranteed (yesterday's "solved" was too strong).
- One failure costs ~0.0075, more than anything the optimiser gains at fixed margin (~0.001). Scores at 0.95 are a lottery over 0–2 failures.
- The leaderboard lists every submission as its own row (S8 rank 22, the others 61–87), so a worse later submission does not replace a better one.
- Certified all-valid sets so far: S2, S3, S5, S8 (400 grids). S6, S7, S9 are 98/100 valid. This is usable training signal (every advisor >= 0.75 on those grids).
- Ops: `mit_normal` estimated start for a 4-CPU job was Nov 15, so it is unusable now. `srun --jobid=<running job> --overlap` needs `-n 1`, otherwise it runs one copy per task (8 copies raced here; log checked intact).

## 2026-11-10 (23:30–midnight): unconstrained ceiling, S6–S10, "work backwards" runs

- **Unconstrained ceiling** (`cp2-optU` 28663463, tag `u1`, validity check off, 40k steps from b2): 0.7542 -> **0.7752**, about the leaders' 0.7749. So the 0.021 gap is the cost of our validity margin. `ckpt/u1/final.npz` is not valid (every grid P(valid) ~ 0).
- **Submissions S6–S10 built at 23:41** (no slot had been used on Nov 10; user is uploading): S6 = b2 (0.7541836735), S7 = x4 (0.7543061224), S8 = b1 (0.7540000000), S9 = x1 (0.7534285714), S10 = b4 (0.7550000000, margin 0.90). Scores to be filled in when reported.
- **New `src/optimize_soft.py`** (user: "work backwards to find a valid region with the highest score"): maximises score − lam × validity deficit, deficit = max(0, log pmin − log P(valid)) per grid; lam rises geometrically, then the constraint is hard. All 196 neighbours ranked by exact gain and CNN-estimated validity change, top 8 confirmed with the blend.
- Jobs `cp2-optW1` (28665054) and `cp2-optW2` (28665056), 150k steps each, ~5 h:
  - w1–w4: from `u1` (0.7752, invalid), lam0 0.003–0.03, different soft fractions and temperatures.
  - w5–w7: from valid sets b2/x4 with a soft constraint (lam0 0.5–5).
  - w8: from `u1`, margin 0.90.
- Timing note: `mit_normal_gpu` took 5 min to start a short srun at 23:36.

## 2026-11-10 (midday): rotation / flip / shift test (user question)

`scratch_tests/sym_test.py` on sets b2 and x4 (job `cp2-sym`):
- Re-orienting any single grid (7 rotations/flips each) never raises the score: 0 of 100 grids changed in both sets.
- Giving every grid a random orientation lowers the score to ~0.690 (from 0.754): the optimiser has already placed grids relative to each other.
- A grid is on average 36.3 cells from its own rotations/flips, closer than the average pair in the set (39.4).
- Cyclic shifts (wrap-around translation) are not a symmetry: the models rate only 4% of shifted grids at P(valid) >= 0.95; mean Wellness prediction falls from 0.91 to 0.78.
- GBM model is not exactly symmetric: only ~60% of rotated/flipped copies keep P(valid) >= 0.95 (all stay above 0.76).

## 2026-11-09 (evening): first models, exploration, submission S1

**Plan approved** (`~/.claude/plans/hazy-moseying-perlis.md`). User rule: use all 5 submissions every day, including today.

**Exploration** (job 28509441, `results/gbm0_summary.json`)
- No grid is labelled by all four advisors; 283 have 2 labels, 3 have 3.
- Each advisor passes ~24–25% of its labelled grids.
- Wellness vs Tax scores correlate at −0.53 on 51 shared grids; none passed both. Wellness-valid grids average 10.5 parks, Tax-valid 4.6.
- Scores take few distinct values (488 / 360 / 241 / 818 unique among 5,000), so the advisors are probably functions of integer quantities.

**Surrogates, 5-fold out-of-fold R²**

| Advisor | GBM (features) | CNN no aug | CNN + rot/flip aug | Blend | Blend error near top (sigma) |
|---|---|---|---|---|---|
| Wellness | 0.898 | 0.761 | 0.858 | 0.906 | 0.076 |
| Tax | 0.966 | 0.953 | 0.973 | 0.977 | 0.030 |
| Transportation | 0.943 | 0.672 | 0.951 | 0.966 | 0.049 |
| Business | 0.964 | 0.816 | 0.988 | 0.990 | 0.027 |

- Rotation/flip augmentation helps every advisor a lot, and invariant-only features match all features: the advisors are rotation/flip symmetric.
- Most important features: distances between district types and district counts.
- Wellness is the weakest model and the main validity risk.
- Jobs: `cp2-gbm0` 28509441 (4 min), `cp2-cnn0` 28509442 (12 min). Blend in `results/blend_b0.json`.

**Dataset candidates**
- 1,041 dataset grids have blended min-advisor prediction >= 0.75; 349 have model P(valid) >= 0.99.
- They are tightly clustered: the best 100-subset of those 349 scores only 0.3158 even if all are valid. High scores need generated grids.

**Submission S1** — `submissions/S1_20261109_164304.npy` (sha 47fb50c4)
- 100 dataset grids, P(valid) >= 0.99. All-valid score would be exactly 0.3157551020 (integer 15472).
- Purpose: test the surrogates on in-distribution grids. Exact match = all 100 valid. Each invalid grid lowers the score by roughly 0.003.
- **Result: 0.3136530612** (integer 15369, deficit 103; leaderboard row 194, submitted 21:52 UTC). Decode: exactly 1 invalid grid, one of positions 27 / 40 / 99 (dataset ids 417209 / 103040 / 56313). The other 97 are certified valid. Conclusion: surrogates are trustworthy on dataset grids at P(valid) >= 0.99.

**Generated candidates, round 0 of `cp2-search0`** (`cache/cands_s0.npz`, verify job 28518997)
- 3,388 unique grids reached the CNN target. Median 25 cells from the nearest of the top-2000 dataset grids.
- Wellness is the weakest advisor for 3,358 of them. CNN mean Wellness prediction 0.916 vs GBM 0.835: the search exploits the CNN, so GBM agreement is required.
- Blended P(valid) >= 0.99: 345; >= 0.95: 1,776.
- Search speed: ~10 s per step, ~20 min per round. Needs speeding up.

**Submission S2** — `submissions/S2_20261109_171634.npy` (sha 56bb5199)
- 100 generated grids, blended P(valid) >= 0.99 (pool 345). All-valid score 0.6329183673 (integer 31013).
- Purpose: do the surrogates hold on out-of-dataset grids at the high-confidence margin?
- **Result: 0.6329183673, exact match. 100/100 valid.**

**Submission S3** — `submissions/S3_20261109_171637.npy` (sha faa70001)
- 100 generated grids, P(valid) in [0.95, 0.99) (pool 1,431 before selection). All-valid score 0.6811224490 (integer 33375). Model expects ~97 valid.
- Purpose: failure count at a lower margin, to decide how aggressive S4/S5 can be.
- **Result: 0.6811224490, exact match. 100/100 valid** (model expected ~3 failures: surrogates are conservative).

**Set optimiser** (`src/optimize_set.py`)
- Changes one cell of one grid at a time; computes the exact change in the set's 10-NN score for all 196 single-cell changes; cheap CNN prefilter, then GBM check; accepts only if blended P(valid) >= `--pmin`.
- First version (strict improvements only), from S3, 40k steps: `o1` (pmin 0.95) 0.7301; `o2` (pmin 0.85) 0.7348. Both stalled near 0.73.
- Added simulated annealing (Gumbel-perturbed ranking + Metropolis threshold, T 3 -> 0.2), best-state tracking, per-interval logging/checkpointing (earlier version only logged/checkpointed on accepted steps).
- Annealing runs `a1`–`a4` (job 28527169, 4 runs on one GPU via `jobs/multi.sbatch`), 60k steps: at step 8000 already 0.745 (pmin 0.95), 0.750 (0.90), 0.750 (0.85).

**Submission S4** — `submissions/S4_20261109_181235.npy` (sha 9d300cb6)
- `o2` output, every grid at P(valid) >= 0.85. All-valid score 0.7348163265 (integer 36006). Model expected ~93.5 valid.
- **Result: 0.7202448980** (integer 35292, deficit 714): 2 failures (possibly 3). 395 pairs reproduce the score, so the failed grids are not identified.
- Conclusion: the 0.85 margin gains ~0.005 diversity over 0.95 but loses ~0.015 to failures. Use >= 0.95 (possibly 0.90) for score attempts.

**Annealing runs, final (job 28527169, 60k steps, ~65 min)**

| Run | pmin | Temperature | All-valid score | Model-expected valid |
|---|---|---|---|---|
| a1 | 0.95 | 3 -> 0.2 | 0.7521 | 96.3 |
| a4 | 0.95 | 6 -> 0.3 | 0.7531 | 96.3 |
| a3 | 0.90 | 3 -> 0.2 | 0.7539 | 92.8 |
| a2 | 0.85 | 3 -> 0.2 | 0.7545 | 89.7 |

- The margin barely changes the achievable diversity (0.7521–0.7545), so the safe margin is the right choice.

**Submission S5** — `submissions/S5_20261109_193339.npy` (sha 3b5be900)
- Run `a4`, every grid at P(valid) >= 0.95. All-valid score 0.7531224490 (integer 36903).
- Purpose: main score attempt of day 1.
- **Result: 0.7531224490, exact match. 100/100 valid. Leaderboard rank 16** (scored 02:06 UTC Nov 10).

**Day 1 summary (5 of 5 submissions used)**

| # | Content | Margin | All-valid | Returned | Valid |
|---|---|---|---|---|---|
| S1 | dataset grids | >= 0.99 | 0.3158 | 0.3137 | 99 |
| S2 | generated | >= 0.99 | 0.6329 | 0.6329 | 100 |
| S3 | generated | 0.95–0.99 | 0.6811 | 0.6811 | 100 |
| S4 | set-optimised | >= 0.85 | 0.7348 | 0.7202 | ~98 |
| S5 | annealed set | >= 0.95 | 0.7531 | 0.7531 | 100 |

- 400 grids at P(valid) >= 0.95 submitted, 1 failure (in S1). Surrogates are reliable and conservative at that margin.

**Running overnight (checked 21:35 EST)**
- `cp2-optB` (28533948): 4 longer annealing runs (200k steps). At 100k–134k steps: 0.7540 / 0.7542 (pmin 0.95), 0.7531 (0.97), 0.7550 (0.90). Plateau at ~0.754–0.755: single-cell annealing under the current surrogates is close to exhausted.
- `cp2-search0` finished all 12 rounds (~37k candidates in `ckpt/s0/`).
- `cp2-cnn1` (28546xxx, see `squeue`): larger CNN (width 192, 10 blocks, 300 epochs, augmentation) to compare with `cnn0`, mainly for Wellness.

**Late evening (22:00–22:15 EST)**
- `cp2-cnn1` done (24 min). Larger CNN, OOF R²: Wellness 0.855 (cnn0: 0.858), Tax 0.974 (0.973), Transportation 0.964 (0.951), Business 0.988 (0.988). More capacity does not help Wellness; it needs better features or the actual rule.
- Full candidate pool verified (`cache/cands_s0.npz`, job 28548376): 36,982 unique candidates, 17,918 with blended P(valid) >= 0.95, 3,513 >= 0.99.
- User asked for a separate exploration run to avoid local optima. Optimiser extended: random-pool starts (`--init pool`), whole-grid replacement moves from the pool (`--p-swap`), reheating cycles (`--cycles`).
- `cp2-optX` (28548860) started 22:13: runs x1–x4, 250k steps, pmin 0.95, random starts (start score ~0.62), swap probability 0.03 / 0.10 / 0.03 / 0, cycles 4 / 4 / 8 / 1.
- Queue problem: the verification job sat pending on `mit_normal` (estimated start 04:30; partition 97% allocated, ~1,900 pending). Moved it to the free `mit_normal_gpu` slot, where it started immediately.

**Plan for Nov 10**
1. Improve the Wellness surrogate (sigma 0.076 vs ~0.03 for the others); try to recover the advisors' rules from residuals. A tighter model enlarges the set of grids counted as safe, which is what limits diversity now.
2. Re-run the annealing optimiser under the improved models; add multi-cell moves.
3. Spend the 5 submissions on the best sets, at >= 0.95 unless a better-calibrated model supports less.

**GPU limits**
- `mit_normal_gpu`: 2 GPUs, 32 CPUs per user. `mit_normal`: 96 CPUs. `mit_preemptable`: 4 GPUs (not used; user asked for `mit_normal`/`mit_normal_gpu`).
- Workaround: `jobs/multi.sbatch` runs several small resumable commands on one GPU and requeues them together.

**Changes made alongside these submissions**
- `src/divscore.py`: fixed `decode` for the zero-failure case.
- `src/search.py`: fixed `.min(1)` tuple bug found by the quick test.
- `src/make_submission.py`: added `--pmax` to select a probability band.
- New: `src/verify_cands.py` (GBM check + blend of search output), `src/decode_cli.py` (decode a returned score and write it into the log).

**Correction to the plan's decode idea**
- One score carries about 15 bits. Exact identification of failed grids is reliable for at most ~2 failures; for more, only the *number* of failures is readable (from the size of the deficit). The all-valid certificate is still exact.

**Running**
- `cp2-search0` (28514418, `mit_normal_gpu`): local search on the CNN ensemble for new grids, target pessimistic min-advisor >= 0.88.

**Code added**: `jobs/run.sbatch`, `src/{common,features,train_gbm,train_cnn,surrogate,blend,search,divscore,make_submission}.py`. Submission log: `results/submission_log.jsonl`.

## 2026-11-09

**Setup**
- Cloned `git@github.com:Prad-R/2155-CP2-2026.git` into `/home/pradr/2.156/cp_2` (HEAD `221250c`).
- Read `README.md`, `utils_public.py` and the cell sources of `starter_notebook.ipynb`. Task summary written to `CLAUDE.md`.
- Created `CLAUDE.md` and `UPDATES.md` (both untracked, not committed).

**Cluster facts checked**
- Slurm account: `mit_general`. `mit_normal`: 12 h limit, CPU. `mit_normal_gpu`: 6 h limit (L40S / H100 / H200 / B200).
- Login slice: 4 CPUs, 10 GB RAM. No jobs currently queued.
- `uv`, `conda`, `module` not on PATH in the non-interactive shell.

**Environment (done)**
- User said to use conda via modules. Built env `2.156_cp_2` with `scripts/setup_env.sh` on `mit_normal` (job 28503608, node3106, ~10 min). Log: `logs/setup_env.log`.
- Added `torch`, `lightgbm`, `xgboost` on top of `requirements.txt`.
- GPU smoke test on `mit_normal_gpu` (job 28506320, L40S): `torch.cuda.is_available()` True, matmul ran.

**Information from the user**
- 5 submissions per day; each returns only the diversity score.
- Leaderboard screenshot: top score 0.7749, top 10 >= 0.7566; scores shown to 10 decimals and are exact multiples of 1/49000.
- Deadline Nov 13 end of day, with report time needed before it. Plan: lock submission by end of Nov 12.

**Decisions**
- Pipeline will log every submission and use the returned scores (calibration, confirming valid grids).
- Flagged to the user: designing submissions purely to decode hidden labels may conflict with course rules; user to check.

**Not done yet**
- No data loaded, no modelling code, no modelling jobs.

**Open questions for the user**
- How much of the solution Claude should build versus guide (course collaboration/AI policy).
- Whether any of today's 5 submissions have been used.
