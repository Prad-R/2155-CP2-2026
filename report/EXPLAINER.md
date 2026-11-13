# What has been done in this project (plain-language guide)

## The task in one paragraph
Submit 100 city layouts (7x7 grids of 5 district types). A layout only counts if all four hidden "advisors" rate it >= 0.75. Among the valid layouts, the score rewards how *different* each one is from its 10 most similar neighbours. So you need many valid layouts that are all far apart from each other. One invalid layout loses its whole contribution, and we only see one number per upload.

## The story in four stages

1. **Guess the advisors with ML (Nov 9-10).** Only 5,000 of 500,000 grids have ratings. We trained tree models and CNNs to predict each advisor, then searched for 100 grids the models thought were valid and diverse. Best result: 0.754. It stopped improving because the models are never fully sure, and being wrong on even a few grids costs a lot. Probes (S15 etc.) showed that the truly valid region is not much bigger than what the models predicted.

2. **Find the advisors' exact formulas (Nov 11).** This was the breakthrough. Two clues:
   - The ratings are a *rank transform* of an integer score (lots of identical values), so the real score is an integer formula.
   - I had assumed the advisors treat rotated/flipped grids the same. That was wrong. They care about direction (for example "park immediately to the left of residential"). Once direction was included, the formulas could be recovered *exactly* from the labels.
   The formulas are in `src/rules.py`. They reproduce all labels and every past submission score to the integer. Validity is now certain, no guessing.

3. **Optimise the whole set under exact validity (Nov 11-12).** A simulated-annealing search (a C++ program, `src/anneal.cpp`, running 96 copies on the cluster) changes single cells of the 100 grids, keeping every grid valid, and keeps changes that raise the diversity score. This got 0.7753 (S19), which was rank 1 when uploaded.

4. **Break the 0.7755 ceiling with structure (Nov 12 night).** Plain annealing gets stuck at 38000/49000 = 0.7755, because every grid ends up with too many neighbours at distance exactly 38. A mathematical construction (125 patterns from a code over GF(5)) has all pairwise distances 39, 40 or 49, which would give 0.796 if all were valid. Only 49 of them could be made valid. But using those 49 as a **fixed core** and annealing the other 51 grids *hot* (a high "temperature", so the search moves a lot) gave **38048 = 0.7764897959 (S22)**. This is the best set.

## Which file to upload
`submissions/S22_20261112_233848.npy` (score predicted exactly: 0.7764897959, all 100 grids valid). It is gitignored, so it exists only on the cluster. Upload it shortly before the deadline, as you decided. Upload it as is; do not edit it.

## Important numbers
| Item | Value |
|---|---|
| Best uploaded (returned by the leaderboard) | S19, 0.7753061224 |
| Best built (final) | S22, 0.7764897959 (38048) |
| Held-back files | S20 (37996), S21 (37997), S22 (38048) |
| Overnight run `cp2-optR5` | prepared and tested, **not started** (needs your go-ahead); it could add a few more points |

## What's in the repo (short map)
- `src/rules.py`: the exact advisor formulas and the validity test.
- `src/anneal.cpp`, `src/anneal.py`, `src/islands.py`: the optimiser and its multi-stage runner.
- `explore/core_prep.py`, `core_scan.sh`, `core_cool.sh`: the fixed-core method that produced S22.
- `src/make_exact_submission.py`: builds a submission file and refuses unless all 100 grids are distinct and valid.
- `UPDATES.md`: full dated log. `CLAUDE.md`: stable rules and facts. `results/submission_log.jsonl`: every submission with its predicted/returned score.
- `report/report.tex`: the 4-page report. `report/EXPLAINER.md`: this file.

## About the report
`report/report.tex` is plain LaTeX (article class, 11pt, 1in margins). There is no LaTeX compiler on the cluster, so **I could not compile it or check the page count**; I aimed for about 4 pages. Compile it on your machine with `pdflatex report.tex` (run twice) or paste it into Overleaf. If it is more than 4 pages, trim Section 3.1 or the table first.

Things to check before submitting the report:
- Replace the author line with your name(s).
- If you have the leaderboard score for S22 by then, add it to Section 1.
- The report states the S22 score as predicted, not returned, because the file was held back.
