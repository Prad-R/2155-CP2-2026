"""Verify the exact rules on every label (incl. tails), extract the raw thresholds, and test all past submissions."""
import numpy as np, json, os, sys
from lib import *
from blocks import exact_check
from rules import raw_scores, NAMES

np.set_printoptions(linewidth=220, suppress=True)
G, Y = data()
ADV_COL = {"Wellness": 0, "Tax": 1, "Transportation": 2}
thr = {}
for j, nm in enumerate(NAMES):
    a = ADV_COL[nm]
    idx = np.where(~np.isnan(Y[:, a]))[0]
    y = Y[idx, a]
    raw, ok = raw_scores(G[idx])
    f = raw[:, j]
    print(f"===== {nm}: labelled {len(idx)}, with R,I,C,P present {ok.sum()}")
    exact_check(f[ok], y[ok])
    # threshold
    hi = f[ok][y[ok] >= 0.75].min(); lo = f[ok][y[ok] < 0.75].max()
    print(f"   smallest raw with label >= 0.75: {hi} (label {y[ok][f[ok] == hi].min():.6f});  largest raw with label < 0.75: {lo} (label {y[ok][f[ok] == lo].max():.6f})")
    thr[nm] = int(hi)
    # raw -> label table near the threshold
    for r in range(hi - 3, hi + 4):
        m = ok & (f == r)
        if m.any():
            print(f"      raw {r}: label {np.unique(np.round(y[m], 6))} n={m.sum()}")
    # grids missing a type: do they still follow the monotone map?
    bad = ~ok
    if bad.any():
        # predicted label from the map (nearest known raw)
        fr, yr = f[ok], y[ok]
        o = np.argsort(fr); fr, yr = fr[o], yr[o]
        pred = np.interp(f[bad], fr, yr)
        print(f"   grids missing a type: {bad.sum()}; |label - map(raw)| quantiles {np.quantile(np.abs(pred - y[bad]), [.5, .9, 1]).round(3)}")
print("thresholds:", thr)
json.dump(thr, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "exact_thresholds.json"), "w"))
# past submissions
log = [json.loads(l) for l in open("results/submission_log.jsonl")]
T = np.array([thr[nm] for nm in NAMES])
print("\nsubmission check (number of grids failing each exact rule):")
for d in log:
    S = np.load(d["file"])
    raw, ok = raw_scores(S)
    fail = raw < T[None]
    anyf = fail.any(1) | ~ok
    marg = (raw - T[None])
    print(f"  {d['name']:4s} returned {d.get('returned_score')}  expected {d['expected_score']:.4f} | missing-type {int((~ok).sum())} | fail W {fail[:, 0].sum():3d} T {fail[:, 1].sum():3d} Tr {fail[:, 2].sum():3d} | any {anyf.sum():3d} | "
          f"min margin W {marg[:, 0].min():4d} T {marg[:, 1].min():4d} Tr {marg[:, 2].min():4d} | failing positions {np.where(anyf)[0][:14].tolist()}")
