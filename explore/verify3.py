"""All four exact rules: label check, then every past submission (exact score reproduction with the label thresholds)."""
import numpy as np, json, sys
from lib import *
from blocks import exact_check
from rules import raw_scores, NAMES, THRESH, THRESH_LABEL, valid
from divscore import int_score, DENOM
G, Y = data()
for j, nm in enumerate(NAMES):
    idx = np.where(~np.isnan(Y[:, j]))[0]
    raw, ok = raw_scores(G[idx]); f = raw[:, j]; y = Y[idx, j]
    m = ok if j in (0, 2) else np.ones(len(idx), bool)
    print(nm, "grids used", m.sum(), end=" "); exact_check(f[m], y[m])
    print(f"    labels at THRESH-1 {np.unique(np.round(y[m & (f == THRESH[j] - 1)], 5))}, at THRESH {np.unique(np.round(y[m & (f == THRESH[j])], 5))}; min label with raw >= THRESH: {y[m & (f >= THRESH[j])].min():.5f}; max label below: {y[m & (f < THRESH[j])].max():.5f}")
log = [json.loads(l) for l in open("results/submission_log.jsonl")]
allok = True
for d in log:
    S = np.load(d["file"]); raw, ok = raw_scores(S)
    v = ok & (raw >= THRESH_LABEL[None]).all(1)
    pred = int_score(S, v); ret = d.get("returned_score")
    mg = raw - THRESH_LABEL[None]
    tag = "not submitted" if ret is None else ("EXACT MATCH" if pred == int(round(ret * DENOM)) else "MISMATCH")
    if ret is not None:
        allok &= pred == int(round(ret * DENOM))
    print(f"  {d['name']:4s} invalid {int((~v).sum())} [W {int((mg[:,0]<0).sum())} T {int((mg[:,1]<0).sum())} Tr {int((mg[:,2]<0).sum())} B {int((mg[:,3]<0).sum())}]  min margins {mg.min(0).tolist()}  predicted {pred/DENOM:.10f}  {tag}; safe-threshold valid {int(valid(S).sum())}")
print("all scored submissions reproduced exactly with four rules:", allok)
