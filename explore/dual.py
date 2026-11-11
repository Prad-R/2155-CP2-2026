"""Raw values that carry two different labels: does the label depend on the dataset file (chunk of 100k)?"""
import numpy as np, sys
from lib import *
sys.path.insert(0, "src")
from rules import raw_scores
G, Y = data()
def bus_raw(g):
    cnt = np.stack([(g == k).sum((1, 2)) for k in range(5)], 1).astype(np.int64); nR, nI, nC, nP, nO = cnt.T
    I = (g == 1); C = (g == 2)
    II = (I[:, :, :-1] & I[:, :, 1:]).sum((1, 2)) + (I[:, :-1, :] & I[:, 1:, :]).sum((1, 2))
    return -24 * nI - 37 * nC - 32 * nP - 22 * nO - 8 * np.maximum(0, 10 * nR - 3 * nI - 3 * nC - 4 * nO) + 10 * II - 2 * C[:, :, 1:].sum((1, 2)) - 5 * C[:, :-1, :].sum((1, 2))
for a, nm in enumerate(ADVISORS):
    idx = np.where(~np.isnan(Y[:, a]))[0]
    y = np.round(Y[idx, a], 7)
    if a == 3:
        f = bus_raw(G[idx]); ok = np.ones(len(idx), bool)
    else:
        raw, ok = raw_scores(G[idx]); f = raw[:, {0: 0, 1: 1, 2: 2}[a]]
    chunk = idx // 100000
    multi = 0; bychunk = 0; shown = 0
    for r in np.unique(f[ok]):
        m = ok & (f == r)
        u = np.unique(y[m])
        if len(u) > 1:
            multi += 1
            # is label a function of chunk within this raw value?
            consistent = all(len(np.unique(y[m & (chunk == c)])) <= 1 for c in range(5))
            bychunk += consistent
            if shown < 4 and 0.3 < u[0] < 0.9:
                shown += 1
                print(f"   {nm} raw {r}: " + "; ".join(f"chunk {c}: {np.unique(y[m & (chunk == c)])} (n={int((m & (chunk == c)).sum())})" for c in range(5)))
    print(f"{nm}: raw values with >1 label: {multi} of {len(np.unique(f[ok]))}; of those, label is a function of the dataset file: {bychunk}")
