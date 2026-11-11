"""Check the explicit integer Business formula on all labelled grids."""
import numpy as np, sys
from lib import *
from blocks import exact_check
np.set_printoptions(linewidth=220, suppress=True, precision=3)
g, y, z, lev, idx = labelled(3)
cnt = counts(g).astype(np.int64); nR, nI, nC, nP, nO = cnt.T
I = (g == 1); C = (g == 2)
II = (I[:, :, :-1] & I[:, :, 1:]).sum((1, 2)) + (I[:, :-1, :] & I[:, 1:, :]).sum((1, 2))
Cleft = C[:, :, 1:].sum((1, 2)); Cbelow = C[:, :-1, :].sum((1, 2))
f = -24 * nI - 37 * nC - 32 * nP - 22 * nO - 8 * np.maximum(0, 10 * nR - 3 * nI - 3 * nC - 4 * nO) + 10 * II - 2 * Cleft - 5 * Cbelow
s, tot, mono = exact_check(f, y)
vals, inv = np.unique(np.round(y, 7), return_inverse=True)
lo = np.full(len(vals), np.inf); hi = np.full(len(vals), -np.inf)
np.minimum.at(lo, inv, f); np.maximum.at(hi, inv, f)
bad = np.where(lo != hi)[0]
print("levels with spread:", len(bad), " non-monotone steps:", int((lo[1:] < hi[:-1]).sum()))
for k in bad[:12]:
    u, c = np.unique(f[inv == k], return_counts=True)
    print(f"   label {vals[k]:.5f}: f values {dict(zip(u.astype(int), c))}")
thr_hi = f[y >= 0.75].min(); thr_lo = f[y < 0.75].max()
print(f"smallest raw with label >= 0.75: {thr_hi} (label {y[f == thr_hi].min():.6f}); largest raw with label < 0.75: {thr_lo} (label {y[f == thr_lo].max():.6f})")
for r in range(thr_hi - 4, thr_hi + 5):
    m = f == r
    if m.any():
        print(f"   raw {r}: labels {np.unique(np.round(y[m], 6))} n={m.sum()}")
d = np.diff(np.sort(np.unique(f[(y > 0.3) & (y < 0.9)])))
print("gaps between consecutive distinct raw values in the central range:", np.unique(d, return_counts=True))
