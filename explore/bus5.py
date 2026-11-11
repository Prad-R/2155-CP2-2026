"""Business: greedy forward search over a library of count features (min of scaled subset sums, hinges, products)
on top of linear local blocks."""
import numpy as np, sys, itertools
from lib import *
from blocks import *

np.set_printoptions(linewidth=220, suppress=True, precision=2)
g, y, z, lev, idx = labelled(3)
cen = (y > 0.05) & (y < 0.95)
g, y, z, lev = g[cen], y[cen], z[cen], lev[cen]
n = len(g)
t = (lev - lev.mean()).astype(float)
cnt = counts(g).astype(float)
base = np.hstack([cnt[:, 1:], ordered_pairs(g)[0], near_profile(g, MAN, list(range(1, 9)), "man")[0], np.ones((n, 1))])
keep = base.std(0) > 0; keep[-1] = True
base = base[:, keep]
subsets = [s for k in range(1, 5) for s in itertools.combinations(range(5), k)]
sname = lambda s: "+".join("n" + SYM[i] for i in s)
ssum = {s: cnt[:, list(s)].sum(1) for s in subsets}
lib = {}
ratios = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (1, 4), (4, 1), (1, 5), (5, 1), (3, 4), (4, 3), (2, 5), (5, 2)]
for s1, s2 in itertools.combinations(subsets, 2):
    if set(s1) & set(s2):
        continue
    for a, b in ratios:
        lib[f"min({a}*({sname(s1)}), {b}*({sname(s2)}))"] = np.minimum(a * ssum[s1], b * ssum[s2])
for s in subsets:
    for c in range(1, 31):
        lib[f"relu({sname(s)} - {c})"] = np.maximum(ssum[s] - c, 0)
for i in range(5):
    for j in range(i, 5):
        lib[f"n{SYM[i]}*n{SYM[j]}"] = cnt[:, i] * cnt[:, j]
names = list(lib)
F = np.stack([lib[k] for k in names], 1)
print(f"library: {F.shape[1]} features; base cols {base.shape[1]}; sd(level) {t.std():.1f}")
X = base
start = sys.argv[1:] if len(sys.argv) > 1 else []
for nm in start:
    X = np.hstack([X, lib[nm][:, None]])
for it in range(8):
    Q, _ = np.linalg.qr(X)
    r = t - Q @ (Q.T @ t)
    Fp = F - Q @ (Q.T @ F)
    nrm = (Fp ** 2).sum(0)
    gain = np.where(nrm > 1e-6, (r @ Fp) ** 2 / np.maximum(nrm, 1e-6), 0)
    o = np.argsort(-gain)[:6]
    print(f"iter {it}: current resid sd {r.std():.3f}; best additions: " + " | ".join(f"{names[j]} -> {np.sqrt(max(r.var() - gain[j] / n, 0)):.3f}" for j in o), flush=True)
    X = np.hstack([X, F[:, o[0]][:, None]])
