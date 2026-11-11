"""Business: ACE (alternating conditional expectations) to recover the raw scale: find a monotone transform theta(level)
that is best fit by an additive model in the chosen feature blocks."""
import numpy as np, sys
from lib import *
from blocks import *
from sklearn.isotonic import IsotonicRegression

np.set_printoptions(linewidth=220, suppress=True, precision=2)
a = int(sys.argv[1]) if len(sys.argv) > 1 else 3
g, y, z, lev, idx = labelled(a)
n = len(g)
cnt = counts(g).astype(int)
Of = onehot(g)
B = {}
B["cnt"] = cnt[:, 1:].astype(float)
B["cntoh"] = np.concatenate([(cnt[:, k:k + 1] == np.arange(1, 50)[None]).astype(np.float32) for k in range(5)], 1)
B["ord4"] = ordered_pairs(g)[0]
B["ord8"] = ordered_pairs(g, ((1, 1), (1, -1)))[0]
B["prof_man"] = near_profile(g, MAN, list(range(1, 9)), "man")[0]
B["cnt2"] = np.stack([cnt[:, i] * cnt[:, j] for i in range(5) for j in range(i, 5)], 1).astype(float)
nl = lev.max() + 1
wts = np.bincount(lev, minlength=nl).astype(float)


def ace(X, iters=40, lam=1e-2, verbose=False):
    X = np.hstack([X, np.ones((n, 1))]).astype(np.float64)
    keep = X.std(0) > 0; keep[-1] = True
    X = X[:, keep]
    G = X.T @ X + lam * np.eye(X.shape[1])
    theta = z.copy()
    for it in range(iters):
        w = np.linalg.solve(G, X.T @ theta)
        p = X @ w
        m = np.bincount(lev, weights=p, minlength=nl) / wts
        iso = IsotonicRegression().fit(np.arange(nl), m, sample_weight=wts)
        th = iso.predict(np.arange(nl))[lev]
        th = (th - th.mean()) / th.std()
        theta = th
    w = np.linalg.solve(G, X.T @ theta)
    p = X @ w
    r2 = 1 - ((theta - p) ** 2).sum() / (theta ** 2).sum()
    return r2, theta, p, w, keep


for combo in (["cnt"], ["cntoh"], ["cntoh", "ord4"], ["cnt", "cnt2", "ord4"], ["cntoh", "cnt2", "ord4"], ["cntoh", "ord4", "ord8"], ["cntoh", "ord4", "prof_man"], ["cntoh", "cnt2", "ord4", "prof_man"]):
    X = np.hstack([B[k] for k in combo])
    r2, theta, p, w, keep = ace(X)
    cen = (y > 0.02) & (y < 0.98)
    r2c = 1 - ((theta - p)[cen] ** 2).sum() / ((theta[cen] - theta[cen].mean()) ** 2).sum()
    print(f"  ACE {'+'.join(combo):40s} cols {X.shape[1]:4d}  R2 {r2:.5f}  (central {r2c:.5f})  resid sd / sd {np.sqrt(1 - r2):.4f}")
