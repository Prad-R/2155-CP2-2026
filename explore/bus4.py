"""Business: which count non-linearity?  Central-only ACE with linear local blocks + candidate count features."""
import numpy as np, sys
from lib import *
from blocks import *
from sklearn.isotonic import IsotonicRegression

np.set_printoptions(linewidth=220, suppress=True, precision=2)
g, y, z, lev, idx = labelled(3)
cen = (y > 0.05) & (y < 0.95)
g, y, z, lev = g[cen], y[cen], z[cen], lev[cen]
lev = lev - lev.min()
n = len(g)
cnt = counts(g).astype(float)
nR, nI, nC, nP, nO = cnt.T
nl = lev.max() + 1
wts = np.maximum(np.bincount(lev, minlength=nl).astype(float), 1e-9)
levsd = lev.std()


def ace(X, iters=30, lam=1e-3):
    X = np.hstack([X, np.ones((n, 1))]).astype(np.float64)
    keep = X.std(0) > 0; keep[-1] = True
    X = X[:, keep]
    Gm = X.T @ X + lam * np.eye(X.shape[1])
    theta = (lev - lev.mean()) / levsd
    for it in range(iters):
        w = np.linalg.solve(Gm, X.T @ theta)
        p = X @ w
        m = np.bincount(lev, weights=p, minlength=nl) / wts
        th = IsotonicRegression().fit(np.arange(nl), m, sample_weight=wts).predict(np.arange(nl))[lev]
        theta = (th - th.mean()) / th.std()
    w = np.linalg.solve(Gm, X.T @ theta)
    return np.std(theta - X @ w) * levsd, theta, X @ w


def lin(X, lam=1e-3):
    X = np.hstack([X, np.ones((n, 1))]).astype(np.float64)
    keep = X.std(0) > 0; keep[-1] = True
    X = X[:, keep]
    t = lev.astype(float)
    w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ t)
    return np.std(t - X @ w)


base = np.hstack([cnt[:, 1:], ordered_pairs(g)[0], near_profile(g, MAN, list(range(1, 9)), "man")[0]])
hinge = lambda v: np.maximum(v, 0)
fam = {
    "base (counts linear + ord4 + prof_man)": np.zeros((n, 0)),
    "+ cnt2": np.stack([cnt[:, i] * cnt[:, j] for i in range(5) for j in range(i, 5)], 1),
    "+ cntoh": np.concatenate([(cnt[:, k:k + 1] == np.arange(1, 50)[None]).astype(float) for k in range(5)], 1),
}
for k in (1, 1.5, 2, 2.5, 3, 4):
    fam[f"+ min({k}nR, nI+nO)"] = np.minimum(k * nR, nI + nO)[:, None]
    fam[f"+ min({k}nR, nI+nO+nC)"] = np.minimum(k * nR, nI + nO + nC)[:, None]
    fam[f"+ min({k}nR, nI), min({k}nR, nO), min({k}nR, nC)"] = np.stack([np.minimum(k * nR, nI), np.minimum(k * nR, nO), np.minimum(k * nR, nC)], 1)
fam["+ hinges nR-c, c=6..14"] = np.stack([hinge(nR - c) for c in range(6, 15)], 1)
fam["+ hinges on every count (c=1..25)"] = np.concatenate([np.stack([hinge(cnt[:, t] - c) for c in range(1, 26)], 1) for t in range(5)], 1)
fam["+ cntoh + cnt2"] = np.hstack([fam["+ cntoh"], fam["+ cnt2"]])
pairoh = []
for i in range(5):
    for j in range(i + 1, 5):
        code = (np.minimum(cnt[:, i], 20) * 21 + np.minimum(cnt[:, j], 20)).astype(int)
        u, inv = np.unique(code, return_inverse=True)
        M = np.zeros((n, len(u))); M[np.arange(n), inv] = 1
        pairoh.append(M[:, M.sum(0) >= 3])
fam["+ free 2-D count tables (all pairs)"] = np.hstack(pairoh)
for nm, F in fam.items():
    X = np.hstack([base, F])
    r_ace, _, _ = ace(X)
    print(f"  {nm:55s} cols {X.shape[1]:5d}  level-linear resid {lin(X):7.2f}   ACE resid {r_ace:7.2f} (level units)", flush=True)
