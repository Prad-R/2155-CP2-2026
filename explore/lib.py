"""Shared helpers for rule-recovery experiments."""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import load_data, ADVISORS, folds_for
from scipy.stats import norm

SYM = "RICPO"
T = 5
RC = np.array([(r, c) for r in range(7) for c in range(7)])
DR = np.abs(RC[:, None, 0] - RC[None, :, 0])
DC = np.abs(RC[:, None, 1] - RC[None, :, 1])
MAN = DR + DC
CHE = np.maximum(DR, DC)
EUC2 = DR ** 2 + DC ** 2

_cache = {}


def data():
    if "d" not in _cache:
        _cache["d"] = load_data()
    return _cache["d"]


def labelled(a):
    """-> grids (n,7,7) int8, y (n,), z (n,), level index (n,), dataset idx"""
    G, Y = data()
    idx = np.where(~np.isnan(Y[:, a]))[0]
    y = Y[idx, a]
    z = norm.ppf(np.clip(y, 0.0005, 0.9995))
    vals, lev = np.unique(y, return_inverse=True)
    return G[idx], y, z, lev, idx


def onehot(g):
    g = np.asarray(g).reshape(-1, 49)
    return (g[:, None, :] == np.arange(T)[None, :, None]).astype(np.float32)   # n,5,49


def counts(g):
    return onehot(g).sum(2)


def near_dist(g, D=MAN, self_zero=True, cap=None):
    """distance from each cell to the nearest cell of each type: (n,5,49); absent type -> cap (default max+1)"""
    O = onehot(g) > 0
    cap = (D.max() + 1) if cap is None else cap
    Dx = D.astype(np.float32).copy()
    if not self_zero:
        np.fill_diagonal(Dx, np.inf)
    out = np.empty(O.shape, np.float32)
    for b in range(T):
        out[:, b] = np.where(O[:, b][:, None, :], Dx[None], np.inf).min(2)
    return np.minimum(out, cap)


def ring_counts(g, D=MAN, radii=(1, 2, 3)):
    """count of type-b cells at distance exactly r from each cell: (n, len(radii), 5, 49)"""
    O = onehot(g)
    out = []
    for r in radii:
        M = (D == r).astype(np.float32)
        out.append(O @ M)
    return np.stack(out, 1)


def ridge_cv(X, t, lams=(1e-3, 1e-2, 1e-1, 1, 10, 100, 1000), k=5, seed=0, standardize=True):
    """K-fold ridge; returns best (r2, lam) and oof predictions. Uses primal or dual depending on shape."""
    X = np.asarray(X, np.float64)
    n = len(t)
    f = folds_for(n, k, seed)
    best = (-9, None, None)
    for lam in lams:
        oof = np.zeros(n)
        for j in range(k):
            tr, te = f != j, f == j
            Xtr, Xte = X[tr], X[te]
            mu = Xtr.mean(0)
            sd = Xtr.std(0) if standardize else np.ones(X.shape[1])
            sd[sd < 1e-9] = 1
            A, B = (Xtr - mu) / sd, (Xte - mu) / sd
            tm = t[tr].mean()
            if A.shape[1] <= A.shape[0]:
                w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (t[tr] - tm))
                oof[te] = B @ w + tm
            else:
                al = np.linalg.solve(A @ A.T + lam * np.eye(A.shape[0]), t[tr] - tm)
                oof[te] = B @ (A.T @ al) + tm
        r2 = 1 - ((oof - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        if r2 > best[0]:
            best = (r2, lam, oof)
    return best


def descr_onehot(codes, ncode=None):
    """codes: (n,49) integer descriptor per cell -> (n, n_distinct) counts of each descriptor value per grid."""
    u, inv = np.unique(codes, return_inverse=True)
    inv = inv.reshape(codes.shape)
    X = np.zeros((codes.shape[0], len(u)), np.float32)
    rows = np.repeat(np.arange(codes.shape[0]), codes.shape[1])
    np.add.at(X, (rows, inv.ravel()), 1)
    return X, u


def gstr(g):
    return ["".join(SYM[v] for v in r) for r in np.asarray(g).reshape(7, 7)]
