"""Feature blocks for rule recovery (directional + distance profiles) and a small OLS/ridge CV workbench."""
import numpy as np, sys
from lib import *

EUC2 = DR ** 2 + DC ** 2


def ordered_pairs(g, offs=((0, 1), (1, 0))):
    """#(type a at cell, type b at cell+(dr,dc)) for each offset -> (n, len(offs)*25), names"""
    n = len(g); O = onehot(g).reshape(n, 5, 7, 7)
    cols, names = [], []
    for dr, dc in offs:
        A = O[:, :, max(0, -dr):7 - max(0, dr), max(0, -dc):7 - max(0, dc)]
        B = O[:, :, max(0, dr):7 + min(0, dr) if dr < 0 else 7, max(0, dc):7 + min(0, dc) if dc < 0 else 7]
        B = O[:, :, max(0, dr):, max(0, dc):][:, :, :A.shape[2], :A.shape[3]]
        cols.append(np.einsum("nahw,nbhw->nab", A, B).reshape(n, 25))
        names += [f"{SYM[a]}>{SYM[b]}@({dr},{dc})" for a in range(5) for b in range(5)]
    return np.hstack(cols), names


def near_profile(g, D, dvals, tag, self_zero=False):
    """indicator counts: #(a-cells whose nearest (other) b is at distance d) for d in dvals[1:] (first = baseline), + 'none'"""
    Of = onehot(g)
    dn = near_dist(g, D, self_zero=self_zero, cap=9999)
    cols, names = [], []
    for a in range(5):
        for b in range(5):
            for d in list(dvals[1:]) + [9999]:
                cols.append((Of[:, a] * (dn[:, b] == d)).sum(1)[:, None]); names.append(f"{tag}:{SYM[a]}->{SYM[b]}={d}")
    return np.hstack(cols), names


def near_lin(g, D, tag, power=1, self_zero=False, cap=None):
    """sum over a-cells of (distance to nearest other b)^power ; absent -> cap"""
    Of = onehot(g)
    cap = (D.max() + 1) if cap is None else cap
    dn = near_dist(g, D, self_zero=self_zero, cap=cap) ** power
    X = np.einsum("nac,nbc->nab", Of, dn).reshape(len(g), 25)
    return X, [f"{tag}^{power}:{SYM[a]}->{SYM[b]}" for a in range(5) for b in range(5)]


def absent(g):
    C = counts(g)
    return (C == 0).astype(np.float32), [f"no{SYM[t]}" for t in range(5)]


def ray_dist(g):
    """distance from each cell to the first cell of type b looking in each of 4 directions (W,E,N,S); 0 if none. -> (n,4,5,49)"""
    n = len(g); out = np.zeros((n, 4, 5, 7, 7), np.float32)
    for b in range(5):
        m = (g == b)
        for c in range(7):
            # west: nearest col < c with m
            for cc in range(c - 1, -1, -1):
                hit = m[:, :, cc] & (out[:, 0, b, :, c] == 0)
                out[:, 0, b, :, c][hit] = c - cc
            for cc in range(c + 1, 7):
                hit = m[:, :, cc] & (out[:, 1, b, :, c] == 0)
                out[:, 1, b, :, c][hit] = cc - c
        for r in range(7):
            for rr in range(r - 1, -1, -1):
                hit = m[:, rr, :] & (out[:, 2, b, r, :] == 0)
                out[:, 2, b, r, :][hit] = r - rr
            for rr in range(r + 1, 7):
                hit = m[:, rr, :] & (out[:, 3, b, r, :] == 0)
                out[:, 3, b, r, :][hit] = rr - r
    return out.reshape(n, 4, 5, 49)


def cv_fit(X, t, ci, lam=1e-3, k=5):
    f = folds_for(len(ci), k, 0)
    X = np.hstack([X, np.ones((len(X), 1))])
    keep = X[ci].std(0) > 0; keep[-1] = True
    Xc = X[ci][:, keep].astype(np.float64)
    oof = np.zeros(len(ci))
    for j in range(k):
        A = Xc[f != j]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ t[ci[f != j]])
        oof[f == j] = Xc[f == j] @ w
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ t[ci])
    wfull = np.zeros(X.shape[1]); wfull[keep] = w
    return (t[ci] - oof).std(), (t[ci] - Xc @ w).std(), wfull


def exact_check(f, y, verbose=True):
    """is the label a single-valued monotone function of integer score f?"""
    vals, inv = np.unique(np.round(y, 7), return_inverse=True)
    lo = np.full(len(vals), np.inf); hi = np.full(len(vals), -np.inf)
    np.minimum.at(lo, inv, f); np.maximum.at(hi, inv, f)
    single = (lo == hi).sum()
    mono = (lo[1:] >= hi[:-1]).mean()
    if verbose:
        print(f"   exact check: {single}/{len(vals)} levels single-valued; monotone steps {mono:.4f}; mean within-level spread {np.mean(hi - lo):.3f}")
    return single, len(vals), mono
