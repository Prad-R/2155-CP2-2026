"""Test hypothesis classes: raw = sum over cells of h(descriptor of the cell). Ridge on descriptor counts, z target."""
import numpy as np, time
from lib import *

np.set_printoptions(linewidth=200, suppress=True)


def nb_counts(g, kind):
    """counts of each type (and 'outside') among neighbours: (n,6,49)"""
    n = len(g)
    P = np.full((n, 9, 9), 5, np.int8)
    P[:, 1:8, 1:8] = g
    offs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    offs8 = offs4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    offs = offs4 if kind == 4 else offs8
    out = np.zeros((n, 6, 7, 7), np.int16)
    for dr, dc in offs:
        nb = P[:, 1 + dr:8 + dr, 1 + dc:8 + dc]
        for t in range(6):
            out[:, t] += nb == t
    return out.reshape(n, 6, 49)


def code_of(parts, bases):
    c = np.zeros(parts[0].shape, np.int64)
    for p, b in zip(parts, bases):
        c = c * b + p.astype(np.int64)
    return c


for a, name in enumerate(ADVISORS):
    g, y, z, lev, idx = labelled(a)
    gf = g.reshape(len(g), 49).astype(np.int64)
    t0 = time.time()
    res = {}
    C = counts(g)
    res["counts"] = ridge_cv(C, z)[:2]
    Coh = np.concatenate([(C[:, t:t + 1] == np.arange(50)[None]).astype(np.float32) for t in range(5)], 1)
    res["counts onehot (additive, any shape)"] = ridge_cv(Coh, z)[:2]
    n4 = nb_counts(g, 4); n8 = nb_counts(g, 8)
    X4, u4 = descr_onehot(code_of([gf] + [n4[:, t] for t in range(6)], [5] + [5] * 6))
    res[f"D1 (type, 4-nbr multiset) [{X4.shape[1]}]"] = ridge_cv(X4, z)[:2]
    X8, u8 = descr_onehot(code_of([gf] + [n8[:, t] for t in range(6)], [5] + [9] * 6))
    res[f"D2 (type, 8-nbr counts) [{X8.shape[1]}]"] = ridge_cv(X8, z)[:2]
    # nearest distances (Manhattan), capped at 4, self excluded? use distance to nearest of each type incl. self=0
    dm = near_dist(g, MAN, cap=5)
    X3, u3 = descr_onehot(code_of([gf] + [np.minimum(dm[:, b], 4) for b in range(5)], [5] + [5] * 5))
    res[f"D3 (type, nearest man dist to each type cap4) [{X3.shape[1]}]"] = ridge_cv(X3, z)[:2]
    dc = near_dist(g, CHE, cap=5)
    X3c, _ = descr_onehot(code_of([gf] + [np.minimum(dc[:, b], 3) for b in range(5)], [5] + [4] * 5))
    res[f"D3c (type, nearest cheb dist cap3) [{X3c.shape[1]}]"] = ridge_cv(X3c, z)[:2]
    res["counts + D1"] = ridge_cv(np.hstack([Coh, X4]), z)[:2]
    res["counts + D1 + D3"] = ridge_cv(np.hstack([Coh, X4, X3]), z)[:2]
    res["counts + D2 + D3"] = ridge_cv(np.hstack([Coh, X8, X3]), z)[:2]
    print(f"\n===== {name} ({time.time() - t0:.0f}s)")
    for k, v in res.items():
        print(f"  {k:62s} R2(z) {v[0]:.4f}  lam {v[1]}")
