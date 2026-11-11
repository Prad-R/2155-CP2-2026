"""Does the residual variance depend on composition?  (per-cell noise => Var = sum_t n_t s_t^2)"""
import numpy as np, sys
from lib import *

np.set_printoptions(linewidth=220, suppress=True, precision=3)
for a in ([int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1]):
    g, y, z, lev, idx = labelled(a)
    cen = (y > 0.08) & (y < 0.92)
    t = lev.astype(float); n = len(g)
    O = onehot(g).reshape(n, 5, 7, 7)
    Pd = np.pad(O, ((0, 0), (0, 0), (1, 1), (1, 1)))
    n4 = Pd[:, :, :-2, 1:-1] + Pd[:, :, 2:, 1:-1] + Pd[:, :, 1:-1, :-2] + Pd[:, :, 1:-1, 2:]
    C = counts(g)
    cols = [C[:, 1:]]
    for aa in range(5):
        for b in range(5):
            for k in range(1, 5):
                cols.append((O[:, aa] * (n4[:, b] == k)).sum((1, 2))[:, None])
    X = np.hstack(cols + [np.ones((n, 1))])
    ci = np.where(cen)[0]
    w, *_ = np.linalg.lstsq(X[ci], t[ci], rcond=None)
    res = t[ci] - X[ci] @ w
    r2 = res ** 2
    A = C[ci]
    v, *_ = np.linalg.lstsq(A, r2, rcond=None)
    pred = A @ v
    print(f"\n{ADVISORS[a]}: resid sd {res.std():.2f}; Var(resid) ~ counts: per-cell variance by type {dict(zip(SYM, v.round(3)))}  R2 of fit {1 - ((r2 - pred) ** 2).sum() / ((r2 - r2.mean()) ** 2).sum():.4f}")
    for tt in range(5):
        o = np.argsort(A[:, tt]); parts = np.array_split(o, 5)
        print(f"   by n{SYM[tt]}: " + "  ".join(f"n~{A[p, tt].mean():4.1f}: sd {res[p].std():5.2f}" for p in parts))
    # by number of RP adjacent relations etc: generic 'activity'
