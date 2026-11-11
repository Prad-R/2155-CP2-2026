"""Is there a position-dependent (non-symmetric) component?  Add per-position type indicators to the neighbour-shape model."""
import numpy as np, sys
from lib import *

np.set_printoptions(linewidth=220, suppress=True, precision=2)
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
    Xs = np.hstack(cols + [np.ones((n, 1))])
    Xp = onehot(g)[:, 1:].reshape(n, -1)                          # 4*49 positional indicators (R baseline)
    ci = np.where(cen)[0]; f = folds_for(len(ci), 5, 0)

    def cv(X, lam=1e-6):
        oof = np.zeros(len(ci))
        for k in range(5):
            A = X[ci[f != k]]; w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ t[ci[f != k]])
            oof[f == k] = X[ci[f == k]] @ w
        return (t[ci] - oof).std()

    print(f"\n{ADVISORS[a]}: OOF resid sd  shape-model {cv(Xs):.3f}   + positional {cv(np.hstack([Xs, Xp]), 1e-3):.3f}   positional only {cv(np.hstack([Xp, np.ones((n, 1))]), 1e-3):.3f}")
    A = np.hstack([Xs, Xp])[ci]
    w = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ t[ci])
    wp = w[Xs.shape[1]:].reshape(4, 7, 7)
    for i, tt in enumerate("ICPO"):
        print(f"  positional weight map for type {tt} (relative to R), mean removed:")
        print((wp[i] - wp[i].mean()).round(1))
