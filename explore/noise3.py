"""What drives the residual variance of the Tax shape model?  Gaussian likelihood of variance models."""
import numpy as np, sys
from lib import *
from scipy.optimize import minimize

np.set_printoptions(linewidth=220, suppress=True, precision=3)
a = int(sys.argv[1]) if len(sys.argv) > 1 else 1
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
pair = lambda s, u: (O[:, s] * n4[:, u]).sum((1, 2))            # number of (s-cell, u-neighbour) relations
F = {
    "const": np.ones(n), "nR": C[:, 0], "nI": C[:, 1], "nC": C[:, 2], "nP": C[:, 3], "nO": C[:, 4],
    "RP": pair(0, 3), "RI": pair(0, 1), "RR": pair(0, 0) / 2, "PP": pair(3, 3) / 2, "RC": pair(0, 2), "RO": pair(0, 4), "PI": pair(3, 1), "PC": pair(3, 2), "PO": pair(3, 4),
    "R_anyP": (O[:, 0] * (n4[:, 3] > 0)).sum((1, 2)), "P_anyR": (O[:, 3] * (n4[:, 0] > 0)).sum((1, 2)),
    "R_noP": (O[:, 0] * (n4[:, 3] == 0)).sum((1, 2)), "P_noR": (O[:, 3] * (n4[:, 0] == 0)).sum((1, 2)),
}


def fit(names):
    A = np.stack([F[k][ci] for k in names], 1)

    def nll(v):
        var = np.maximum(A @ v, 0.5)
        return 0.5 * (np.log(var) + r2 / var).sum()
    v0, *_ = np.linalg.lstsq(A, r2, rcond=None)
    best = minimize(nll, v0, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-4, fatol=1e-4))
    return best.fun, best.x


base, _ = fit(["const"])
for names in (["const"], ["nR", "nI", "nC", "nP", "nO"], ["const", "nP"], ["const", "nR"], ["const", "nP", "nR"], ["const", "RP"], ["const", "RP", "nP", "nR"],
              ["const", "R_anyP", "P_anyR"], ["const", "R_anyP", "P_anyR", "R_noP", "P_noR"], ["const", "nP", "nR", "RP", "RI", "PP", "RR"],
              ["const", "nP", "nR", "RP", "RI", "RC", "RO", "PI", "PC", "PO", "RR", "PP"]):
    nl, v = fit(names)
    print(f"  dNLL {base - nl:8.1f}  " + "  ".join(f"{k}:{x:+.2f}" for k, x in zip(names, v)))
