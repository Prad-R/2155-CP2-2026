"""Direction-dependent adjacency?  Ordered pair counts split by horizontal / vertical and by order."""
import numpy as np, sys
from lib import *

np.set_printoptions(linewidth=220, suppress=True, precision=2)
for a in ([int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1]):
    g, y, z, lev, idx = labelled(a)
    cen = (y > 0.08) & (y < 0.92)
    t = lev.astype(float); n = len(g)
    C = counts(g)
    H = np.zeros((n, 5, 5)); V = np.zeros((n, 5, 5))
    for s in range(5):
        for u in range(5):
            H[:, s, u] = ((g[:, :, :-1] == s) & (g[:, :, 1:] == u)).sum((1, 2))      # s left of u
            V[:, s, u] = ((g[:, :-1, :] == s) & (g[:, 1:, :] == u)).sum((1, 2))      # s above u
    sym = (H + H.transpose(0, 2, 1) + V + V.transpose(0, 2, 1))
    iu = np.triu_indices(5)
    Xsym = sym[:, iu[0], iu[1]]
    hv = np.concatenate([(H + H.transpose(0, 2, 1))[:, iu[0], iu[1]], (V + V.transpose(0, 2, 1))[:, iu[0], iu[1]]], 1)
    full = np.concatenate([H.reshape(n, 25), V.reshape(n, 25)], 1)
    ci = np.where(cen)[0]; f = folds_for(len(ci), 5, 0)

    def cv(X):
        X = np.hstack([C[:, 1:], X, np.ones((n, 1))])
        oof = np.zeros(len(ci))
        for k in range(5):
            w, *_ = np.linalg.lstsq(X[ci[f != k]], t[ci[f != k]], rcond=None)
            oof[f == k] = X[ci[f == k]] @ w
        w, *_ = np.linalg.lstsq(X[ci], t[ci], rcond=None)
        return (t[ci] - oof).std(), w

    s0, _ = cv(Xsym); s1, _ = cv(hv); s2, w = cv(full)
    print(f"\n{ADVISORS[a]}: OOF resid sd  symmetric pairs {s0:.3f} | horizontal vs vertical {s1:.3f} | fully ordered (left-of, above) {s2:.3f}")
    Wh = w[4:29].reshape(5, 5); Wv = w[29:54].reshape(5, 5)
    print("  ordered horizontal coef [row = left type, col = right type] (minus mean):"); print((Wh - Wh.mean()).round(2))
    print("  ordered vertical coef [row = upper type, col = lower type] (minus mean):"); print((Wv - Wv.mean()).round(2))
