"""Shape of neighbour effects: level ~ counts + sum_k coef[a,b,k] * #(a-cells with exactly k b-neighbours in N4)."""
import numpy as np, sys
from lib import *

np.set_printoptions(linewidth=220, suppress=True, precision=2)
a = int(sys.argv[1]) if len(sys.argv) > 1 else 1
g, y, z, lev, idx = labelled(a)
cen = (y > 0.08) & (y < 0.92)
t = lev.astype(float)
n = len(g)
O = onehot(g).reshape(n, 5, 7, 7)
Pd = np.pad(O, ((0, 0), (0, 0), (1, 1), (1, 1)))
n4 = Pd[:, :, :-2, 1:-1] + Pd[:, :, 2:, 1:-1] + Pd[:, :, 1:-1, :-2] + Pd[:, :, 1:-1, 2:]     # n,5,7,7 count of b-neighbours
C = counts(g)
cols, names = [C[:, 1:]], [f"n{SYM[i]}" for i in range(1, 5)]
for aa in range(5):
    for b in range(5):
        for k in range(1, 5):
            v = (O[:, aa] * (n4[:, b] == k)).sum((1, 2))
            cols.append(v[:, None]); names.append(f"{SYM[aa]}|{k}{SYM[b]}")
X = np.hstack(cols + [np.ones((n, 1))]); names.append("const")
ci = np.where(cen)[0]
Xc = X[ci]
# drop empty columns
keep = Xc.std(0) > 0; keep[-1] = True
Xc = Xc[:, keep]; names = [nm for nm, k in zip(names, keep) if k]
w, *_ = np.linalg.lstsq(Xc, t[ci], rcond=None)
res = t[ci] - Xc @ w
s2 = res.var() * len(ci) / (len(ci) - Xc.shape[1])
cov = s2 * np.linalg.pinv(Xc.T @ Xc)
se = np.sqrt(np.diag(cov))
print(f"{ADVISORS[a]}: resid sd {res.std():.3f} with {Xc.shape[1]} columns (central n={len(ci)})")
W = dict(zip(names, zip(w, se)))
print("counts:", {k: f"{v[0]:+.2f}±{v[1]:.2f}" for k, v in W.items() if k.startswith("n")})
print("rows: own type | columns: neighbour type b ; entries coef for k=1,2,3,4 neighbours of b (± se)")
for aa in range(5):
    for b in range(5):
        s = []
        for k in range(1, 5):
            key = f"{SYM[aa]}|{k}{SYM[b]}"
            s.append(f"{W[key][0]:+6.2f}±{W[key][1]:.2f}" if key in W else "   --   ")
        print(f"  {SYM[aa]} with k {SYM[b]}-nbrs: " + "  ".join(s))
