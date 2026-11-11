import numpy as np
from collections import Counter
r = np.load("datasets/scores.npy")
for a, nm in enumerate(["Wellness", "Tax", "Transportation", "Business"]):
    y = r[~np.isnan(r[:, a]), a]
    u, c = np.unique(y, return_counts=True)
    print(f"== {nm}: n {len(y)} unique {len(u)} min {y.min():.6f} max {y.max():.6f} mean {y.mean():.4f}")
    print("  first 14 unique:", " ".join(f"{v:.6f}" for v in u[:14]))
    print("  mid 14 unique:  ", " ".join(f"{v:.6f}" for v in u[len(u)//2:len(u)//2+14]))
    print("  last 8 unique:  ", " ".join(f"{v:.6f}" for v in u[-8:]))
    o = np.argsort(-c)[:8]; print("  most common:", " ".join(f"{u[i]:.6f}x{c[i]}" for i in o))
    print("  counts-of-counts:", sorted(Counter(c.tolist()).items())[:12])
    print("  decimals needed:", {d: int((np.abs(u - np.round(u, d)) < 1e-9).sum()) for d in (2, 3, 4, 5, 6)})
    h = np.histogram(y, bins=20, range=(0, 1))[0]; print("  hist20:", h.tolist())
