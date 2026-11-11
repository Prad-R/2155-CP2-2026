"""Are central levels consecutive raw values?  Compare landmark-midpoint gaps with tie counts."""
import numpy as np
from lib import *

np.set_printoptions(linewidth=220, suppress=True, precision=2)
for a in range(4):
    g, y, z, lev, idx = labelled(a)
    vals, c = np.unique(y, return_counts=True)
    m = vals * 999
    cen = (vals > 0.08) & (vals < 0.92)
    d = np.diff(m)
    e = (c[:-1] + c[1:]) / 2 * 999 / 5000
    ok = cen[:-1] & cen[1:]
    exc = (d - e)[ok]
    print(f"\n===== {ADVISORS[a]}: central levels {ok.sum()}  mean gap {d[ok].mean():.2f} landmarks, mean expected from counts {e[ok].mean():.2f}")
    print("  excess (gap - expected) quantiles:", np.quantile(exc, [0, .05, .25, .5, .75, .95, 1]).round(2))
    big = np.where(ok & (d - e > 0.6 * d[ok].mean()))[0]
    print("  levels followed by a suspicious hole:", len(big), [(round(m[i], 1), int(c[i]), round(d[i], 1), round(e[i], 1)) for i in big[:20]])
    # spans via chain: span_n = 2*(m_n - start_n) ; reconstruct assuming contiguous tiling from the median level
    print("  first 40 central (midpoint, count, count*999/5000):", [(round(mm, 1), int(cc), round(cc * .1998, 1)) for mm, cc in zip(m[cen][:40], c[cen][:40])])
