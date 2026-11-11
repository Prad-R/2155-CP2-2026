"""If the raw score is discrete, the rank of each distinct label value ("level") is the raw score up to
an unknown monotone map that is linear wherever all levels are observed. Is a level a simple function of integer features?"""
import os, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from scipy.special import ndtri
from common import ROOT, ADVISORS, load_data, folds_for
from features2 import features2
grids, ratings = load_data(); n2 = features2(grids[:2])[1]
for a in range(4):
    idx = np.where(~np.isnan(ratings[:, a]))[0]; y = ratings[idx, a]; fold = folds_for(len(idx))
    X = np.load(os.path.join(ROOT, "ckpt", "feat2", f"X_a{a}.npy")); n1 = X.shape[1] - len(n2)
    u, inv, c = np.unique(y, return_inverse=True, return_counts=True)
    mid = (y > 0.08) & (y < 0.92); L = inv.astype(float)
    um = u[(u > 0.08) & (u < 0.92)]; gaps = np.diff(um) * 999
    # level spacing in quantile units should track the level's population if levels are consecutive integers
    print(f"== {ADVISORS[a]}: {len(um)} levels in the middle 84% ; grids per level median {np.median(c[(u>0.08)&(u<0.92)]):.0f}")
    z = ndtri(np.clip(y, 0.0005, 0.9995))
    # is z linear in level index? (expected if raw is integer-spaced and roughly normal)
    co = np.polyfit(L[mid], z[mid], 1); print(f"   z vs level index (middle): R2 {r2_score(z[mid], np.polyval(co, L[mid])):.5f}; one level = {co[0]:.4f} z")
    pair = np.r_[np.arange(0, 35), n1 + np.array([i for i, s in enumerate(n2) if s.startswith('o')])]
    for nm, cols in [("counts", np.arange(5)), ("counts + pair counts (all ranges)", pair), ("all features", np.arange(X.shape[1]))]:
        o = np.zeros(len(y))
        for k in range(5):
            tr, te = (fold != k) & mid, fold == k
            mu, sd = X[tr][:, cols].mean(0), X[tr][:, cols].std(0) + 1e-6
            o[te] = RidgeCV(alphas=np.logspace(-3, 2, 10)).fit((X[tr][:, cols] - mu) / sd, L[tr]).predict((X[te][:, cols] - mu) / sd)
        r = (L - o)[mid]
        print(f"   linear fit of level on {nm:34s}: R2 {r2_score(L[mid], o[mid]):.4f}  residual sd {r.std():.2f} levels  |resid|<0.5: {(np.abs(r)<0.5).mean():.2f}", flush=True)
