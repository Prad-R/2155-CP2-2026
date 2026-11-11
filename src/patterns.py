"""Pattern analysis: what drives each advisor, and what our optimised sets look like.

Writes results/patterns/{patterns.json, report.md, fig*.png}. Light (labelled grids
and our own sets only); run through srun.

  python -u src/patterns.py
"""
import glob, json, os, pickle
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT, ADVISORS, THRESH, load_data, folds_for
from features import features, PAIRS, RING
from divscore import hamming_matrix, int_score, DENOM, K

OUT = os.path.join(ROOT, "results", "patterns")
os.makedirs(OUT, exist_ok=True)
DN = ["Res", "Ind", "Com", "Park", "Off"]
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # one fixed colour per district
R = {}
md = []
P = lambda *a: (print(*a, flush=True), md.append(" ".join(str(x) for x in a)))


def table(head, rows):
    P("| " + " | ".join(head) + " |")
    P("|" + "---|" * len(head))
    for r in rows:
        P("| " + " | ".join(str(x) for x in r) + " |")
    P("")


def feat_names(groups):
    n = []
    for g, (s, e) in groups.items():
        for j in range(e - s):
            if g == "onehot": n.append(f"cell({j//5//7},{j//5%7})={DN[j%5]}")
            elif g == "count": n.append(f"n {DN[j]}")
            elif g in ("adj4", "adjd"):
                a, b = PAIRS[j]; n.append(f"{'edge' if g=='adj4' else 'diagonal'} pairs {DN[a]}-{DN[b]}")
            elif g == "dist":
                b, a, s_ = j // 20, j // 4 % 5, j % 4
                n.append([f"mean dist {DN[a]}->nearest {DN[b]}", f"max dist {DN[a]}->nearest {DN[b]}",
                          f"n {DN[a]} touching a {DN[b]}", f"n {DN[a]} within 2 of a {DN[b]}"][s_])
            elif g == "comp": n.append(f"{'n clusters' if j%2==0 else 'largest cluster'} {DN[j//2]}")
            elif g == "ring": n.append(f"n {DN[j%5]} in ring {j//5} (0=edge)")
            else: n.append(f"n {DN[j%5]} in {g} {j//5}")
    return n


grids, ratings = load_data()
X0, groups = features(grids[:2])
names = feat_names(groups)
gslice = lambda ks: np.concatenate([np.arange(*groups[k]) for k in ks])
lab = [np.where(~np.isnan(ratings[:, a]))[0] for a in range(4)]
Y = [ratings[lab[a], a] for a in range(4)]
XF = [features(grids[lab[a]])[0] for a in range(4)]
CNT = [x[:, groups["count"][0]:groups["count"][1]] for x in XF]

# ---------------------------------------------------------------- A. score values
P("# Pattern analysis\n")
P("## A. Structure of the score values\n")
rows = []
for a in range(4):
    y = Y[a]; u = np.unique(y)
    best = None
    for N in range(2, 3001):
        fr = np.mean(np.abs(u * N - np.round(u * N)) < 1e-6)
        if fr > 0.99: best = N; break
    gaps = np.diff(u)
    R[f"values/{ADVISORS[a]}"] = dict(n_unique=len(u), grid_N=best, frac0=float((y == 0).mean()), frac1=float((y == 1).mean()),
                                    min_gap=float(gaps.min()), median_gap=float(np.median(gaps)), valid_frac=float((y >= THRESH).mean()))
    rows.append([ADVISORS[a], len(u), best, f"{(y==0).mean():.3f}", f"{(y==1).mean():.3f}", f"{gaps.min():.2e}", f"{np.median(gaps):.2e}", f"{(y>=THRESH).mean():.3f}"])
table(["advisor", "distinct values", "values are k/N for N=", "share = 0", "share = 1", "min gap", "median gap", "share >= 0.75"], rows)

# ---------------------------------------------------------------- B. counts
P("## B. District counts: score by count, and the range valid grids occupy\n")
fig, ax = plt.subplots(4, 5, figsize=(13, 9), sharey=True, facecolor="#fcfcfb")
rows = []
for a in range(4):
    for d in range(5):
        c = CNT[a][:, d].astype(int); y = Y[a]
        xs = [v for v in np.unique(c) if (c == v).sum() >= 15]
        m = [y[c == v].mean() for v in xs]
        pv = [(y[c == v] >= THRESH).mean() for v in xs]
        ax[a, d].set_facecolor("#fcfcfb")
        ax[a, d].plot(xs, m, color=COL[d], lw=2)
        ax[a, d].axhline(THRESH, color="#888", lw=0.8, ls="--")
        ax[a, d].grid(alpha=0.2); ax[a, d].spines[["top", "right"]].set_visible(False)
        if a == 0: ax[a, d].set_title(f"number of {DN[d]} cells", fontsize=10)
        if d == 0: ax[a, d].set_ylabel(f"{ADVISORS[a]}\nmean score", fontsize=9)
        v = y >= THRESH
        q = np.quantile(c[v], [0.1, 0.5, 0.9]).astype(int)
        rho = spearmanr(c, y)[0]
        bestc = xs[int(np.argmax(pv))]
        R[f"count/{ADVISORS[a]}/{DN[d]}"] = dict(spearman=float(rho), valid_q10_50_90=q.tolist(), best_count=int(bestc), best_valid_rate=float(max(pv)),
                                                curve={int(x): round(float(v_), 3) for x, v_ in zip(xs, m)})
        rows.append([ADVISORS[a], DN[d], f"{rho:+.2f}", f"{q[0]}-{q[2]} (median {q[1]})", f"{bestc} ({max(pv):.0%} valid)"])
fig.suptitle("Mean advisor score against the number of cells of each district (labelled grids; dashed = 0.75)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_score_vs_count.png"), dpi=130); plt.close(fig)
table(["advisor", "district", "Spearman(count, score)", "count range of valid grids (10-90%)", "count with highest valid rate"], rows)

# ---------------------------------------------------------------- C. single features
P("## C. Strongest single features per advisor (Spearman with the score)\n")
for a in range(4):
    X = XF[a]; inv = gslice(["count", "adj4", "adjd", "dist", "comp", "ring"])
    rho = np.array([spearmanr(X[:, j], Y[a])[0] if X[:, j].std() > 0 else 0 for j in inv])
    o = np.argsort(-np.abs(rho))[:10]
    R[f"top_features/{ADVISORS[a]}"] = [(names[inv[j]], round(float(rho[j]), 3)) for j in o]
    P(f"**{ADVISORS[a]}**\n")
    table(["feature", "Spearman"], [[names[inv[j]], f"{rho[j]:+.3f}"] for j in o])

# ---------------------------------------------------------------- D. group ablation
P("## D. How much each kind of information explains (3-fold R2)\n")
sets = {"counts (linear)": ["count"], "counts": ["count"], "counts+edge pairs": ["count", "adj4"], "counts+distances": ["count", "dist"],
        "counts+clusters": ["count", "comp"], "counts+rings": ["count", "ring"], "all position-free": ["count", "adj4", "adjd", "dist", "comp", "ring"],
        "everything": list(groups)}
rows = []
for a in range(4):
    f = folds_for(len(Y[a]), 3, 1); r = [ADVISORS[a]]
    for nm, ks in sets.items():
        X = XF[a][:, gslice(ks)]; oof = np.zeros(len(Y[a]))
        for k in range(3):
            tr, te = f != k, f == k
            m = LinearRegression() if "linear" in nm else lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=15, subsample=0.8,
                                                                         subsample_freq=1, colsample_bytree=0.6, min_child_samples=10, verbose=-1, n_jobs=8)
            oof[te] = m.fit(X[tr], Y[a][tr]).predict(X[te])
        r2 = r2_score(Y[a], oof); R[f"ablation/{ADVISORS[a]}/{nm}"] = float(r2); r.append(f"{r2:.3f}")
    rows.append(r)
table(["advisor"] + list(sets), rows)

# ---------------------------------------------------------------- E. model residuals
P("## E. Where the blended model is wrong (out-of-fold residual = truth - prediction)\n")
cfg = json.load(open(os.path.join(ROOT, "results", "blend_b0.json")))
rows = []
for a in range(4):
    d = pickle.load(open(os.path.join(ROOT, "ckpt", cfg["gbm"], f"cv_a{a}_all.pkl"), "rb"))
    oc = np.load(os.path.join(ROOT, "ckpt", cfg["cnn"], f"a{a}_aug{cfg['augs'][a]}_oof.npy"))
    b = cfg["blend"][a]; pred = b["w_gbm"] * d["oof"] + b["w_cnn"] * oc + b["b"]
    assert np.allclose(d["y"], Y[a])
    res = Y[a] - pred
    bands = [(0, .25), (.25, .5), (.5, .65), (.65, .75), (.75, .85), (.85, 1.01)]
    sd = [float(res[(pred >= lo) & (pred < hi)].std()) if ((pred >= lo) & (pred < hi)).sum() > 5 else None for lo, hi in bands]
    nearv = (pred >= 0.75)
    fp = float((Y[a][nearv] < THRESH).mean())
    fn = float((pred[Y[a] >= THRESH] < 0.75).mean())
    X = XF[a]; inv = gslice(["count", "adj4", "adjd", "dist", "comp", "ring"])
    hi = pred >= 0.6
    rho = np.array([spearmanr(X[hi][:, j], res[hi])[0] if X[hi][:, j].std() > 0 else 0 for j in inv])
    o = np.argsort(-np.abs(rho))[:4]
    R[f"residual/{ADVISORS[a]}"] = dict(sd_by_pred_band=sd, false_valid_rate=fp, missed_valid_rate=fn, kurtosis=float(((res - res.mean()) ** 4).mean() / res.var() ** 2),
                                      top_resid_feats=[(names[inv[j]], round(float(rho[j]), 3)) for j in o])
    rows.append([ADVISORS[a]] + [f"{s:.3f}" if s is not None else "-" for s in sd] + [f"{fp:.1%}", f"{fn:.1%}", "; ".join(f"{names[inv[j]]} ({rho[j]:+.2f})" for j in o[:3])])
table(["advisor"] + [f"sd, pred {lo}-{min(hi,1)}" for lo, hi in bands] + ["pred>=0.75 but invalid", "valid but pred<0.75", "features most correlated with residual (pred>=0.6)"], rows)

# ---------------------------------------------------------------- F. our optimised sets
P("## F. What our optimised sets look like\n")
S = {}
for t in ["b1", "b2", "b3", "b4", "x1", "x2", "x3", "x4"]:
    z = np.load(os.path.join(ROOT, "ckpt", t, "final.npz")); S[t] = (z["grids"].reshape(100, 7, 7).astype(np.int8), z["pred"])
s5 = sorted(glob.glob(os.path.join(ROOT, "submissions", "S5_*.npy")))
rows = []
for t, (g, pred) in S.items():
    D = hamming_matrix(g).astype(float); iu = np.triu_indices(100, 1)
    p = (g.reshape(100, 49)[:, :, None] == np.arange(5)).mean(0)          # 49,5 per-cell frequencies
    comp = p.mean(0)
    Dn = D.copy(); np.fill_diagonal(Dn, 1e6); nn = np.sort(Dn, 1)
    z_ = (pred - THRESH) / np.array([b["sigma"] for b in cfg["blend"]])
    bind = np.bincount(z_.argmin(1), minlength=4)
    R[f"sets/{t}"] = dict(score=int_score(g) / DENOM, mean_pairwise=float(D[iu].mean() / 49), cell_bound=float((1 - (p ** 2).sum(1)).mean()),
                         comp_bound=float(1 - (comp ** 2).sum()), nearest=float(nn[:, 0].mean() / 49), tenth=float(nn[:, K - 1].mean() / 49),
                         composition=comp.round(3).tolist(), binding=bind.tolist(), pred_mean=pred.mean(0).round(3).tolist())
    rows.append([t, f"{int_score(g)/DENOM:.4f}", f"{nn[:,0].mean()/49:.3f}", f"{nn[:,K-1].mean()/49:.3f}", f"{D[iu].mean()/49:.3f}", f"{(1-(p**2).sum(1)).mean():.3f}",
                 f"{1-(comp**2).sum():.3f}", " / ".join(f"{c*49:.1f}" for c in comp), " / ".join(map(str, bind)), " / ".join(f"{v:.3f}" for v in pred.mean(0))])
table(["set", "score (10-NN mean)", "nearest", "10th nearest", "mean over all pairs", "limit from per-cell mix", "limit from overall mix",
       "cells per grid Res/Ind/Com/Park/Off", "tightest advisor W/T/Tr/B (grids)", "mean prediction W/T/Tr/B"], rows)

allg = np.concatenate([g for g, _ in S.values()]); allp = np.concatenate([p for _, p in S.values()])
Xs = features(allg)[0]; cs = Xs[:, groups["count"][0]:groups["count"][1]]
P("Count of each district per grid in our 800 optimised grids, against grids each advisor labels valid:\n")
rows = [["our optimised grids"] + [f"{cs[:,d].mean():.1f} ({int(cs[:,d].min())}-{int(cs[:,d].max())})" for d in range(5)]]
pool = np.load(os.path.join(ROOT, "cache", "cands_s0.npz")); pk = pool["pvalid"] >= 0.95
cp = (pool["grids"][pk].reshape(-1, 49)[:, :, None] == np.arange(5)).sum(1)
rows.append(["candidate pool, P(valid)>=0.95"] + [f"{cp[:,d].mean():.1f} ({int(np.quantile(cp[:,d],.02))}-{int(np.quantile(cp[:,d],.98))})" for d in range(5)])
for a in range(4):
    v = Y[a] >= THRESH
    rows.append([f"{ADVISORS[a]}-valid labelled"] + [f"{CNT[a][v][:,d].mean():.1f} ({int(np.quantile(CNT[a][v][:,d],.02))}-{int(np.quantile(CNT[a][v][:,d],.98))})" for d in range(5)])
rows.append(["all dataset grids"] + [f"{CNT[0][:,d].mean():.1f} ({int(np.quantile(CNT[0][:,d],.02))}-{int(np.quantile(CNT[0][:,d],.98))})" for d in range(5)])
table(["group"] + [f"{d} mean (range)" for d in DN], rows)
R["sets/pooled_counts"] = dict(mean=cs.mean(0).tolist(), min=cs.min(0).tolist(), max=cs.max(0).tolist(), sd=cs.std(0).tolist())

# fig2: composition by group
fig, ax = plt.subplots(1, 5, figsize=(14, 3.2), sharey=True, facecolor="#fcfcfb")
labs = ["all dataset"] + [f"{A}-valid" for A in ADVISORS] + ["pool P>=0.95", "our optimised sets"]
for d in range(5):
    vals = [CNT[0][:, d].mean()] + [CNT[a][Y[a] >= THRESH][:, d].mean() for a in range(4)] + [cp[:, d].mean(), cs[:, d].mean()]
    ax[d].set_facecolor("#fcfcfb"); ax[d].barh(labs[::-1], vals[::-1], color=COL[d], height=0.6)
    for i, v in enumerate(vals[::-1]): ax[d].text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=8, color="#333")
    ax[d].set_title(f"{DN[d]} cells per grid", fontsize=10); ax[d].spines[["top", "right"]].set_visible(False); ax[d].set_xlim(0, max(vals) * 1.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_composition.png"), dpi=130); plt.close(fig)

# fig3: where each district sits in our sets
freq = (allg[:, :, :, None] == np.arange(5)).mean(0)
fig, ax = plt.subplots(1, 5, figsize=(14, 3.3), facecolor="#fcfcfb")
for d in range(5):
    im = ax[d].imshow(freq[:, :, d], cmap="Blues", vmin=0, vmax=freq.max())
    for i in range(7):
        for j in range(7):
            ax[d].text(j, i, f"{freq[i,j,d]*100:.0f}", ha="center", va="center", fontsize=7, color="#fff" if freq[i, j, d] > 0.6 * freq.max() else "#222")
    ax[d].set_title(f"{DN[d]}: % of grids with this district in the cell", fontsize=9); ax[d].set_xticks([]); ax[d].set_yticks([])
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_cell_maps.png"), dpi=130); plt.close(fig)
ringf = np.stack([freq[RING.reshape(7, 7) == r].mean(0) for r in range(4)])
R["sets/ring_freq"] = ringf.round(3).tolist()
P("Share of each district by ring (0 = outer edge, 3 = centre cell) in our optimised grids:\n")
table(["ring"] + DN, [[r] + [f"{ringf[r,d]:.2f}" for d in range(5)] for r in range(4)])

# top features of each advisor: our sets vs that advisor's valid / all labelled grids
P("Each advisor's strongest features: our grids against labelled grids\n")
rows = []
for a in range(4):
    for nm, rho in R[f"top_features/{ADVISORS[a]}"][:6]:
        j = names.index(nm); v = Y[a] >= THRESH
        rows.append([ADVISORS[a], nm, f"{rho:+.2f}", f"{XF[a][:,j].mean():.2f}", f"{XF[a][v][:,j].mean():.2f}", f"{Xs[:,j].mean():.2f}", f"{Xs[:,j].min():.1f}-{Xs[:,j].max():.1f}"])
table(["advisor", "feature", "Spearman", "mean, all labelled", "mean, advisor-valid", "mean, our grids", "range, our grids"], rows)

# cluster structure in our grids
ci = groups["comp"][0]
P("Cluster structure (connected same-district regions), our grids against all dataset grids:\n")
table(["district", "clusters, ours", "clusters, dataset", "largest cluster, ours", "largest cluster, dataset"],
      [[DN[d], f"{Xs[:,ci+2*d].mean():.1f}", f"{XF[0][:,ci+2*d].mean():.1f}", f"{Xs[:,ci+2*d+1].mean():.1f}", f"{XF[0][:,ci+2*d+1].mean():.1f}"] for d in range(5)])

# do independent runs find the same grids?
P("## G. Do independent runs converge to the same grids?\n")
tags = list(S); rows = []
for t in tags:
    g = S[t][0]; within = hamming_matrix(g).astype(float); np.fill_diagonal(within, 99)
    r = [t, f"{within.min(1).mean():.1f}"]
    for u in ("b2", "x4"):
        r.append("-" if u == t else f"{hamming_matrix(g, S[u][0]).min(1).mean():.1f}")
    rows.append(r)
if s5:
    g5 = np.load(s5[0]); rows.append(["S5 (submitted)", "", f"{hamming_matrix(g5, S['b2'][0]).min(1).mean():.1f}", f"{hamming_matrix(g5, S['x4'][0]).min(1).mean():.1f}"])
table(["set", "mean distance to nearest grid in same set (cells)", "to nearest grid in b2", "to nearest grid in x4"], rows)

json.dump(R, open(os.path.join(OUT, "patterns.json"), "w"), indent=1)
open(os.path.join(OUT, "report.md"), "w").write("\n".join(md) + "\n")
print("saved", OUT)
