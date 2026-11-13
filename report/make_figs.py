"""Figures for report.tex. Run from the repo root (on a compute node): python report/make_figs.py"""
import json, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, "src"); sys.path.insert(0, ".")
from rules import raw_scores, valid
from utils_public import load_grids, diversity_score

OUT = "report/figs/"
import os; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})
BLUE, ORANGE, GREEN, GREY, RED = "#1f77b4", "#e07b00", "#2a9d5c", "#777777", "#c0392b"

def pair_dist(g):
    f = g.reshape(len(g), 49)
    d = (f[:, None, :] != f[None, :, :]).sum(2)
    return d

# ---------------- Fig 1: submission progress ----------------
rows = [json.loads(l) for l in open("results/submission_log.jsonl")]
pts = []
for r in rows:
    m = re.fullmatch(r"S(\d+)", r["name"])
    if not m: continue
    n = int(m.group(1))
    ret = r.get("returned_score")
    try: ret = float(ret)
    except (TypeError, ValueError): ret = None
    pts.append((n, ret, r.get("expected_score")))
pts.sort()
fig, ax = plt.subplots(figsize=(6.6, 3.0))
xs = [n for n, ret, _ in pts if ret is not None]; ys = [ret for _, ret, _ in pts if ret is not None]
ax.plot(xs, ys, "-o", color=BLUE, ms=4, lw=1.2, label="score returned by the leaderboard")
held = [(n, 38048/49000 if n == 22 else e) for n, ret, e in pts if ret is None and n >= 20]
ax.plot([h[0] for h in held], [h[1] for h in held], "*", color=ORANGE, ms=10, ls="", label="exact score, held back (S20--S22)")
ax.axhline(38000/49000, color=GREY, ls="--", lw=1); ax.text(1.2, 38000/49000-0.022, "annealing ceiling 38000/49000", color=GREY, fontsize=8)
ax.axhline(0.7959, color=GREEN, ls=":", lw=1); ax.text(1.2, 0.7959+0.006, "ideal code construction 0.796 (not all valid)", color=GREEN, fontsize=8)
ax.axvspan(0.5, 17.5, color="#f0f0f0", zorder=0); ax.text(9, 0.30, "surrogate models", ha="center", color=GREY, fontsize=8)
ax.text(20, 0.30, "exact rules", ha="center", color=GREY, fontsize=8)
ax.set_xlabel("submission number"); ax.set_ylabel("diversity score"); ax.set_ylim(0.25, 0.84)
ax.set_xticks(range(1, 23, 3)); ax.legend(loc="center right", fontsize=7.5, frameon=False, bbox_to_anchor=(1.0, 0.32))
fig.tight_layout(); fig.savefig(OUT + "fig1_progress.pdf"); plt.close(fig)

# ---------------- Fig 2: labels are a rank transform of integer scores ----------------
G = load_grids()
S = np.load("datasets/scores.npy")
names = ["Wellness", "Tax", "Transportation", "Business"]
fig, axs = plt.subplots(1, 4, figsize=(6.8, 2.1))
for a in range(4):
    idx = np.where(~np.isnan(S[:, a]))[0]
    raw, ok = raw_scores(G[idx])
    sel = ok
    axs[a].scatter(raw[sel, a], S[idx[sel], a], s=1.5, color=BLUE, alpha=0.5, rasterized=True)
    lo, hi = np.percentile(raw[sel, a], [0.5, 99.5]); axs[a].set_xlim(lo - 0.05*(hi-lo), hi + 0.05*(hi-lo))
    axs[a].axhline(0.75, color=RED, lw=0.8, ls="--")
    axs[a].set_title(names[a], fontsize=8); axs[a].set_xlabel("exact integer score", fontsize=7.5)
    axs[a].tick_params(labelsize=7)
    if a == 0: axs[a].set_ylabel("label")
fig.tight_layout(); fig.savefig(OUT + "fig2_labels.pdf"); plt.close(fig)

# ---------------- load submissions ----------------
def load(name):
    import glob
    f = sorted(glob.glob(f"submissions/{name}_2*.npy"))[0]
    return np.load(f)
S21, S22, S19, S8 = load("S21"), load("S22"), load("S19"), load("S8")
for nm, s in [("S22", S22), ("S21", S21), ("S19", S19), ("S8", S8)]:
    v = valid(s)
    print(nm, "valid", int(v.sum()), "score", diversity_score(s, k=10, N=100))

# ---------------- Fig 3: example grids ----------------
cmap = ListedColormap(["#e4572e", "#7a7a7a", "#f2b134", "#59a96a", "#4b7bb5"])
names5 = ["Residential", "Industrial", "Commercial", "Park", "Office"]
fig, axs = plt.subplots(2, 4, figsize=(6.6, 3.5))
for i, ax in enumerate(axs.ravel()):
    gi = i * 6 if i < 4 else 49 + (i - 4) * 12
    ax.imshow(S22[gi], cmap=cmap, vmin=0, vmax=4)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_xticks(np.arange(-.5, 7, 1), minor=True); ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8); ax.tick_params(which="minor", length=0)
    ax.set_title(("core " if i < 4 else "free ") + f"grid {gi}", fontsize=8)
fig.legend(handles=[Patch(color=cmap(k), label=names5[k]) for k in range(5)], loc="lower center", ncol=5, fontsize=7.5, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(OUT + "fig3_grids.pdf"); plt.close(fig)

# ---------------- Fig 4: distance structure ----------------
fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.3))
bins = np.arange(33, 51)
for s, c, lab in [(S19, GREY, "S19 (0.7753)"), (S22, ORANGE, "S22 (0.7765)")]:
    d = pair_dist(s)[np.triu_indices(100, 1)]
    h = np.bincount(d, minlength=50)[33:50]
    axs[0].plot(np.arange(33, 50), h, "-o", ms=3, color=c, label=lab)
axs[0].set_xlabel("Hamming distance of a pair"); axs[0].set_ylabel("number of pairs (of 4950)")
axs[0].legend(fontsize=7, frameon=False); axs[0].set_title("(a) pair distances", fontsize=8)
for s, c, lab in [(S19, GREY, "S19"), (S22, ORANGE, "S22")]:
    d = pair_dist(s); np.fill_diagonal(d, 99)
    close = np.sort((d <= 38).sum(1))
    axs[1].plot(np.arange(100), close, color=c, label=lab)
axs[1].axhline(10, color=RED, ls="--", lw=0.8); axs[1].text(2, 11, "10 = k", color=RED, fontsize=7.5)
axs[1].set_xlabel("grid (sorted)"); axs[1].set_ylabel("neighbours within distance 38")
axs[1].legend(fontsize=7, frameon=False); axs[1].set_title("(b) crowding", fontsize=8)
for s, c, lab in [(S19, GREY, "S19"), (S22, ORANGE, "S22")]:
    d = pair_dist(s); np.fill_diagonal(d, 99)
    contrib = np.sort(np.sort(d, 1)[:, :10].sum(1))
    axs[2].plot(np.arange(100), contrib, color=c, label=lab)
axs[2].axhline(380, color=RED, ls="--", lw=0.8); axs[2].text(2, 381, "380", color=RED, fontsize=7.5)
axs[2].set_xlabel("grid (sorted)"); axs[2].set_ylabel("sum of 10-NN distances")
axs[2].legend(fontsize=7, frameon=False); axs[2].set_title("(c) contribution", fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "fig4_distances.pdf"); plt.close(fig)

# ---------------- Fig 5: composition of valid grids vs the dataset ----------------
sub = G[::10]                                             # 50,000 dataset grids
fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.0))
for k, (t, nm) in enumerate([(3, "park"), (0, "residential"), (1, "industrial")]):
    cd = (sub.reshape(len(sub), 49) == t).sum(1); cs = (S22.reshape(100, 49) == t).sum(1)
    b = np.arange(0, 30) - 0.5
    axs[k].hist(cd, bins=b, density=True, color=GREY, alpha=0.6, label="dataset")
    axs[k].hist(cs, bins=b, density=True, color=ORANGE, alpha=0.8, label="S22")
    axs[k].set_xlabel(f"{nm} cells per grid"); axs[k].set_xlim(0, 25)
    if k == 0: axs[k].set_ylabel("share of grids"); axs[k].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(OUT + "fig5_composition.pdf"); plt.close(fig)
print("done")
