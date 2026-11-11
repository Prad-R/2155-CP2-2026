"""Static figures of the submitted designs -> results/viz/*.png, and the interactive page (template + data)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from common import ROOT
V = os.path.join(ROOT, "results", "viz")
R = json.load(open(os.path.join(V, "designs.json")))
open(os.path.join(V, "designs.html"), "w").write(open(os.path.join(V, "template.html")).read().replace("/*DATA*/[]", json.dumps(R, separators=(",", ":"))))
DC = ["#f2d43d", "#3b3e48", "#d9482b", "#1f9e5a", "#2f7fdc"]; DN = ["Residential", "Industrial", "Commercial", "Park", "Office"]
CC = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
cm = ListedColormap(DC); INK, MUT = "#1c2321", "#55605b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#d5d9d3", "text.color": INK})
G = lambda r: np.array(list(r["g"]), int).reshape(7, 7)
def tile(ax, r, title=None):
    ax.pcolormesh(G(r)[::-1], cmap=cm, vmin=0, vmax=4, edgecolors="white", linewidth=0.6); ax.set_aspect("equal"); ax.axis("off")
    if title: ax.set_title(title, fontsize=7, color=MUT, pad=2)
def dlegend(fig, y=0.995):
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in DC], labels=[f"{i} {n}" for i, n in enumerate(DN)], loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, y), fontsize=9)
ok = [r for r in R if r["st"] in ("confirmed", "near")]
for b, nm, ttl in (("c", "map_character", "Layouts grouped by character (district counts and which districts touch)"), ("l", "map_layout", "Layouts grouped cell by cell (the distance the score uses)")):
    fig, ax = plt.subplots(figsize=(11, 7.5))
    co = np.array([r["co"] for r in ok]); s = 10 + 60 * (co - co.min()) / (co.max() - co.min())
    for st, al in (("near", 0.35), ("confirmed", 0.95)):
        m = np.array([r["st"] == st for r in ok])
        ax.scatter(np.array([r["x" + b] for r in ok])[m], -np.array([r["y" + b] for r in ok])[m], s=s[m], c=[CC[r["c" + b] - 1] for r, k in zip(ok, m) if k], alpha=al, edgecolors="white", linewidths=0.5)
    bad = [r for r in R if r["st"] == "invalid"]
    ax.scatter([r["x" + b] for r in bad], [-r["y" + b] for r in bad], marker="x", c=INK, s=70, linewidths=2)
    for k in sorted({r["c" + b] for r in ok}):
        L = [r for r in ok if r["c" + b] == k]
        ax.text(np.mean([r["x" + b] for r in L]), -np.mean([r["y" + b] for r in L]), f"{b.upper()}{k}", ha="center", va="center", fontsize=12, fontweight="bold", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=CC[k - 1], lw=2.5))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl + "\nsolid = confirmed valid, faded = set with 2-4 unknown failures, x = confirmed invalid, larger = contributes more", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(V, nm + ".png"), dpi=150); plt.close(fig)
for b in ("c", "l"):
    ks = sorted({r["c" + b] for r in ok}); fig, axs = plt.subplots(len(ks), 9, figsize=(13, 1.55 * len(ks) + 0.6), gridspec_kw=dict(width_ratios=[2.6] + [1] * 8))
    for row, k in zip(axs, ks):
        L = [r for r in ok if r["c" + b] == k]; mx, my = np.mean([r["x" + b] for r in L]), np.mean([r["y" + b] for r in L])
        L2 = sorted(L, key=lambda r: (r["x" + b] - mx) ** 2 + (r["y" + b] - my) ** 2)[:8]; mc = np.mean([r["cnt"] for r in L], 0)
        row[0].barh(range(5), mc[::-1], color=DC[::-1]); row[0].set_yticks([]); row[0].set_xlim(0, 20)
        for i, v in enumerate(mc[::-1]): row[0].text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=7)
        row[0].set_title(f"{b.upper()}{k}: {len(L)} layouts, mean contribution {np.mean([r['co'] for r in L]):.0f}", fontsize=8, loc="left", color=CC[k - 1], fontweight="bold")
        row[0].spines[["top", "right"]].set_visible(False); row[0].tick_params(labelsize=6)
        for ax, r in zip(row[1:], L2): tile(ax, r, f"{r['set']} {r['co']}")
    dlegend(fig); fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join(V, f"clusters_{'character' if b == 'c' else 'layout'}.png"), dpi=140); plt.close(fig)
def sheet(L, cols, fn, ttl):
    rows_ = int(np.ceil(len(L) / cols)); fig, axs = plt.subplots(rows_, cols, figsize=(cols * 0.82, rows_ * 0.95 + 0.7)); axs = np.atleast_2d(axs)
    for ax in axs.ravel(): ax.axis("off")
    for ax, r in zip(axs.ravel(), L): tile(ax, r, f"{r['co']} {r['set']} C{r['cc']}")
    fig.suptitle(ttl, fontsize=10, y=0.999, x=0.01, ha="left"); dlegend(fig, 1 - 0.35 / (rows_ * 0.95 + 0.7)); fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / (rows_ * 0.95 + 0.7))); fig.savefig(os.path.join(V, fn), dpi=130); plt.close(fig)
conf = sorted([r for r in R if r["st"] == "confirmed"], key=lambda r: -r["co"])
sheet(conf, 25, "gallery_confirmed.png", f"All {len(conf)} confirmed-valid layouts, highest contribution first (label: contribution in own set, set, character cluster)")
best = sorted([r for r in R if r["set"] in ("S5", "S8", "S11", "S13")], key=lambda r: -r["co"])
sheet(best[:30] + best[-30:], 15, "gallery_top_bottom.png", "Our four best sets (S5, S8, S11, S13): 30 highest contributors (top two rows) and 30 lowest (bottom two rows)")
print("wrote", sorted(os.listdir(V)), flush=True)
