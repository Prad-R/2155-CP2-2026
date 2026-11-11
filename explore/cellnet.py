"""Structured per-cell model: score = sum_i MLP(own type, counts of each type per offset class around i, nearest distances, position)
+ MLP(global counts).  D4-invariant by construction.  Target = integer level (central labels)."""
import numpy as np, sys, time, torch, torch.nn as nn, torch.nn.functional as F
from lib import *

dev = "cuda" if torch.cuda.is_available() else "cpu"


def offset_classes(R):
    ks = sorted({(min(abs(dr), abs(dc)), max(abs(dr), abs(dc))) for dr in range(-R, R + 1) for dc in range(-R, R + 1)} - {(0, 0)},
                key=lambda k: (k[0] ** 2 + k[1] ** 2, k))
    mats = []
    for k in ks:
        M = ((np.minimum(DR, DC) == k[0]) & (np.maximum(DR, DC) == k[1])).astype(np.float32)
        mats.append(M)
    return ks, np.stack(mats)                                       # K,49,49


POS = np.stack([np.minimum(RC[:, 0], 6 - RC[:, 0]), np.minimum(RC[:, 1], 6 - RC[:, 1])], 1)
POS = np.stack([POS.min(1), POS.max(1)], 1).astype(np.float32) / 3.0          # D4-invariant position descriptor


class Feat:
    def __init__(s, R=3, near=True, metric="cls"):
        s.ks, M = offset_classes(R)
        s.M = torch.as_tensor(M, device=dev)
        s.near = near
        s.man = torch.as_tensor(MAN.astype(np.float32), device=dev)
        s.pos = torch.as_tensor(POS, device=dev)
        s.dim = 5 + 5 * len(s.ks) + 2 + (5 if near else 0)

    def __call__(s, g):
        """g: (n,7,7) long tensor -> (n,49,dim)"""
        n = len(g)
        O = F.one_hot(g.reshape(n, 49).long(), 5).float()               # n,49,5
        cnt = torch.einsum("njb,kij->nikb", O, s.M).reshape(n, 49, -1)   # n,49,K*5
        parts = [O, cnt / 2.0, s.pos[None].expand(n, -1, -1)]
        if s.near:
            d = torch.where(O.permute(0, 2, 1)[:, :, None, :] > 0, s.man[None, None], torch.full((), 14.0, device=dev)).amin(3)   # n,5,49
            parts.append(d.permute(0, 2, 1) / 4.0)
        return torch.cat(parts, 2)


class CellNet(nn.Module):
    def __init__(s, dim, w=128, depth=3, link=False):
        super().__init__()
        L, c = [], dim
        for i in range(depth):
            L += [nn.Linear(c, w), nn.GELU()]; c = w
        L += [nn.Linear(c, 1)]
        s.f = nn.Sequential(*L)
        s.cnt = nn.Sequential(nn.Linear(5, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        s.b = nn.Parameter(torch.zeros(1))

    def forward(s, X, C):
        return s.f(X).sum((1, 2)) / 7 + s.cnt(C).squeeze(1) + s.b


def train(X, C, t, tr, te, w=128, depth=3, epochs=300, bs=256, lr=2e-3, wd=1e-5, seed=0):
    torch.manual_seed(seed)
    m = CellNet(X.shape[2], w, depth).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    idx = torch.where(tr)[0]; n = len(idx); steps = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    for ep in range(epochs):
        perm = idx[torch.randperm(n, device=dev)]
        for i in range(n // bs):
            b = perm[i * bs:(i + 1) * bs]
            loss = F.mse_loss(m(X[b], C[b]), t[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    with torch.no_grad():
        p = m(X, C)
    return m, p


if __name__ == "__main__":
    advs = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0]
    nfold = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    cfgs = [("R1", dict(R=1, near=False)), ("R2", dict(R=2, near=False)), ("R3", dict(R=3, near=False)), ("R1+near", dict(R=1, near=True)),
            ("R2+near", dict(R=2, near=True)), ("R3+near", dict(R=3, near=True)), ("R6+near", dict(R=6, near=True))]
    if len(sys.argv) > 3:
        cfgs = [c for c in cfgs if c[0] in sys.argv[3].split(",")]
    for a in advs:
        g, y, z, lev, idx = labelled(a)
        cen = (y > 0.05) & (y < 0.95)
        g, lev = g[cen], lev[cen].astype(np.float64)
        sd = lev.std(); t = torch.as_tensor((lev - lev.mean()) / sd, dtype=torch.float32, device=dev)
        f = folds_for(len(g), 5, 0)
        G = torch.as_tensor(g, device=dev)
        C = F.one_hot(G.reshape(len(G), 49).long(), 5).float().sum(1) / 10
        print(f"===== {ADVISORS[a]}: central n={len(g)} sd(level)={sd:.1f}", flush=True)
        for nm, kw in cfgs:
            X = Feat(**kw)(G)
            t0 = time.time(); r = []
            for k in range(nfold):
                tr, te = torch.as_tensor(f != k, device=dev), torch.as_tensor(f == k, device=dev)
                m, p = train(X, C, t, tr, te, seed=k)
                r.append((float((p - t)[te].std()) * sd, float((p - t)[tr].std()) * sd))
            r = np.array(r)
            print(f"  {nm:10s} dim {X.shape[2]:3d}  held-out resid sd {r[:, 0].mean():6.2f} levels (R2 {1 - (r[:, 0].mean() / sd) ** 2:.4f})   train {r[:, 1].mean():6.2f}  [{time.time() - t0:.0f}s]", flush=True)
