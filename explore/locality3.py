"""Locality test v2: target = integer level index, central labels only (no link distortion).
Reports held-out residual sd in level units (exact rule => < 0.5)."""
import numpy as np, sys, time, torch, torch.nn as nn, torch.nn.functional as F
from lib import *
from locality import aug

dev = "cuda" if torch.cuda.is_available() else "cpu"


class SumNet(nn.Module):
    def __init__(s, nconv, w=96, k=3, cnt=True, link=False):
        super().__init__()
        layers, c = [], 5
        for i in range(nconv):
            layers += [nn.Conv2d(c, w, k, padding=k // 2), nn.GELU()]; c = w
        layers += [nn.Conv2d(c, w, 1), nn.GELU(), nn.Conv2d(w, w, 1), nn.GELU(), nn.Conv2d(w, 1, 1)]
        s.f = nn.Sequential(*layers)
        s.cnt = nn.Sequential(nn.Linear(5, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1)) if cnt else None
        s.b = nn.Parameter(torch.zeros(1))
        s.link = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1)) if link else None

    def forward(s, g):
        x = F.one_hot(g.long(), 5).permute(0, 3, 1, 2).float()
        out = s.f(x).sum((1, 2, 3)) / 7 + s.b
        if s.cnt is not None:
            out = out + s.cnt(x.sum((2, 3)) / 10).squeeze(1)
        if s.link is not None:
            out = out + s.link(out[:, None]).squeeze(1)
        return out


def run(model, gtr, ttr, gte, tte, epochs=300, bs=256, lr=2e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n = len(gtr); steps = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(n // bs):
            b = perm[i * bs:(i + 1) * bs]
            loss = F.mse_loss(model(gtr[b]), ttr[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        p = model(gte)
        ptr = model(gtr)
    return float((p - tte).std()), float((ptr - ttr).std()), p


if __name__ == "__main__":
    advs = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2, 3]
    nfold = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    configs = [("rf1", dict(nconv=0)), ("rf3", dict(nconv=1)), ("rf5", dict(nconv=2)), ("rf7", dict(nconv=3)),
               ("rf9", dict(nconv=4)), ("rf13", dict(nconv=6)), ("rf5k5", dict(nconv=1, k=5)), ("rf7k7", dict(nconv=1, k=7)),
               ("rf13k13", dict(nconv=1, k=13))]
    for a in advs:
        g, y, z, lev, idx = labelled(a)
        cen = (y > 0.05) & (y < 0.95)
        g, lev = g[cen], lev[cen].astype(np.float64)
        sd = lev.std(); t = (lev - lev.mean()) / sd
        f = folds_for(len(g), 5, 0)
        G = torch.as_tensor(g, device=dev); Tt = torch.as_tensor(t, dtype=torch.float32, device=dev)
        print(f"===== {ADVISORS[a]}: central n={len(g)}  sd(level)={sd:.1f}", flush=True)
        for nm, kw in configs:
            if names and nm not in names:
                continue
            t0 = time.time(); r = []
            for k in range(nfold):
                tr, te = torch.as_tensor(f != k, device=dev), torch.as_tensor(f == k, device=dev)
                torch.manual_seed(k); np.random.seed(k)
                m = SumNet(**kw).to(dev)
                r.append(run(m, G[tr], Tt[tr], G[te], Tt[te])[:2])
            r = np.array(r) * sd
            print(f"  {nm:10s} held-out resid sd {r[:, 0].mean():6.2f} levels (R2 {1 - (r[:, 0].mean() / sd) ** 2:.4f})   train resid sd {r[:, 1].mean():6.2f}  [{time.time() - t0:.0f}s]", flush=True)
