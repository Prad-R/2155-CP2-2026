"""How local are the advisor rules?  Sum-pooled CNNs with a restricted receptive field, z target, held-out R2."""
import numpy as np, sys, time, torch, torch.nn as nn, torch.nn.functional as F
from lib import *

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True


class SumNet(nn.Module):
    def __init__(s, nconv, w=96, cnt=True, pool="sum"):
        super().__init__()
        layers, c = [], 5
        for i in range(nconv):
            layers += [nn.Conv2d(c, w, 3, padding=1), nn.GELU()]; c = w
        layers += [nn.Conv2d(c, w, 1), nn.GELU(), nn.Conv2d(w, w, 1), nn.GELU(), nn.Conv2d(w, 1, 1)]
        s.f = nn.Sequential(*layers)
        s.cnt = nn.Sequential(nn.Linear(5, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1)) if cnt else None
        s.b = nn.Parameter(torch.zeros(1))

    def forward(s, g):
        x = F.one_hot(g.long(), 5).permute(0, 3, 1, 2).float()
        out = s.f(x).sum((1, 2, 3)) / 7 + s.b
        if s.cnt is not None:
            out = out + s.cnt(x.sum((2, 3)) / 10).squeeze(1)
        return out


def aug(g):
    k = np.random.randint(4)
    g = torch.rot90(g, k, (1, 2))
    if np.random.rand() < 0.5:
        g = g.flip(2)
    return g


def run(model, gtr, ztr, gte, zte, epochs=200, bs=256, lr=2e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(gtr); steps = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(n // bs):
            b = perm[i * bs:(i + 1) * bs]
            loss = F.mse_loss(model(aug(gtr[b])), ztr[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        p = torch.stack([model(torch.rot90(gte, k, (1, 2)).flip(2) if fl else torch.rot90(gte, k, (1, 2))) for k in range(4) for fl in (0, 1)]).mean(0)
        ptr = model(gtr)
    r2 = 1 - ((p - zte) ** 2).sum() / ((zte - zte.mean()) ** 2).sum()
    r2tr = 1 - ((ptr - ztr) ** 2).sum() / ((ztr - ztr.mean()) ** 2).sum()
    return float(r2), float(r2tr), p.cpu().numpy()


if __name__ == "__main__":
    advs = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2, 3]
    nfold = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    configs = [("rf1 (counts only)", 0), ("rf3", 1), ("rf5", 2), ("rf7", 3), ("rf9", 4), ("rf13", 6)]
    for a in advs:
        g, y, z, lev, idx = labelled(a)
        zs = (z - z.mean()) / z.std()
        f = folds_for(len(g), 5, 0)
        G = torch.as_tensor(g, device=dev); Z = torch.as_tensor(zs, dtype=torch.float32, device=dev)
        print(f"===== {ADVISORS[a]}", flush=True)
        for nm, nconv in configs:
            t0 = time.time(); r = []
            for k in range(nfold):
                tr, te = torch.as_tensor(f != k, device=dev), torch.as_tensor(f == k, device=dev)
                torch.manual_seed(k); np.random.seed(k)
                m = SumNet(nconv).to(dev)
                r.append(run(m, G[tr], Z[tr], G[te], Z[te])[:2])
            r = np.array(r)
            print(f"  {nm:20s} held-out R2(z) {r[:, 0].mean():.4f}  (folds {np.round(r[:, 0], 4)})  train {r[:, 1].mean():.4f}  [{time.time() - t0:.0f}s]", flush=True)
