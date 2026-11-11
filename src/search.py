"""Generate candidate grids by local search on the CNN ensemble (GPU).

Phase A: noisy hill-climb on obj = min_a(mean_a - kappa*std_a) from good dataset grids.
Phase B: once obj >= target, random walk among single-cell neighbours that stay >= target,
         which spreads candidates through the feasible region.
Resumable per round (ckpt/<tag>/round_XXX.npz).

  python -u src/search.py --tag s0 --cnn cnn0 --augs 1,1,1,1 [--quick]
"""
import argparse, os, time
import numpy as np
import torch

from common import ROOT, load_data
from surrogate import CNNEns

p = argparse.ArgumentParser()
p.add_argument("--tag", default="s0")
p.add_argument("--cnn", default="cnn0")
p.add_argument("--augs", default="1,1,1,1")
p.add_argument("--rounds", type=int, default=20)
p.add_argument("--batch", type=int, default=384)
p.add_argument("--steps", type=int, default=150)
p.add_argument("--target", type=float, default=0.85)
p.add_argument("--kappa", type=float, default=1.0)
p.add_argument("--noise", type=float, default=0.01)
p.add_argument("--pool", type=int, default=20000, help="start from top-N dataset grids")
p.add_argument("--every", type=int, default=5, help="record feasible grids every N steps")
p.add_argument("--quick", action="store_true")
args = p.parse_args()
if args.quick:
    args.rounds, args.batch, args.steps = 2, 32, 10

out = os.path.join(ROOT, "ckpt", args.tag + ("_quick" if args.quick else ""))
os.makedirs(out, exist_ok=True)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
augs = [int(x) for x in args.augs.split(",")]
ens = CNNEns(args.cnn + ("_quick" if args.quick else ""), augs)
dev = ens.dev
grids, _ = load_data()

# dataset predictions from training outputs -> starting pool
cdir = os.path.join(ROOT, "ckpt", args.cnn + ("_quick" if args.quick else ""))
P = np.stack([np.stack([np.load(os.path.join(cdir, f"a{a}_aug{augs[a]}_f{k}_all.npy")) for k in range(5)], 1)
              for a in range(4)], 1)  # n,4,5
score_ds = (P.mean(2) - args.kappa * P.std(2)).min(1)
pool = np.argsort(-score_ds)[:args.pool]
log("dataset obj: max %.3f, #>=0.75: %d, #>=target: %d" % (score_ds.max(), (score_ds >= 0.75).sum(), (score_ds >= args.target).sum()))

# neighbour generator: for each grid all 49*4 single-cell changes
cell = torch.arange(49, device=dev).repeat_interleave(4)          # 196
shift = torch.arange(1, 5, device=dev).repeat(49)                 # 196


def neighbours(g):  # g (B,7,7) -> (B,196,7,7)
    B = len(g)
    f = g.reshape(B, 1, 49).repeat(1, 196, 1)
    idx = cell.view(1, 196, 1).expand(B, 196, 1)
    new = (f.gather(2, idx) + shift.view(1, 196, 1)) % 5
    f.scatter_(2, idx, new)
    return f.reshape(B, 196, 7, 7)


def obj_of(g):
    pr = ens.predict_t(g)                                         # B,4,5
    mean, std = pr.mean(2), pr.std(2)
    return (mean - args.kappa * std).min(1).values, mean, std


for r in range(args.rounds):
    unit = os.path.join(out, f"round_{r:03d}.npz")
    if os.path.exists(unit):
        continue
    gen = torch.Generator(device=dev).manual_seed(1000 + r)
    rs = np.random.RandomState(1000 + r)
    g = torch.as_tensor(grids[rs.choice(pool, args.batch, replace=False)], device=dev).long()
    rec = []
    for step in range(args.steps):
        nb = neighbours(g)                                        # B,196,7,7
        o, _, _ = obj_of(nb.reshape(-1, 7, 7))
        o = o.view(len(g), 196)
        cur, _, _ = obj_of(g)
        feas = o >= args.target
        anyf = feas.any(1)
        # phase B: uniform random feasible neighbour; phase A: noisy best neighbour
        rnd = torch.rand(o.shape, device=dev, generator=gen)
        pickB = torch.where(feas, rnd, torch.full_like(rnd, -1.0)).argmax(1)
        pickA = (o + args.noise * torch.randn(o.shape, device=dev, generator=gen)).argmax(1)
        inB = (cur >= args.target) & anyf
        pick = torch.where(inB, pickB, pickA)
        g = nb[torch.arange(len(g), device=dev), pick]
        if step % args.every == 0 or step == args.steps - 1:
            c, m, s = obj_of(g)
            ok = c >= args.target
            if ok.any():
                rec.append((g[ok].cpu().numpy().astype(np.int8), m[ok].cpu().numpy(), s[ok].cpu().numpy()))
            if step % 25 == 0:
                log(f"round {r} step {step}: obj mean {c.mean():.3f} max {c.max():.3f} feasible {int(ok.sum())}/{len(g)}")
    if rec:
        G_ = np.concatenate([x[0] for x in rec]); M_ = np.concatenate([x[1] for x in rec]); S_ = np.concatenate([x[2] for x in rec])
        _, ui = np.unique(G_.reshape(len(G_), -1), axis=0, return_index=True)
        G_, M_, S_ = G_[ui], M_[ui], S_[ui]
    else:
        G_, M_, S_ = np.zeros((0, 7, 7), np.int8), np.zeros((0, 4), np.float32), np.zeros((0, 4), np.float32)
    tmp = unit + ".tmp.npz"
    np.savez(tmp, grids=G_, mean=M_, std=S_)
    os.replace(tmp, unit)
    log(f"round {r}: saved {len(G_)} feasible candidates")
open(os.path.join(out, "DONE"), "w").write("ok\n")
log("DONE")
