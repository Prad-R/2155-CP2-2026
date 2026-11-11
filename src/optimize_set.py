"""Set-level optimisation of a 100-grid submission on the exact 10-NN diversity score.

Each step: pick a grid, score all 196 single-cell changes by the exact change in the
set's diversity score, keep those the CNN ensemble still rates safe, then confirm the
best few with the independent GBM ensemble and accept the best one whose blended
P(valid) >= pmin. Resumable (state checkpoint every --ckpt-every steps).

  python -u src/optimize_set.py --tag o1 --init submissions/S3_x.npy --pmin 0.95 --steps 20000
"""
import argparse, json, os, signal, time
import numpy as np
import torch

from common import ROOT
from surrogate import CNNEns, GBMEns
from blend import blend_apply, p_valid
from divscore import hamming_matrix, int_score, DENOM, K

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--init", required=True)
p.add_argument("--blend", default="b0")
p.add_argument("--pmin", type=float, default=0.95)
p.add_argument("--cnn-target", type=float, default=0.78, help="cheap CNN prefilter: min_a(mean) >= this")
p.add_argument("--t0", type=float, default=3.0, help="start temperature (integer score units)")
p.add_argument("--t1", type=float, default=0.2)
p.add_argument("--ncand", type=int, default=24)
p.add_argument("--cycles", type=int, default=1, help="number of anneal/reheat cycles")
p.add_argument("--pool", default="", help="candidate .npz (grids, pvalid) for whole-grid replacement moves")
p.add_argument("--p-swap", type=float, default=0.03, help="probability of a whole-grid replacement move")
p.add_argument("--swap-cands", type=int, default=256)
p.add_argument("--swap-temp", type=float, default=5.0, help="swap temperature as a multiple of T")
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--topk", type=int, default=6, help="candidates per step sent to the GBM check")
p.add_argument("--p-flat", type=float, default=0.05, help="probability of accepting a zero-gain move")
p.add_argument("--ckpt-every", type=int, default=250)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--quick", action="store_true")
args = p.parse_args()
if args.quick:
    args.steps, args.ckpt_every = 60, 20

out = os.path.join(ROOT, "ckpt", args.tag + ("_quick" if args.quick else ""))
os.makedirs(out, exist_ok=True)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
cfg = json.load(open(os.path.join(ROOT, "results", f"blend_{args.blend}.json")))
B = cfg["blend"]
cnn, gbm = CNNEns(cfg["cnn"], cfg["augs"]), GBMEns(cfg["gbm"])
procs = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))


def blended(g):
    """g (n,7,7) -> blended pred (n,4), pvalid (n,)"""
    mc = cnn.predict(g).mean(2)
    mg = gbm.predict(g, procs=1).mean(2)
    pred = blend_apply(B, mg, mc)
    return pred, p_valid(B, pred)[0]


def total_scores(Dc):
    """Dc (m,n,n) int with large diagonal -> (m,) integer scores."""
    return np.partition(Dc, K - 1, axis=2)[:, :, :K].sum((1, 2))


poolG = None
if args.pool:
    _p = np.load(os.path.join(ROOT, args.pool))
    _k = _p["pvalid"] >= args.pmin
    poolG, poolpv = _p["grids"][_k].reshape(-1, 49).astype(np.int8), _p["pvalid"][_k]
    log(f"pool: {len(poolG)} candidates with pvalid >= {args.pmin}")

state_f = os.path.join(out, "state.npz")
if os.path.exists(state_f):
    z = np.load(state_f)
    G, step0, pv = z["G"], int(z["step"]), z["pv"]
    rng = np.random.RandomState(args.seed + step0)
    log(f"resumed at step {step0}")
else:
    step0 = 0
    rng = np.random.RandomState(args.seed)
    if args.init.startswith("pool"):
        _i = rng.choice(len(poolG), 100, replace=False)
        G = poolG[_i].copy()
    else:
        G = np.load(os.path.join(ROOT, args.init))
        G = (G["grids"] if args.init.endswith(".npz") else G).astype(np.int8).reshape(100, 49)
    _, pv = blended(G.reshape(-1, 7, 7))
log(f"start score {int_score(G.reshape(-1,7,7))/DENOM:.4f}  pvalid min {pv.min():.3f} mean {pv.mean():.3f}")


def save(step):
    tmp = state_f + ".tmp.npz"
    np.savez(tmp, G=G, step=step, pv=pv, bestG=bestG, bestpv=bestpv, best=best)
    os.replace(tmp, state_f)


stop = {"flag": False}
signal.signal(signal.SIGTERM, lambda *a: stop.update(flag=True))

n = 100
D = hamming_matrix(G).astype(np.int32)
np.fill_diagonal(D, 10**6)
cur = int(total_scores(D[None])[0])
cell = np.repeat(np.arange(49), 4)
shift = np.tile(np.arange(1, 5), 49)
acc = tried = nocnn = nogbm = nswap = 0
cyc = max(args.steps // max(args.cycles, 1), 1)
best, bestG, bestpv = cur, G.copy(), pv.copy()
if os.path.exists(state_f) and "best" in np.load(state_f) and int(np.load(state_f)["best"]) > cur:
    _z = np.load(state_f)
    best, bestG, bestpv = int(_z["best"]), _z["bestG"], _z["bestpv"]
step = step0 - 1
for step in range(step0, args.steps):
    if stop["flag"]:
        break
    if step % 2000 == 0 and step > step0:
        log(f"step {step}: T {T:.2f} best {best/DENOM:.4f} score {cur/DENOM:.4f} accepted {acc}/{tried} (cnn-reject {nocnn}, gbm-reject {nogbm}, swaps {nswap}) pvalid min {pv.min():.3f} mean {pv.mean():.3f}")
    if step % args.ckpt_every == 0 and step > step0:
        save(step)
    T = args.t0 * (args.t1 / args.t0) ** ((step % cyc) / cyc)
    # pick a grid, biased to those contributing least
    contrib = np.partition(D, K - 1, axis=1)[:, :K].sum(1)
    w = np.exp(-(contrib - contrib.min()) / 15.0)
    i = rng.choice(n, p=w / w.sum())
    if poolG is not None and rng.rand() < args.p_swap:
        # large move: replace grid i by the best of a random batch of pool candidates
        ci = rng.choice(len(poolG), min(args.swap_cands, len(poolG)), replace=False)
        rows = (poolG[ci][:, None, :] != G[None]).sum(2).astype(np.int32)
        rows[:, i] = 10**6
        Dc = np.repeat(D[None], len(ci), 0)
        Dc[:, i, :] = rows
        Dc[:, :, i] = rows
        gain = total_scores(Dc) - cur
        Ts = args.swap_temp * T
        k = int(np.argmax(gain + Ts * rng.gumbel(size=len(ci))))
        if gain[k] >= -Ts * rng.exponential() and rows[k].min() > 0:
            G[i], pv[i] = poolG[ci[k]], poolpv[ci[k]]
            D[i, :] = rows[k]
            D[:, i] = rows[k]
            cur += int(gain[k])
            nswap += 1
            if cur > best:
                best, bestG, bestpv = cur, G.copy(), pv.copy()
        continue
    old = G[i, cell]                                   # 196
    new = (old + shift) % 5
    # exact new distance rows for the 196 neighbours
    diff_old = (G[:, cell] != old[None]).T             # 196, n
    diff_new = (G[:, cell] != new[None]).T
    rows = D[i][None] - diff_old + diff_new            # 196, n
    rows[:, i] = 10**6
    Dc = np.repeat(D[None], 196, 0)
    Dc[:, i, :] = rows
    Dc[:, :, i] = rows
    gain = total_scores(Dc) - cur
    # annealed choice: Gumbel-perturbed ranking + Metropolis threshold on the true gain
    order = np.argsort(-(gain + T * rng.gumbel(size=196)))[:args.ncand]
    order = order[gain[order] >= -T * rng.exponential()]
    tried += 1
    if not len(order):
        continue
    cand = np.repeat(G[i][None], len(order), 0)
    cand[np.arange(len(order)), cell[order]] = new[order]
    pc = cnn.predict(cand.reshape(-1, 7, 7))           # m,4,5
    okc = pc.mean(2).min(1) >= args.cnn_target
    if not okc.any():
        nocnn += 1
        continue
    pick = np.where(okc)[0][:args.topk]
    _, pvc = blended(cand[pick].reshape(-1, 7, 7))
    good = np.where(pvc >= args.pmin)[0]
    if not len(good):
        nogbm += 1
        continue
    j = good[0]
    k = order[pick[j]]
    G[i] = cand[pick[j]]
    pv[i] = pvc[j]
    D[i, :] = rows[k]
    D[:, i] = rows[k]
    cur += int(gain[k])
    acc += 1
    if cur > best:
        best, bestG, bestpv = cur, G.copy(), pv.copy()

save(step + 1 if not stop["flag"] else step)
assert int_score(G.reshape(100, 7, 7)) == cur, (int_score(G.reshape(100, 7, 7)), cur)
Gf, cur = bestG.reshape(100, 7, 7), best
pred, pvf = blended(Gf)
tmp = os.path.join(out, "final.tmp.npz")
np.savez(tmp, grids=Gf, pvalid=pvf, pred=pred, score=cur)
os.replace(tmp, os.path.join(out, "final.npz"))
log(f"final score {cur/DENOM:.10f} (int {cur}) pvalid min {pvf.min():.3f} mean {pvf.mean():.3f} expected valid {pvf.sum():.1f}")
if not stop["flag"]:
    open(os.path.join(out, "DONE"), "w").write("ok\n")
