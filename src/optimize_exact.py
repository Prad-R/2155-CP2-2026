"""Set optimisation under EXACT validity for all four advisors (src/rules.py).  Simulated annealing on the exact
integer 10-NN score; a grid may only move to a state that is valid for every advisor.  Invalid grids in the start
set are repaired first.

  python -u src/optimize_exact.py --tag e1 --init submissions/S8_20261110_234101.npy --steps 400000
"""
import argparse, json, os, signal, time
import numpy as np

from common import ROOT
from rules import raw_scores, NAMES, THRESH
from divscore import hamming_matrix, int_score, DENOM, K

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--init", required=True)
p.add_argument("--steps", type=int, default=400000)
p.add_argument("--t0", type=float, default=1.0)
p.add_argument("--t1", type=float, default=0.05)
p.add_argument("--nswap", type=int, default=200)
p.add_argument("--chunk", type=int, default=24)
p.add_argument("--max-chunks", type=int, default=6)
p.add_argument("--margin", type=int, nargs=4, default=[0, 0, 0, 0], help="extra raw margin for Wellness, Tax, Transportation, Business on top of rules.THRESH")
p.add_argument("--allswaps", action="store_true", help="evaluate all two-cell swaps of the chosen grid instead of --nswap random ones")
p.add_argument("--focus", type=float, default=15.0, help="scale of the grid-selection weights exp(-(contribution - min)/focus)")
p.add_argument("--dmin", type=int, default=38, help="pairs closer than this are penalised when --lam > 0")
p.add_argument("--lam", type=float, default=0.0, help="weight of the pair penalty sum relu(dmin - d) added to the objective")
p.add_argument("--init-key", default="", help="array name inside an .npz init file (default: grids, or bestG for a state file)")
p.add_argument("--ckpt-every", type=int, default=5000)
p.add_argument("--log-every", type=int, default=10000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--quick", action="store_true")
args = p.parse_args()
if args.quick:
    args.steps, args.ckpt_every, args.log_every = 3000, 1000, 1000

out = os.path.join(ROOT, "ckpt", args.tag + ("_quick" if args.quick else ""))
os.makedirs(out, exist_ok=True)
t_start = time.time()
log = lambda *a: print(f"[{time.time()-t_start:7.1f}s]", *a, flush=True)
T3 = THRESH + np.array(args.margin)
SC3 = np.array([78.0, 56.0, 32.0, 160.0])
BIG = 10**6


def deficit(g):
    """normalised violation per grid (0 = valid)"""
    raw, ok = raw_scores(g)
    return (np.maximum(0, T3[None] - raw) / SC3[None]).sum(1) + 50.0 * (~ok)


state_f = os.path.join(out, "state.npz")
if os.path.exists(os.path.join(out, "DONE")):
    log("already done"); raise SystemExit
if os.path.exists(state_f):
    z = np.load(state_f)
    G, step0, best, bestG = z["G"], int(z["step"]), int(z["best"]), z["bestG"]
    log(f"resumed at step {step0}")
else:
    G = np.load(os.path.join(ROOT, args.init))
    if args.init.endswith(".npz"):
        G = G[args.init_key or ("grids" if "grids" in G.files else "bestG")]
    G = G.astype(np.int8).reshape(100, 49)
    step0, best, bestG = 0, -1, G.copy()
rng = np.random.RandomState(args.seed + step0)
n = 100
D = hamming_matrix(G).astype(np.int32)
np.fill_diagonal(D, BIG)
cell = np.repeat(np.arange(49), 4)
shift = np.tile(np.arange(1, 5), 49)
SW1, SW2 = np.triu_indices(49, 1)


def total(Dm):
    return int(np.partition(Dm, K - 1, axis=1)[:, :K].sum())


cur = total(D)


def candidates(i, nswap):
    """all single-cell changes of grid i plus random two-cell swaps -> cand (M,49), rows (M,100), gain (M,)"""
    old = G[i, cell]
    new = (old + shift) % 5
    rows = D[i][None] - (G[:, cell] != old[None]).T + (G[:, cell] != new[None]).T
    cand = np.repeat(G[i][None], 196, 0)
    cand[np.arange(196), cell] = new
    if nswap:
        if args.allswaps:
            okk = G[i, SW1] != G[i, SW2]
            c1, c2 = SW1[okk], SW2[okk]
        else:
            c1, c2 = rng.randint(49, size=3 * nswap), rng.randint(49, size=3 * nswap)
            okk = G[i, c1] != G[i, c2]
            c1, c2 = c1[okk][:nswap], c2[okk][:nswap]
        a1, a2 = G[i, c1], G[i, c2]
        rows2 = (D[i][None] - (G[:, c1] != a1[None]).T - (G[:, c2] != a2[None]).T + (G[:, c1] != a2[None]).T + (G[:, c2] != a1[None]).T)
        cand2 = np.repeat(G[i][None], len(c1), 0)
        cand2[np.arange(len(c1)), c1] = a2
        cand2[np.arange(len(c1)), c2] = a1
        rows, cand = np.concatenate([rows, rows2]), np.concatenate([cand, cand2])
    rows[:, i] = BIG
    Dx = D.copy(); Dx[:, i] = BIG
    part = np.sort(np.partition(Dx, K - 1, axis=1)[:, :K], axis=1)
    S9, t10 = part[:, :K - 1].sum(1), part[:, K - 1]
    oth = S9[None] + np.minimum(rows, t10[None])
    oth[:, i] = 0
    own = np.partition(rows, K - 1, axis=1)[:, :K].sum(1)
    gain = oth.sum(1) + own - cur
    if args.lam > 0:
        pen = np.maximum(0, args.dmin - rows).sum(1) - np.maximum(0, args.dmin - D[i]).sum()
        return cand, rows, gain, gain - args.lam * pen
    return cand, rows, gain, gain


def apply(i, c, row, g):
    global cur
    G[i] = c
    D[i, :] = row
    D[:, i] = row
    cur += int(g)


# ---------- repair phase: make every grid valid ----------
DEF = deficit(G)
log(f"start score {cur/DENOM:.4f}; invalid grids {int((DEF > 0).sum())}; total deficit {DEF.sum():.2f}")
rep = 0
while (DEF > 0).any() and rep < 20000:
    i = int(rng.choice(np.where(DEF > 0)[0]))
    cand, rows, gain, _ = candidates(i, args.nswap)
    dc = deficit(cand)
    u = gain - 2000.0 * (dc - DEF[i]) + 0.5 * rng.gumbel(size=len(cand))
    k = int(np.argmax(u))
    apply(i, cand[k], rows[k], gain[k]); DEF[i] = dc[k]
    rep += 1
    if rep % 200 == 0:
        log(f"repair {rep}: invalid {int((DEF > 0).sum())} deficit {DEF.sum():.2f} score {cur/DENOM:.4f}")
if rep:
    log(f"repair done after {rep} moves: invalid {int((DEF > 0).sum())} score {cur/DENOM:.4f}")
assert not (DEF > 0).any(), "could not repair the start set"
if best < 0:
    best, bestG = cur, G.copy()


def save(step):
    tmp = state_f + ".tmp.npz"
    np.savez(tmp, G=G, step=step, best=best, bestG=bestG)
    os.replace(tmp, state_f)


stop = {"flag": False}
for s in (signal.SIGTERM, signal.SIGUSR1):
    signal.signal(s, lambda *a: stop.update(flag=True))

acc = 0
step = step0 - 1
t_loop = time.time()
for step in range(step0, args.steps):
    if stop["flag"]:
        break
    T = args.t0 * (args.t1 / args.t0) ** (step / args.steps)
    if step % args.log_every == 0 and step > step0:
        Dt = D[np.triu_indices(n, 1)]
        log(f"step {step}: T {T:.3f} score {cur/DENOM:.5f} (int {cur}) best {best} accepted {acc} pairs<38 {int((Dt < 38).sum())} pairs<37 {int((Dt < 37).sum())} "
            f"min {int(Dt.min())} {(step - step0) / (time.time() - t_loop):.0f} steps/s")
    if step % args.ckpt_every == 0 and step > step0:
        save(step)
    contrib = np.partition(D, K - 1, axis=1)[:, :K].sum(1)
    w = np.exp(-(contrib - contrib.min()) / args.focus)
    i = rng.choice(n, p=w / w.sum())
    cand, rows, gain_s, gain = candidates(i, args.nswap)
    order = np.argsort(-(gain + T * rng.gumbel(size=len(cand))))
    thresh = -T * rng.exponential()
    pick = -1
    for c in range(args.max_chunks):
        ids = order[c * args.chunk:(c + 1) * args.chunk]
        if len(ids) == 0 or gain[ids[0]] + 6 * T < thresh:
            break
        raw, ok = raw_scores(cand[ids])
        v = ok & (raw >= T3[None]).all(1)
        for q in np.where(v)[0]:
            k = ids[q]
            if gain[k] < thresh:
                continue
            pick = k
            break
        if pick >= 0:
            break
    if pick < 0:
        continue
    apply(i, cand[pick], rows[pick], gain_s[pick])
    acc += 1
    if cur > best:
        best, bestG = cur, G.copy()

save(step + 1 if not stop["flag"] else step)
assert total(D) == cur == int_score(G.reshape(100, 7, 7))
if not stop["flag"]:
    Gf = bestG.reshape(100, 7, 7)
    raw, ok = raw_scores(Gf)
    assert ok.all() and (raw >= T3[None]).all()
    sc = int_score(Gf)
    tmp = os.path.join(out, "final.tmp.npz")
    np.savez(tmp, grids=Gf, raw=raw, score=sc)
    os.replace(tmp, os.path.join(out, "final.npz"))
    log(f"final score {sc/DENOM:.10f} (int {sc}); min raw margins over rules.THRESH W/T/Tr/B {(raw - THRESH[None]).min(0).tolist()}")
    open(os.path.join(out, "DONE"), "w").write("ok\n")
