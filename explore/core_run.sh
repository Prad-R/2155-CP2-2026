#!/bin/bash
# Frozen-core test (Nov 12 evening): 12 chains from a start set of explore/core_prep.py. The first K grids (the valid
# grids of a constructed design, pairwise >= 39 apart) stay fixed (--frozen K); the other grids start as invalid code
# grids, are repaired, then annealed around the core. Chains 11 and 12 leave the core free for comparison.
# The chains run until they are stopped (they checkpoint on SIGTERM, e.g. at the end of the job).
# Run inside a job allocation:
#   WBNAME=<job name> explore/wb.sh -c 96 bash explore/core_run.sh <start.npy> <tag prefix> <seed base> [steps]
cd /home/pradr/2.156/cp_2 || exit 1
start=${1:?start set}; pre=${2:?tag prefix}; sb=${3:?seed base}; steps=${4:-300000000}
K=$(cat "${start%.npy}.k") || exit 1
export ANNEAL_BIN=/home/pradr/2.156/cp_2/bin/anneal6
COMMON="--init $start --steps $steps --dmin 39 --log-every 2000000 --ckpt-every 2000000 --special $K"
run() { n=$1; shift; python -u src/anneal.py --tag "$pre$n" $COMMON --seed $((sb + 10#$n)) "$@" >> "logs/core-$pre$n.out" 2>&1 & }
F="--frozen $K"
run 01 $F --t0 0.35 --t1 0.35 --focus 2 --lam 0.15 --boost 1
run 02 $F --t0 0.45 --t1 0.45 --focus 2 --lam 0.25 --boost 1
run 03 $F --t0 0.45 --t1 0.45 --focus 5 --lam 0.25 --boost 3
run 04 $F --t0 0.55 --t1 0.55 --focus 5 --lam 0.3 --boost 1
run 05 $F --t0 0.55 --t1 0.55 --focus 5 --lam 0.3 --boost 3
run 06 $F --t0 0.65 --t1 0.65 --focus 5 --lam 0.3 --boost 1
run 07 $F --t0 0.65 --t1 0.65 --focus 5 --lam 0.3 --boost 3
run 08 $F --t0 0.75 --t1 0.75 --focus 5 --lam 0.3 --boost 2
run 09 $F --t0 0.75 --t1 0.75 --focus 5 --lam 0.3 --lam2 0.1 --dmin2 41 --boost 1
run 10 $F --t0 0.45 --t1 0.45 --focus 5
run 11 --t0 0.45 --t1 0.45 --focus 2 --lam 0.25 --boost 1
run 12 --t0 0.60 --t1 0.60 --focus 5 --lam 0.3 --boost 3
wait
echo "core chains $pre ended at $(date)"
