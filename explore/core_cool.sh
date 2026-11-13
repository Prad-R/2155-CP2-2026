#!/bin/bash
# Fixed-core chains with a cooling schedule (Nov 12, 21:30): does heat help the free grids settle around the core?
# (The constant-temperature chains of explore/core_run.sh flattened at 37755-37896 within 10M steps.)
# 14 chains, all with the first K grids fixed; T falls from 0.9-1.5 to 0.45-0.6 over <steps> steps.
#   WBNAME=<job name> explore/wb.sh -c 96 bash explore/core_cool.sh <start.npy> <tag prefix> <seed base> <steps>
cd /home/pradr/2.156/cp_2 || exit 1
start=${1:?start set}; pre=${2:?tag prefix}; sb=${3:?seed base}; steps=${4:?steps}
K=$(cat "${start%.npy}.k") || exit 1
export ANNEAL_BIN=/home/pradr/2.156/cp_2/bin/anneal6
COMMON="--init $start --steps $steps --dmin 39 --log-every 2000000 --ckpt-every 2000000 --special $K --frozen $K --focus 5"
run() { n=$1; shift; python -u src/anneal.py --tag "$pre$n" $COMMON --seed $((sb + 10#$n)) "$@" >> "logs/core-$pre$n.out" 2>&1 & }
run 01 --t0 1.2 --t1 0.5 --lam 0.3 --boost 1
run 02 --t0 1.2 --t1 0.5 --lam 0.3 --boost 2
run 03 --t0 1.0 --t1 0.5 --lam 0.3 --boost 1
run 04 --t0 1.0 --t1 0.5 --lam 0.3 --boost 2
run 05 --t0 1.0 --t1 0.45 --lam 0.3 --lam2 0.1 --dmin2 41 --boost 1
run 06 --t0 0.9 --t1 0.5 --lam 0.3 --boost 1
run 07 --t0 0.9 --t1 0.5 --lam 0.3 --boost 2
run 08 --t0 0.9 --t1 0.6 --lam 0.5 --boost 1
run 09 --t0 1.5 --t1 0.5 --lam 0.3 --boost 1
run 10 --t0 1.5 --t1 0.6 --lam 0.5 --boost 1
run 11 --t0 1.0 --t1 0.5 --lam 0.3 --boost 0.5
run 12 --t0 1.0 --t1 0.5 --lam 0.15 --boost 1
run 13 --t0 0.9 --t1 0.9 --lam 0.3 --boost 1
run 14 --t0 1.0 --t1 0.5 --lam 0.3 --boost 1 --nswap 400
wait
echo "cooling chains $pre ended at $(date)"
