#!/bin/bash
# Last-hour run (Nov 13): hot fixed-core chains started from S22 on mit_quicktest. 32 chains, ~3.5M steps each.
cd /home/pradr/2.156/cp_2 || exit 1
source /etc/profile.d/modules.sh; module load miniforge/26.7.2-0; conda activate 2.156_cp_2
export ANNEAL_BIN=/home/pradr/2.156/cp_2/bin/anneal6
start=cache/core_S24.npy; K=49
COMMON="--init $start --steps 2200000 --dmin 39 --log-every 500000 --ckpt-every 500000 --special $K --boost 1 --frozen $K --focus 5"
i=0
for cfg in "2.0 0.3" "2.0 0.3" "2.0 0.3" "1.75 0.3" "2.25 0.3" "1.5 0.3" "2.0 0.6" "1.6 0.3"; do
  set -- $cfg
  for rep in 1 2 3 4; do
    i=$((i+1)); n=$(printf "%02d" $i)
    python -u src/anneal.py --tag s$n $COMMON --seed $((9700+i)) --t0 $1 --t1 $1 --lam $2 >> logs/core-s$n.out 2>&1 &
  done
done
wait
