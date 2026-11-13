#!/bin/bash
# Construction probe (Nov 12): 12 searches of 16M moves (~1 h) for a layout + relabelling that makes the 125 code grids valid.
# Run inside a job allocation: WBNAME=<job name> explore/wb.sh -c 12 bash explore/crystal_run.sh
cd /home/pradr/2.156/cp_2 || exit 1
mkdir -p ckpt/crystal
run() { n=$1; shift; bin/crystal --crystal --state ckpt/crystal/c$n.bin --steps 16000000 --t0 2 --t1 0.05 --log-every 500000 --thr 70 17 -187 -1155 --seed $((10#$n)) "$@" >> logs/crystal-c$n.out 2>&1 & }
run 01 --keep 100 --pairing 0; run 02 --keep 100 --pairing 0; run 03 --keep 100 --pairing 0; run 04 --keep 100 --pairing 0
run 05 --keep 100 --pairing 1; run 06 --keep 100 --pairing 1; run 07 --keep 100 --pairing 1; run 08 --keep 100 --pairing 1
run 09 --keep 80 --pairing 0; run 10 --keep 80 --pairing 1; run 11 --keep 60 --pairing 0; run 12 --keep 60 --pairing 1
wait
echo "all probe searches finished at $(date)"
grep -h "^finished" logs/crystal-c*.out | cut -c1-200
python explore/crystal_check.py ckpt/crystal/c[0-9]*.bin
