#!/bin/bash
# Second design search (Nov 12 evening): 14 searches of 10M moves (~35 min); 12 continue from the best probe design
# (ckpt/crystal/c11.bin, 47 valid) with a fixed extra cost per invalid grid (--cstep), 2 start fresh. Afterwards the
# frozen-core test (explore/core_run.sh) runs from the design with the most valid grids (tags kb01..kb12), or, if no
# design beats c11, 12 more chains on the c11 core with other seeds (tags kc01..kc12).
# Run inside a job allocation: WBNAME=<job name> explore/wb.sh -c 96 bash explore/crystal_run2.sh
cd /home/pradr/2.156/cp_2 || exit 1
BIN=bin/anneal6; STEPS=${STEPS:-10000000}
run() { n=$1; shift; $BIN --crystal --state ckpt/crystal/d$n.bin --steps "$STEPS" --log-every 500000 --thr 70 17 -187 -1155 --seed $((100 + 10#$n)) "$@" >> logs/crystal-d$n.out 2>&1 & }
C="--cinit ckpt/crystal/c11.bin"
run 01 $C --keep 60 --cstep 0.2 --t0 0.15 --t1 0.02
run 02 $C --keep 60 --cstep 0.2 --t0 0.15 --t1 0.02
run 03 $C --keep 60 --cstep 0.2 --t0 0.25 --t1 0.02
run 04 $C --keep 65 --cstep 0.2 --t0 0.15 --t1 0.02
run 05 $C --keep 65 --cstep 0.2 --t0 0.25 --t1 0.02
run 06 $C --keep 70 --cstep 0.2 --t0 0.15 --t1 0.02
run 07 $C --keep 70 --cstep 0.2 --t0 0.25 --t1 0.02
run 08 $C --keep 70 --cstep 0.2 --t0 0.50 --t1 0.02
run 09 $C --keep 80 --cstep 0.2 --t0 0.25 --t1 0.02
run 10 $C --keep 80 --cstep 0.2 --t0 0.50 --t1 0.02
run 11 $C --keep 60 --cstep 0.0 --t0 0.15 --t1 0.02
run 12 $C --keep 70 --cstep 0.0 --t0 0.25 --t1 0.02
run 13 --pairing 0 --keep 60 --cstep 0.2 --t0 2 --t1 0.03
run 14 --pairing 0 --keep 60 --cstep 0.2 --t0 2 --t1 0.03
wait
echo "design searches finished at $(date)"
python explore/crystal_check.py ckpt/crystal/c11.bin ckpt/crystal/d[0-9]*.bin | tee logs/crystal-d-summary.txt
set -- $(awk '{ if ($3 + 0 > m) { m = $3 + 0; f = $1 } } END { sub(":", "", f); print f, m }' logs/crystal-d-summary.txt)
echo "best design: $1 with $2 valid"
if [ "$2" -gt 47 ]; then
    python explore/core_prep.py "$1" cache/core_B.npy && bash explore/core_run.sh cache/core_B.npy kb 5200
else
    bash explore/core_run.sh cache/core_c11.npy kc 5300
fi
