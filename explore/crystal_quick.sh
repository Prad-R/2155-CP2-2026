#!/bin/bash
cd /home/pradr/2.156/cp_2 || exit 1
bin/crystal --crystal --state ckpt/crystal/quick.bin --steps 60000 --t0 2 --t1 0.2 --seed 1 --log-every 10000 --keep 100 --pairing 0 --thr 70 17 -187 -1155 2>&1 | cut -c1-200
python explore/crystal_check.py ckpt/crystal/quick.bin
