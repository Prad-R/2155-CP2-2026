#!/bin/bash
# Quick end-to-end test of src/islands.py with tiny stages (test tags zt001..zt003, removed afterwards by hand).
cd /home/pradr/2.156/cp_2 || exit 1
A="--t0 0.45 --t1 0.45 --focus 2 --lam 0.25 --dmin 39 --log-every 50000 --ckpt-every 25000"
echo "### two islands, 3 stages x 150k steps (zt001 rule best, zt002 rule own)"
python -u src/islands.py zt001 ckpt/v073/state.bin best 9001 3x150000 $A > /tmp/zt001.$$ 2>&1 &
p1=$!
python -u src/islands.py zt002 ckpt/v069/state.bin own 9002 1x150000 --t0 0.75 --t1 0.75 --focus 5 --lam 0.3 --dmin 39 --lam2 0.1 --dmin2 41 --log-every 50000 --ckpt-every 25000 :: 2x150000 $A > /tmp/zt002.$$ 2>&1 &
p2=$!
echo "### signal test: zt003, TERM after 8 s, then run again"
python -u src/islands.py zt003 ckpt/v073/state.bin best 9003 2x150000 $A > /tmp/zt003.$$ 2>&1 &
p3=$!
sleep 8; kill -TERM $p3; wait $p3; echo "zt003 first run exit code $? ; DONE markers: $(ls ckpt/zt003s*/DONE 2>/dev/null | wc -l)"
python -u src/islands.py zt003 ckpt/v073/state.bin best 9003 2x150000 $A >> /tmp/zt003.$$ 2>&1
echo "zt003 second run exit code $? ; DONE markers: $(ls ckpt/zt003s*/DONE 2>/dev/null | wc -l)"
wait $p1; echo "zt001 exit code $?"; wait $p2; echo "zt002 exit code $?"
for n in 1 2 3; do echo "----- zt00$n"; grep -v "^step" /tmp/zt00$n.$$ | cut -c1-330; echo "(step lines: $(grep -c '^step' /tmp/zt00$n.$$); last: $(grep '^step' /tmp/zt00$n.$$ | tail -n 1 | cut -c1-150))"; rm -f /tmp/zt00$n.$$; done
echo "----- stage dirs:"; ls -d ckpt/zt00* | tr '\n' ' '; echo; ls ckpt/zt00*/DONE | wc -l
