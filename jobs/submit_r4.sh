#!/bin/bash
# Queue the island run (jobs/opt_r4.txt, written by jobs/make_r4.sh: y001..y096, 300M steps each, ~10.5 h) and a
# small job behind it that builds the next submission if the run produced a better set.
# Usage: jobs/submit_r4.sh [partition]     (default mit_preemptable; never pi_faez: account mit_general only)
# Cancel with: scancel -n cp2-optR4 ; scancel -n cp2-next
cd /home/pradr/2.156/cp_2 || exit 1
part=${1:-mit_preemptable}
[ "$part" = mit_preemptable ] || [ "$part" = mit_normal ] || { echo "partition must be mit_preemptable or mit_normal"; exit 1; }
if ls -d ckpt/y[0-9]* >/dev/null 2>&1; then echo "ckpt/y* already exists: the run was started before (resubmit by hand if that is intended)"; exit 1; fi
unset CP2_SKIP_IF
dep=$(squeue -u "$USER" -h -n cp2-optR3 -o %i | head -1)
# One chain per physical core: on nodes that count hardware threads as CPUs (node2004 type) a plain -c 96 is only
# 48 cores and the chains run at half speed; --threads-per-core=1 makes Slurm hand out whole cores there.
out=$(sbatch -p "$part" -J cp2-optR4 -c 96 --threads-per-core=1 --mem=32G -t 12:00:00 ${dep:+--dependency=afterany:$dep} jobs/multi_cpu.sbatch jobs/opt_r4.txt) || { echo "$out"; exit 1; }
echo "$out"
jid=${out##* }
sbatch -p "$part" -J cp2-next --dependency=afterany:"$jid" jobs/build_best.sbatch next "best set of cp2-optR4 (island chains)"
