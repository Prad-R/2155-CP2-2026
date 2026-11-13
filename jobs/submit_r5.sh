#!/bin/bash
# Queue the overnight fixed-core run (jobs/opt_r5.txt, written by jobs/make_r5.sh: za001..zd096, 240M steps each,
# ~9.5 h) and a small job behind it that builds the next submission if the run produced a better set.
# It waits for cp2-optR4 if that job is still running, so the two never hold cores at the same time.
# Usage: jobs/submit_r5.sh [partition]     (default mit_preemptable; never pi_faez: account mit_general only)
# Cancel with: scancel -n cp2-optR5 ; scancel -n cp2-nextR5
cd /home/pradr/2.156/cp_2 || exit 1
part=${1:-mit_preemptable}
[ "$part" = mit_preemptable ] || [ "$part" = mit_normal ] || { echo "partition must be mit_preemptable or mit_normal"; exit 1; }
if ls -d ckpt/z[a-d][0-9][0-9][0-9]s[0-9]* >/dev/null 2>&1; then echo "ckpt/z* already exists: the run was started before (resubmit by hand if that is intended)"; exit 1; fi
[ "$(wc -l < jobs/opt_r5.txt)" = 96 ] || { echo "jobs/opt_r5.txt does not have 96 lines: run jobs/make_r5.sh first"; exit 1; }
unset CP2_SKIP_IF
dep=$(squeue -u "$USER" -h -n cp2-optR4 -o %i | head -1)
# One chain per physical core (see jobs/submit_r4.sh): --threads-per-core=1.
out=$(sbatch -p "$part" -J cp2-optR5 -c 96 --threads-per-core=1 --mem=32G -t 12:00:00 ${dep:+--dependency=afterany:$dep} jobs/multi_cpu.sbatch jobs/opt_r5.txt) || { echo "$out"; exit 1; }
echo "$out"
jid=${out##* }
sbatch -p "$part" -J cp2-nextR5 --dependency=afterany:"$jid" jobs/build_best.sbatch next "best set of cp2-optR5 (fixed-core chains)"
