#!/bin/bash
# Queue the follow-up chains (jobs/opt_r3.txt: r801..r896, 300M steps, best penalty settings of cp2-optR2).
# Starts when cp2-optR2 has finished (or at once if it already has).  Usage: jobs/submit_r3.sh
# Cancel with: scancel -n cp2-optR3      (never pi_faez: multi_cpu.sbatch uses mit_general / mit_preemptable)
cd /home/pradr/2.156/cp_2 || exit 1
dep=$(squeue -u "$USER" -h -n cp2-optR2 -o %i | head -1)
# Skip rule: at start (and after any requeue) the job exits without running if some chain already holds a 38000 set.
export CP2_SKIP_IF="explore/reached.sh 38000"
# One chain per physical core: on hyperthreaded nodes 96 "CPUs" are only 48 cores and the chains run at half speed.
sbatch -J cp2-optR3 -c 96 --threads-per-core=1 --mem=24G -t 12:00:00 ${dep:+--dependency=afterany:$dep} jobs/multi_cpu.sbatch jobs/opt_r3.txt
