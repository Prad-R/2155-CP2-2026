#!/bin/bash
# Build a submission file from the best verified chain, as a small batch job (so it can wait on another job).
# Usage: sbatch -J cp2-s20 [--dependency=afterany:<jobid>] jobs/build_best.sbatch S20 "purpose text"   (never pi_faez)
#SBATCH -A mit_general
#SBATCH -p mit_preemptable
#SBATCH -t 0:15:00
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/%x-%j.out

source /etc/profile.d/modules.sh
module load miniforge/26.7.2-0
conda activate 2.156_cp_2
cd /home/pradr/2.156/cp_2 || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=/home/pradr/2.156/cp_2:/home/pradr/2.156/cp_2/src:/home/pradr/2.156/cp_2/explore:$PYTHONPATH
# a requeued job must not build the same name twice
if ls "submissions/${1}_"*.npy >/dev/null 2>&1; then echo "=== $(date) $1 already built: $(ls submissions/${1}_*.npy)"; exit 0; fi
echo "=== $(date) job $SLURM_JOB_ID on $(hostname): building $1"
CP2_INSIDE=1 bash explore/build_best.sh "$@"
