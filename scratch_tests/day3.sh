set -e
source /etc/profile.d/modules.sh; module load miniforge/26.7.2-0; conda activate 2.156_cp_2; export PYTHONPATH=src
python -u src/xeval.py z1,z2,z3,z4 b0,z0
python -u src/make_submission.py --name S11 --source ckpt/z2/final.npz --pmin 0 --iters 10 --purpose "z2: z0 models, margin 0.98, swap moves"
python -u src/make_submission.py --name S12 --source ckpt/z1/final.npz --pmin 0 --iters 10 --purpose "z1: z0 models, margin 0.95, swap moves"
python -u src/make_submission.py --name S13 --source ckpt/z3/final.npz --pmin 0 --iters 10 --purpose "z3: soft penalty, 43 grids below margin 0.95 (min 0.873); tests whether z0 margin can be loosened"
python -u src/make_probe.py --name S14 --core submissions/S8_20261110_234101.npy --risky ckpt/z3/final.npz --hi 0.95 --n 10 --purpose "targeted labelling: S8 core (90) + 10 z3 grids with z0 P(valid) below 0.95"
