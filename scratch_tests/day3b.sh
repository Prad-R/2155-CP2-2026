set -e
source /etc/profile.d/modules.sh; module load miniforge/26.7.2-0; conda activate 2.156_cp_2; export PYTHONPATH=src
python -u src/make_probe.py --name S14 --core submissions/S8_20261110_234101.npy --risky ckpt/z1/final.npz --pfile results/xeval_z1.npz --pkey b0 --lowest --n 8 --hi 2 --purpose "targeted labelling: S8 core (92) + the 8 z1 grids the first-generation models trust least (z0 says >= 0.95): which generation is right?"
python -u src/optimize_soft.py --tag dq --init ckpt/z2/final.npz --blend z0 --pmin 0.95 --blend2 b0 --pmin2 0.90 --lam0 5 --lam1 300 --lam-hard 3000 --soft-frac 0.4 --nswap 200 --quick
