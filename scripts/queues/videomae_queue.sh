#!/usr/bin/env bash
# VideoMAE v2-giant: cache -> pixel baselines -> L0 -> L2 -> L1 (queued by Claude)
cd /root/meta/physlaw
export HF_HOME=/root/hf_cache HF_HUB_OFFLINE=1
P=/root/envs/physics_venv/bin/python
M=configs/models/videomae_v2_giant.yaml
set -e
$P scripts/04_cache_activations.py --model $M --data data/f1 > /root/envs/cache_videomae.log 2>&1; echo "[vq] cache done $(date +%T): $(tail -1 /root/envs/cache_videomae.log)"
$P scripts/05_surprise.py --model $M --data data/f1 > /root/envs/pixel_videomae.log 2>&1; echo "[vq] pixel baselines done $(date +%T)"
$P scripts/06_eval.py --model $M --level configs/eval/l0_states.yaml --data data/f1 > /root/envs/l0_videomae.log 2>&1; echo "[vq] L0 done $(date +%T)"
$P scripts/06_eval.py --model $M --level configs/eval/l2_surprise.yaml --data data/f1 > /root/envs/l2_videomae.log 2>&1; echo "[vq] L2 done $(date +%T)"
OMP_NUM_THREADS=32 $P scripts/06_eval.py --model $M --level configs/eval/l1_forecast_laws.yaml --data data/f1 > /root/envs/l1_videomae.log 2>&1; echo "[vq] L1 done $(date +%T)"
echo "[vq] ALL DONE"
