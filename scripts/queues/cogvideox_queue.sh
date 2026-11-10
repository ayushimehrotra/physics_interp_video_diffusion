#!/usr/bin/env bash
# CogVideoX-2b: cache -> pixel baselines -> L0 -> L2 -> L1 (queued by Claude)
cd /root/meta/physlaw
export HF_HOME=/root/hf_cache HF_HUB_OFFLINE=1
P=/root/envs/physics_venv/bin/python
M=configs/models/cogvideox_2b.yaml
set -e
$P scripts/04_cache_activations.py --model $M --data data/f1 > /root/envs/cache_cogvideox2b.log 2>&1; echo "[cq] cache done $(date +%T): $(tail -1 /root/envs/cache_cogvideox2b.log)"
$P scripts/05_surprise.py --model $M --data data/f1 > /root/envs/pixel_cogvideox2b.log 2>&1; echo "[cq] pixel baselines done $(date +%T)"
$P scripts/06_eval.py --model $M --level configs/eval/l0_states.yaml --data data/f1 > /root/envs/l0_cogvideox2b.log 2>&1; echo "[cq] L0 done $(date +%T)"
$P scripts/06_eval.py --model $M --level configs/eval/l2_surprise.yaml --data data/f1 > /root/envs/l2_cogvideox2b.log 2>&1; echo "[cq] L2 done $(date +%T)"
OMP_NUM_THREADS=32 $P scripts/06_eval.py --model $M --level configs/eval/l1_forecast_laws.yaml --data data/f1 > /root/envs/l1_cogvideox2b.log 2>&1; echo "[cq] L1 done $(date +%T)"
echo "[cq] ALL DONE"
