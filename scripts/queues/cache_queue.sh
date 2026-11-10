#!/usr/bin/env bash
cd /root/meta/physlaw
export HF_HOME=/root/hf_cache HF_HUB_OFFLINE=1
for m in vjepa2_vitl untrained_vitl; do
  /root/envs/physics_venv/bin/python scripts/04_cache_activations.py --model configs/models/$m.yaml --data data/f1 > /root/envs/cache_$m.log 2>&1
  echo "[queue] cache $m exit $? $(date +%T): $(tail -1 /root/envs/cache_$m.log)"
done
echo "[queue] ALL DONE $(date +%T)"
