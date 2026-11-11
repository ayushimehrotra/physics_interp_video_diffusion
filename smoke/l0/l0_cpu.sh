#!/bin/bash
cd /root/wt/smoke
export HF_HUB_OFFLINE=1 PHYSLAW_MODELS=/workspace/models
PY=/root/envs/physics_venv/bin/python
OUT=/workspace/meta/physlaw/results_smoke
for cfg in untrained_vitl vjepa2_vitl videomae_v2_giant cogvideox_2b cogvideox15_5b_i2v; do
  name=$($PY -c "import sys;sys.path.insert(0,'.');from physlaw.config import load_config;print(load_config('configs/models/$cfg.yaml').name)")
  echo "=== $(date +%T) $name" >> /root/smoke_logs/cpu_queue.log
  $PY scripts/06_eval.py --model configs/models/$cfg.yaml --level configs/eval/l0_states.yaml --data data/f1 \
      --workers 32 --out $OUT/$name/f1 > /root/smoke_logs/$name.l0.log 2>&1
  echo "    $(date +%T) exit=$? $name" >> /root/smoke_logs/cpu_queue.log
done
