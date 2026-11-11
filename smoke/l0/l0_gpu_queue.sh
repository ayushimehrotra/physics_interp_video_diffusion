#!/bin/bash
# L0 smoke: stage 4 (8 groups, no forecast) + stage 6 L0 for every model without a cache.
cd /root/wt/smoke
export HF_HUB_OFFLINE=1 PHYSLAW_MODELS=/workspace/models
PY=/root/envs/physics_venv/bin/python
OUT=/workspace/meta/physlaw/results_smoke
for cfg in "$@"; do
  name=$($PY -c "import sys;sys.path.insert(0,'.');from physlaw.config import load_config;print(load_config('configs/models/$cfg.yaml').name)")
  log=/root/smoke_logs/$name.l0.log
  echo "=== $(date +%T) $cfg ($name)" | tee -a /root/smoke_logs/queue.log
  { $PY scripts/04_cache_activations.py --model configs/models/$cfg.yaml --data data/f1 --limit 8 --no-forecast \
        --out /root/smoke_cache/$name/f1 \
    && $PY scripts/06_eval.py --model configs/models/$cfg.yaml --level configs/eval/l0_states.yaml --data data/f1 \
        --cache /root/smoke_cache/$name/f1 --out $OUT/$name/f1 ; } > $log 2>&1
  echo "    $(date +%T) exit=$? ($name)" | tee -a /root/smoke_logs/queue.log
done
