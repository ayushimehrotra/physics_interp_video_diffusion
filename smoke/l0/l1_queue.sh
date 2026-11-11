#!/bin/bash
# L1 smoke: wait for the L0 GPU queue, then stage 4 pass 2 (forecast) on each 8-group smoke cache + L1.
while pgrep -f l0_gpu_queue.sh > /dev/null; do sleep 60; done
cd /root/wt/smoke
export HF_HUB_OFFLINE=1 PHYSLAW_MODELS=/workspace/models
PY=/root/envs/physics_venv/bin/python
OUT=/workspace/meta/physlaw/results_smoke
for cfg in "$@"; do
  name=$($PY -c "import sys;sys.path.insert(0,'.');from physlaw.config import load_config;print(load_config('configs/models/$cfg.yaml').name)")
  C=/root/smoke_cache/$name/f1; log=/root/smoke_logs/$name.l1.log
  echo "=== $(date +%T) $cfg ($name)" | tee -a /root/smoke_logs/l1_queue.log
  {
    if [ ! -f $C/_maskfeat_v2.npz ]; then
      $PY scripts/06_eval.py --model configs/models/$cfg.yaml --level configs/eval/l0_states.yaml --data data/f1 \
          --cache $C --out /root/smoke_scratch/$name.l0 || exit 1
    fi
    $PY scripts/04_cache_activations.py --model configs/models/$cfg.yaml --data data/f1 --limit 8 --out $C \
    && $PY scripts/06_eval.py --model configs/models/$cfg.yaml --level configs/eval/l1_forecast_laws.yaml --data data/f1 \
          --cache $C --out $OUT/$name/f1
  } > $log 2>&1
  echo "    $(date +%T) exit=$? ($name)" | tee -a /root/smoke_logs/l1_queue.log
done
