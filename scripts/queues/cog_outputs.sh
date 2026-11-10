#!/usr/bin/env bash
# after the CogVideoX-2b queue: gate (+ more seeds only if it passes) -> output-level eval
cd /root/meta/physlaw
until grep -q "ALL DONE" /root/envs/cogvideox_queue.log 2>/dev/null; do sleep 30; done
export HF_HOME=/root/hf_cache HF_HUB_OFFLINE=1
P=/root/envs/physics_venv/bin/python
$P scripts/07_generate_outputs.py --model configs/models/cogvideox_2b.yaml --data data/f1 > /root/envs/gen_cogvideox2b.log 2>&1; echo "[co] 07 done $(date +%T): $(tail -1 /root/envs/gen_cogvideox2b.log)"
$P scripts/09_eval_outputs.py --model configs/models/cogvideox_2b.yaml --data data/f1 > /root/envs/outputs_cogvideox2b.log 2>&1; echo "[co] 09 done $(date +%T)"
echo "[co] ALL DONE"
