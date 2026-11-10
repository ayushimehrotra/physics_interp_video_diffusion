#!/usr/bin/env bash
# Queued by Claude: runs when both F1 render drivers have exited.
cd /root/meta/physlaw
until ! pgrep -f "[0]1_generate_data.py" > /dev/null; do sleep 20; done
echo "[queue] render drivers done $(date +%T); complete groups: $(ls data/f1/*/_GROUP_COMPLETE | wc -l)"
/root/envs/kubric_venv/bin/python scripts/02_validate_data.py --data data/f1 > /root/envs/f1_subset_validate.log 2>&1
echo "[queue] validation exit $? $(date +%T): $(grep -E 'passed|failed' data/f1/_validation/pytest.txt | tail -1)"
export HF_HOME=/root/hf_cache HF_HUB_OFFLINE=1
for m in vjepa2_vitl untrained_vitl; do
  /root/envs/physics_venv/bin/python scripts/04_cache_activations.py --model configs/models/$m.yaml --data data/f1 > /root/envs/cache_$m.log 2>&1
  echo "[queue] cache $m exit $? $(date +%T): $(tail -1 /root/envs/cache_$m.log)"
done
echo "[queue] ALL DONE $(date +%T)"
