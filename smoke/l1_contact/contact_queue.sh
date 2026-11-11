#!/bin/bash
# Contact-pool re-cache + L1 (relayed job, 2026-11-11). One model at a time on the GPU.
cd /workspace/meta/physlaw
export PHYSLAW_MODELS=/workspace/models HF_HUB_OFFLINE=1
PY=/root/envs/physics_venv/bin/python
L=smoke/l1_contact/logs; Q=$L/queue.log
for cfg in vjepa2_vitl videomae_v2_giant untrained_vitl cogvideox_2b; do
  name=$($PY -c "import sys;sys.path.insert(0,'.');from physlaw.config import load_config;print(load_config('configs/models/$cfg.yaml').name)")
  echo "=== $(date +%T) $cfg ($name) stage4" >> $Q
  $PY scripts/04_cache_activations.py --model configs/models/$cfg.yaml --data data/f1 --force > $L/$name.stage4.log 2>&1 \
    || { echo "    $(date +%T) FAIL stage4 $name" >> $Q; continue; }
  if [ $cfg = vjepa2_vitl ] || [ $cfg = videomae_v2_giant ]; then
    echo "    $(date +%T) pred-real $name" >> $Q
    $PY scripts/04_cache_activations.py --model configs/models/$cfg.yaml --data data/f1 --pred-real --force > $L/$name.predreal.log 2>&1 \
      || { echo "    $(date +%T) FAIL pred-real $name" >> $Q; continue; }
  fi
  echo "    $(date +%T) maskfeat $name" >> $Q
  $PY -c "import json,sys;from pathlib import Path;sys.path.insert(0,'.');from physlaw.config import load_config;from physlaw.models.base import token_spec;from physlaw.eval.l0_states import load_index,complete_groups,mask_features;m=load_config('configs/models/$cfg.yaml');c=Path('cache')/m.name/'f1';r=complete_groups(load_index(c));l=[json.loads((Path('data/f1')/x['clip']/'labels.json').read_text()) for x in r];print(sorted(k for k in mask_features(c,Path('data/f1'),r,l,token_spec(m)) if k.startswith('w.')))" > $L/$name.maskfeat.log 2>&1
  grep -q "w.contact.post" $L/$name.maskfeat.log || { echo "    $(date +%T) FAIL maskfeat (no w.contact.post) $name" >> $Q; continue; }
  echo "    $(date +%T) L1 $name" >> $Q
  $PY scripts/06_eval.py --model configs/models/$cfg.yaml --level configs/eval/l1_forecast_laws.yaml --data data/f1 \
      --out smoke/l1_contact/$name/f1 > $L/$name.l1.log 2>&1 \
    && echo "    $(date +%T) DONE $name" >> $Q || echo "    $(date +%T) FAIL L1 $name" >> $Q
done
echo "=== $(date +%T) queue finished" >> $Q
