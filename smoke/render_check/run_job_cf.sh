#!/bin/bash
# collision families at the reduced format: pin master.sensor_width_mm=41.6 (the master's FOV), else
# kubric_scene.configure defaults to 36*W/720 mm and renders 2x zoomed relative to the labels (see MANIFEST.md)
cd /workspace/meta/physlaw
export KUBRIC_USE_GPU=1 OMP_NUM_THREADS=4
t0=$(date +%s)
/root/envs/kubric_venv/bin/python physlaw/data/generate.py --config configs/data/$1.yaml --seed $2 \
  --out smoke/render_check/$3 --set nuisance_frac=0 master.width=416 master.height=240 master.samples_per_pixel=32 \
  master.sensor_width_mm=41.6 > smoke/render_check/logs/$3.log 2>&1
echo "$3 rc=$? $(( $(date +%s) - t0 ))s $(grep '^RESULT' smoke/render_check/logs/$3.log | cut -c1-300)"
