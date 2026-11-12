#!/bin/bash
# fam_check.sh <famdir> [frames...]: build viz, print twin labels, write a stills strip to the scratchpad
d=$1; shift; S=/tmp/claude-0/-workspace/1e09f4b9-65a0-4d0e-87b2-f3a82065f4e4/scratchpad/stills
cd /workspace/meta/physlaw/smoke/render_check
/root/envs/kubric_venv/bin/python make_videos.py $d || exit 1
/root/envs/kubric_venv/bin/python -c "
import json;d=json.load(open('$d/viz/summary.json'))
for g,v in d.items():
  print(g,v['grid'],v['compare_bytes'],'crf',v['compare_crf'])
  for t in v['twins']: print(' ',{k:t[k] for k in ['twin','legal','violated_law','magnitude','epsilon','event_kind','clean_until_s','event_frame','render_device']})
"
for c in $d/viz/*__compare.mp4; do bash stills.sh $c $S/$(basename $d).png ${@:-60 128 180 230 288}; done
