#!/usr/bin/env bash
# Render every benchmark family at master quality (stage 01), then validate each (stage 02).
# Run from the repo root after scripts/00_setup_envs.sh:
#
#   bash scripts/queues/render_all.sh                 # every family, full size (configs/data/f*.yaml n_groups)
#   SMOKE=10 bash scripts/queues/render_all.sh        # 10 groups per family into data/<fam>_smoke (check first)
#   PC1=1 bash scripts/queues/render_all.sh           # also the PC1 training set (data/pc1/<fam>)
#
# Env: NGPU (default: all visible), WORKERS per GPU (default 4; the GPU saturates at ~4),
#      FAMS (space-separated config names, default all eleven), SMOKE, PC1.
# Each GPU runs its share of families one after another (heaviest first). GPUs beyond one per family add a
# second driver (--reverse, same --out) to the heaviest families. Logs: logs/render_<fam>[_rev].log.
# Resumable: rerun the same command; finished groups (_GROUP_COMPLETE) are skipped, failed ones retried.
# Check labels.json render_device = "OPTIX: ..." (a silent CPU fallback otherwise).
# F1's config asks for 400 groups; copy an existing data/f1 here first to keep its rendered groups.
set -u
cd "$(dirname "$0")/../.."
K=${K:-/root/envs/kubric_venv/bin/python}
export KUBRIC_USE_GPU=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
WORKERS=${WORKERS:-4}
SMOKE=${SMOKE:-}
# Heaviest first (groups x twins): F7 / F6 / F9 / F8 / F2 carry 11-21-twin ladders.
FAMS=(${FAMS:-f7_freefall f6_friction f9_pendulum f8_ramp f2_restitution f3_hidden_mass f1_collisions \
              f4_galilean f11_support f12_spin f10_rolling})
mkdir -p logs

render() {  # render <config name> [--reverse]
  local cfg=$1; shift
  local fam; fam=$(grep -m1 '^family:' "configs/data/$cfg.yaml" | awk '{print $2}')
  local extra=() tag=$fam
  [ -n "$SMOKE" ] && extra=(--n-groups "$SMOKE" --out "data/${fam}_smoke") && tag=${fam}_smoke
  [ "${1:-}" = --reverse ] && extra+=(--reverse) && tag=${tag}_rev
  echo "[render_all] $(date +%F\ %T) GPU $CUDA_VISIBLE_DEVICES start $cfg ${extra[*]}"
  $K scripts/01_generate_data.py --config "configs/data/$cfg.yaml" --workers "$WORKERS" "${extra[@]}" \
    > "logs/render_$tag.log" 2>&1
  local rc=$?
  echo "[render_all] $(date +%F\ %T) GPU $CUDA_VISIBLE_DEVICES done $cfg (exit $rc)"
}

validate() {  # validate <config name>
  local fam; fam=$(grep -m1 '^family:' "configs/data/$1.yaml" | awk '{print $2}')
  local d=data/$fam; [ -n "$SMOKE" ] && d=data/${fam}_smoke
  $K scripts/02_validate_data.py --data "$d" > "logs/validate_$(basename "$d").log" 2>&1
  local rc=$?
  echo "[render_all] $(date +%F\ %T) validate $d exit $rc"
}

pids=()
for ((g = 0; g < NGPU; g++)); do
  (
    export CUDA_VISIBLE_DEVICES=$g
    for ((i = g; i < ${#FAMS[@]}; i += NGPU)); do render "${FAMS[$i]}"; validate "${FAMS[$i]}"; done
    # Spare GPU: second driver on the heaviest families, taking groups from the end.
    if ((g >= ${#FAMS[@]})); then render "${FAMS[$(((g - ${#FAMS[@]}) % ${#FAMS[@]}))]}" --reverse; fi
  ) &
  pids+=($!)
done
wait "${pids[@]}"

if [ -n "${PC1:-}" ]; then
  echo "[render_all] $(date +%F\ %T) PC1 training set"
  CUDA_VISIBLE_DEVICES=0 $K scripts/01_generate_data.py --config configs/data/pc1_train.yaml --workers "$WORKERS" \
    ${SMOKE:+--n-groups $SMOKE --out data/pc1_smoke} > logs/render_pc1.log 2>&1
fi
echo "[render_all] $(date +%F\ %T) ALL DONE"
