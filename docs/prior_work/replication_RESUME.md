# Replication with larger datasets + Adam probes — RESUME (written 2026-11-02 12:35 UTC, before pod stop)

Goal (user, 2026-11-02): replicate the relational-physics results on the larger datasets with
Adam-trained linear probes alongside ridge, for: direct vs composed, latent parameter via the
law, one factor at a time, transfer, surprise. Causal intervention and subspace experiments were
explicitly dropped by the user ("not done correctly"). Steering is a separate line (see
`outputs/steering_checkpoint/RESUME.md`).

## Layout
- `replication/code/` — everything (was `/root/fast/repl`). `lib.py` = probe library (ridge via
  `_fast_cv.DualRidge`, Adam = batched AdamW, nested grouped CV, nested cell selection,
  GPU via `ADAM_DEVICE`). `cache.py` reads one capture set sequentially into `/dev/shm/fast/cache`.
  `expN_*.py` = experiments. `render/` = the render scheduler + capture watcher used today.
  `surprise/` = VoE scripts + results (momentum_*.json, ladder_*.json, ladder_motion_control.json).
- `replication/results/` — all result JSONs (`expN_<capture>_<probe>.json`).
- `replication/logs/` — run logs (the printed tables are the quickest read).

## Results so far (all on held-out, seed/group-grouped CV; Adam and ridge agree everywhere)
1. **Direct vs composed** (exp1; cons_set_1000 + ke_factor_set, V-JEPA2-L + VideoMAEv2-giant):
   momentum never stored — composed beats direct by 0.06–0.27 in every cell; KE ~tie on the KE set
   (±0.01), composed +0.03–0.07 on the conservation set.
2. **Direction test** (exp5, non-vacuous linear-space version): KE / momentum gain ≈0 from
   activations orthogonal to mass & speed directions — same as made-up pseudo-quantities;
   positive controls (v_Ax +0.63–0.75, restitution +0.33–0.36) prove sensitivity.
3. **One factor at a time** (exp3, ke_factor_set): S->M / M->S transfer slopes 0.03–0.19 (Adam),
   ≤0 (ridge); both-routes 0.96–1.00. Caveat: within a family KE, p and the moving factor are the
   same variable, so this alone cannot prove "no KE axis" — exp5 does.
4. **Transfer** (exp4): mass/velocity/momentum/KE transfer across elasticity, material and
   dataset at ~in-domain levels; only failure = extrapolating from the narrow same-material
   subset. Adam does not transfer better than ridge.
5. **Latent mass** (exp2 / exp2b, NEW latent_mass_set_v1, 800 clips, fresh seeds): the mass ratio
   is recovered exactly as well as the model's own decoded deflections allow (kinematics-only
   0.86–0.88 giant, 0.75–0.78 V-JEPA2 ≥ direct 0.85 / 0.59–0.62); residual gain +0.002 (ridge),
   ≤0 (Adam), same as the in-span control; radius/restitution controls gain +0.17–0.96. The
   naive-quotient comparison in exp2 made the giant look like direct > structured — that was
   quotient noise amplification, not mass information.
6. **Surprise** (V-JEPA2-L and -g predictors): momentum null holds at 497 pairs (vitg reversed);
   the energy "dissipation prior" (§46f) does NOT survive the full ladder — surprise rises
   monotonically with e (minimum at e=0.2), no jump at e=1, and a copy-last-frame baseline
   reproduces every impossible-vs-legal asymmetry.

## Pending, in order
1. **Visible-mass positive control** for exp2: captures exist (`outputs/visible_mass_set_v1_*`,
   200 clips, same seeds as latent scenes 0–199). Cache them, run `exp2_latent.py` and
   `exp2b_latent_resid.py` on them (direct q should read high there, from appearance).
2. **Friction transfer at 300/300** (redo §35c): captures exist (`outputs/slide_set_v2_*`,
   `outputs/incline_set_v2_*`, 300 each, both encoders). Script NOT yet written: fit mu on one
   set, test on the other, both directions, ridge + Adam, per-domain standardisation, `size`
   control, and score the transferred probe against the target set's acceleration (§35c design).
   Also add leave-environment-out / pooled training. Note seeds 0–299 = the deleted originals'
   seeds (first 150/300 reproduce them).
3. Steering — see `outputs/steering_checkpoint/RESUME.md`.
4. Write the results into `docs/physics/*.md` + CLAUDE.md (not done; nothing here is committed).

## How to restart (fresh pod: /root and /dev/shm are wiped)
```bash
# 1. git dir + envs (project convention)
tar -xf /workspace/envs/git-dirs.tar -C /root          # restores /root/git-dirs
bash /workspace/envs/rebuild_envs.sh                     # physics_venv + kubric_venv
# 2. code back to its expected path (scripts hard-code /root/fast/repl)
mkdir -p /root/fast && cp -r /workspace/meta/physics_interp_video_diffusion/replication/code /root/fast/repl
mkdir -p /root/fast/results /root/fast/logs
# 3. HF cache: the snapshot files under /workspace/hf_cache are 0-byte stubs since the
#    2026-11-02 remount (symlinks lost); rebuild a working cache in RAM from the blobs
/root/envs/physics_venv/bin/python /root/fast/repl/rebuild_hf.py
S=$(ls -d /dev/shm/hfhub/hub/models--OpenGVLab--VideoMAEv2-giant/snapshots/*/)
mv $S/video_preprocessor_config.json $S/preprocessor_config.json   # known misclassification
export HF_HOME=/dev/shm/hfhub HF_HUB_OFFLINE=1
# 4. cache captures into RAM, ONE AT A TIME (sequential reads; ~3–5 min each)
cd /root/fast/repl && for n in visible_mass_set_v1_vjepa2-vitl visible_mass_set_v1_videomaev2-giant \
    slide_set_v2_vjepa2-vitl slide_set_v2_videomaev2-giant incline_set_v2_vjepa2-vitl incline_set_v2_videomaev2-giant; do
  /root/envs/physics_venv/bin/python cache.py $n; done
# 5. run with capped threads; Adam on a GPU
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_THREADS=2
CUDA_VISIBLE_DEVICES=1 ADAM_DEVICE=cuda:0 /root/envs/physics_venv/bin/python exp2_latent.py visible_mass_set_v1_vjepa2-vitl --probes adam
```

## Traps hit today (do not rediscover)
- **geesefs** (`/workspace`) died once under heavy concurrent I/O (bulk `cp -r` + renders +
  agents). Never bulk-copy through it in parallel. Per-file open latency is ~0.4 s: capturing
  straight off the mount ran at 13.6 s/clip; prefetching the set into `/dev/shm` with 8 threads
  then capturing ran at 0.8 s/clip (`render/capture_watcher.py: prefetch`).
- **HF snapshot stubs** after the remount (above) — symptom: "config file ... is not a valid JSON file".
- **/root has ~8 GB free** — keep large scratch in `/dev/shm` (57 GB).
- **BLAS oversubscription**: uncapped numpy under render load made ridge 12× slower; cap threads.
- **`pkill -f <text>` / `pgrep -f <text>` match your own shell's command line** and killed the
  shell twice. Kill by PID or anchor the pattern to the binary path
  (`^/root/envs/physics_venv/bin/python exp`).
- The `exp2_latent.py` "label check" line prints a wrong R² (−1599) from a broadcast bug in that
  print only; the identity itself holds to 6e-16 (verified separately).
