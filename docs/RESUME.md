# PhysLaw: state, results and plan

Benchmark of physical laws inside video models. Design reference: `docs/SPEC.md` (§5.0, the v3 metrics of
2026-11-11, is authoritative). **This is the only status doc.** Related work: `papers/` (117 PDFs) indexed in
`papers/INDEX.md` (physics in video models, sections A–D; benchmarks of model internals, last section), plus
`docs/references/`; paper draft `docs/physlaw.tex` (Related Work and Methods) with `docs/references.bib` (111 entries). Metric audit page: https://claude.ai/artifact/PwUENMP5r9r5fixh3YfhBY (pre-v3; §5.0 supersedes it).
One branch: `physlaw-v2-merged`. Last updated 2026-11-12.

---

## 0. Resume on a new pod (do this first)

```bash
cd /workspace/meta/physlaw                       # repo, data, cache, results and smoke outputs all live on /workspace
KUBRIC_SRC=/workspace/kubric bash scripts/00_setup_envs.sh     # venvs in /root/envs (~12 min; /root is not persistent)
export PHYSLAW_MODELS=/workspace/models HF_HUB_OFFLINE=1
OMP_NUM_THREADS=8 /root/envs/physics_venv/bin/python -m pytest -q -p no:cacheprovider tests    # ~20-30 min
```

Known state:
1. **No cache matches the v3 code.** Every model cache must be rebuilt with stage 4 `--force` (per-frame pools
   `pooled_frame_t`, diffusion `fc_pooled_frame_t`, the contact pool) before L0–L4 (§5 below).
2. **Nothing but F1 is rendered at master quality.** F2–F12 exist only as `--no-render` label smoke sets
   (`smoke/newlaws/`) and a few CPU 320×184 test groups.
3. Code wired but never run on rendered data: every L0–L4 path and output readout of F6 glide / slide / pair, F9, F11 and
   F12 is tested on synthetic labels and synthetic clips only. Validate the readouts on real clips (stage 08 `--real`)
   before scoring generated videos; F11 alpha_ratio from tracks is ±15% on synthetic renders (tip_consistency is the
   robust statistic). L1 windows need ≥ 3 token-frames: at DiT token rates (~2 token-frames/s) many F9 / F11 windows
   come back unscored (`n_clips.windows_*`).
4. Stage 4 pools roi_A / roi_B from the first two NON-aux objects (`tokens/project.roi_object_ids`); caches built
   before 2026-11-11 pooled the F9 rod / F11 prop as roi_B: rebuild (item 1).
5. Cosmos-Predict2.5 weights are gated: accept the licence on HF, `hf auth login`, download to `/workspace/models`.
6. Keep uv caches off /workspace (`UV_CACHE_DIR=/root/.cache/uv`); geesefs refuses chmod / symlinks / `sed -i`.

## 1. What is built

| Part | State |
|---|---|
| **Families** | F1 collisions: 64 groups rendered, frozen (`data/f1`). Kubric two-sphere scene (`generate.py`): F2 restitution ladder 0.0–1.5, F3 hidden mass, F4 Galilean boosts. Closed form (`physlaw/data/families/`, PyBullet cross-check where Bullet is a valid reference): F6 friction (v2, 2026-11-12: `slide` legal / reverse / viscous / decel_x0.7…x1.3 and `pair` legal / heavy_x13 on one uniform textured floor, block already sliding at frame 0, invisible branch mid-slide; glide `legal`/`speed_x0.7…x1.3`/`turn_20`), F7 projectile / drop / bounce (21-rung ε ladder), F8 Galileo's ramp (21-rung ε ladder), F9 pendulum (`pair`: legal, period_x0.8, period_x1.25, heavy_x0.8; `swing`: 21-rung damper ladder, apex states in `edit.apex_states`), F10 rolling (κ per group), F11 support (overhang grid 0.55…1.45; twins stay / tip; prop at 3g as the clip's g reference), F12 spin (legal e 0.9, legal_e04, spin_x0.7…x1.3, KE-conserving spin_ke_x0.8 / x1.2). Configs `configs/data/f*.yaml`; `--no-render` acceptance passes for every family (`smoke/newlaws/accept_*`). |
| **Laws** | `physlaw/eval/laws/` (SPEC §5.0.1): momentum (+ Newton I inertia_speed / inertia_turn on F6 glide), energy (generalised restitution ε for collision / bounce / pad / swing; ρ\* = ε²), gravity, friction (decel_const, stop_hold, mass_ratio), oscillation (period_ratio, mass_period_ratio), rolling (order, slip), statics (tip_consistency from the rotation angle; alpha_ratio with g measured in the clip), angular_momentum (L_ratio), galilean. Every statistic is length- and time-scale free (tests/test_laws.py); `ratio_score` for floor-relative scoring. `physlaw/data/io.law_inputs(labels)` feeds every law from any labels file and drops `aux` objects (F9 rods, F11 prop). |
| **Models** | Adapters for V-JEPA 2 (L, g, untrained), VideoMAE-v2 (base, giant), CogVideoX (2b, 5b, 1.5-5B-I2V), Wan (2.1 T2V 1.3B/14B, 2.1 I2V 14B, 2.2 TI2V 5B, 2.2 I2V A14B), PhysRVG, VideoREPA-5B, Cosmos-Predict2.5-2B, Wan2.1-VACE-1.3B / 14B, LTX-Video 0.9.8 13B distilled, positive controls PC1 / PC2. Weights for all but Cosmos in `/workspace/models`. Output tracks per config (`output_track`, `track_a_mode`, `physlaw/models/track_a.py`). |
| **Levels** | All nine laws are wired into every level (SPEC §5.0.4): L0 (`l0_states.py`: targets incl. F9 bob angle / ω, F11 block a_cm / ω, F12 bar ω; mask floor, untrained, one-twin refit, selectivity), L1 (`l1_forecast_laws.py`: S_mom; energy creation / tails / median ε̂ incl. F8 and F9 swing; ratio scores for gravity, friction, oscillation, rolling, statics, angular_momentum, glide inertia; P-pred-real / as_is instrument), L2 (`l2_surprise.py`: contrasts with speed-matched violators and the pixel-baseline motion control; ε kinks on F2 / F7 bounce / F8 / F9 swing; F11 across-group kink on overhang x; L2-rep), L2b (`l2b_detection.py`: momentum, gravity, friction, oscillation, angular_momentum with the kinematic baseline), L3 (`l3_invariance.py`: Galilean, mass independence for F7 drop / F6 pair / F9 pair, transfer clean → nuisance), L4 (`l4_causal.py`: one manifold-steering routine per law with precondition, swap test, patch arm, steering reliability, controls; plus **localization** (`localize`, `localize:` config block, json `localization`; gated on the model passing steering / swap / the L2 kink): swap IIA per layer (DiTs per SNR × layer, live patch per block × SNR) vs random / shuffled-target / size / passive / untrained controls, token groups, heads and MLPs by attribution pre-screen + exact patching + mean-ablation knock-out, DAS subspace vs k, agreement rule; tested on tiny planted / no-law stand-ins only, never run on a real model), outputs (`outputs.py`: scale-free readouts for every family incl. F12 bar-angle tracking), leaderboard (`analysis/aggregate.py`, `figures.py`, stage 11). |
| **Shared instrument** | `physlaw/probes/ridge.py`: one calibrated ridge instrument (R1) per object and quantity, selectivity control tasks (scene and object controls), held-out selection (`probes/select.py`). |
| **Positive controls** | PC1 (ViT-S V-JEPA-style, legal clips only) and PC2 (Wan2.2-TI2V-5B LoRA, RigidBench recipe): built and code-path tested, not trained. |

## 2. Environment and pipeline

* `/workspace` is geesefs: no `sed -i`, symlinks, chmod, rsync. Use Python or the Edit tool for file edits.
* Venvs: `/root/envs/kubric_venv` (py3.11, bpy, pybullet, no torch: `physlaw/data/*`),
  `/root/envs/physics_venv` (py3.12, torch 2.8 cu128: models, eval). Keep `physlaw/data/*` torch-free.
* Rendering: `KUBRIC_USE_GPU=1`, check `render_device` = `OPTIX: ...` in labels; ~5 frames/s aggregate per H100 at
  4–6 workers, so a 10-group family with ~5 twins takes ~30–45 min per GPU (ε-ladder families have 21 twins: ~4×).
  The 2026-11-11/12 render checks (`smoke/render_check/`, `smoke/render_check_full/`) ran on an **A100 80GB PCIe**
  (`OPTIX: NVIDIA A100 80GB PCIe` in every label). Their MANIFESTs record no aggregate throughput; the labels'
  `render_seconds` give a median of ~58 s per twin at 416×240 / 32 spp and ~170 s per twin at 832×480 / 64 spp
  (one render process each, 289 frames; twins after the first render only their post-event frames).
* Never `pkill -f` a pattern that appears in your own command line.

```bash
K=/root/envs/kubric_venv/bin/python; P=/root/envs/physics_venv/bin/python
KUBRIC_USE_GPU=1 $K scripts/01_generate_data.py --config configs/data/f7_freefall.yaml --n-groups 10 --workers 6
$K scripts/02_validate_data.py --data data/f7
$P scripts/04_cache_activations.py --model configs/models/vjepa2_vitl.yaml --data data/f7 --force [--pred-real] [--limit N]
$P scripts/05_surprise.py ... --force                             # pixel baselines for L2
$P scripts/06_eval.py --model <cfg> --level configs/eval/<l0_states|l1_forecast_laws|l2_surprise|l2b_detection|l3_invariance|l4_causal>.yaml --data data/<fam> [--out smoke/...]
$P scripts/07_generate_outputs.py / 08_track_outputs.py / 09_eval_outputs.py    # output tracks A / B
$P scripts/11_aggregate.py                                        # results/leaderboard.md
```
Order per model: L0 (builds `_maskfeat_v2.npz`), L2, L2b, L1, then L3 / L4. Smoke runs: `--limit 8` in stage 4,
`--out smoke/<level>/<model>/<family>` in stage 6.

## 3. Decisions

* F1 = 64 groups, frozen. One simulation per group at e = 1; later twins re-keyframed, so pre-contact frames are
  byte-identical. Momentum violation = common drift, KE-reducing sign; energy-creating `mom_violate_e015` draws
  rejected. Master format 48 fps × 289 frames, 832×480, event frames 112–128.
* Data deviations (2026-11-10): F1 `nuisance_frac` 0 (others 0.2); F4 widens the camera, not the floor; F7 projectiles
  enter from off-screen and get a `legal_e04` control twin; F6 pairs use a visibly heavier metal cube; F10 tube
  κ = (1 + i²)/2 per group.
* Smoke models: V-JEPA 2, CogVideoX, Cosmos-Predict2.5 (user, 2026-11-11).
* **2026-11-11, v3 metrics** (SPEC §5.0; supersedes every earlier metric):
  * **Nine laws, scale-free statistics.** Every statistic is a ratio of one kinematic quantity measured twice in the
    same clip by the same instrument, so it is unchanged by a length scale, a time scale and (with documented
    exceptions) a common probe gain; no metric g, fps or speed enters. The two remaining violations were fixed:
    statics now decides "tipped" from the block's rotation angle (radians) and scores α / α_legal with g measured in
    the clip (the block's free fall after separation, else the prop's 3g descent / 3; none → only tip_consistency);
    the F9 swing ε uses apex heights only (v = 0 there, so no g term).
  * **Energy = generalised restitution ε on ladders.** ε ≤ 1 legal, the boundary interior; F7 bounce, F8 and F9 swing
    ladders have 21 rungs (eps0.5, eps0.55, …, eps1.5; also the samplers' default), F2 0.0–1.5. Clean windows and
    stop-dose exclusion. Primary energy evidence is the L2 kink (and L2-rep), not L1 or L2b (R5).
  * **F8 = Galileo's ramp** (descent, pad, climb to ε²h); the old time-rescaled / path-dependence twins are removed
    (each was a playback-speed change).
  * **Control cuts** (user): no-event observer rows, L0 kinematic observers, permutation nulls, L1 pre-contact
    references, the L2 β motion adjustment (replaced by the model − pixel-baseline win rate and speed-matched
    violators), runtime kink comparison observers (now unit tests), L4 random-subspace / PC-matched controls.
  * **P-pred-real** is the L1 instrument wherever the cache has it (the model's own forecast-space outputs for masked
    real post frames); as_is (one probe on pre + post rows) otherwise; moment matching is a diagnostic only;
    P-forecast never scores a law. S_mom stays the momentum score (S_outcome and r_p dropped).
  * **L4 manifold steering**: one routine per law (precondition, steer along the fitted θ manifold, swap test =
    interchange-intervention accuracy with non-target selectivity, patch arm, steering reliability = per-scene slopes,
    wrong-way share, off-manifold distance, steering vs showing the same Δθ through the input).
  * **Selectivity** (Hewitt & Liang): every probe target also fitted on scene- and object-control tasks.
  * **New families** F9 (pendulum, closed form), F11 (support / torque balance), F12 (angular momentum); F6 gains
    glide (Newton I) and slides `viscous`; F6 belongs to law friction (not gravity); F10 κ random per group and the
    rotation readout via the wedge-marker angle tracker (reused for the F12 bar).
* **2026-11-11, methodology fixes** (found while writing the Methods draft, now `docs/physlaw.tex`; caches and label sets must be rebuilt):
  R1 instrument = INVERSE calibration (ŷ ≈ c + d·y, decode (ŷ − c)/d; forward y ~ ŷ left a ρ² shrinkage), also used
  by L3 mass independence (one map for both objects); L1 / outputs gravity windows = free-flight, in-frame frames only
  (after the release, before clean_until); gravity null = Aristotle's fall (a′ = 0, v_h′ = 0: axis_ratio scorable);
  statics tip window = min(2√(a/g_clip), time to leave the ledge) in clip time; L2 / stage 5 `obj` region = one
  region per group (union of the twins' object regions over the clean window); F7 in L2 / L2b, F7 heights in L0;
  violated_law labels `gravity` (F7) / `friction` (F6 slide / pair); one sensor-width default (41.6 mm); clean_until_s
  for every sub-family; L4 swap test scores invariance for zero-prediction laws and encoder energy steering is
  injected where the curve is fitted (predictor post tokens).
* **2026-11-11, scene framing** (render check `smoke/render_check/`, full-res re-check `smoke/render_check_full/`):
  F9 and F7 bounce cameras fitted per group to the whole motion (largest rung's apex) with larger bobs / balls
  (F9 bob >= 70 px, bounce ball >= 50 px at 832); F4 camera 30 -> 13 m (spheres >= 46 px); F10 camera without yaw
  (200 mm at 80 m); F12 puck comes in 20-40 deg off the face normal (slides off past the bar end: no second contact
  in the clip, bar in frame to the end); top-view sun at 45 deg elevation (no ice hotspot). Earlier renders of these
  families are stale.
* **2026-11-12, F4 camera boosts and F8 framing** (full-res check `smoke/render_check_full/{f4_camboost,f8_ramp_v2}`):
  F4 gains camera-boost twins `cam_usmall_e1/_e015`, `cam_ularge_e1/_e015` (unboosted collision, camera translating
  at -u: the same ball image motion as the ball-boost twin, the floor slides; F4 floor now has a 1 m checker, every
  twin). Labels `boost_kind` ball | camera; camera twins `frame: camera` = primary states in the camera's rest frame
  (what readouts / probes / levels use), world states under `*_world` (`io.world_view`, `io.law_inputs(frame=
  "world")`), `camera_trajectory`. L3 galilean: `cam_<level>` levels vs u0 plus `camera_vs_ball` pairs. Track B does
  not run F4 (moving background). F8: every length scales with h, camera fitted to release .. eps = 1.5 apex, block
  >= 70 px (was 32); fillets 4 x the block's half-diagonal, the block in exact two-corner contact (never enters the
  ramp; its rotation in the energy), ramps one swept mesh each (`Static.profile`); Bullet descent check back to 1-2 mm
  (size-independent 0.05 m tolerance). Earlier F4 / F8 renders are stale (F4: 5 boosts x 2 e = 10 twins per group).
* **2026-11-12, F6 v2** (SPEC §2.3 F6, §5.0): the flat / incline / band-crossing slides are deleted (the band's
  look did the work, friction switching on at a line is unphysical, legal-only twins). One uniform concrete-like floor
  (`generate.surface_texture`, a static named `slab` carries μ in the PyBullet check: ~2 mm agreement), oblique camera
  fitted to every twin's path (block >= 70 px), block already decelerating at frame 0, invisible branch mid-slide,
  legal stop >= 1.4 s before the clip end. `slide`: legal, reverse, viscous, (no_friction, dropped later), decel_x0.7/0.85/1.15/1.3;
  `pair`: legal, heavy_x13. Friction law: decel_const (moving / pre), new stop_hold (mean speed over the stop window /
  speed at the branch; legal 0, Stokes null e^-1), mass_ratio; windows `io.friction_windows`. L0 adds the F6
  deceleration target; L1 friction kind = pre / moving / stop windows; L2 friction primary = reverse vs legal (motion
  control decel_x0.85 + x1.15; F6 has no legal-vs-legal control); L3 mass independence on pairs, the F6 transfer is
  gone (transfer stays for F1 / F7 clean → nuisance); outputs readout and leaderboard row `friction_stop`. L4
  friction must move to `subfamily: pair`, `steer_twin: legal`, no `invariance_ref` (config block not yet applied in
  `configs/eval/l4_causal.yaml`). Master-quality render check: `smoke/render_check_full/f6_v3/` (below).
* **2026-11-12, audit fixes** (user's metric audit: forecast-based Galilean test, mask-only checks, twins that break two laws, and the code check list; user decisions):
  F6 `no_friction` dropped; slide legal stop 2.1–2.4 s before the clip end (reverse ends >= ~150 px, median ~205, from
  the legal rest point), pair branch in frames 112–120 and stop 0.5–0.6 s before the end (heavy_x13 ~35–50 px short);
  re-rendered `smoke/render_check_full/f6_v3/` (f6_v2 deleted). F7 legal-vs-legal L2 control dropped (identical inside
  the clean window). F12 gains KE-conserving `spin_ke_x0.8 / x1.2` (bar spin x k paid by the radial relative
  translation: p, orbital L and KE = legal; primary L2 contrast); spin_x > 1 also create energy (flagged). L2: halves
  of speed-matched pairs (`[half <twin>]`), energy-creating share per violator, one leave-one-group-out x̂ for the
  speed-match exclusion and K. L2b verdict requires the ROI mask floor. L3 Galilean law test on the forecast
  (`galilean.forecast`; the real-video d is the floor); L3 mass independence must beat the mask-only score.
* **2026-11-12, audit fixes L0 / L1 / leaderboard** (user's metric audit: pre-event mass/inertia gate, event-extrapolated velocities, negative-control forecasts, mask-only checks, filter pass shares, non-positive scores, same-instrument S_mom, sanity-only mass-free checks): L1 F3 has no
  S_mom (one forecast per set serves all five density-only mass ratios; json `momentum.status = not scored`), F3 energy
  still scored. F7 bounce and F8 pad are per-token-frame kinds `bounce` / `pad` (no window averages): bounce = velocity
  lines over the free frames on each side extrapolated to the contact (post up to the 2nd contact; forecast to the
  group's earliest); pad = √(h′/h) from decoded apex heights like the F9 swing (forecast to the group's latest apex).
  H4 negative controls in every L1 package (`ctl_static` last context frame repeated / pre pool, `ctl_other_forecast`,
  `ctl_other_real`; same instrument, same statistic) and a `verdict` (claim only if the forecast beats all three by a
  paired group-bootstrap CI, and the mask-only floor for roi). M1 `pass_share` + `S_sensitivity` /
  `creation_rate_sensitivity` (gated-out clips = failures). M2: S clipped to [−1, 1]; undefined / non-positive log T
  where the truth is defined = −1 (`laws.ratio_score(floor=)`). M4: S_mom from decoded pre AND post (`S_true_pre`
  diagnostic); rolling / F12 one (v, ω) instrument; statics g_clip from the decoded fall when available, φ over exactly
  the tip window. Lower tail = ε̂ < (per set, lowest floor-decoded legal ε) − τ. N1: antiparallel / normal = sanity.
  L0 `mass_pre_event` (+ F10 `inertia_factor` target); leaderboard `l0.mass_pre` / `l0.kappa_pre`, L1 `*_mass` rows,
  "uninformative" flags on mass-independence / rolling scores, `✗claim`. R² of multi-output probes = mean per-output R²
  (`ridge.r2_score`, `grid._r2_cols`). Gravity g_ratio cannot beat controls (ii) / (iii) by construction (one g).
* **2026-11-12, L4 gaps** (`l4_causal.py`): collision patch has a verdict (`patch_verdict`: B's change along n closer
  to the momentum-required change than to none, group-bootstrap CI + sign-flip p; high/low-impulse slope-ratio CI excludes 1
  and covers physics; beats patch_random / passive on the first); required change includes the donor's A mass (full-ROI
  swap). Swap required change on theta_c (beyond the nuisance: the swap keeps the base's size). DiT zero-change swap reads
  theta from the activations at the swapped block / SNR (`site_theta`). Slope-0 steering and swap pass only if the push
  moved the model's theta readout (>= `min_effect` of its real-video change) or a control direction moved the outcome,
  else "inert push (uninformative)" (`null_verdict`). `readout_validity`: the readout on real clips vs the true theta.

## 4. Results (pre-v3 metrics, F1 only; to be re-run under v3)

* **L0.** States decodable from own-ball ROIs (held-out R²): V-JEPA 2 0.85–0.97, CogVideoX-2b 0.91–0.99,
  VideoMAE-v2 0.79–0.95, above mask floors; one-twin refit V-JEPA 2 0.81–0.95, untrained −0.02.
* **L1 S_mom** (frame vs ROI pool, legal twins, `smoke/l1_frame_roi/`): frame pooling cannot score momentum (probe on
  real legal video ≈ mass-blind); with ROI, every trained model's forecast breaks momentum worse than the mass-blind
  rule (V-JEPA2-L −1.00, CogVideoX-2b −0.58, VideoMAEv2-g −1.00).
* **L2.** V-JEPA 2: momentum violations more surprising (raw win 0.66 / 0.70 for e1 / e015, Holm-significant);
  CogVideoX-2b one borderline effect; VideoMAE-v2 and untrained nothing. Ideal-observer audit: e015 should be the
  primary momentum contrast; mass-blindness does not pass L2.
* **L2b.** Momentum violations barely decodable (AUC 0.55–0.58).
* **Outputs.** CogVideoX-2b track A gate 0/20 (now `output_track: none`). Wan2.2-I2V-A14B track B: strict 0/22,
  loose 8/22, all 8 with violations.
* **PC1 500-step test**: predictor not yet better than copy-last; `--check` on V-JEPA 2 fails.
* **L4 momentum validation (v3 code, 2026-11-12, `smoke/l4_momentum_validation/`)**: V-JEPA2-L runs end to end
  (precondition LOSO R² 0.23 vs null −0.02, layer 8 roi_A). Steering slope −0.01 [−0.06, +0.01] (random +0.005), swap IIA
  0.23 vs random 0.14 (fraction 0.00), patch fail (closer −0.39 [−0.48, −0.31]): V-JEPA2-L does not use its mass
  direction. Instrument limit: the frame-pool R1 log deflection ratio is finite on only 25/64 real clips (slope 0.54
  [0.16, 1.17]); the input-level reference 0.40 [−0.62, +0.99] (41 scenes). Needs a per-object instrument / more groups.
  Untrained V-JEPA2-L: precondition passes (appearance; LOSO R² 0.16), readout validity −0.06 [−0.28, +0.14] (33/64),
  steering / swap / patch fail: not passing, as required.

## 5. Plan (in order; GPU work)

1. **Render every family at master quality** (`KUBRIC_USE_GPU=1`, 832×480, 48 fps): F2, F3, F4, F6, F7, F8, F9, F10,
   F11, F12, 10 groups each for the smoke pass (more later), one family per GPU; `02_validate_data.py` on each.
2. **Stage 4 `--force`** on every model × family to rebuild caches with per-frame pools (`pooled_frame_t`; diffusion
   `fc_pooled_frame_t`) and the contact pool; `--pred-real` for encoders with a predictor / decoder. Re-cache F1 too.
3. **Stage 05 `--force`** (pixel baselines for L2) on every family.
4. **L0 → L1 → L2 / L2-rep → L2b → L3 → L4** per model (smoke models first: V-JEPA2-L, untrained, CogVideoX-2b,
   Cosmos), then stages 07–09 (output tracks, readout validation on real clips first) and stage 11 (leaderboard).
5. **PC1 / PC2 training** on the re-rendered training set (`configs/data/pc1_train.yaml`, legal clips only), then PC1
   through L0–L2b; check the §5.0.5 pass criteria before claiming any large model fails.
