# PhysLaw: state, results and plan

Benchmark of physical laws inside video models. Design reference: `docs/SPEC.md` (v2). **This is the only status
doc** (STATUS, HANDOFF_F1, SPEC_v1, prior_work, claude_memory were folded in on 2026-11-11; see git history).
Metric audit (every law × level, confounds, proposed metrics): https://claude.ai/artifact/PwUENMP5r9r5fixh3YfhBY
One branch: `physlaw-v2-merged`. Last updated 2026-11-11, before the pod was stopped.

---

## 0. Resume on the new pod (do this first)

```bash
cd /workspace/meta/physlaw                       # repo, data, cache, results and smoke outputs all live on /workspace
KUBRIC_SRC=/workspace/kubric bash scripts/00_setup_envs.sh     # venvs in /root/envs (~12 min; /root is not persistent)
export PHYSLAW_MODELS=/workspace/models HF_HUB_OFFLINE=1
/root/envs/physics_venv/bin/python -m pytest -q -p no:cacheprovider tests      # sanity
```

Known state to repair before running anything:
1. **`cache/vjepa2-vitl/f1` is half re-cached**: 299/320 clip files have the new `contact` pool (15 pools), 21 still
   have 12. Re-run `04_cache_activations.py --model configs/models/vjepa2_vitl.yaml --data data/f1 --force` and
   `--pred-real --force` before L0/L1 on it. VideoMAE-g and untrained caches were not re-cached (no contact pool yet).
2. **Cosmos-Predict2.5 weights are not downloaded** (gated): accept the licence on its HF page, `hf auth login`,
   then download to `/workspace/models` (config `configs/models/cosmos_predict25_2b.yaml`).
3. Nothing of F2–F10 is rendered on /workspace at full quality (the 2026-11-11 F2 render was stopped at 0/10).
4. Keep uv caches off /workspace (`UV_CACHE_DIR=/root/.cache/uv`); geesefs refuses chmod.

## 1. What is built

| Part | State |
|---|---|
| **Families** | F1 collisions: 64 groups rendered, frozen (`data/f1`). F2 restitution, F3 hidden mass, F4 Galilean (Kubric two-sphere scene, `generate.py`); F6 friction, F7 free fall / projectile / bounce, F8 ramp, F10 rolling (closed form, `physlaw/data/families/`, PyBullet cross-check). Configs `configs/data/f*.yaml`. Only 2–3 low-quality test groups per family exist (CPU, 320×184, 2026-11-10). **F9 pendulum not built**: Kubric's PyBullet bodies refuse every constraint (hinge impossible). |
| **Models** | Adapters for V-JEPA 2 (L, g, untrained), VideoMAE-v2 (base, giant), CogVideoX (2b, 5b, 1.5-5B-I2V), Wan (2.1 T2V 1.3B/14B, 2.1 I2V 14B, 2.2 TI2V 5B, 2.2 I2V A14B), PhysRVG, VideoREPA-5B, **Cosmos-Predict2.5-2B**, **Wan2.1-VACE-1.3B / 14B**, **LTX-Video 0.9.8 13B distilled**, positive controls PC1 / PC2. Weights for all but Cosmos, VACE-14B and LTX (VACE-1.3B / LTX downloading) in `/workspace/models`, verified 2026-11-11. |
| **Output tracks** | Every generator config has `output_track` + `track_a_mode` (2026-11-11, `physlaw/models/track_a.py`, spec §4.2 group 4). Track A: Cosmos (native), PhysRVG (native, its trained 5-frame / 2-latent context), Wan2.2-TI2V-5B at PhysRVG settings (**hack**, same clamp), Wan2.1-VACE-1.3B / 14B (variant), LTX-Video 13B distilled (native, 49 frames at 24 fps). Track B: Wan2.2-I2V-A14B, Wan2.1-I2V-14B, CogVideoX1.5-5B-I2V. CogVideoX-2b / 5b, VideoREPA, Wan2.1-T2V, Wan2.2-TI2V-5B: none (stage 07 `--allow-hack` only). |
| **Levels** | L0, L1, L2, L2b, **L3** (Galilean F4, mass independence F6/F7, transfer), **L4** (mass-direction steering, F1 direction → F3), outputs track A / B (stages 07–09, readouts for F1–F3, F7, F8), aggregation (stage 11, `results/leaderboard.md`). |
| **Laws** | `physlaw/eval/laws/`: momentum (+ S_mom), energy, CoM, angular momentum, Galilean, gravity, axis independence, mass independence, bounce, pendulum, ramp energy, rolling. `physlaw/data/io.law_inputs(labels)` feeds them for every family. |
| **Positive controls** | PC1 (ViT-S V-JEPA-style, trained on legal clips only) and PC2 (Wan2.2-TI2V-5B LoRA, RigidBench recipe): built, code-path tested, not trained. Pass: L1 S_family ≥ 0.5 and L2 adjusted win ≥ 0.7 on F1 momentum. |

## 2. Environment and pipeline

* `/workspace` is geesefs: no `sed -i`, symlinks, chmod, rsync. Use Python or the Edit tool for file edits.
* Venvs: `/root/envs/kubric_venv` (py3.11, bpy, pybullet, no torch: `physlaw/data/*`),
  `/root/envs/physics_venv` (py3.12, torch 2.8 cu128: models, eval). Keep `physlaw/data/*` torch-free.
* Rendering: `KUBRIC_USE_GPU=1`, check `render_device` = `OPTIX: ...` in labels; ~5 frames/s aggregate per H100 at
  4–6 workers, so a 10-group family with ~5 twins takes ~30–45 min per GPU.
* Never `pkill -f` a pattern that appears in your own command line.

```bash
K=/root/envs/kubric_venv/bin/python; P=/root/envs/physics_venv/bin/python
KUBRIC_USE_GPU=1 $K scripts/01_generate_data.py --config configs/data/f7_freefall.yaml --n-groups 10 --workers 6
$K scripts/02_validate_data.py --data data/f7
$P scripts/04_cache_activations.py --model configs/models/vjepa2_vitl.yaml --data data/f1 [--pred-real] [--limit N]
$P scripts/05_surprise.py ...                                     # pixel baselines for L2
$P scripts/06_eval.py --model <cfg> --level configs/eval/<l0_states|l2_surprise|l2b_detection|l1_forecast_laws|l3_invariance|l4_causal>.yaml --data data/<fam> [--out smoke/...]
$P scripts/07_generate_outputs.py / 08_track_outputs.py / 09_eval_outputs.py    # output tracks A / B
$P scripts/11_aggregate.py                                        # results/leaderboard.md
```
Order per model: L0 (builds `_maskfeat_v2.npz`), L2, L2b, L1, then L3/L4. Smoke runs: `--limit 8` in stage 4,
`--out smoke/<level>/<model>/<family>` in stage 6. The 8-group queue scripts are in `smoke/l0/`.

## 3. Decisions

* F1 = 64 groups, frozen (no time to render more). One simulation per group at e = 1; later twins re-keyframed, so
  pre-contact frames are byte-identical. Momentum violation = common drift, KE-reducing sign; energy-creating
  `mom_violate_e015` draws rejected. Master format 48 fps × 289 frames, 832×480, event frames 112–128.
* Data deviations (built 2026-11-10): F1 `nuisance_frac` 0 (others 0.2); F4 widens the camera, not the floor;
  F7 projectiles enter from off-screen and get a `legal_e04` control twin; F6 slides use a visibly heavier metal
  cube; F8 has no shared ledge (pre-event identity per ramp geometry); F10 tube I/(mr²) = 0.905; F3 restitution 1.
* **2026-11-11, metrics** (audit page):
  * L1 momentum: S_outcome and the r_p residual are dropped (S_outcome's ceiling is set by the hidden e:
    1 − |e − e*|/(1+e); r_p is perfect for the no-collision null).
  * **S_mom** (implemented): η = |q_A+q_B| / (|q_A|+|q_B|), q_i = m_i Δv̂_i; interaction gate D ≥ κ·2μu_n;
    S_mom = 1 − (η_fc − η_floor)/(η_blind − η_floor) with the mass-blind reference η = |m_A−m_B|/(m_A+m_B); unequal
    masses only (ratio ≥ 1.25: 54/64 groups); mass-ratio bins; κ sensitivity; per-group sums for common-subset
    comparison (`momentum_on_common_groups`); flagged ‡ / "undefined" when η_floor ≥ 0.8·η_blind.
    S_family kept as the secondary collision-law score.
  * **`contact` pool** (implemented, `tokens/pool.contact_mask`): both balls' last pre-contact ROIs, dilated 2
    tokens, held fixed after contact. Local like `roi`, leak-free like `frame`. Needs a stage 4 re-cache.
  * **`+cal` probe variants** (implemented): fold-proper affine de-shrinking of probe outputs.
* Smoke models: V-JEPA 2, CogVideoX, Cosmos-Predict2.5 (user, 2026-11-11).

## 4. Results

Smoke outputs (2026-11-11) are in `smoke/`; full-protocol tables are in `results/<model>/<family>/*.md` and
`results/leaderboard.md`.

### 4.1 Full-protocol results (F1, 64 groups)

* **L0.** States decodable from own-ball ROIs (held-out R²): V-JEPA 2 0.85–0.97, CogVideoX-2b 0.91–0.99,
  VideoMAE-v2 0.79–0.95, above mask floors. One-twin refit of post targets: V-JEPA 2 0.81–0.95, untrained −0.02.
  Mass is visible in F1 (density tied to material).
* **L2 (motion-adjusted, held-out distance).** V-JEPA 2: momentum violations more surprising, adjusted win 0.66 (e1)
  / 0.70 (e015), Holm-significant; 0.43 on the 21 energy-creating groups; no dose-response. energy_create 0.59 n.s.
  CogVideoX-2b: one borderline effect (e015, 0.66 [0.52, 0.77]). VideoMAE-v2, untrained: nothing.
* **L2b.** Momentum violations barely decodable (AUC 0.55–0.58). energy_create 0.94–0.97, but the legal pair is
  almost as separable (0.89–0.94): motion, not law.
* **L1 (old scores).** No decoded forecast is closer to the legal family than "no collision" on the frame pool;
  the forecast does hold outcome information (V-JEPA 2 forecast-trained ROI probe R² 0.74).
* **Outputs.** CogVideoX-2b track A gate 0/20 (latent replacement; now `output_track: none`, `--allow-hack` only). Wan2.2-I2V-A14B track B: strict 0/22, loose 8/22, all 8 with
  violations (loose medians: momentum residual 1.00, CoM 0.72).
* **Readout validation on real clips:** F1 312/320 strict; F2 44/44; F3 10/10; F7 11/11; F8 5/5; F4 11/18
  (short windows: readout not validated for boosts).
* **L3 / L4 stand-ins on F1** (code paths only): L3 mass-independence probe floor recovers the slope (−0.62 vs
  −0.82), the forecast does not; L4 decoded forecast barely moves under steering (2 groups).
* **PC1 500-step test**: predictor not yet better than copy-last (0.011 vs 0.008); `--check` on V-JEPA 2: FAIL
  (S_family −1.00; L2 0.67 / 0.75).

### 4.2 Metric audit (2026-11-11, from labels, no model)

L2 ideal observers (velocity-space surprise), F1 64 groups:

| Contrast | Perfect | Law, e=1 | Law, e=0.5 | Law, e=0 | Mass-blind | Mass-blind, ratio ≥1.25 | Violator slower |
|---|---|---|---|---|---|---|---|
| mom_violate_e1 vs conserve_e1 | 1.00 | 1.00 | 0.84 | **0.53** | 0.50 | 0.41 | 72% |
| mom_violate_e015 vs conserve_e015 | 1.00 | 0.95 | 0.98 | 1.00 | 0.50 | 0.41 | 81% |

* **e015 should be the primary momentum contrast** (e1 depends on the model's belief about e).
* Mass-blindness does not pass L2. Violators are mostly slower (KE-reducing drift), so raw win rates are conservative.
* Dose r_p mostly tracks |p_pre| (rank corr 0.36 with the edit size): use η of the violating twin.
* The L2 motion adjustment fits β on relative-velocity changes and applies it to CoM changes (unidentifiable):
  report model − pixel-baseline win rate as the primary motion control.

### 4.3 Smoke tests (2026-11-11)

* Unit tests: 814 passed on physlaw-v2 incl. new S_mom and contact-pool tests; laws + tokens tests pass on this branch.
* **L1 S_mom, frame vs ROI pool** (`smoke/l1_frame_roi/`, P-real-mm, best layer, legal twins):

| Model | Pool | Probe R² | η floor | margin over mass-blind (0.47) | η forecast | S_mom | mask-only R² |
|---|---|---|---|---|---|---|---|
| V-JEPA2-L | frame | 0.48 | 0.47 | +0.00 | 0.71 | undefined ‡ | — |
| V-JEPA2-L | roi | 0.90 | 0.30 | +0.17 | 0.71 | −1.00 | 0.18 |
| CogVideoX-2b | frame | 0.37 | 0.44 | +0.03 | 0.81 | ‡ | — |
| CogVideoX-2b | roi | 0.87 | 0.30 | +0.17 | 0.57 | −0.58 | 0.40 |
| VideoMAEv2-g (pred-real) | roi | 0.84 | 0.36 | +0.11 | 0.81 | −1.00 | 0.06 |
| untrained | roi | 0.58 | 0.52 | −0.05 | 0.83 | undefined | 0.18 |

  * Frame pooling cannot score momentum (probe on REAL legal video ≈ mass-blind). ROI can; every trained model's
    forecast breaks momentum worse than the mass-blind rule. ROI leaks the outcome location (CogVideoX strongly),
    which can only flatter the forecast. Gate passed by everything (κ = 0.25 too lax under probe noise).
  * The contact-pool + `+cal` run was stopped mid re-cache (see §0.1): **not run yet**.
* **L0 smoke** (`smoke/l0/`):
  * PASS on the 64-group caches: untrained, V-JEPA2-L, VideoMAEv2-g, CogVideoX1.5-5B-I2V (pass-1 cache). Held-out
    post-velocity R² (A): V-JEPA2-L 0.915, VideoMAE-g 0.927, CogVideoX1.5 0.982 (mask floor 0.47).
  * PASS, new model, 8 groups: V-JEPA2-g (sha256 OK). Weak numbers (pre velocity 0.30), but V-JEPA2-L on the same
    8 groups gets 0.28: sample size.
  * Stopped / not run: CogVideoX-2b L0 re-run (killed), Wan2.1-T2V-1.3B (stage 4 27/40 clips), CogVideoX-5b,
    VideoREPA-5B, Wan2.2-TI2V-5B (+PhysRVG settings), PhysRVG, Wan2.1-T2V-14B. Queue: `smoke/l0/l0_gpu_queue.sh <cfg...>`.
  * L1 for the new DiTs and CogVideoX1.5 needs stage 4 pass 2 (forecasts): `smoke/l0/l1_queue.sh`.

## 5. Plan (smoke test every law × level on V-JEPA 2, CogVideoX, Cosmos, then package)

Time estimates per H100; rendering dominates, so spread families across GPUs.

1. **Render 10 groups per family** (F2, F3, F4, F6, F7, F8, F10) at master quality, one family per GPU:
   ~30–45 min each. Validate with `02_validate_data.py`.
2. **Repair caches** (§0.1) and run stage 4 on the new families, 8–10 groups: V-JEPA2-L (minutes per family),
   CogVideoX-2b (~1 h per family; `--limit`), track A gates on 20 F1 clips: Cosmos (download), PhysRVG +
   TI2V-5B clamp, Wan2.1-VACE-1.3B, LTX-Video 13B distilled (`configs/eval/outputs.yaml` track_a.models).
3. **L1 pool choice**: contact vs roi vs frame on V-JEPA2-L / CogVideoX-2b with `+cal`. Adopt contact if its
   η_floor beats mass-blind by ≥ 0.1 with mask-only R² ≈ 0. Then set the gate from floor noise.
4. **L2 momentum** (F1, no new data): e015 primary, dose η, ideal-observer bands, model − pixel-baseline win rate;
   F2 adjacent levels as held-out legal control once F2 is rendered.
5. **L2b**: kinematic-feature baseline AUC and e1→e015 transfer.
6. **Per-family law metrics** (audit page): gated ratio metrics for axis/mass independence (null-satisfied),
   ĝ ratio for gravity, within-group speed ratios for F8 path independence, two-sided violations in configs.
7. **L3 / L4** on F4 / F3 renders for V-JEPA 2 and CogVideoX-2b.
8. **Outputs**: `eval/outputs.py` still scores r_p and S_outcome; switch to S_mom (track A) and mass-free tests
   (track B: Δv antiparallel along n, ê ≤ 1).
9. `scripts/11_aggregate.py` → leaderboard; package this file, the audit page, `smoke/`, `results/`.
