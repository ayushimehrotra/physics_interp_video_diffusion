# EXPERIMENTS — which script produced which result

Script → what it asks → which § → where the output lands. `CLAUDE.md`'s
"Reproducing what has been run" has the full command lines; this is the map.

The normal sequence is **capture once per (dataset, model), then probe or
intervene many times**. Capture needs a GPU and writes pooled activations;
everything else reads what capture wrote and runs in seconds to minutes.

---

## The one script every experiment goes through

**`scripts/pipeline/run_experiment.py`** — three modes.

```bash
# 1. capture: invert/encode every scene, store pooled activations (GPU)
.venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml \
    --mode capture --model videomae-base \
    --dataset-dir datasets/probe_set_v3_latent -o outputs/v3_videomae-base

# 2. probe: fit linear probes on what capture wrote (no GPU, seconds)
.venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml \
    --mode probe -o outputs/v3_videomae-base --tag v3grid \
    --pools "A.pre,A.post,A.all,B.pre,B.post,B.all,all.pre,all.post,all.all" \
    --targets momentum_pre,momentum_delta,mass_A,velocity_A_pre

# 3. generate: prompt x seed sweeps (the default mode)
```

Capture and generate shard across GPUs and skip finished work, so an
interrupted sweep resumes. Probe is single-process and re-runnable with
different targets or strata without touching a GPU.

**Storage layout.** `outputs/<tag>_<model>/momentum_probe/activations/scene_*.npz`,
one file per scene: `act.<site>` arrays of shape `[steps, pools, dim]`, plus a
`meta` blob holding the scene's whole `labels.json`. Keeping the full labels is
what lets a new question be asked of an already-captured set without re-running
a GPU.

**Where the probe protocol lives.** `physics_interp/interp/__init__.py` —
`fit_probe` (60/40 scene split, ridge alpha selected on an inner 75% carve-out
of *train*, `shuf*` permutation control), `probe_grid` (every site × step × pool
× target), `select_scenes` (`--where`), and `DERIVED` (every probe target that
is computed from `labels.json` rather than stored in it).

**Output dir naming.** `outputs/<experiment>_<model>`, where `<experiment>` is
the dataset or intervention family — `v3`, `slide`, `incline`, `patch`, `steer`,
`predict`. `outputs/` with no suffix is CogVideoX-2b on v1.

---

## Probing

| script | asks | § | output |
|---|---|---|---|
| `run_experiment.py --mode probe` | is quantity X linearly decodable, and where | §1–§4, §13, §35 | `probes_<tag>.json` |
| `mask_only_floor.py` | how much of an object-local score is the pooling mask alone | §20 | `mask_floor_concat.json` |
| `compose_incline.py` | mu direct vs composed from decoded (a, θ) | §35 | printed + JSON |
| `lawful_errors.py` | are the model's errors *lawful* — one readout tied across the collision | new | `lawful_errors.json` |
| `probe_conservation_set.py` | is a momentum violation detectable, paired twins, exact paired null | §42 | `conservation_probes.json` |
| `probe_energy.py` | is the clip losing ENERGY — the full 9-cell pool grid, and does it beat a camera | §44 | `energy_probes*.json`, `energy_fixed_e.json` |
| `energy_composition.py` | does an energy readout survive a change of SCENARIO, and is energy stored or composed | §45 | `energy_composition_{A,B}.json` |

**`mask_only_floor.py` is not optional.** Any object-local number without its
mask floor beside it is not reportable — `cube_size` is 99% recoverable from the
mask alone on the slide set.

**`lawful_errors.py`** is the one test here that does not need a decodable
label. It fits a single readout tied across the pre- and post-contact pools and
asks whether the two readouts' errors are shared. Run both arms — the placebo is
the control the result stands on:

```bash
.venv/bin/python old_experiments/scripts/lawful_errors.py outputs/v3_videomae-base/momentum_probe \
    --n-perm 500 --out outputs/v3_videomae-base/momentum_probe/lawful_errors.json
.venv/bin/python old_experiments/scripts/lawful_errors.py outputs/v3_videomae-base/momentum_probe \
    --placebo ke
```

`--min-r2` gates cells on readout quality and must not be lowered casually: with
no working readout the errors *are* the target and every statistic degenerates.
See RESULTS §6.

**`probe_energy.py` — four modes, and the last two are what make the first
readable.** The bare grid says the binary "is this clip losing energy" decodes at
AUC 0.95–0.98; `--vs-kinematics` says a camera does it at 1.000. Run all four:

```bash
.venv/bin/python scripts/kinetic_energy/probe_energy.py --n-perm 200            # the 9-cell grid, all sites
.venv/bin/python scripts/kinetic_energy/probe_energy.py --mask-floor            # §20's floor, same grid
.venv/bin/python scripts/kinetic_energy/probe_energy.py --transfer              # fit honest, predict violating
.venv/bin/python scripts/kinetic_energy/probe_energy.py --vs-kinematics         # probe vs a mass-free observer
.venv/bin/python scripts/kinetic_energy/probe_energy.py --fixed-e               # the magnitude, at e pinned
```

No GPU and no torch — numpy throughout, plus Pillow for `--mask-floor`. Pools are the
nine the captures already store, so the region × phase crossing needs no re-capture.

**Do not read a number from this script without its mass-free ceiling**, printed
next to every target. On the honest half the label *is* a visible mass-free scalar
(§12 again), so the ceiling there is 1.000 and the probe's job is only to show that
relative speed is legible. The cells worth reading are the violating ones, where
elasticity and energy loss come apart. See RESULTS §7.

**`energy_composition.py` — ask the question where §44's identity does not reach.**
The friction sets have one cube and one floor, so the floor's recoil is unobservable
and nothing hands the observer a mass:

```bash
.venv/bin/python scripts/kinetic_energy/energy_composition.py --test A   # slide -> collision transfer
.venv/bin/python scripts/kinetic_energy/energy_composition.py --test B   # direct vs composed, in-set
```

`--test A` is the one that works. **Read its two CONTROL rows first** — apparent size
and speed must cross the domain gap or the KE row is void rather than negative.
`--test B` is at ceiling on this set (mass decodes at 0.994) and its verdict is gated
on both a bootstrap CI *and* a cross-convention sign check, because two read
conventions produced significant gaps pointing opposite ways. See RESULTS §8.

**`--fixed-e` is the one that kills the obvious rescue.** "Masses vary 0.059–24.2 here,
so surely the *magnitude* of the loss needs them" — no: momentum conservation gives
`m_A/m_B = |Δv_B|/|Δv_A|` (max relative error 1.02e−15 on all 150 honest clips), and
substituting it rebuilds the KE ratio from visible velocities at **R² = 1.000000**. The
mode prints that identity check, then races the probe against a three-rung observer
ladder at `e` pinned to 0.15. Read the ladder's descending K1 > K2 > K3 as *capacity*
(54 / 77 / 119 quadratic columns against n = 75), never as "mass hurts". §44c-bis.

---

## Causal — steering and patching

All of these write at `block.N.resid_post` through a **binary** token mask (never
the normalised pooling row, which sums to 1 and silently rescales alpha per
scene), with `scale="sigma"` so alpha means the same relative perturbation at
every depth. Directions and readouts are fit on train; every intervention is
evaluated on held-out scenes.

| script | asks | § | output |
|---|---|---|---|
| `steer_velocity.py` | 1-D steering along the probe direction; alpha ladder + depth sweep | §18a | `steer_pilot_vx_read11.json`, `steer_sweep_vx_read11.json` |
| `steer_multidim.py` | does the *generative* direction work where rank-1 does not | §18c | `multidim_block<N>.json` |
| `patch_velocity.py` | transplant one scene's A-local representation into another | §18d | `patch_block<N>_k16.json` |
| `steer_polar.py` | polar (speed × heading) or Cartesian (v_x, v_y) | §19 | `polar_block<N>.json` |
| `steer_friction.py` | steer mu under Joseph et al.'s held-out-probe protocol | §30a | `steer_friction_block<N>.json` |
| `patch_surface.py` | is mu bound to the surface or the object | §31 | printed + JSON |
| `subspace_angles.py` | do A and B occupy different subspaces inside the same tokens | §18b | printed |

**The design worth copying is `patch_surface.py`'s.** Patch and readout use
**disjoint** token pools, so the passive channel is not argued away but
*computed* — the read pool can only move by the sum of its pooling weights over
the patched tokens, which comes out exactly 0.000 for floor→cube. Contrast
`steer_velocity.py` §18a, where edit and readout shared tokens and a purely
passive effect scored observed/passive = 1.02 and looked like success.

---

## Capability — what the models can do at all

| script | asks | § |
|---|---|---|
| `predict_pre_only.py` | can the aftermath be predicted from pre-contact frames | §15 |
| `predict_object_local.py` | the same, through object-local pools (§15's readout was 12× too blunt) | §21 |
| `render_prediction.py` | what does the model actually predict, in pixels (VideoMAE only — V-JEPA2 has no decoder) | §21 |
| `analyse_collide_friction.py` | is mu a surface property or a per-object, per-phase kinematic estimate | §31 |

All prediction routes came back closed. See RESULTS §8.

---

## Dataset generation

Under `datasets/`. Rendering uses the **kubric venv**
(`/workspace/kubric/.venv/bin/python`), not the project venv.

| script | set |
|---|---|
| `generate_scene.py` | `probe_set` (v1), `probe_set_v2/v3_latent` — `--arm legacy\|latent_mass\|visible_mass` |
| `generate_slide_scene.py` | `slide_set_v1/v2` |
| `generate_incline_scene.py` | `incline_set_v1/v2` |
| `generate_friction_scene.py` | superseded two-ball top-down friction set |
| `generate_collide_friction_scene.py` | `collide_set_v1` |
| `run_batch.py` | driver — resumes by seed, re-renders only what is missing |
| `validate*.py` | per-set validation checks; write `_validation.json` |
| `contact_sheet.py` | **run this and look at it before probing** |

```bash
KUBRIC_USE_GPU=1 /workspace/kubric/.venv/bin/python datasets/run_batch.py \
    --num-scenes 300 --output-dir datasets/<set> \
    --seed-offset 0 --violate-frac 0.25 --workers 3 --arm <arm>
```

`--arm` and `--violate-frac` are both non-default and both matter.

**Two of three rendering bugs in this project were invisible to every validator
and obvious in one glance at the clips.** The contact sheet is a required step,
not a nicety.

---

### `energy_voe_v1` — conservation of energy as a LAW

Three scripts, and the design note lives in `plan_energy_voe_set.py`'s docstring:

```bash
python datasets/plan_energy_voe_set.py --seeds 60 --k 1.25 1.5 2.0   # physics only
python datasets/run_energy_voe_set.py --workers 4                     # render
python datasets/validate_energy_voe_set.py                            # 6 checks
```

The plan step runs the generator with `render=False` and applies the edit, so
feasibility (in-frame, post-contact travel, momentum, realised `e`) is checked by the
same code that will run in the real render rather than by a reimplementation.

**`validate_energy_voe_set.py` before anything else.** Check 1 is that momentum is
conserved in *every* clip — if it is not, the set has reintroduced the confound it
exists to remove. It also prints the mass-free separability of impossible-vs-legal,
which is ~1.000 by design: **this set is for violation of expectation, not probing.**

---

## Reading a result

1. Is the target an exact function of what a camera can see? If so the bar is an
   **oracle**, not `shuf*` — and name the oracle's basis (§34c).
2. Is the target a **linear combination** of other probed targets? If so
   composition, subspace containment and INLP confirm by algebra (§34b).
3. Is the target a norm or a threshold on one? A linear probe cannot reach it —
   put the nonlinearity in the readout *after* a linear fit (§34a).
4. Object-local pool? Print the mask-only floor (§20).
5. Did the number come from an argmax over the grid? Then the null is the
   **grid-max** on shuffled labels, not the winning cell's own `shuf` — and any
   probe with per-cell hyperparameter selection needs an empirical
   multi-permutation null (§34f).

---

## Files that are not results

`docs/artifacts/` builds the HTML result pages (`build_index.py`,
`build_incline_page.py` and their templates/rows JSON). `tests/` holds 27 unit
tests over the hook substrate and probe protocol. `configs/momentum_probe.yaml`
is the capture/probe config every experiment above starts from.
