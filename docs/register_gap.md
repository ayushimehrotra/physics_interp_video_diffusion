# This repo against the Momentum Protocol Register (compiled 2026-09-24)

The register (the artifact produced on the other GPU instance) describes the
project as of 2026-09-24. This repository, on both `origin/main` and
`runpod-rebuild`, is the state of **2026-08-07**: five commits, the last one
"Document exact flags for regenerating probe_set". Nothing from the other box
was ever pushed to this remote. Everything below the first table was therefore
rebuilt here from the register's prose on 2026-10-11 (branch `register-catchup`);
it is a second implementation, not the other box's code, and its numbers will
not be the register's numbers until it is run on real captures.

## What was here vs what the register describes

| | this repo (2026-08-07) | register (2026-09-24) |
|---|---|---|
| model under study | CogVideoX-2b, activations along an inverted DDIM trajectory | three frozen encoders: VideoMAE-base, V-JEPA2-ViT-L, VideoMAEv2-giant |
| datasets | `probe_set` (150 scenes, 75 violating, random-direction common drift, continuous e) | `conservation_set_v1` (150 seed pairs, e pinned to 1.00 / 0.15, split-changing edit), `energy_voe_450`, `energy_ladder_v1` (20 × 11), `scene_ladder_v1` (1557 clips) |
| captures | none | all four sets × three encoders, plus truncation captures (v1) |
| detection statistic | accuracy of a thresholded ridge (`fit_probe`) | raw AUC over every clip with a label-permutation null; paired accuracy removed (§55) |
| folds | random 60/40 by scene | grouped K-fold by seed (twins never split) |
| protocols | one binary probe + variable decodability | §2a, §2c, §2g, §47, §48, §49, §50, §51, §52, §54b, §55, §56, §57, §58, §59, §60, §61 |
| scripts | `run_experiment.py` | ~13 named analysis scripts, `plan_energy_voe_set_register.py --from-conservation-plan`, a VoE pipeline |
| docs | `CLAUDE.md`, dataset README | `docs/variables/momentum.md` (§1–§55), `docs/conservation_set_v1.md`, `docs/artifacts/*.json`, `old/` archives |

The physics in `momentum_dataset/generate_scene.py` (zero gravity, planar,
frictionless, contact on frame 20, pre/post frames 18/22) is the same physics the
register's sets are built on, and is reused unchanged.

## What was built here, mapped to the register

| register item | where it lives now | verified how |
|---|---|---|
| outcome space, legal line, {normal, tangential, drift} basis, closed-form drift–legal cosine (Part 2 §4, §57h) | `physics_interp/momentum/physics.py` | `tests/test_momentum.py`: true outcome decomposes to drift 0, tangential 0, implied e exact (the §47 geometry control) |
| the split-changing violation, the (1+e)/(1+e_sim) restitution edit (Part 1, §58) | `physics.split_edit`, `physics.restitution_edit`; wired into `generate_scene.py` (`--violation-mode split`, `--edit-restitution`, `--restitution`) | tests: edit keeps separation speed, breaks momentum, equals a common drift along n; restitution edit exact, dKE closed form, identity at e_sim |
| raw AUC, permutation null, grouped folds, inner alpha selection (§55) | `physics_interp/momentum/stats.py` | tests on signal vs shuffled labels |
| capture format + clip-set index, `concat` = `A.all+B.all` | `physics_interp/momentum/captures.py` | used by every script below on synthetic captures |
| §2a direct probe, §2c fitted norm, every layer | `scripts/probe_conservation_set_register.py` | runs on synthetic captures |
| §2a diagnosis / §61 (exact dp linear vs exact \|dp\|, quadratic ingredient oracle, terciles, sample-size curve) | `scripts/direct_probe_failure.py` | exact \|dp\| → 1.000 on synthetic; exact dp vector linear → ~0.5 |
| §2g Q1–Q4 at every layer | `scripts/momentum_failure_register.py`, `detectors.failure_localisation` | runs; radial/tangential/cos/tercile table |
| §47 valid-collision subspace on truncation rungs | `scripts/valid_collision_subspace.py`; truncation via `capture_encoder.py --truncate k` | refuses unless the geometry control passes; on synthetic rungs the gap is positive pre-contact and closes post-contact |
| §48 composed norm | `scripts/geometric_violation_detector.py` | runs; self-calibration floor reported |
| §49 subspace membership, `--fit-honest` | `scripts/subspace_membership.py` | runs; honest-only fit predicts every clip |
| §50 four routes on identical features, label-free site choice | `scripts/momentum_by_elasticity.py` | runs; object-local vs frame-wide |
| §51 lawful errors ratio, second arm per stratum | `scripts/lawful_errors_register.py` | runs; bootstrap CI |
| §54b three-point chord test (register: "never written to a script") | `scripts/chord_test.py` | returns 0.400 ± 0.001 on a near-linear noiseless synthetic encoder; test asserts the physics value |
| §56 manifold shape, §56h legal-six / PC2 LOO-R² audit | `scripts/elasticity_manifold.py` | runs on a synthetic 20 × 11 ladder |
| §57 population γ with chord/random ablations and the cancellation diagnostic | `scripts/curve_detector.py --ladder` | runs; null on synthetic |
| §60 per-scene curve, matched-legal control, chord ablation, kinematic oracle | `scripts/curve_detector.py --per-scene` | kinematic oracle 1.000 with honest 0.000 on synthetic |
| §59 VoE rank statistic, residualised twin difference, boundary slopes | `scripts/voe_ladder.py` (statistics) + `capture_encoder.py --surprise` (V-JEPA2 predictor) | statistics run on a synthetic surprise table; the predictor call is **unverified** |
| encoder captures with pools × phases, `embed.out`/`block.i.resid_post`, straddle refusal | `physics_interp/momentum/encoders.py`, `scripts/capture_encoder.py` | frame specs and token counts checked (1568 / 8192 / 2048); pool weights sum to 1; **no model run here yet** |
| conservation_set_v1 plan (seeds 101+, e alternating 1.00/0.15) | `momentum_dataset/plan_conservation_set_register.py` | 300 jobs |
| energy_voe_450, energy_ladder_v1 (seeds 9500+), scene_ladder_v1 `--from-conservation-plan` | `momentum_dataset/plan_energy_voe_set_register.py`, `run_plan.py` | 450 / 220 / 1950 jobs; 1950 − the register's 393 drops = its 1557 |
| §58's validation checks, label-trap check, relative in-frame rule | `momentum_dataset/validate_sets.py`; rule enforced in `generate_scene.py` (`_DROPPED`) | all checks pass on synthetic label sets; an injected trap fails the right check |
| the label trap itself | fixed at the source: new clips write `restitution` = realised and keep `restitution_simulated`; `physics.clip_restitution` prefers `restitution_realised` and refuses to fall back on an edited arm | tested |

## Not built, and why

* **The register's numbers.** No real captures exist here. Every script has been run only on `physics_interp/momentum/synthetic.py`'s toy encoder.
* **The other box's exact frame sampling.** The register says only that captures are "identical in pools / frames / phase split". The specs here (`10:41:2` for the 16-frame encoders, `5:37:1x2` for V-JEPA2) are the choices that keep every tubelet on one side of contact; they are documented assumptions in `encoders.py`.
* **VideoMAEv2-giant loading.** `OpenGVLab/VideoMAEv2-giant` with `trust_remote_code`; the block list is located generically. Needs a real run.
* **V-JEPA2 predictor surprise.** Written against `VJEPA2Model.forward(context_mask=, target_mask=)` and `predictor_output.{last_hidden_state, target_hidden_state}`; raises with the available fields if the API differs.
* **§52** (scale) is §48–§51 run on the giant; no separate script. **§55** is a method change and is enforced by `stats.py` having no paired accuracy.
* **Rerun queue item 4** ("a set where the violation is NOT geometrically off the rendered family") is a dataset design question, not written.
* **`docs/variables/momentum.md`, `docs/conservation_set_v1.md`, `old/`** — the notebook and archives exist only on the other box.

## The fastest way to actually catch up

Push from the other box. `autopush.sh` was written for exactly this and the
remote is `github.com/ayushimehrotra/physics_interp_video_diffusion`. If that
history arrives, prefer it over this branch for everything the register
quotes, and keep from here only what the register says was missing (the chord
test, the §47 truncation capture path, the set validator).
