# Benchmark Spec v2: Physical Laws Inside Video Models

Working name: **PhysLaw-Internal**. Implementation spec for Claude Code. Supersedes v1 (`physlaw_benchmark_spec.md`, 2026-11-04).
**§5.0 (v3 metrics, 2026-11-11) is authoritative** for every law statistic, probe rule and score; §5.1–§6 only point to it.
State, results and the run plan: `docs/RESUME.md` (the only status doc).

---

## How to apply this spec to the existing repo

**Do not rebuild anything that is already done.** Decisions and deviations are recorded in `docs/RESUME.md` §3.

Rules:
1. Milestones marked **DONE** in Section 10 stay as they are. Re-run them only where Section 10 or `docs/RESUME.md` §5 says so.
2. Where this spec and the built code disagree, keep the built behaviour and record the disagreement in `docs/RESUME.md`.

---

## 0. Goal

Build a benchmark that tests whether video models obey physical laws **in their internal representations**, and compare that with whether their **generated outputs** obey the same laws.

Hypothesis under test: models encode per-object physical *states*, while the *laws relating those states* hold weakly or not at all in what the models compute internally.

The benchmark must be able to return either answer. A trained positive control (Section 4.4) shows the tests are passable; an untrained negative control shows they cannot be passed from pixel layout alone.

### 0.1 Deliverables

1. A data generator producing labeled synthetic videos with matched twin clips, for eleven scene families (F1–F4, F6–F12; F5 merged into F6 / F7).
2. A model-adapter layer exposing activations, forecasts, surprise and generation for every model.
3. Five evaluation levels (L0 to L4) plus a trained-detection companion to L2 (L2b).
4. Output-level evaluation in two tracks (A: scene continuation, B: self-consistency).
5. A results pipeline: per-model JSON, a leaderboard and standard figures.

### 0.2 Changes from v1

| Area | Change | Section |
|---|---|---|
| Rendering | 48 fps master, 289 frames, contact frames 112 to 128 (was 24 fps, 81 frames) | 2.1 |
| Twins | One simulation per group; closed-form legal outcomes; KE-reducing drift sign for momentum violations; `energy_physical` flag | 2.2 |
| F1 size | Target restored to 400 groups (currently 64); confirm with user | 2.3, 11 |
| F1 laws | Center-of-mass velocity and angular momentum added (no new rendering) | 2.3 |
| F5 | Merged: drop pairs move to F7, slide pairs move to F6 | 2.3 |
| New families | F7 free fall / projectiles / bounces; F8 ramp energy exchange; F9 pendulum (optional); F10 rolling (optional) | 2.3 |
| Models | Grouped by research question; added Wan2.1-1.3B, Wan2.1-14B, V-JEPA 2 ViT-g, two physics post-training pairs (PhysRVG, VideoREPA), Cosmos-Predict2.5 for track A, Wan2.1-I2V-14B for track B | 4 |
| Controls | Second positive control (simulator-fine-tuned DiT); positive control moved onto the critical path | 4.4 |
| L0 | Noised-input-latent baseline for DiTs; one-twin-per-group refit; matched velocity protocol check | 5.1 |
| L1 | Superseded by v3 (§5.0): S_mom for momentum, ratio scores per law, P-pred-real / as_is instrument, held-out space selection | 5.0 |
| L2 | Superseded by v3 (§5.0): contrasts with speed-matched violators and the model − pixel-baseline win rate; energy kinks on ε ladders | 5.0 |
| L2b | New: trained violation detection, reported beside zero-fit surprise | 5.4 |
| L4 | Superseded by v3 (§5.0): manifold steering per law, swap test, steering reliability | 5.0 |
| Metrics v3 (2026-11-11) | Nine laws as scale-free statistics; ε ladders; F8 Galileo ramp; new families F9, F11, F12, F6 glide / viscous; F10 κ per group | 2.3, 5.0 |
| F6 v2 (2026-11-12) | F6 = slide (legal / reverse / viscous / decel ladder) + pair + glide on one uniform surface, invisible branch mid-slide; flat / incline / band slides and the F6 L3 transfer removed; friction law gains stop_hold | 2.3, 5.0 |
| Audit fixes (2026-11-12, user) | F6 `no_friction` twin dropped, slide / pair timing for visible effects; F7 legal-vs-legal control dropped; F12 KE-conserving `spin_ke_x0.8 / x1.2` (primary); L2 speed-matched halves and energy flags; one x̂ for the kink's exclusion and K; L2b verdict needs the ROI mask floor; L3 Galilean law test on the forecast (real videos = floor); L3 mass independence must beat the mask-only score | 2.3, 5.0 |
| Outputs | Two tracks (A continuation, B self-consistency); colour-tracker readout replaces SAM 2 / CoTracker; strict and loose gates | 6 |

---

## 1. Repository layout

```
physlaw/
  data/        generate.py, twins.py, io.py (law_inputs), kubric_scene.py (F1-F4), families/f6..f12 (closed form)
  tokens/      project.py (mask -> token grid), pool.py (frame, contact, roi pools)
  models/      one adapter per model family (vjepa2, videomae2, cogvideox, wan, ltx, cosmos_predict, dit, track_a,
               positive_control, untrained), registry.py
  probes/      ridge.py (shared R1 instrument, selectivity), grid.py, select.py (held-out selection)
  eval/        laws/ (collision.py, freefall.py, incline.py: the nine laws), l0_states.py, l1_forecast_laws.py,
               l2_surprise.py, l2b_detection.py, l3_invariance.py, l4_causal.py, outputs.py
  analysis/    aggregate.py, figures.py
scripts/       00_setup_envs.sh ... 11_aggregate.py (one CLI per stage)
configs/       data/, models/, eval/ (YAML with defaults: inheritance)
```

Every stage stays a separate resumable CLI command reading from and writing to disk.

---

## 2. Data

### 2.1 Rendering defaults (as implemented)

- Simulator: Kubric (PyBullet physics, Blender Cycles rendering), with the existing fast post-processing (`physlaw/data/fast_post.py`), Blender thread cap of 8, and persistent render data.
- Master render: **48 fps, 289 frames (6.0 s), 832 × 480**. Each adapter subsamples to its native rate by exact stride (8, 16 or 24 fps) and resizes by whole-frame squash (letterbox available).
- Event frame (contact, branch or release, depending on family): sampled in frames 112 to 128 and stored per clip.
- Per clip: `video.mp4`, `masks.npz`, `labels.json` (per-object per-frame position, velocity, angular velocity, mass, radius, density; event frame; contact normal and point where relevant; scene parameters; camera intrinsics and extrinsics; seed; twin group ID; family; split; `physlaw.*` flags).
- Splits by hash of twin group ID (about 70/15/15), stable as sets grow.

### 2.2 Twin construction (as implemented, generalized)

A **twin group** is a set of clips that are byte-identical up to the event frame and differ only afterwards.

General rules for every family:
1. **One simulation per group.** Legal and violating outcomes after the event frame are computed in closed form from the simulated pre-event state, then re-keyframed exactly at the event time. This guarantees identical pre-event frames.
2. **Each violating twin breaks exactly one law.** Where a violation could also break a second law (for example, a momentum violation that also creates energy), choose the edit sign that avoids it, reject draws that still break it, and record a flag (for example `physlaw.energy_physical`) so L2 can stratify.
3. **At least two legal twins per group where the family allows it.** Two legal outcomes with different amounts of motion give L2 its legal-vs-legal control. Exceptions: F6 (one surface, one μ: the outcome is unique), F7 projectile / drop (its second legal twin `legal_e04` differs only after the first floor contact, which ends the clean window: no L2 control), F9 pair, F10, F11.
4. **Violation magnitude is recorded** per twin so L2 can test dose-response.

### 2.3 Scene families

Priority: **P0** required for the main paper, **P1** strongly recommended, **P2** optional.

#### F1. Two-body collisions (P0, built)

Two spheres on a frictionless plane, zero gravity, radius and density sampled independently.

| Twin | Construction | Laws broken |
|---|---|---|
| `conserve_e1` | Closed-form elastic outcome | None |
| `conserve_e015` | Closed-form outcome at e = 0.15 | None |
| `mom_violate_e1` | Common drift of both balls, sign chosen to lower \|v_cm\| | Momentum (energy too in about 1/3 of groups; flagged) |
| `mom_violate_e015` | Same edit on the e = 0.15 outcome; energy-creating draws rejected | Momentum |
| `energy_create` | Both Δv scaled by λ = (1 + 1.5)/(1 + e_sim) | Energy |

Laws scored on F1 (§5.0.1): momentum (η, which also covers the CoM velocity; mass-free antiparallel and normal alignment, which cover angular momentum about the contact) and energy (ε, ρ\* = ε²).

Size: **400 groups target** (currently 64; see Section 11).

#### F2. Restitution sweep (P1)

Same scene as F1; restitution ladder e ∈ {0.0, 0.1, …, 1.5} (e ≤ 1 legal, e > 1 creates energy, §5.0.2); identical pre-contact frames. 100 groups. Used for the L2 energy kink, L1 and L4.

#### F3. Hidden mass (P1)

As F1 legal twins, with **identical material, texture and radius** for both balls. Mass ratio m_A/m_B ∈ {0.33, 0.5, 1, 2, 3} through density only (geometric mean fixed), all five sharing byte-identical pre-event frames. 300 groups × 5. **Hidden mass by design:** mass is invisible before the event and shows only in the outcome, so F3 tests whether the mass ratio is decodable from the post-event frames (L0) and serves L1 energy. It is NOT scored for L1 momentum (one forecast per set cannot depend on the ratio: S_mom would fail by construction, audit H1; `l1_forecast_laws.MASS_HIDDEN_FAMILIES`), and it is not an L2 / L2b / L3 / L4 family (every twin is legal: no violation contrast; `configs/eval/l2_surprise.yaml`, `l4_causal.yaml` families exclude f3).

#### F4. Galilean boosts (P0)

Take F1 legal groups (e = 1 and e = 0.15). **Ball boost** (twins `u0_*`, `usmall_*`, `ularge_*`): add a common velocity u to both initial velocities, u ∈ {0, small, large}, directions sampled; the camera is static. **Camera boost** (2026-11-12; twins `cam_usmall_e1`, `cam_usmall_e015`, `cam_ularge_e1`, `cam_ularge_e015`): the collision is the unboosted one and the camera translates at constant velocity −u (same u as the ball-boost level, no rotation, keyframed in Blender), so relative to the camera the balls move exactly as in the ball-boost twin (projected centres equal to < 0.5 px in every frame up to clean_until_s; on the frictionless floor the physics is identical) while the floor slides across the frame. The floor carries a mild 1 m checker (every F4 twin) so the camera motion is visible. The 40 m floor fills the view and the camera (13 m above the sphere plane, spheres ≥ 46 px at 832 × 480) keeps every boost's spheres in frame from frame 0 to 1 s after contact. Restitution and center-of-mass-frame scattering angle are invariant. pre_segment = the boost level (`u0`, `usmall`, `ularge`, `cam_usmall`, `cam_ularge`). Labels: `boost_kind` ball | camera, `edit.boost` = u (the apparent boost; camera twins also `edit.camera_velocity` = −u), and for camera twins `frame: camera` (the primary states — per-frame position / velocity, pre / post_velocity, event_state — are in the camera's rest frame, i.e. what the clip shows and what every readout, probe target and level uses; the world states, equal to the u0 twin's, are kept under `*_world`, `io.law_inputs(frame="world")`) and `camera_trajectory` (per-frame camera position, velocity −u, t_ref = the contact). L3 treats `cam_<level>` as boost levels of their own against u0 and pairs each with its ball-boost twin (`camera_vs_ball`: the same ball image motion, only the background differs; law 0). Track-B output readouts do not run on F4 (not in the outputs families: the moving background breaks the context-background cut test and background-subtracted tracking; `_world_maps` would read the camera-frame positions). 200 seeds × 2 e-values × 5 boosts = 2,000 clips. **No other benchmark tests this.**

#### F6. Friction (slide, pair) and glides (P1)

Redesigned 2026-11-12 (the banded floor, the incline and the band-crossing slides are removed: a band's look did the work, friction switching on at a line is unphysical, and their legal-only twins had no violators). One uniform, textured, concrete-like floor (`generate.surface_texture`: world-space two-scale noise, identical everywhere and in every twin; the PyBullet check runs the legal twin from frame 0 on a slab with friction μ and agrees to ~2 mm). The block is **already sliding at frame 0** and decelerates at a = μg (Coulomb), so the pre-event frames show the clip's own friction rate. Event = an **invisible branch** on a frame mid-slide (event_kind `branch`). Timing (2026-11-12, user: effects clearly visible): **slide** branch in frames 112–128, legal stop 0.9–1.6 s later and 2.1–2.4 s before the clip end, so `reverse` ends ≥ ~150 px (median ~205) from the legal rest point at 832 px; **pair** branch in frames 112–120 and legal stop 0.5–0.6 s before the clip end, so most of the slide is after the branch and `heavy_x13`'s heavier block stops ~35–50 px (median ~43) short. Every twin that stops does so ≥ 0.5 s before the clip end. Oblique camera (pitch 30–42°, yaw 0: the motion runs along image x at constant depth) fitted to the union of every twin's whole path (96% of the width), block ≥ 70 px at 832 × 480 (median ~90), every twin in frame to the clip end (clean_until_s = clip end). Lengths scale with the block (union of the paths 5.5–8 block widths), so μ is small (slide ≈ 0.05–0.11, pair ≈ 0.03–0.06); no statistic uses μ or g. Render check: `smoke/render_check_full/f6_v3/`. Labels: constants `t_stop_s` (the legal stop, shared by the group), `v0`, `a`, `mu`, `direction`, `start`; per twin `t_stop_twin_s` (when it comes to rest, if it does) and an exact event_state.
- **Slide** (one block; twins identical before the branch): `legal` (same deceleration, stops, stays stopped); `reverse` (the same acceleration continued through v = 0: slides back; the naive constant-acceleration extrapolation, which friction cannot produce; creates energy, magnitude 1); `viscous` (drag ∝ v from the branch, matched to the legal deceleration there: v = v_e e^(−λτ), λ = a/v_e; creeps on, magnitude e^(−1) = the speed left at the legal stop); `decel_x{0.7,0.85,1.15,1.3}` (deceleration × k from the branch; an equality law: contrasts, not a kink; x0.85 + x1.15 is the speed-matched pair). violated_law `friction`. No `no_friction` twin (dropped 2026-11-12, user: its long glide set the camera span and shrank every other twin's effect). 300 groups.
- **Pair** (mass independence, from v1 F5): two blocks of the same size and visibly different material / density (the heavier is metal, mass ratio 2–5) side by side on the same surface, the same speed and deceleration at frame 0 (camera pitch 45–55° and a lateral gap that keep the two silhouettes apart). Twins `legal` (both stop together), `heavy_x13` (the heavier decelerates 1.3× from the branch; violated_law `friction`, statistic mass_ratio). 200 groups.
- **No legal-vs-legal control in F6:** with one surface and one μ the Coulomb outcome is unique given the shared pre-event frames (which show the deceleration), and twins of different groups do not share a context (a pair legal vs a slide legal is not a paired contrast). The pixel baselines and the speed-matched pairs (glide speed_x0.9 + x1.1, slide decel_x0.85 + x1.15) are F6's motion controls.
- **Glide (Newton's first law, law momentum):** one puck slides on frictionless ice, top-down; a branch on a frame mid-glide, nothing visible happens. Twins: `legal` (same velocity), `speed_x{0.7,0.8,0.9,1.1,1.2,1.3}` (|v′| = s|v|; equality law, so L2 reports contrasts, not a kink; s > 1 also creates kinetic energy with no force: `energy_physical` False, so L2 reports each half of the speed-matched pair separately), `turn_20` (same speed, direction turned 20°). violated_law `momentum`; clean_until_s = the last frame the puck is wholly in frame. 150 groups.

#### F7. Free fall, projectiles and bounces (P0, new)

The most-tested law across the literature (Principia, Morpheus, LikePhys, RigidBench, Kneifl et al.).

Gravity on, standard g. Sub-families, each with a branch frame in 112 to 128 (mid-flight):

| Sub-family | Legal twin(s) | Violating twins | Laws |
|---|---|---|---|
| **Projectile** | `legal` (closed-form parabola, floor bounce e_floor 0.8), `legal_e04` (bounce 0.4) | `gravity_x15` (g ×1.5 after the branch), `vx_x06` (horizontal velocity ×0.6 at the branch); violated_law `gravity` | gravity: g_ratio, axis_ratio |
| **Drop pairs** (from v1 F5) | `legal`, `legal_e04`: two balls of identical size and look, different density, released together | `heavy_x13` (the heavier ball's acceleration ×1.3 after the branch; violated_law `gravity`) | gravity: mass_ratio |
| **Floor bounce** | ε ladder `eps0.5`, `eps0.55`, …, `eps1.0` (first-bounce restitution; event = floor contact) | `eps1.05` … `eps1.5` (energy created) | energy: ε (L2 kink) |

Sizes: projectile 300 groups, drop pairs 200 groups, bounce 200 groups. Projectile launch angle and speed sampled so the full arc stays in frame. Later bounces at 0.8, after the clean window. Projectile / drop clean_until_s = the first floor contact or the last frame a ball is wholly in frame, whichever comes first (bounce: the second floor contact). Bounce framing: camera fitted per group so floor contact .. the ε = 1.5 apex fill 85% of the height; ball radius 0.17–0.22 × the drop height (≥ 50 px across at 832 × 480).

#### F8. Galileo's ramp (P0, redesigned 2026-11-11)

A frictionless cube is released from rest on ramp A at height h, descends (pre-event: the clip shows its own g and scale), crosses a dissipation pad on the flat (event = entering the pad; rendered identically in every twin; speed s → εs) and climbs ramp C to an apex at ε²h. Twins: the 21-rung ladder `eps0.5`, `eps0.55`, …, `eps1.5` (ε ≤ 1 legal, > 1 violated_law `energy`), byte-identical before the event; clean_until_s = the apex on ramp C. Geometry and framing (2026-11-12): the ramp surface is straight parts joined by concave fillets of radius 4 × the block's half-diagonal, rendered as one smooth swept mesh per ramp; the block rides it in exact two-corner contact (flush on straight parts, its bottom face a chord where it bridges a fillet: no corner ever enters the surface) and turns there, the rotation entering the energy (one-degree-of-freedom Lagrangian, `f8_ramp.Contact` / `motion`), so (|v|² + (I/m)|ω|²)/2 + g z is conserved off the pad. Every length scales with h (block half-size 0.38–0.45 h, ramps 36–50° / 45–60°, pad 0.7–0.9 h, the block flush over the whole pad) and the side camera is fitted per group (`framed_side_camera`) to the release point .. the top of the block at the ε = 1.5 apex (85% of the binding dimension), so the block is ≥ 70 px across at 832 × 480 and in frame from frame 0 to clean_until_s at every rung. PyBullet descent check (until the front corner reaches the fillet): ~1–2 mm. The old twins (slope B, kink, ×1.25, path dependence, time-rescaled speed ladder) are removed: each was the legal motion under gravity k²g, i.e. a playback-speed change. 250 groups. `physlaw/data/families/f8_ramp.py`.

#### F9. Pendulum (P2, built closed form)

Kubric's PyBullet bodies refuse every constraint, so there is no hinge: the bob (sphere) and the rod (a thin box, `aux`, massless in the physics) are keyframed from the exact large-angle motion (Jacobi elliptic, period 4K(sin²(A/2))/ω₀, ω₀² = gL/(L² + 2r²/5)). Side view, camera fitted per group to the pendulums' full range (swing: up to the ε = 1.5 apex) plus beam and damper (85% of the binding dimension); bob radius 0.15–0.2 L, ≥ 70 px across at 832 × 480. `physlaw/data/families/f9_pendulum.py`.

| Sub-family | Twins | Law |
|---|---|---|
| **pair** (150) | Two pendulums, same L, amplitude and phase, bobs of different density (heavier = metal), swinging from frame 0; branch on a frame at a common apex. `legal`; `period_x0.8`, `period_x1.25` (both periods ×k after the branch, same amplitude); `heavy_x0.8` (the heavier bob's period ×0.8) | oscillation: period_ratio, mass_period_ratio |
| **swing** (150) | Held at A₀, released; event = the bob entering a damper zone \|θ\| ≤ δ at the bottom (pad strip, identical in every twin; between frames, event_kind `swing`). A constant torque in the zone sets the next apex to ε²·h₀. Ladder `eps0.5`, `eps0.55`, …, `eps1.5` (21 rungs); ε ≤ 1 legal, > 1 violated_law `energy`. Later crossings at ε = 0.9; clean_until_s = the next damper entry. Labels `edit.apex_states` = the release apex and the first apex after the damper (v = 0) | energy: ε = √(h_apex′/h_apex), heights above the lowest point only (no g) |

#### F10. Rolling (P2, new)

Solid vs hollow cylinder released together down the same incline. Per group the RNG draws which object (A or B) is solid and the tube's inner ratio i ∈ [0.6, 0.95] (κ = (1+i²)/2 ∈ [0.68, 0.95]), so κ_A varies across groups (`constants.solid`, `constants.hollow`, `inertia_factors`). Legal: solid arrives first (rotational inertia). Violating: hollow arrives first, or both arrive together. 150 groups. Camera looks squarely across the lanes (45° pitch, no yaw, 200 mm lens at 80 m), so equal distance along the slab is equal screen x (perspective residual ≤ 2.3% of the offset from the image centre). Gate on L0: if angular velocity is not decodable (R² < 0.3) in V-JEPA 2, report F10 as L0-only.

#### F11. Static equilibrium / torque balance (P2, new)

A uniform block (half-length a, half-height b = βa) on a thin shelf, overhang x = (CoM distance beyond the edge)/a + 1, so the boundary (CoM over the edge) is **x = 1**; held by a prop that drops away at the event (a frame; same in every twin). Groups on the grid x ∈ {0.55, 0.65, …, 1.45} (symmetric about 1: across-group kink at x = 1). Twins `stay` (legal iff x < 1) and `tip` (legal iff x > 1); violated_law `statics`, magnitude \|x − 1\|. Tipping is closed form on a frictionless edge (Lagrangian in the rotation φ and the contact coordinate u; DOP853) until the edge reaction vanishes, then free flight; the reaction is vertical at the release, so \|a_cm\|/g = d²/(d² + (1+β²)/3), d = \|x − 1\| (in units of a). The violating tip at x < 1 mirrors the lever until the CoM passes over the edge, so tip kinematics are smooth in \|x − 1\| across the boundary. Draws in which the falling block would cut the shelf or the prop are rejected. The prop descends at 3g (`constants.prop_acc_over_g`); with the block's own free fall after it leaves the ledge it is the clip's g reference (the statics law never uses a metric g). The event state carries the exact initial CoM acceleration and angular acceleration. 220 groups. `physlaw/data/families/f11_support.py`.

#### F12. Angular momentum (P2, new)

Top-down frictionless table. A puck (disc, no spin) hits a free bar at rest off-centre (impact parameter 0.35–0.8 half-lengths) on its long face; event = contact (between frames, event_kind `contact`). Closed-form frictionless rigid impulse with restitution e. Bar marked (quadrant wedge) so rotation is visible. Twins `legal` (e = 0.9), `legal_e04` (e = 0.4), `spin_x{0.7,0.8,0.9,1.1,1.2,1.3}` (bar spin ×k, linear velocities legal: linear momentum conserved, L_ratio = 1 + (k−1)I_Bω′/L; equality law: contrasts, not a kink; k > 1 adds rotational KE and creates energy in ~half the groups at ×1.1 and nearly all at ×1.2 / ×1.3, flagged `energy_physical` False), and the **KE-conserving L violators** `spin_ke_x{0.8,1.2}` (audit M3; the primary F12 contrast): bar spin ×k, the spin energy taken from / given to the radial part (along r_A − r_B at the contact) of the relative translational velocity, so the CoM velocity (linear momentum), the orbital angular momentum and the total KE (= the legal twin's) are unchanged and only L breaks (L_ratio as spin_x k); draws whose radial relative KE cannot pay for k = 1.2 are resampled (~0.6 extra draws per group). The puck comes in 20–40° off the face normal towards the nearer bar end, so it slides off past the end; draws in which any twin would touch again before the clip end are rejected and the bar stays in frame to the clip end (the clip is physically clean throughout). clean_until_s = the last frame before the puck leaves the frame (clip end if never). 200 groups. `physlaw/data/families/f12_spin.py`.

#### Nuisance variants (all families)

For 20% of seeds per family, render an extra variant with randomized camera pose, lighting and floor/object textures, physics identical. Tagged `nuisance=1`.

### 2.4 Data acceptance tests

Existing F1 tests stay. For each new family add:
- Twins byte-identical in every frame before the event frame.
- Legal twins satisfy their law to closed-form precision (relative error < 1e-6 in simulator units).
- Each violating twin breaks its target law by the specified factor and breaks no other scored law beyond tolerance (or carries a flag).
- ε ladders (F2, F7 bounce, F8, F9 swing): every rung's realised ε equals its label to 1e-6; every rung shares the pre-event frames and `stop_dose`.
- F7 drop pairs and F3: per-object mean RGB inside masks matches within tolerance.
- F4: restitution and CoM-frame angle match across boosts to 1e-6; every camera-boost twin's projected ball centres match its ball-boost twin's to < 0.5 px in every frame up to clean_until_s, and its world states are the u0 twin's.
- Labels validate against the JSON schema.

---

## 3. Token projection and pooling (as implemented)

`tokens/project.py` maps each object's mask to each model's token grid (patch size, tubelet, resize). Pools (mean over tokens):

- **ROI per object:** disk at the projected centre, radius 1.5× projected object radius.
- **Pair ROI:** union of both ROIs.
- **Frame-wide:** all tokens.

Temporal windows: `pre`, `post`, `all`.

- **Contact:** both objects' last pre-event ROIs, dilated 2 tokens, held fixed after the event (local and leak-free).

Leakage rule: an ROI placed at an object's **true post-event location** leaks the outcome into a forecast readout, so L1 scores frame, contact and roi pools and every roi number is read beside its mask-only floor (§5.0.3 R3).

Mask-only floors (probes on the pooling mask itself) are reported for every ROI result.

---

## 4. Models

### 4.1 Adapter interface (as implemented)

```python
class ModelAdapter:
    def encode(self, video, noise_level=None) -> dict[int, Tensor]
    def forecast(self, video, context_end_frame) -> dict[int, Tensor]
    def surprise(self, video, context_end_frame, object_masks) -> dict[str, float]
    def generate(self, video, context_end_frame, seed) -> Video   # generators only
```

Implementation rules (as built):
- **Encoders: encode context frames alone.** Never use `transformers`' `VJEPA2Model.forward(context_mask=...)` for context, which encodes the whole clip and leaks the future.
- **DiTs: context latents come from VAE-encoding the context frames alone** (the VAE's temporal GroupNorm otherwise leaks post-event frames).
- **DiT encode():** noise to the timestep matching each SNR in {−20, −8, 0, +8} dB, one conditional pass.
- **DiT surprise():** the model's own training loss on post-event latents over the SNR timesteps × 2 noise draws (identical across twins), with clean context and with noisy context.
- **Two-pass caching for diffusion models:** `--no-forecast` caches encode + surprise; a later pass adds forecasts and generations without recomputing.
- Weights hash-verified; `HF_HUB_OFFLINE=1` for runs.

### 4.2 Models, grouped by the question each answers

Status: **DONE**, **RUNNING**, **ADD**. Priority as in Section 2.3.

#### Group 1. Does the training objective matter? (encoders)

| Model | Objective | Priority | Status | Notes |
|---|---|---|---|---|
| V-JEPA 2 ViT-L | Predictive latent | P0 | DONE | |
| V-JEPA 2 ViT-g | Predictive latent, larger | P1 | ADD | Scale within objective (Joseph et al. used L, H, g) |
| VideoMAE-v2-giant (pre-training checkpoint with decoder) | Pixel reconstruction | P0 | DONE | |
| V-JEPA 2-AC | Action-conditioned predictor | P2 | ADD | Encoder is the frozen V-JEPA 2 encoder; only the predictor differs. Passive scenes need a zero or null action, which is off-distribution. Run only L1/L2 with that caveat stated |

#### Group 2. Do results replicate on the literature's own models? (DiTs)

| Model | Used by | Priority | Status |
|---|---|---|---|
| CogVideoX-2b | Esmati, LikePhys, Seeking Physics | P0 | DONE |
| CogVideoX1.5-5B-I2V | Kneifl | P1 | RUNNING (pass 1) |
| **Wan2.1-T2V-1.3B** | Esmati (best model), LikePhys | **P0** | ADD |
| Wan2.1-T2V-14B | Kneifl (main model), LikePhys | P1 | ADD |
| Open-Sora 2.0, LTX-Video, HunyuanVideo | Kneifl, Esmati, LikePhys | P2 | ADD if time allows |

#### Group 3. Does physics post-training change internals? (pairs)

| Base | Physics-trained | Training signal | Priority |
|---|---|---|---|
| Wan2.2-TI2V-5B | PhysRVG (weights: HappyP4nda/PhysRVG) | RL reward for rigid-body collisions | P0 |
| CogVideoX-5B | VideoREPA-5B (released LoRA adapter, github.com/aHapBean/VideoREPA) | Aligns DiT token relations to VideoMAE-v2 | P0 |

Run L0, L1, L2, L2b on both members of each pair with identical settings. VideoREPA's teacher (VideoMAE-v2) is at chance on L2 in this benchmark; report whether the student gains law sensitivity the teacher lacks. Output track A: PhysRVG is natively video-conditioned (its v2v pipeline holds the first 2 latent frames = 5 frames clean), so it runs track A on exactly that context (Group 4); its base runs the same 2-latent clamp, flagged `track_a_mode: hack`, as the pair comparison. The CogVideoX-5B / VideoREPA pair is text-to-video: no track A (latent replacement only with `--allow-hack`).

Follow-up (not in this spec's milestones): a VideoREPA variant aligned to V-JEPA 2 instead of VideoMAE-v2, using VideoREPA's public training code.

#### Group 4. Output track A (scene continuation)

Track A needs a model that continues several REAL frames (velocities visible); a model conditioned on one frame cannot be track A. Every generator config states `output_track` (A / B / none) and `track_a_mode`: `native` (the model's own trained multi-frame conditioning), `variant` (an official frame-conditioned sibling of a text-only model), `hack` (conditioning the model was not trained for; flagged in every output record) or `n/a`. Stage 07 refuses track A for `none` / B models unless `--allow-hack`. Context = frames handed to the model, ending at the last pre-contact frame of its input window (`physlaw/models/track_a.py`; contact at master frame 120 in the examples).

| Model (config) | Mode | Context | fps | Priority | Notes |
|---|---|---|---|---|---|
| Cosmos-Predict2.5-2B Video2World (`cosmos_predict25_2b`) | native | last 5 frames (2 latents) of the whole pre-contact latents | 16 | P0 | Gated weights; own venv (diffusers ≥ 0.39) |
| PhysRVG (`physrvg`) | native | 5 frames = 2 latents (its trained v2v context), masters 105–117 | 16 | P0 | Paper: 15 fps; 15 does not divide the 48 fps master |
| Wan2.2-TI2V-5B at PhysRVG settings (`wan22_ti2v_5b_physrvg`) | **hack** | same 2-latent clamp | 16 | P0 | First frame only natively; pair comparison with PhysRVG |
| Wan2.1-VACE-1.3B / 14B (`wan21_vace_1_3b`, `wan21_vace_14b`) | variant | every pre-contact window frame (40, masters 0–117), mask 0; rest grey, mask 1 | 16 | P0 / P1 | Official siblings of Wan2.1-T2V-1.3B / 14B (text only) |
| LTX-Video 0.9.8 13B distilled (`ltx_video_13b_distilled`) | native | last 49 pre-contact frames (8k+1), masters 22–118 | 24 | P1 | Single pass at 480 × 832 (the card's multi-scale refinement skipped) |

Not track A: CogVideoX-2b / 5b, VideoREPA-5B, Wan2.1-T2V-1.3B / 14B, Wan2.2-TI2V-5B (own settings) are `output_track: none` (latent replacement, gate 0/20 on CogVideoX-2b, only with `--allow-hack`).

#### Group 5. Output track B (self-consistency from frame 0)

| Model | Priority | Status |
|---|---|---|
| Wan2.2-I2V-A14B | P0 | Started (22 clips). Finish 20 groups × 4 seeds |
| Wan2.1-I2V-14B | P1 | ADD |
| CogVideoX1.5-5B-I2V (official I2V pipeline, `track_b_adapter`) | P2 | ADD |

#### Group 6. Controls (Section 4.4)

| Control | Priority | Status |
|---|---|---|
| Untrained V-JEPA 2 (same architecture, random init) | P0 | DONE |
| PC1: small predictive transformer trained on the render distribution | **P0, critical path** | ADD |
| PC2: a DiT fine-tuned on simulator data | P1 | ADD |

### 4.3 Checkpoint pinning

Pin exact repo IDs, revisions and sha256 for every model in its config, as done for VideoMAE-v2.

### 4.4 Positive controls

**PC1. Small predictive transformer (critical path).**
- Architecture: V-JEPA-style encoder + predictor, ViT-S scale, same input format as V-JEPA 2.
- Training data: a **separate training render set** (never the benchmark's evaluation twins): legal clips only from F1, F2, F3, F4, F7 and F8, at least 3,000 clips per family. Lower resolution (256 × 256) and lower samples per pixel are allowed for this set.
- Objective: masked latent prediction, with future-frame masking so the predictor learns to forecast past the event frame.
- Report training curves and in-distribution prediction error.
- Pass criterion for "the test is passable": §5.0.5 (momentum: L1 S_mom and the L2 win rate; energy: PC1 kink K with CI > 0 beyond every baseline on F2 or F7 bounce). If PC1 fails, the test is too hard or broken: stop and report before claiming that any large model fails.

**PC2. Simulator-fine-tuned DiT.**
Fine-tune Wan2.2-TI2V-5B (or CogVideoX-2b if memory requires) on the PC1 training render set, following RigidBench's recipe (first-frame pinning; full fine-tune or rank-32 LoRA). Shows whether a full-scale generator can pass when trained on the right data.

---

## 5. Evaluation levels

### Shared protocol (as implemented)

- Ridge probes, penalty by nested CV on training folds. Features standardised, targets centred on the training fold; a multi-output R² is the mean of the per-output R² (targets standardised per output, `probes.ridge.r2_score`), so no output is weighted by its variance.
- Cross-validation grouped by twin group ID; 12 fold seeds; with fewer than 100 groups, 5-fold × 4 seeds cross-fitting is allowed and must be stated.
- Permutation null: 500 shuffled refits, max over the searched grid (layers, pools, noise levels), corrected p-values.
- Kinematic observers fit through the identical pipeline (`obs_vel`, `obs_speed`, `obs_quad`, `obs_normal`).
- **Selection rule (new):** whenever a score is reported for a best layer, pool or noise level, select it on held-out groups and report the score on different held-out groups. Report the number of variants searched.

### 5.0 Law statistics and confound rules (v3, 2026-11-11; supersedes 5.1–5.6 and §6 where they conflict)

**Goal.** Every number answers one question: does the model represent the law, beyond what any correlate of the law
(speed, position, layout, playback speed, familiarity, the measuring probe) would produce?

#### 5.0.1 Nine laws

Every law is reduced to **dimensionless statistics that are ratios of one kinematic quantity measured twice by the
same instrument in the same clip.** Theorem the code tests: each statistic is unchanged by (i) a length scale
(metres ↔ pixels), (ii) a time scale (fps, playback speed, slow motion), (iii) a common gain on every measured vector
(uniform probe shrinkage). So no statistic needs metric calibration, and none can be passed by a playback-speed prior.

| Law (module) | Families | Statistic(s) | Legal | Merged from |
|---|---|---|---|---|
| **momentum** (collision.py) | F1–F4; F6 glide | η = \|q_A+q_B\|/(\|q_A\|+\|q_B\|), q_i = m_i Δv_i; mass-free: antiparallel = −cos(Δv_A, Δv_B), normal alignment \|Δv·n\|/\|Δv\|; Newton I on a force-free segment (ctx force_free): inertia_speed = \|v′\|/\|v\|, inertia_turn = ∠(v, v′)/π | η = 0; antiparallel = 1; alignment = 1; inertia_speed = 1, inertia_turn = 0 (null: Aristotelian stop, v′ = 0) | momentum, CoM (same Δp), angular momentum about contact (= normal alignment), inertia |
| **energy** (collision.py) | F1–F4 collisions, F7 bounce, F8 ramp, F9 swing | generalised restitution ε of the exchange event; ρ\* = ε² (fraction of the exchanged energy returned) | ε ≤ 1 (one-sided) | energy, bounce height ratio, ramp bottom speed, pendulum damper |
| **gravity** (freefall.py) | F7 projectile/drop | g_ratio = a_post·ĝ / a_pre·ĝ; axis_ratio = \|v_h′\|/\|v_h\|; mass_ratio = a_heavy/a_light along ĝ; fit windows (L1): free token-frames only (after the release, every ball in frame, last master frame ≤ clean_until_s), ≥ 3 per side | each = 1 (null: Aristotle's fall, a′ = 0 and v_h′ = 0: g_ratio 0, axis_ratio 0; mass_ratio m_heavy/m_light) | gravity, axis independence, mass independence |
| **friction** (freefall.py) | F6 slide / pair | windows (`io.friction_windows`): pre = before the branch; moving = [branch, min(legal stop, the twin's own stop) − 0.1 s]; stop = [legal stop + 0.1 s, clean_until_s]. decel_const = Σ(a·v̂)_moving / Σ(a·v̂)_pre; stop_hold = max_i \|v̄_stop,i\| / \|v_pre,i\| (v̄_stop = mean velocity over the stop window, v_pre = at the branch); mass_ratio = (a·v̂)_heavy / (a·v̂)_light while moving | decel_const 1, stop_hold 0, mass_ratio 1 (null: Stokes drag F = −cv matched at the branch: decel_const = e^(−cΔt/m) < 1, stop_hold = e^(−1) for one body, mass_ratio = m_light/m_heavy). Twins: reverse stop_hold > 0 with decel_const 1; viscous both off | Coulomb friction (constant deceleration, a stop that holds), mass independence of slides (moved from gravity) |
| **oscillation** (freefall.py) | F9 pair | period_ratio = T_after/T_before (geometric mean over bobs; T = 2 × the interval between passes through the bottom); mass_period_ratio = T_heavy/T_light after the branch | each = 1 (null: heavier swings faster, T ∝ m^(−1/2)) | pendulum (revived) |
| **statics** (incline.py) | F11 | tip_consistency = [(φ > 0.05 rad) == (x > 1)], φ the block's rotation angle (radians) over the tip window after the release, T_w = min(2√(a/g_clip), time until the block leaves the ledge) in the clip's own time unit (no g_clip: the twin-shared time to the clean-window end), so φ is time-scale invariant; alpha_ratio = α / α_legal for x > 1 (NaN for x ≤ 1), α the initial angular acceleration, α_legal = g_clip·d/(d² + (1+β²)/3)/a, d = \|x − 1\|, g_clip a free fall measured in the same clip (the block after it leaves the ledge, else the prop's 3g descent / 3; none → alpha_ratio NaN, only tip_consistency scored), a the half-length in the same units | 1; 1 (null: stays, alpha_ratio 0) | torque balance |
| **angular_momentum** (collision.py) | F12 | L_ratio = L_z after / before about the system CoM (Σ m(r−R)×(v−V) + Σ Iω; masses and inertias from the labels) | 1 (null: rotation ignored, the legal impulse without the bar's spin) | angular momentum (new) |
| **rolling** (incline.py) | F10 | order = log(a_s/a_h) / log((1+κ_h)/(1+κ_s)); slip = \|ω\|·r/\|v\| per object (rolling without slipping; both sides rad/time) | order 1 (null "inertia ignored" = 0); slip 1 (null "no spin" = 0) | rolling (+ rotation, 2026-11-11) |
| **galilean** (collision.py) | F4 | Δε and Δθ_cm between a boosted clip and its unboosted partner | 0 | galilean |

F9 is rebuilt in closed form (no hinge constraint). **ε per event kind** (n = event normal, A → B):
collision ε = −(v_A′ − v_B′)·n / (v_A − v_B)·n (mass-free); bounce ε = −v′·ĝ / v·ĝ (floor = infinite mass);
pad (F8) ε = s′/s, speed after / before the dissipation pad; swing (F9) ε = √(h′/h), h = (p − p_bottom)·(−ĝ) the
apex height above the lowest point, read at the apexes before and after the damper (v = 0, so no g enters). No
statistic uses the scene's g: statics measures g_clip in the clip. Gain exceptions to the invariance theorem (a
measured term vs a differently measured one): rolling slip, L_ratio, statics alpha_ratio. With momentum conserved,
KE − KE′ = ½μu_n²(1 − ε²), so ρ\* = ε² is the internal (CoM-frame, normal) energy ratio: boost-invariant and
mass-free, unlike KE′/KE (which changes under a boost and is diluted by CoM motion). Never use KE′/KE as a score.

#### 5.0.2 Energy data (all three families share one design)

Pre-event phase shows the scene's own scales (approach speeds; the fall; the descent), so g and fps are calibrated in
the clip. A visible exchange event (contact; floor contact; a pad on the run-out, rendered identically in every twin)
licenses an abrupt change. Twins form a **ladder ε ∈ {0.50, 0.55, …, 1.50}** (F7 bounce, F8, F9 swing: 21 rungs, twins eps0.5, eps0.55, …, eps1.5; F2: 0.0 … 1.5) with identical pre-event
frames; ε < 1 is legal (dissipation), ε > 1 creates energy; **the law boundary ε = 1 is interior**. Inside the clean
window (below) post-event kinematics are affine in ε for collisions (F2: Δv ∝ 1 + e) and the F7 bounce (v′ = ε|v|, then
ballistic), and smooth (analytic) but not affine for the F8 pad and the F9 swing (the pad / damper acceleration is
∝ ε² − 1, and the pendulum after the damper is a nonlinear function of its energy); either way every kinematic
correlate is smooth through ε = 1, which is what the kink needs (its curvature trend absorbs smooth curvature).
* **F8 is redesigned as Galileo's ramp**: release from rest on ramp A (height h), descend (pre-event), cross the pad
  (event; speed s → εs), climb ramp C to apex ε²h. Old F8 twins (time-rescaled speed ladder, path dependence) are
  removed: each was the legal motion with gravity k²g, i.e. a playback-speed change, so energy was not identifiable.
* **Clean window.** Labels record per twin `clean_until_s` = the first secondary event after the exchange (an object
  leaves the frame, a second contact, the end of the climb / apex+, a wall). Per group, the analysis window is
  [event, min over rungs of clean_until_s]. Labels record per object `stop_dose`: the ε at which |v′| is minimal (a
  ball stops; |v′| kinks there even though v′ is smooth). Groups with |stop_dose − 1| < exclude_radius are excluded
  from kink statistics (reported separately).

#### 5.0.3 Probe confound rules (L0, L1, L3, L4)

* **R1 Same instrument.** Both terms of a ratio are decoded by the same linear map: one probe per (object, quantity)
  trained on real pre- AND post-event windows jointly. Ridge shrinks toward the mean (v̂ = μ + γ(v − μ)), so the
  statistic is computed after a fold-proper INVERSE calibration of the probe on real clips (fitted on training
  groups' inner out-of-fold predictions: ŷ ≈ c + d·y, decode (ŷ − c)/d, so E[decode | y] = y; the forward fit
  y ~ ŷ would leave a ρ² shrinkage toward the mean; the inverse map removes the shrinkage in expectation but
  amplifies the noise by 1/d, and it is applied per output only where d is significantly positive, t = d/se(d) ≥ 3
  with se clustered by group; a weaker output keeps the raw, shrunk ridge prediction). L3 mass independence uses the same instrument (one map for
  both objects).
* **R2 Floor.** Every decoded statistic is reported beside the same statistic decoded by the same probe from the real
  frames of the same clips ("floor"); scores are relative to the floor. A model whose floor cannot measure the
  statistic (floor error above the threshold in the config) is "unmeasurable" for that law, not "fails".
* **R3 Object patches with their mask floor.** ROI (object-patch) pools are used alongside frame and contact pools,
  and every patch result is reported beside the mask-only floor (the same probe on the pooling mask alone), which
  measures how much the patch's placement alone gives away. L1 roi scores (S_mom, creation rate) must BEAT the same
  score computed from the mask-only features (verdict `beats_mask_only`); frame / contact pools are unaffected.
* **R3b Selectivity (Hewitt & Liang).** Every probe target is also fitted on two control tasks that keep the data's
  structure but no physics — values permuted across twin groups as whole blocks (scene control) and a fixed random
  value per object identity (object control). Selectivity = real held-out R² − control R²; a high control R² means
  the probe reads identity, not the quantity. L4 also reports the swap test (interchange-intervention accuracy with
  non-target selectivity) and steering reliability (per-scene slopes and wrong-way share, distance of steered tokens
  from the natural activation distribution, steering vs showing the same Δθ through the input).
* **R4 Reference models.** A law claim needs the model above the untrained model and the real-frame floor's
  measurement error. (Cut 2026-11-11 by the user: the no-event observer rows — covered by the interaction gate and each
  law's null — L0 kinematic observers, permutation nulls — covered by held-out selection + group-bootstrap CIs — and
  the L1 pre-contact reference rows.)
* **R5 Energy evidence is probe-free first.** A one-sided inequality (ε ≤ 1) is satisfied by any dissipative or static
  forecast, and a threshold on a monotone dose (ε > 1) is decodable from any correlate of the dose. So L1 energy is
  necessary-not-sufficient (creation rate, two-sided tails, expected ε̂), L2b does not score energy, and the primary
  energy evidence is the kink (behavioural L2, representational L2-rep), which no smooth correlate can produce.

#### 5.0.4 Scores per level

Laws per family everywhere: F1–F4 momentum + energy (F4 also galilean); F6 slide / pair friction, F6 glide
momentum (Newton I: inertia_speed, inertia_turn); F7 projectile / drop gravity, F7 bounce energy; F8 energy; F9 pair
oscillation, F9 swing energy; F10 rolling; F11 statics; F12 angular_momentum (+ η). Objects labelled `aux` (F9 rods,
F11 prop) are never probe targets or scored objects.

* **L0** decodability of every ingredient with R1–R4 and selectivity (pre and post velocity with one probe; height for
  F7/F8; bob angle and ω for F9; block CoM acceleration and ω for F11; bar ω for F12; ω for F10; F6 block deceleration along the motion before / after the branch), mask floor, untrained
  model, one-twin refit. **Mass before the event** (`mass_pre_event`): pre-event log_mass (F10: κ = I/(m r²) per
  cylinder) is decodable when its held-out R² CI > 0, every selectivity > 0 and R² beats the untrained model's;
  mass-independence scores (gravity / friction mass_ratio, oscillation mass_period_ratio, L3 mass independence, L4
  log-mass steering) and rolling scores of a model without it are "uninformative" (a mass-blind model passes them).
  The same-instrument ε check uses the labelled event-instant velocities.
* **L1** (decoded forecast; one forecast per group and pre_segment): momentum S_mom (F1, F2, F4; dv = decoded post −
  decoded pre velocities, one calibrated instrument; the true-pre mix is a diagnostic). **F3 has no L1 momentum
  score**: its five mass ratios share look and pre-event frames (masses by density), so one forecast serves all of
  them; the mass-free checks (antiparallel, normal alignment) cannot see a momentum edit and are sanity checks only;
  F3 is an L2 / L4 family. Energy creation rate P(ε̂ > 1 + τ), lower tail P(ε̂ < ε_lo,g − τ) with ε_lo,g = per set
  the lowest floor-decoded ε of its gated legal twins (the floor of the same set: the forecast is shared by every
  rung, so a per-twin reference would score knowledge of the hidden rung), median ε̂ (expected ε) — all forecast vs
  floor. ε estimators (no window averages): collisions from the pooled pools (constant velocity on both sides);
  F7 bounce from per-token-frame velocity lines over the free frames on each side (after the release, ball in
  frame, post up to the second contact; the forecast up to the group's earliest), both extrapolated to the contact;
  F8 pad and F9 swing from decoded apex heights, ε = √(h′/h) (the forecast read up to the group's latest apex).
  Ratio laws (gravity, friction decel_const / stop_hold / mass_ratio, oscillation, rolling, angular_momentum):
  S = 1 − |log T_fc − log T_legal| / |log T_null − log T_legal| per statistic, floor-relative; statistics whose null is 0
  or whose legal value is 0 use the linear form 1 − |T − T_legal| / |T_null − T_legal| (friction stop_hold: legal 0,
  null e^(−1)); statistics already centred on 1 (glide inertia, statics alpha_ratio) use 1 − |T − 1|; statics tip_consistency is
  an accuracy. Per clip S is clipped to [−1, 1]; a decoded T that is non-finite, or ≤ 0 in the log form, where the
  labelled statistic is defined scores −1 (never dropped). Rolling and F12 read v and ω with one instrument per object;
  statics φ is integrated over exactly the tip window (end interpolated), g_clip for alpha_ratio from the decoded free
  fall where the block leaves the ledge in the clean window (else labelled). **Negative-control forecasts** (every
  law): (i) the last context token-frame's features repeated (pooled collisions: the pre-event pool), (ii) another
  set's forecast, (iii) another set's real post-event features (same twin type), decoded by the identical instrument
  and scored with the same statistic; a law claim (json `verdict.claim_allowed`) needs the forecast to beat all three
  (paired group-bootstrap CI of the difference > 0) and, for roi, the mask-only floor; otherwise "probe fills in".
  Where a statistic is scene-independent (gravity g_ratio: one g for every set) (ii) and (iii) obey the law by
  construction and L1 cannot support the claim. Forecast-dependent gates (momentum interaction, energy gates):
  report the pass share and a sensitivity score counting removed clips as failures (S −1; creation 1).
* **L2** (zero-fit surprise, clean window): contrasts per violating twin vs legal (momentum, gravity (F7 projectile gravity_x15 and vx_x06, drop heavy_x13;
  no F7 legal-vs-legal control: `legal_e04` and `legal` are identical inside the clean window), F6 glide speed /
  turn with speed-matched pairs 0.9 / 1.1, F6 slide reverse (primary) / viscous / decel_x0.85 + x1.15
  (speed-matched) / decel_x0.7, x1.3 and pair heavy_x13 (F6 has no legal-vs-legal control, §2.3), F9 pair period / heavy, F12
  spin_ke_x0.8 + spin_ke_x1.2 (primary, KE-conserving) and the spin_x ladder, with `legal_e04` as
  the legal-vs-legal control), speed-matched violators and the model − pixel-baseline win rate as the motion control.
  A contrast with several violators (a speed-matched pair) is also reported per half (`[half <twin>]`), and every
  contrast records the share of groups whose violator also creates energy (`energy_physical` False: F6 glide
  speed_x > 1, F12 spin_x > 1, F2 `*_pos`, F6 reverse), flagged in the md (audit M3);
  energy **kink** K on the ε axis (F2, F7 bounce, F8, F9 swing) with pixel baselines; inconclusive when the model's
  expected dose x̂ is within exclude_radius of 1. x̂ is fitted once per group and source, leave-one-group-out over the
  groups kept by the stop / minimum rules, and that same x̂ drives both the speed-match exclusion and K's x̂
  neighbourhood (`l2_surprise.logo_x_hat`). F11 is an **across-group kink** on overhang x of the per-group contrast
  c = log S(violating) − log S(legal) (a log ratio: scale-free); the mean curve c̄(x) is standardised over the dose
  grid (zero mean, unit sd) before D²; the side means use raw c. The law predicts c > 0 on both sides of x = 1 (the
  violating twin switches there); a motion correlate is linear through 1.
* **L2-rep** (energy, replaces L2b for energy): the kink statistic on the ladder trajectory of post-event activations,
  K_rep = (‖D²a(1)‖ − median ‖D²a(x)‖) / median ‖D²a(x)‖ over interior x away from 1 and x̂, D²a(x) = a(x+h) −
  2a(x) + a(x−h), layer selected on held-out groups; baselines: raw pixels, label kinematics, untrained model.
* **L2b** momentum (incl. F6 glide), gravity (F7 projectile / drop), friction (F6 slide / pair), oscillation (F9 pair), rolling (F10), angular_momentum (F12) only, with
  the kinematic baseline and speed-matched violators. Verdict `detected` only if the selected cell's AUC CI is above
  0.5, its paired CI over the kinematic baseline is above 0 and, for a selected ROI cell, its paired CI over the ROI
  mask floor (the same probe on the pooling weights alone) is above 0 (audit M0).
* **L3** Galilean: ε and θ_cm computed from decoded velocities (R1, probes valid on boosted real clips by the floor),
  never from probes trained to output ε. The law test reads the model's **forecast** (audit H2): per forecast space
  one R1 velocity instrument per object decodes the real pre-event velocities and the forecast's post-event pools of
  each boost level (and cam_<level>); d_ε, d_θ of every boosted forecast vs its u = 0 forecast (and camera vs ball
  boost) are compared, pair by pair, with the same d from the real post-event frames (the measurability floor; real
  videos are invariant by construction): not invariant when d(forecast) − d(floor) has a scene-bootstrap CI above 0
  (signed ε shifts also reported). Mass independence: gravity mass_ratio (F7 drop), friction mass_ratio (F6
  pair), oscillation mass_period_ratio (F9 pair), floor-relative; the forecast's score through ROI pools must beat the
  same score from the mask-only instrument (ROI pooling weights alone): S − S_mask CI > 0 (audit M0). Transfer: clean →
  nuisance (F1, F7).
* **L4** one manifold-steering routine per law (precondition, steer, swap test, patch arm, steering reliability,
  controls): momentum mass-direction steering; energy restitution steering (the law predicts ε̂ saturates at 1, beyond a
  control direction's saturation and passive propagation); friction θ = log mass (F6 pair; slope 0 on the deceleration after the branch);
  oscillation θ = log bob mass (period slope 0); statics θ = overhang x (tip probability steps at x = 1); angular
  momentum θ = impact parameter / bar inertia (L_ratio stays 1); rolling θ = κ_A.
* **L4 localization** (`localize`, json `localization`; only for a model whose steering or swap test passes, or whose
  L2 / L2-rep kink is "beyond baselines", else the reason is recorded): (1) swap IIA per layer (DiTs: per (SNR, layer);
  DiT momentum also the live patch per block × SNR) with group-bootstrap CIs vs random direction, shuffled targets,
  a size span fitted beyond θ, passive, untrained → peak and flagged layers; (2) the swap per token group (ROIs,
  contact, background, pre / post) at the top layers; (3) per head / MLP: attribution-patching pre-screen, exact
  patching on the top-k, mean-ablation knock-out re-scored on the law (dependence on θ; energy creation rate and
  saturation; invariance), off-distribution distance; (4) DAS (k ∈ {1, 4, 16}) at the top layer, held-out IIA vs k,
  principal angles to the curve. Localized = swap IIA AND knock-out both beat their controls. Caps in the config.
* **Outputs** score the same statistics from tracked videos (scale-free: no true g, fps or metric speed enters).

#### 5.0.5 Controls

Kept (one per confound): untrained V-JEPA 2-L (every level); PC1 and PC2 (§4.4); real-frame floor and same-instrument
decoding; mask-only floor for patch pools; L1 negative-control forecasts (last context frame repeated, another set's
forecast, another set's real post-event features); L0 pre-event mass / κ decodability beside every mass-independence
and rolling score (leaderboard flag "uninformative"); held-out selection + group-bootstrap CIs; mass-blind rule (S_mom);
pixel baselines (copy-last, constant velocity) and speed-matched violators (both drift signs) as the L2 motion
controls; legal-vs-legal contrast (F1, F2, F12, F4 per boost level; none in F6, F7, F9, F10, F11); the inconclusive flag; twins, ε ladders, clean window, stop-dose exclusion and
scale-free statistics (design); L2b kinematic baseline with the paired CI; L2-rep pixel, label-kinematics and untrained
baselines; L4 passive propagation, one norm-matched random direction, and the speed direction for restitution
steering; output readout validation on real clips.
Cut (2026-11-11, user): no-event observer rows, L0 kinematic observers, permutation nulls, L1 pre-contact references,
the L2 β motion adjustment, runtime kink comparison observers (novelty, speed-only, motion, law: now unit tests that
assert their by-construction values), L4 random-subspace and PC-matched controls. Pass criteria: momentum (L1 S_mom, L2 win rate) and **energy: PC1 kink K with
CI > 0 beyond every baseline on F2 or F7 bounce, untrained K CI containing 0**.

### 5.1–5.6 Levels (pointers; §5.0 is authoritative)

* **L0 States** (`eval/l0_states.py`): §5.0.3–5.0.4. Also required: the noised-input-latent baseline for DiTs (probe on
  the model's own noised VAE latent, same pooling) and, for Wan2.2-TI2V-5B, the velocity protocol check against
  RigidBench's (unregularised linear probe on actor-mask-averaged tokens).
* **L1 Forecast law consistency** (`eval/l1_forecast_laws.py`): instrument P-pred-real (the model's own forecast-space
  outputs for masked real post-event frames; primary wherever the cache has it) or as_is (one probe on pre and post rows
  of the same layer); moment-matched forecasts are a diagnostic only; P-forecast (a probe trained on forecasts) is never
  used. S_mom is primary for momentum; S_family (distance to the legal collision family) is secondary. DiTs: one space
  per (SNR, layer), selected on held-out groups.
* **L2 Violation sensitivity, zero-fit** (`eval/l2_surprise.py`): `surprise()` on the clean post-event window for every
  twin with identical context, masks and noise (the object region is one per group: the union of the group's twins'
  object masks over the clean window); contrasts, kinks and the motion controls of §5.0.4–5.0.5 (paired win
  rate, bootstrap CI, Wilcoxon, Cliff's delta; Holm within family).
* **L2b Trained violation detection** (`eval/l2b_detection.py`): logistic probe on post-event activations (roi and
  frame pools), grouped CV, the kinematic baseline and the paired CI.
* **L3 Invariance** (`eval/l3_invariance.py`) and **L4 Causal propagation** (`eval/l4_causal.py`): §5.0.4. For DiTs, L4
  also generates after the intervention and tracks the result with the output readout.

---

## 6. Output-level evaluation

Readout (`eval/outputs.py`): a colour tracker initialised from the real context (blob splitting, shadow folding,
rejections for size change, extra objects, scene cuts, jitter, short windows), plus a marker-angle tracker for rotation
(F10 wedge, F12 bar). Each family's readout is validated on real legal and violating clips before use
(`scripts/09_eval_outputs.py`). Statistics are those of §5.0.1, measured from tracks (scale-free; g only where measured
in the clip).

* **Track A, scene continuation**: generate from the real pre-event context and score with the same laws as L1. A
  feasibility gate first (20 clips; < 50% measurable → report the gate and skip extra seeds). Report `track_a_mode` with
  every result; keep `hack` rows apart.
* **Track B, self-consistency from frame 0**: generate from frame 0 with the minimal prompt; score whether the generated
  event obeys each law given its own generated pre-event state, with mass-free tests (Δv antiparallel along n, ε ≤ 1);
  strict and loose gates.
* External output benchmarks (optional): Principia / Morpheus on Group 2 and 3 models where phenomena overlap.

**Key analysis:** correlation across models, and within each post-training pair, between internal scores (L1, L2 win
rate, L2b) and output scores.

---

## 7. Statistics

- Bootstrap 95% CIs over twin groups.
- Holm correction across contrasts within each family and level.
- Every "best" variant follows the held-out selection rule (Section 5).
- Report the number of groups, test groups per fold and variants searched for every number.

---

## 8. Results format

Per model: `results/<model>/<family>/<level>.{json,md}` with config hash, checkpoint, metrics, CIs, nulls, and the layers, pools and noise levels used.

Leaderboard (`scripts/11_aggregate.py`): one row block per law (momentum, energy, gravity, friction, oscillation, rolling, statics, galilean, angular_momentum) with L0 R², the L1 score, L2 win rate / kink K, L2b AUC, L3 invariance, L4 steering, output scores (tracks A and B).

Figures:
1. L0 vs L1 per model: states decodable, laws weak.
2. L2 model vs pixel-baseline win rate: the motion confound.
3. L2 win rate vs L2b AUC.
4. Galilean invariance error vs boost size.
5. Internal vs output adherence scatter.
6. Post-training pairs: base vs physics-trained deltas on every level.
7. Objective comparison: predictive (V-JEPA 2) vs reconstruction (VideoMAE-v2) vs denoising (DiTs) across laws.
8. Positive control vs large models on the same axes.

---

## 9. Engineering requirements

As in v1, plus:
- Diffusion models run in two passes (Section 4.1).
- Each new family has its own data config and acceptance tests before any model runs on it.
- Every stage records the git hash, config hash and seeds.
- `docs/RESUME.md` is updated after each milestone with results and any deviation from this spec.

---

## 10. Milestones

### Done (keep)

| # | Milestone |
|---|---|
| 1 | F1 generator, twins, acceptance tests; 64 groups rendered |
| 2 | V-JEPA 2 adapter, token projection, overlays, activation caching |
| 3 | L0 + L2 on V-JEPA 2 and untrained control (64 groups) |
| 4 | L1 on V-JEPA 2 (64 groups) |
| 6 | VideoMAE-v2-giant: L0 to L2 |
| 7 (part) | CogVideoX-2b: L0, L1 (preliminary), L2, output gate; track B readout validated; Wan2.2-A14B 22 track B clips |

### Next

The ordered run plan lives in `docs/RESUME.md` §5.

---

## 11. Open decisions for the user

- [ ] **F1 size.** RESUME records a 2026-11-09 decision to keep F1 at 64 groups. This spec recommends 400 because CIs at ~9 test groups per fold are wide and PC1 and L2b need more data. Confirm 400, or pick a number (minimum recommended: 200).
- [ ] PC1 pass criterion values (§5.0.5): accept or adjust.
- [ ] PC2 base model: Wan2.2-TI2V-5B or CogVideoX-2b.
- [ ] F9 and F10: include or drop.
- [ ] Target venue and deadline, which sets how far down the milestone list to go.
