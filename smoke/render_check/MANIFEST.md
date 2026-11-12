# PhysLaw render check (2026-11-11)

One twin group per family / sub-family, rendered to eyeball the scenes. Built by `make_videos.py` (viewers) and `make_manifest.py` (this file, from `<family>/viz/summary.json` + `notes.json`).

**Format (reduced, not the master):** 416x240 at 32 spp, otherwise the master timing: 48 fps x 289 frames (6.0 s), same seeds, so the same scenes the full-res dataset would get. `nuisance_frac=0` (no nuisance variants). GPU: OPTIX on the A100 (`render_device` in every label). Every twin of every group was rendered (no subsets; the 21-rung ladders were cheap at this format). F1 was NOT re-rendered: its viewers are built from `data/f1/f1_100000` (832x480, master format).

Render command (closed-form families F6-F12): `smoke/render_check/run_job.sh <config-stem> <seed> <out>` = `KUBRIC_USE_GPU=1 physlaw/data/generate.py --config configs/data/<config>.yaml --seed <seed> --out smoke/render_check/<out> --set nuisance_frac=0 master.width=416 master.height=240 master.samples_per_pixel=32`. Collision families F2-F4: `run_job_cf.sh`, the same plus `master.sensor_width_mm=41.6` (see the finding below). Seeds: first seed of each sub-family block (F7 projectile 700000 / drop 700300 / bounce 700500; F9 pair 900000 / swing 900150); F11 seeds 1100002 (x = 0.65) and 1100000 (x = 1.35), picked with `--no-render`.

**Viewers:** per twin `<family>/viz/<group>__<twin>.mp4` (H.264 yuv420p, 640 px wide, 48 fps, twin name + [legal / violated law] top-left, frame number bottom-right, red border + yellow `EVENT f<n>` for 6 frames from the event frame). Per group `<family>/viz/<group>__compare.mp4`: every twin tiled, frame-synced, legal twins first. All comparison files are < 0.4 MB.

**Automated checks (`auto_check.py`, decoded mp4s):** in every group, every twin's frames before `first_post_contact_frame` are byte-identical to the first twin of its `pre_segment` (F4: per boost). Long static stretches after the event appear only where physics says so (friction cubes stopped, the `stay` block, a tipped block after leaving frame, bounce balls at rest).

## Finding: reduced-resolution collision scenes render 2x zoomed (code inconsistency, not hit by the master format)

`physlaw/data/generate.py:96` (`master_fmt`) defaults `sensor_width_mm` to `36 * 832 / 720` = 41.6 mm (used for labels / K / closed-form cameras), while `physlaw/data/kubric_scene.py:308` (`configure`) defaults it to `36 * W / 720`. With `master.width != 832` and no explicit `master.sensor_width_mm`, the F1-F4 render has half the field of view of its labels: at 416 px the seg centroids sat at 2x the label pixel offset from centre and seg areas were 4x the projected disc (f2_200000 e1.0, frame 60: label B (209, 88), seg (210, 55)); in-frame acceptance also uses the zoomed FOV, so a different draw gets accepted. Closed-form families (F6-F12) are unaffected (their camera takes `fmt`). The master format (832) is unaffected (both defaults are 41.6 mm), and so is PC1 (`pc1_train.yaml` sets `master.sensor_width_mm: 41.6`). Any other reduced-res collision render must pass `master.sensor_width_mm=41.6`. The first F2-F4 renders here showed it; they were deleted and re-rendered with the override (verified: f3 ratio_1 seg centroid (196.3, 97.4) vs label (196.8, 97.9), area 374 = label count). Not fixed in code (outside this task's scope).

## f1

* group `f1_100000`, 5 twins, source `data/f1/f1_100000`, render 832x480 @ 48 fps
* comparison: `smoke/render_check/f1/viz/f1_100000__compare.mp4` (0.10 MB, grid 3x2, tile 416x240); per-twin mp4s next to it (`<group>__<twin>.mp4`)

| twin | legal | violated_law | magnitude | epsilon | event_kind | event frame | clean_until_s |
|---|---|---|---|---|---|---|---|
| conserve_e1 | True | - | - | 1 | - | 124 | - |
| conserve_e015 | True | - | - | 0.15 | - | 124 | - |
| energy_create | False | energy | - | 1.5 | - | 124 | - |
| mom_violate_e1 | False | momentum | - | 1 | - | 124 | - |
| mom_violate_e015 | False | momentum | - | 0.15 | - | 124 | - |

**Looks right.** Spheres approach, touch at f124, and separate; the energy_create / mom_violate twins separate differently. Spheres are small (~30-40 px at 832) and slow but stay in frame. F1 labels use the older `physlaw-labels-1` schema (no `legal` / `violated_law` / `event_kind` / `clean_until_s`); the legal / law columns were derived from `momentum_conserved` and `e_target`.

## f2

* group `f2_200000`, 23 twins, source `f2/f2_200000`, render 416x240 @ 48 fps, OPTIX: NVIDIA A100 80GB PCIe
* comparison: `smoke/render_check/f2/viz/f2_200000__compare.mp4` (0.33 MB, grid 5x5, tile 332x192); per-twin mp4s next to it (`<group>__<twin>.mp4`)

| twin | legal | violated_law | magnitude | epsilon | event_kind | event frame | clean_until_s |
|---|---|---|---|---|---|---|---|
| e0.0 | True | - | 0 | -1e-16 | collision | 115 | 6 |
| e0.1 | True | - | 0 | 0.1 | collision | 115 | 6 |
| e0.15 | True | - | 0 | 0.15 | collision | 115 | 6 |
| e0.2 | True | - | 0 | 0.2 | collision | 115 | 6 |
| e0.3 | True | - | 0 | 0.3 | collision | 115 | 6 |
| e0.4 | True | - | 0 | 0.4 | collision | 115 | 6 |
| e0.5 | True | - | 0 | 0.5 | collision | 115 | 6 |
| e0.6 | True | - | 0 | 0.6 | collision | 115 | 6 |
| e0.7 | True | - | 0 | 0.7 | collision | 115 | 6 |
| e0.8 | True | - | 0 | 0.8 | collision | 115 | 5.81 |
| e0.9 | True | - | 0 | 0.9 | collision | 115 | 5.65 |
| e1.0 | True | - | 0 | 1 | collision | 115 | 5.5 |
| e1.1 | False | energy | 1.15 | 1.1 | collision | 115 | 5.35 |
| e1.2 | False | energy | 1.31 | 1.2 | collision | 115 | 5.23 |
| e1.3 | False | energy | 1.49 | 1.3 | collision | 115 | 5.12 |
| e1.4 | False | energy | 1.68 | 1.4 | collision | 115 | 5.02 |
| e1.5 | False | energy | 1.88 | 1.5 | collision | 115 | 4.92 |
| mom_violate_e0.15_neg | False | momentum | 0.609 | 0.15 | collision | 115 | 6 |
| mom_violate_e0.15_pos | False | momentum | 0.609 | 0.15 | collision | 115 | 6 |
| mom_violate_e0.5_neg | False | momentum | 0.794 | 0.5 | collision | 115 | 6 |
| mom_violate_e0.5_pos | False | momentum | 0.794 | 0.5 | collision | 115 | 5.31 |
| mom_violate_e1_neg | False | momentum | 1.06 | 1 | collision | 115 | 6 |
| mom_violate_e1_pos | False | momentum | 1.06 | 1 | collision | 115 | 4.62 |

**Looks right.** All 23 twins: 11 legal rungs incl. e0.15, 5 energy rungs e1.1-e1.5, and 6 momentum violators at e = 0.15 / 0.5 / 1 (pos / neg; one split magnitude per group, so pos and neg share `magnitude`). The rungs separate progressively, and pre-contact frames are identical. Sphere A leaves the bottom edge only after its clean_until_s (= labels inframe_until): e1.0 at f265, e1.5 at f237, mom_violate_e1_pos at f223. Both spheres are red in this group (same-material pair), so A and B are distinguished only by size. Rendered with the sensor-width fix (see top). The first, zoomed attempt accepted a different draw; it was deleted.

## f3

* group `f3_300000`, 5 twins, source `f3/f3_300000`, render 416x240 @ 48 fps, OPTIX: NVIDIA A100 80GB PCIe
* comparison: `smoke/render_check/f3/viz/f3_300000__compare.mp4` (0.12 MB, grid 3x2, tile 416x240); per-twin mp4s next to it (`<group>__<twin>.mp4`)

| twin | legal | violated_law | magnitude | epsilon | event_kind | event frame | clean_until_s |
|---|---|---|---|---|---|---|---|
| ratio_0.33 | True | - | 0 | 1 | collision | 116 | 6 |
| ratio_0.5 | True | - | 0 | 1 | collision | 116 | 6 |
| ratio_1 | True | - | 0 | 1 | collision | 116 | 6 |
| ratio_2 | True | - | 0 | 1 | collision | 116 | 6 |
| ratio_3 | True | - | 0 | 1 | collision | 116 | 6 |

**Looks right.** Identical look for both spheres (latent_mass arm). The five ratios share pre-contact frames and diverge after f116. Everything stays in frame. Rendered with the sensor-width fix.

## f4

* group `f4_400000`, 6 twins, source `f4/f4_400000`, render 416x240 @ 48 fps, OPTIX: NVIDIA A100 80GB PCIe
* comparison: `smoke/render_check/f4/viz/f4_400000__compare.mp4` (0.12 MB, grid 3x2, tile 416x240); per-twin mp4s next to it (`<group>__<twin>.mp4`)

| twin | legal | violated_law | magnitude | epsilon | event_kind | event frame | clean_until_s |
|---|---|---|---|---|---|---|---|
| u0_e1 | True | - | 0 | 1 | collision | 115 | 6 |
| u0_e015 | True | - | 0 | 0.15 | collision | 115 | 6 |
| ularge_e1 | True | - | 0 | 1 | collision | 115 | 6 |
| ularge_e015 | True | - | 0 | 0.15 | collision | 115 | 6 |
| usmall_e1 | True | - | 0 | 1 | collision | 115 | 6 |
| usmall_e015 | True | - | 0 | 0.15 | collision | 115 | 6 |

**Looks right, but subtle.** 3 pre-segments (u0 / usmall / ularge) x (e1, e015), and the pre-event frames are identical within each boost. Camera at 30 m: spheres are only ~10 px across at 416 (~20 px at full res), and the boost drift is a few tens of px over the clip. Nothing leaves the frame. Rendered with the sensor-width fix.

## F6 (removed)

The old band-design F6 renders (f6_flat, f6_incline, f6_slides, f6_glide) were deleted: superseded by smoke/render_check_full/f6_v2.

## f7_projectile

* group `f7_700000`, 4 twins, source `f7_projectile/f7_700000`, render 416x240 @ 48 fps, OPTIX: NVIDIA A100 80GB PCIe
* comparison: `smoke/render_check/f7_projectile/viz/f7_700000__compare.mp4` (0.07 MB, grid 2x2, tile 416x240); per-twin mp4s next to it (`<group>__<twin>.mp4`)

| twin | legal | violated_law | magnitude | epsilon | event_kind | event frame | clean_until_s |
|---|---|---|---|---|---|---|---|
| legal | True | - | 0 | - | branch | 122 | - |
| legal_e04 | True | - | 0 | - | branch | 122 | - |
| gravity_x15 | False | gravity | 1.5 | - | branch | 122 | - |
| vx_x06 | False | axis_independence | 0.6 | - | branch | 122 | - |

**Looks right.** The ball is thrown in from off-screen: it is invisible for f0-75 and first visible at f76 (by design, but 1.6 s of empty frame). It branches at f122 mid-flight. gravity_x15 lands sooner and vx_x06 lands short. It bounces on the floor around f165-175 and leaves frame left at about f263 (legal, gravity_x15). Projectile and drop twins carry no clean_until_s.

## f7_drop

* group `f7_700300`, 3 twins, source `f7_drop/f7_700300`, render 416x240 @ 48 fps, OPTIX: NVIDIA A100 80GB PCIe
* comparison: `smoke/render_check/f7_drop/viz/f7_700300__compare.mp4` (0.06 MB, grid 2x2, tile 416x240); per-twin mp4s next to it (`<group>__<twin>.mp4`)

| twin | legal | violated_law | magnitude | epsilon | event_kind | event frame | clean_until_s |
|---|---|---|---|---|---|---|---|
| legal | True | - | 0 | - | branch | 114 | - |
| legal_e04 | True | - | 0 | - | branch | 114 | - |
| heavy_x13 | False | mass_independence | 1.3 | - | branch | 114 | - |

**Looks right.** Two identical red balls hover with no visible holder until release at f97, then branch at f114 mid-fall. heavy_x13: the heavy (left) ball lands first. legal_e04 bounces lower. Both balls are at rest late in the clip.

## f7_bounce

* group `f7_700500`, 21 twins, source `f7_bounce/f7_700500`, render 416x240 @ 48 fps, OPTIX: NVIDIA A100 80GB PCIe
* comparison: `smoke/render_check/f7_bounce/viz/f7_700500__compare.mp4` (0.21 MB, grid 5x5, tile 332x192); per-twin mp4s next to it (`<group>__<twin>.mp4`)

| twin | legal | violated_law | magnitude | epsilon | event_kind | event frame | clean_until_s |
|---|---|---|---|---|---|---|---|
| eps0.5 | True | - | 0 | 0.5 | bounce | 116 | 2.98 |
| eps0.55 | True | - | 0 | 0.55 | bounce | 116 | 3.04 |
| eps0.6 | True | - | 0 | 0.6 | bounce | 116 | 3.1 |
| eps0.65 | True | - | 0 | 0.65 | bounce | 116 | 3.15 |
| eps0.7 | True | - | 0 | 0.7 | bounce | 116 | 3.21 |
| eps0.75 | True | - | 0 | 0.75 | bounce | 116 | 3.26 |
| eps0.8 | True | - | 0 | 0.8 | bounce | 116 | 3.32 |
| eps0.85 | True | - | 0 | 0.85 | bounce | 116 | 3.38 |
| eps0.9 | True | - | 0 | 0.9 | bounce | 116 | 3.43 |
| eps0.95 | True | - | 0 | 0.95 | bounce | 116 | 3.49 |
| eps1.0 | True | - | 0 | 1 | bounce | 116 | 3.55 |
| eps1.05 | False | energy | 1.05 | 1.05 | bounce | 116 | 3.6 |
| eps1.1 | False | energy | 1.1 | 1.1 | bounce | 116 | 3.66 |
| eps1.15 | False | energy | 1.15 | 1.15 | bounce | 116 | 3.72 |
| eps1.2 | False | energy | 1.2 | 1.2 | bounce | 116 | 3.77 |
| eps1.25 | False | energy | 1.25 | 1.25 | bounce | 116 | 3.83 |
| eps1.3 | False | energy | 1.3 | 1.3 | bounce | 116 | 3.88 |
| eps1.35 | False | energy | 1.35 | 1.35 | bounce | 116 | 3.94 |
| eps1.4 | False | energy | 1.4 | 1.4 | bounce | 116 | 4 |
| eps1.45 | False | energy | 1.45 | 1.45 | bounce | 116 | 4.05 |
| eps1.5 | False | energy | 1.5 | 1.5 | bounce | 116 | 4.11 |

**Looks right, but small in frame.** All 21 rungs are pre-identical until the first floor contact at f116. The rebound apex grows monotonically with eps, and clean_until_s runs 2.98 s (eps0.5) to 4.11 s (eps1.5). The scene uses only the bottom ~40% of the frame: the ball is ~24 px across at full res, drops about 2 m, and even the eps1.5 apex stays far below the top edge. A closer camera would make the ladder easier to read.

## f8_ramp

* group `f8_800000`, 21 twins, source `f8_ramp/f8_800000`, render 416x240 @ 48 fps, OPTIX: NVIDIA A100 80GB PCIe
* comparison: `smoke/render_check/f8_ramp/viz/f8_800000__compare.mp4` (0.28 MB, grid 5x5, tile 332x192); per-twin mp4s next to it (`<group>__<twin>.mp4`)

| twin | legal | violated_law | magnitude | epsilon | event_kind | event frame | clean_until_s |
|---|---|---|---|---|---|---|---|
| eps0.5 | True | - | 0 | 0.5 | pad | 116 | 3.57 |
| eps0.55 | True | - | 0 | 0.55 | pad | 116 | 3.59 |
| eps0.6 | True | - | 0 | 0.6 | pad | 116 | 3.61 |
| eps0.65 | True | - | 0 | 0.65 | pad | 116 | 3.63 |
| eps0.7 | True | - | 0 | 0.7 | pad | 116 | 3.66 |
| eps0.75 | True | - | 0 | 0.75 | pad | 116 | 3.7 |
| eps0.8 | True | - | 0 | 0.8 | pad | 116 | 3.73 |
| eps0.85 | True | - | 0 | 0.85 | pad | 116 | 3.77 |
| eps0.9 | True | - | 0 | 0.9 | pad | 116 | 3.81 |
| eps0.95 | True | - | 0 | 0.95 | pad | 116 | 3.85 |
| eps1.0 | True | - | 0 | 1 | pad | 116 | 3.89 |
| eps1.05 | False | energy | 1.05 | 1.05 | pad | 116 | 3.94 |
| eps1.1 | False | energy | 1.1 | 1.1 | pad | 116 | 3.98 |
| eps1.15 | False | energy | 1.15 | 1.15 | pad | 116 | 4.03 |
| eps1.2 | False | energy | 1.2 | 1.2 | pad | 116 | 4.07 |
| eps1.25 | False | energy | 1.25 | 1.25 | pad | 116 | 4.12 |
| eps1.3 | False | energy | 1.3 | 1.3 | pad | 116 | 4.17 |
| eps1.35 | False | energy | 1.35 | 1.35 | pad | 116 | 4.22 |
| eps1.4 | False | energy | 1.4 | 1.4 | pad | 116 | 4.27 |
| eps1.45 | False | energy | 1.45 | 1.45 | pad | 116 | 4.31 |
| eps1.5 | False | energy | 1.5 | 1.5 | pad | 116 | 4.36 |

**Looks right.** The block starts near the right edge on ramp A, runs across the green pad (event f116), and climbs ramp C on the left. Climb height is ordered by eps, and eps1.5 reaches x = 28 px (still in frame). All 21 twins keep in_frame = true.

## f9_pair

* group `f9_900000`, 4 twins, source `f9_pair/f9_900000`, render 416x240 @ 48 fps, OPTIX: NVIDIA A100 80GB PCIe
* comparison: `smoke/render_check/f9_pair/viz/f9_900000__compare.mp4` (0.09 MB, grid 2x2, tile 416x240); per-twin mp4s next to it (`<group>__<twin>.mp4`)

| twin | legal | violated_law | magnitude | epsilon | event_kind | event frame | clean_until_s |
|---|---|---|---|---|---|---|---|
| legal | True | - | 0 | - | branch | 120 | - |
| heavy_x0.8 | False | oscillation | 0.8 | - | branch | 120 | - |
| period_x0.8 | False | oscillation | 0.8 | - | branch | 120 | - |
| period_x1.25 | False | oscillation | 1.25 | - | branch | 120 | - |

**Looks right, but small.** Two pendulums (heavier brass bob, lighter dark bob) swing in phase and branch at f120. period_x0.8 / x1.25 / heavy_x0.8 dephase visibly by f150-f200. The pendulums occupy only the top third of the frame, and the support bar touches the top edge. Rods are thin and faint at 416 px. Pair twins carry no clean_until_s. Rendered from f9_pendulum.py as modified 21:46 (before this job started).

## f9_swing

* group `f9_900150`, 21 twins, source `f9_swing/f9_900150`, render 416x240 @ 48 fps, OPTIX: NVIDIA A100 80GB PCIe
* comparison: `smoke/render_check/f9_swing/viz/f9_900150__compare.mp4` (0.24 MB, grid 5x5, tile 332x192); per-twin mp4s next to it (`<group>__<twin>.mp4`)

| twin | legal | violated_law | magnitude | epsilon | event_kind | event frame | clean_until_s |
|---|---|---|---|---|---|---|---|
| eps0.5 | True | - | 0 | 0.5 | swing | 112 | 3.22 |
| eps0.55 | True | - | 0 | 0.55 | swing | 112 | 3.24 |
| eps0.6 | True | - | 0 | 0.6 | swing | 112 | 3.26 |
| eps0.65 | True | - | 0 | 0.65 | swing | 112 | 3.27 |
| eps0.7 | True | - | 0 | 0.7 | swing | 112 | 3.28 |
| eps0.75 | True | - | 0 | 0.75 | swing | 112 | 3.29 |
| eps0.8 | True | - | 0 | 0.8 | swing | 112 | 3.3 |
| eps0.85 | True | - | 0 | 0.85 | swing | 112 | 3.31 |
| eps0.9 | True | - | 0 | 0.9 | swing | 112 | 3.31 |
| eps0.95 | True | - | 0 | 0.95 | swing | 112 | 3.32 |
| eps1.0 | True | - | 0 | 1 | swing | 112 | 3.32 |
| eps1.05 | False | energy | 1.05 | 1.05 | swing | 112 | 3.33 |
| eps1.1 | False | energy | 1.1 | 1.1 | swing | 112 | 3.33 |
| eps1.15 | False | energy | 1.15 | 1.15 | swing | 112 | 3.34 |
| eps1.2 | False | energy | 1.2 | 1.2 | swing | 112 | 3.34 |
| eps1.25 | False | energy | 1.25 | 1.25 | swing | 112 | 3.35 |
| eps1.3 | False | energy | 1.3 | 1.3 | swing | 112 | 3.35 |
| eps1.35 | False | energy | 1.35 | 1.35 | swing | 112 | 3.35 |
| eps1.4 | False | energy | 1.4 | 1.4 | swing | 112 | 3.36 |
| eps1.45 | False | energy | 1.45 | 1.45 | swing | 112 | 3.36 |
| eps1.5 | False | energy | 1.5 | 1.5 | swing | 112 | 3.37 |

**Correct timing, but very small in frame.** The bob is held at the left apex until release at f92, then crosses the green damper at the bottom (event f112, damper entry). The eps1.0 apex is symmetric (x 150 -> 209 px, apex back at f139). All 21 rungs share the pre-event frames. The pendulum is tiny and high: bob ~16 px across at 416 (~32 at full res), pivot ~40 px from the top, and the whole pendulum spans ~70x40 px of the 416x240 frame. The post-event apex differences between neighbouring rungs are a few px. The scene would benefit from a closer / lower camera. Rendered from f9_pendulum.py as modified 21:46.

## f10_rolling

* group `f10_1000000`, 4 twins, source `f10_rolling/f10_1000000`, render 416x240 @ 48 fps, OPTIX: NVIDIA A100 80GB PCIe
* comparison: `smoke/render_check/f10_rolling/viz/f10_1000000__compare.mp4` (0.12 MB, grid 2x2, tile 416x240); per-twin mp4s next to it (`<group>__<twin>.mp4`)

| twin | legal | violated_law | magnitude | epsilon | event_kind | event frame | clean_until_s |
|---|---|---|---|---|---|---|---|
| legal | True | - | 0 | - | release | 123 | - |
| hollow_first | False | rolling | 1.19 | - | release | 123 | - |
| slip_w2 | False | rolling | 2 | - | release | 123 | - |
| tie | False | rolling | 1.19 | - | release | 123 | - |

**Physically right, but the camera misleads.** Labels: the solid (B) travels 9.49 m and the tube (A) 8.00 m by f288 in legal; tie gives both 9.49 m; hollow_first swaps them; slip_w2 has legal translation at double spin. The render matches the labels (pixel / seg agree). With the 30 deg yaw, though, the front lane is ~26 px left of the back lane at equal world x. On screen, legal (tube 1.5 m behind) looks almost tied, while tie looks staggered. Judge order by distance along the slab, not by screen x. Bodies are marked with a dark quadrant wedge, so spin is visible (slip_w2 visibly over-spins).

## f11_support_x0.65

* group `f11_1100002`, 2 twins, source `f11_support_x0.65/f11_1100002`, render 416x240 @ 48 fps, OPTIX: NVIDIA A100 80GB PCIe
* comparison: `smoke/render_check/f11_support_x0.65/viz/f11_1100002__compare.mp4` (0.04 MB, grid 2x1, tile 416x240); per-twin mp4s next to it (`<group>__<twin>.mp4`)

| twin | legal | violated_law | magnitude | epsilon | event_kind | event frame | clean_until_s |
|---|---|---|---|---|---|---|---|
| stay | True | - | 0 | - | release | 126 | 6 |
| tip | False | statics | 0.35 | - | release | 126 | 4.21 |

**Looks right.** x = 0.65 (CoM on the shelf). The prop drops away at f126: legal `stay` remains, and violator `tip` pivots about the edge and falls, out of frame by f288 (clean_until 4.21 s).

## f11_support_x1.35

* group `f11_1100000`, 2 twins, source `f11_support_x1.35/f11_1100000`, render 416x240 @ 48 fps, OPTIX: NVIDIA A100 80GB PCIe
* comparison: `smoke/render_check/f11_support_x1.35/viz/f11_1100000__compare.mp4` (0.04 MB, grid 2x1, tile 416x240); per-twin mp4s next to it (`<group>__<twin>.mp4`)

| twin | legal | violated_law | magnitude | epsilon | event_kind | event frame | clean_until_s |
|---|---|---|---|---|---|---|---|
| tip | True | - | 0 | - | release | 120 | 3.71 |
| stay | False | statics | 0.35 | - | release | 120 | 6 |

**Looks right.** x = 1.35 (CoM beyond the edge, about 2/3 of the block past it). Legal `tip` pivots about the edge and falls after the prop drops at f120 (clean_until 3.71 s). Violator `stay` hangs in place with most of the block past the edge.

## f12_spin

* group `f12_1200000`, 8 twins, source `f12_spin/f12_1200000`, render 416x240 @ 48 fps, OPTIX: NVIDIA A100 80GB PCIe
* comparison: `smoke/render_check/f12_spin/viz/f12_1200000__compare.mp4` (0.18 MB, grid 3x3, tile 416x240); per-twin mp4s next to it (`<group>__<twin>.mp4`)

| twin | legal | violated_law | magnitude | epsilon | event_kind | event frame | clean_until_s |
|---|---|---|---|---|---|---|---|
| legal | True | - | 0 | - | contact | 123 | 3.75 |
| legal_e04 | True | - | 0 | - | contact | 123 | 4.71 |
| spin_x0.7 | False | angular_momentum | 0.7 | - | contact | 123 | 4.5 |
| spin_x0.8 | False | angular_momentum | 0.8 | - | contact | 123 | 4.5 |
| spin_x0.9 | False | angular_momentum | 0.9 | - | contact | 123 | 3.9 |
| spin_x1.1 | False | angular_momentum | 1.1 | - | contact | 123 | 3.62 |
| spin_x1.2 | False | angular_momentum | 1.2 | - | contact | 123 | 3.54 |
| spin_x1.3 | False | angular_momentum | 1.3 | - | contact | 123 | 3.48 |

**Looks right up to clean_until, then intersects.** The puck hits the marked bar off-centre at f123, and the spin ladder 0.7-1.3 visibly changes the bar's rotation rate. **Interpenetration after the clean window:** in legal (clean_until 3.75 s = f180), the spinning bar's end sweeps through the stopped puck around f180-f190, and in legal_e04 the puck overlaps the bar end late (about f280). The bar also leaves the right edge near the end in several twins. Both happen after clean_until_s, so the labels exclude them, but the full 6 s clips show the overlap. magnitude = k (the spin factor itself, as documented in f12_spin.py), not |k - 1|.

