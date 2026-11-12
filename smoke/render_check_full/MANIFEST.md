# PhysLaw full-resolution render check (2026-11-12)

Re-render, at master quality, of the families whose framing `smoke/render_check/MANIFEST.md` flagged, after the
framing fixes. Format: 832x480, 64 spp (the config's `master.samples_per_pixel`), 48 fps x 289 frames,
`nuisance_frac=0`, `KUBRIC_USE_GPU=1` (every label: `render_device` = `OPTIX: NVIDIA A100 80GB PCIe`). Same seeds as
the reduced check; every seed was still accepted at attempt 0. Command (per group):
`KUBRIC_USE_GPU=1 physlaw/data/generate.py --config configs/data/<config>.yaml --seed <seed> --out smoke/render_check_full/<family> --set nuisance_frac=0`
(logs in `logs/`). The reduced renders in `smoke/render_check/` are untouched (and now stale for these families).

**Viewers** (`<family>/viz/`): per twin `<group>__<twin>.mp4` (832 px, H.264 yuv420p crf 20, all <= 0.2 MB), the
comparison grid `<group>__compare.mp4` (<= 0.7 MB), and a 2-up `<group>__2up.mp4` at 832 px per tile (pair below).
`summary.json` lists every twin's labels. F4 also has `f4_400000__comoving.mp4` (see F4).

**Checks**: `smoke/render_check/auto_check.py`: every group's decoded pre-event frames are identical across twins
(F4: per boost). `PHYSLAW_DATA=smoke/render_check_full/<family> pytest tests/test_data.py`: all 7 pass (12-16 passed,
the rest skipped as not applicable). Static runs after the event: only f7 bounce eps0.5 (ball at rest, 36 frames).

| family | seed / group | 2-up pair | object size at 832 px, before -> after | what changed |
|---|---|---|---|---|
| f9_pair | 900000 / f9_900000 | legal vs period_x0.8 | bob 34 -> 98 px | per-group fitted camera, bigger bobs |
| f9_swing | 900150 / f9_900150 | eps1.0 vs eps1.5 | bob 33 -> 105 px | per-group fitted camera, bigger bob |
| f7_bounce | 700500 / f7_700500 | eps1.0 vs eps1.5 | ball 40 -> 63 px | per-group fitted camera, ball scales with drop height |
| f4 (STALE: see f4_camboost) | 400000 / f4_400000 | u0_e1 vs ularge_e1 | spheres 28 -> 64 px | camera 30 -> 13 m |
| f10_rolling | 1000000 / f10_1000000 | legal vs tie | cylinders 45-57 -> 44 px (same scale, less perspective) | no yaw, 200 mm lens at 80 m |
| f12_spin | 1200000 / f12_1200000 | legal vs spin_x1.3 | puck 33 px, bar 146 px long (unchanged) | oblique approach, no second contact, bar placed in frame |
| f8_ramp (STALE: see f8_ramp_v2) | 800000 / f8_800000 | eps1.0 vs eps1.5 | block 32 px (unchanged, look only) | nothing |
| f4_camboost (2026-11-12) | 400000 / f4_400000 | per level: usmall_e1 vs cam_usmall_e1 (also ularge, e015) | spheres 64 px (unchanged) | + camera-boost twins, checker floor |
| f8_ramp_v2 (2026-11-12) | 800000 / f8_800000 | eps1.0 vs eps1.5 | block 32 -> 75 px | fitted camera, every length ~ h, two-corner contact, swept-mesh ramps |

("before" = the reduced render's labels, scaled to 832 px.)

## f9 pair / swing

`f9_pendulum.py`: the camera is fitted per group (`families.framed_side_camera`) to the pendulums' full range of
motion (pair: both bobs at their amplitude plus the beam; swing: the release apex on one side, the eps = 1.5 first apex
on the other, beam above, damper below), filling 85% of the binding image dimension. Bob radius = 0.15-0.20 x L
(L 0.8-1.2 m), pair amplitude 20-30 deg and bob gap 0.10-0.30 m; draws with a bob under 70 px are rejected (none in
100 seeds). Pair f9_900000: bobs 98 px, beam rows 58-99 (fully inside), bobs down to row 416. Swing f9_900150: bob 105
px, beam rows 28-75, damper bottom row 462, bob swings over columns 190-597 (all 21 rungs). Neighbouring rungs' apexes
now differ by ~2.6x the earlier pixels (326 vs 125 px/m).

## f7 bounce

`f7_freefall.py`: camera fitted per group so floor contact .. the top of the ball at the eps = 1.5 apex (2r + 2.25 H)
fill 85% of the image height; ball radius = 0.17-0.22 x the drop height H (here r 0.32 m, H 1.6 m). Ball 63 px; its
centre spans rows 67 (eps1.5 apex) to 412 (floor). The horizon of the level camera sits mid-frame (floor below).

## f4

`configs/data/f4_galilean.yaml`: `camera_distance` 30 -> 13 m (10.8 x 6.2 m view; r 0.30-0.60 spheres are 46-92 px).
In-frame acceptance (every boost, frame 0 to 1 s after contact) unchanged; ~1 rejected draw per group at 13 m (30
seeds), seed 400000 accepted at attempt 0. clean_until_s: u0 6.0 s, ularge_e1 5.58 s (B leaves the top edge at f269).

**Co-moving viewer** `f4/viz/f4_400000__comoving.mp4` (1.4 MB, 4 cols x 3 rows, 832 px tiles: e1 raw, e1 drift
removed, e015 raw, e015 drift removed; rows u0 / usmall / ularge). Drift = projection of u (t - t_event) through the
labels' camera, removed with a bilinear sub-pixel shift, padded with the floor colour. Numbers in
`f4_400000__comoving_check.json`: labels' per-frame pixel difference to the u0 partner equals the projected drift to
2e-13 px; segmentation centroids minus drift vs u0: median 0.05-0.08 px, max 0.22 px inside the clean window (2.5 px
only at f283-288 in ularge_e1, where B is leaving the frame and its mask is clipped). Stills (f250) confirm the
drift-removed tiles match u0. No data or labelling mismatch.

## f10 rolling

`f10_rolling.py`: no yaw (image x = world x), 45 deg pitch, 200 mm at 80 m (same 50 px/m at the target as 50 mm at
20 m). Equal world x now lands within 2.3% of its offset from the image centre: tie twin max |du| 6.8 px over the clip
(was ~26 px from the yaw). In the 2-up, `tie` shows both bodies level and `legal` shows the solid ahead.

## f12 spin

`f12_spin.py`: the puck comes in 20-40 deg off the face normal towards the nearer bar end, so after the frictionless
contact it slides off past the end instead of lingering in front of the face; draws in which ANY twin's puck and bar
would touch again before the clip end are rejected (head-on, 95% of draws re-touched), and the group is placed after
the motion is known so the bar stays wholly in frame from 1 s before the contact to the last frame in every twin.
Puck speed 1.2-2.0 m/s (was 1.5-2.5) so the bar's drift fits; ~9% of draws rejected. clean_until_s = the last frame
before the puck leaves the frame (legal 4.85 s, legal_e04 5.0 s; the clip end if it never leaves). Stills: no
overlap anywhere, bar in frame to f288.

## light hotspot (f6 glide, f12)

`generate.py` `build_family_kubric`: top-view sun moved from 65 deg to 45 deg elevation (intensity 3.0 -> 3.85 to
keep the floor's irradiance), so its mirror reflection on the glossy ice falls far outside the view. F12 stills: no
hotspot. F6 glide was not re-rendered (same light and ice material).

## f8 ramp (look only, no change)

Block 32 px across; the camera (20 m) shows ramp A on the right, pad at the bottom, ramp C rising out of the top-left
corner; eps1.5 climbs to the upper-left edge (in frame). The scene uses the lower ~2/3 of the frame and the block is
small; the pad is a thin green strip. A fitted camera like F7 bounce would enlarge it (not done here).

## f4_camboost (2026-11-12): F4 camera-boost twins

`twins.f4_twins(camera_boosts=...)`, `configs/data/f4_galilean.yaml` `twins.camera_boosts: [usmall, ularge]`: four new
twins `cam_usmall_e1`, `cam_usmall_e015`, `cam_ularge_e1`, `cam_ularge_e015` = the UNBOOSTED collision (world states =
u0's) filmed by a camera translating at -u (u of that level; Blender camera position keyframed every frame,
`generate.keyframe_twin`), pre_segment `cam_usmall` / `cam_ularge`. The F4 floor now carries a 1 m checker (contrast
0.25, `floor_checker`; every F4 twin) so the camera motion is visible. Group: 10 twins, attempt 0, 2402 s on the A100
(shared with the F8 render), `render_device` OPTIX A100.

Checks (`viz/f4_400000__camboost_check.json`): labelled ball pixels of every camera-boost twin vs its ball-boost twin,
every frame up to clean_until_s: max |diff| 1.1e-13 px; rendered mask centroids (both balls wholly inside): median 0.00
px, max 0.01 px. Camera travel over the clip 1.20 m (usmall, 0.32 px/frame) and 3.13 m (ularge, 0.83 px/frame).
clean_until_s equal to the ball twin's (ularge_e1 5.58 s). `PHYSLAW_DATA=smoke/render_check_full/f4_camboost pytest
tests/test_data.py`: 17 passed, 3 skipped (incl. the new `test_f4_camera_boost_matches_ball_boost`, decoded pre-event
identity per segment, `test_camera_projects_to_pixel_labels` on the camera twins). Stills (f30, f118, f240): balls at the
same pixels in each ball/camera pair, the checker offset between them.

Viewers (`f4_camboost/viz/`, `make_viewers.py`): per twin `f4_400000__<twin>.mp4` (832 px); grid
`f4_400000__compare.mp4` (5 x 2 tiles of 384 px, twin order u0_e1 ... cam_ularge_e015, 0.4 MB); 2-ups ball | camera
`f4_400000__2up__<level>_<e>__cam_<level>_<e>.mp4` for usmall / ularge x e1 / e015 (1664 x 480, 0.38-0.46 MB):
identical ball motion, only the floor moves in the right tile.

The `f4/` render above (plain floor, 6 twins) and `f8_ramp/` are stale; their folders are still on disk (deleting
them was refused by the session's permission check) and should be removed.

## f8_ramp_v2 (2026-11-12): F8 framing and contact geometry

`f8_ramp.py`: the ramp SURFACE is primary (straight parts joined by concave fillets of radius 4 x the block's
half-diagonal) and the block rides it in exact two-corner contact (`Contact`: flush on straight parts; where it bridges
a fillet its bottom face is a chord of the surface, both corners on it, so no corner ever enters the ramp). The block
turns while it bridges, and the rotation is in the energy: `motion` integrates the one-degree-of-freedom Lagrangian
(M(u) = |c'|^2 + (I/m) theta'^2; constant acceleration on straight parts, DOP853 rtol 1e-10 between the four corner
breakpoints of each fillet), so (|v|^2 + (I/m)|w|^2)/2 + g z is conserved off the pad (acceptance checks it). Ramps
render as ONE swept mesh each (`Static.profile`, `families.profile_mesh_obj`; 256 chords per fillet): no overlapping
slabs, no hatch lines. Every length scales with h (block half-size 0.38-0.45 h, ramp A 36-50 deg, ramp C 45-60 deg, pad
0.7-0.9 h, the block flush over the whole pad); camera fitted per group to the release .. the eps = 1.5 apex (fill 0.85);
blocks under 70 px rejected (12 seeds: 70.7-77.8 px, 3 rejections). The Bullet descent check stops when the block's
front corner reaches fillet A (Bullet sees the fillet as chord boxes): its gap is 1.3-2.2 mm in 12 seeds, so
`physics_check_tol_per_size` is gone (size-independent 0.05 m tolerance).

Group f8_800000 (re-rendered with `--force`): h 2.04 m, block half-size 0.90 m (75 px face at 832), fillet radius
5.10 m, attempt 0, Bullet gap 1.8 mm, 21 twins, 1498 s, OPTIX A100. Block mask over frame 0 .. clean_until_s: eps0.5
cols 240-766, eps1.0 157-766, eps1.5 cols 65-766 rows 86-377 (of 832 x 480): in frame at every rung.
`tests/test_f8_ladder.py::test_f8_block_never_enters_the_ramp`: bottom corners never below the rendered surface by
> 0.05 mm and one always touching (every frame to the apex). `PHYSLAW_DATA=smoke/render_check_full/f8_ramp_v2 pytest
tests/test_data.py`: 13 passed, 7 skipped. Stills (f60, f150, f190, f195): smooth fillets, the block bridges the
fillet nose-up with both corners on it, no penetration, no hatch.

Viewers (`f8_ramp_v2/viz/`, `make_viewers.py`): per twin `f8_800000__eps<e>.mp4` (832 px), grid `f8_800000__compare.mp4`
(5 x 5 tiles of 332 px, 0.47 MB), 2-up `f8_800000__2up__eps1.0__eps1.5.mp4` (1664 x 480, 0.24 MB).

## f6_v3 (2026-11-12, F6 without no_friction; f6_v2 deleted)

User decision: drop the `no_friction` twin (its long glide set the camera span and shrank every other effect) and make
the effects clearly visible. `f6_friction.py`: slide legal stop 2.1-2.4 s before the clip end (was 1.4-2.0), so the
slide after the branch is short and the reverse twin slides back far; pair branch in the first half of the event window
(frames 112-120) and legal stop 0.5-0.6 s before the clip end, so most of the slide is after the branch; camera fill
0.96 of the width (was 0.88). Over 100 seeds each: slide block >= 81 px (median 92), reverse at the clip end >= 158 px (median
207 px) from the legal rest point; pair blocks >= 75 px (median 85), heavy_x13's heavier block stops >= 37 px (median 43 px)
short. mu (= a / g; no statistic uses it) is now 0.05-0.11 for slides (was 0.03-0.06), 0.03-0.06 for pairs.

`f6_v3/`: slide group f6_600000 (7 twins: legal, reverse, viscous, decel_x0.7/0.85/1.15/1.3; branch f128, legal stop
3.85 s, mu 0.089; block ~100 px wide incl. its top face; rendered centres at f288: legal 142 px, reverse 328 px (186
px back), viscous 92 px, decel_x0.7 118 px; every twin in frame) and pair group f6_600300 (legal, heavy_x13; branch
f119, legal stop 5.47 s, mu 0.042; heavier (metal) block at f288: legal 99 px, heavy_x13 143 px: 44 px short; no
occlusion). 832x480, 64 spp, OPTIX A100 (561 s + 285 s). `PHYSLAW_DATA=smoke/render_check_full/f6_v3 pytest
tests/test_data.py`: 25 passed, 14 skipped; `auto_check.py`: pre-event frames identical across twins in both groups.
Viewers in `f6_v3/viz/` (`make_viewers.py smoke/render_check_full/f6_v3 --pairs legal:reverse legal:viscous
legal:heavy_x13`): per-twin mp4s, `__compare.mp4` grids, 2-ups legal vs reverse, legal vs viscous (slide) and legal vs
heavy_x13 (pair). Logs `logs/f6_v3_*`.
