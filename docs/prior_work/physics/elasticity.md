# Elasticity (coefficient of restitution)

Split out of `CLAUDE.md` on 2026-10-27, verbatim apart from paths updated to the new layout. Section numbers are unchanged; the index in
`CLAUDE.md` says which file each § lives in. The rest of the elasticity story is told
from the energy side in [`kinetic_energy.md`](kinetic_energy.md) §44, and the restitution
readout before contact in [`momentum.md`](momentum.md) §62.

---

### 36. Elastic vs inelastic on the existing sets — measurements

Run 2026-08-21 on existing labels and captures. No new renders. Recorded because
the four-way elastic/inelastic x conserved/violated grid was proposed and these are
the numbers it would be built on.

**Realised restitution vs the sampled label.** `e_real = -(v_rel_post . n) / (v_rel_pre . n)`,
with `n` the line of centres at `collision.actual_frame`, velocities at frames 18
and 22:

| set | scenes | corr(e_label, e_real) | median abs diff |
|---|---|---|---|
| `probe_set` v1, conserving | 75 | +1.0000 | 0.0001 |
| `probe_set` v1, violating | 75 | +0.9999 | 0.0000 |
| v3, conserving | 225 | +0.9999 | 0.0001 |
| v3, violating | 75 | **+0.6333** | **0.6754** |

v1's violation adds a common `dv` to both balls, which cancels in `v_A - v_B`, so
restitution is untouched. v3's scales the two impulse magnitudes, which does not
cancel.

**`cos(dv_A, dv_B)` on violating scenes.** v1: -0.9999 to **+0.9993**. v3: -1.0000
to +0.2404. Conserving is -1.0000 in both.

**Kinetic energy, conserving scenes only.** `KE_post/KE_pre`, frames 18 -> 22:

| | `probe_set` v1 (n=75) | v3 (n=225) |
|---|---|---|
| median | 0.839 | 0.940 |
| range | 0.180 - 1.000 | 0.119 - 1.000 |
| >= 0.98 | 9 (12%) | 70 (31%) |
| 0.90-0.98 | 18 (24%) | 69 (31%) |
| 0.70-0.90 | 20 (27%) | 52 (23%) |
| < 0.70 | 28 (37%) | 34 (15%) |

`KE_loss = 0.5 * mu_red * (1 - e^2) * u_n^2` reproduces the observed loss at
corr **+1.00000**, median relative error 1e-4, in both sets.

**KE ratio is not restitution.** corr(e, KE ratio): v1 conserving **+0.548**, v1
violating **+0.058**; v3 conserving +0.307. On violating scenes the aftermath is
overwritten, so KE ratio runs to **2.63** — energy is created. By impact parameter
`b = |b|/(r_A+r_B)`, v1 conserving median KE ratio: b<0.3 **0.69** (n=23),
0.3-0.6 0.81 (n=31), b>0.6 **0.92** (n=21).

**Probes for restitution and energy, VideoMAE on `probe_set` v1.** Mean held-out R2
over 20 random 60/40 splits, standardised features, fixed alpha 10 (NOT the
`fit_probe` protocol — a separate one-off):

| cell | restitution | KE ratio |
|---|---|---|
| blk 5 all.pre | -0.040 | **0.459** |
| blk 5 all.post | 0.014 | 0.417 |
| blk 9 all.pre | 0.235 | 0.334 |
| blk 11 all.pre | **0.248** | 0.222 |
| blk 9 all.all | -0.112 | 0.025 |

**Retrospective 2x2 on `probe_set`, split at the median restitution 0.53.**
Violation AUC via the §34a route (regress `momentum_delta`, take the norm of the
prediction, score AUC against the conserved/violated label), VideoMAE all.pre
block 9, mean over 40 splits:

| cell | n | violating | AUC |
|---|---|---|---|
| all 150 | 150 | 75 | 0.675 |
| e > 0.53 | 75 | 42 | 0.531 |
| e <= 0.53 | 75 | 33 | 0.579 |

Cell sizes in the existing set: elastic/conserved 33, elastic/violated 42,
inelastic/conserved 42, inelastic/violated 33.

---

### 56i. The population restitution curve on CogVideoX-2b — shared in early blocks, scene-specific from block ~15, and a LINE, not a cubic

Run 2026-10-28 on the RTX 4090 pod. Step 1 of the mentor's "manifold steering" plan:
before steering along a population e-curve, check that one exists. **Nothing has been
steered yet.** Artifacts: `docs/artifacts/ladder_manifold_cogvideox-2b/manifold_step{0..8}.{json,txt}`.

**The mentor's design.** 200 seeds x ~20 restitution values with identical pre-collision
frames; centre each seed's embeddings on the seed mean; centroid per `e` across seeds;
fit a cubic through the centroids; steer along it. Run here at 20 seeds x 6 legal arms,
which turned out to be enough to answer the question (see "what more seeds would buy").

**Data — `datasets/energy_ladder_v1`.** 20 seeds (9500-9519, disjoint from the
conservation sets), 11 arms each: legal `e` = 0.2, 1/3, 0.5, 2/3, 0.8, 1.0 and impossible
`e` = 1.25, 1.5, 2, 3, 5. Every arm simulates the same collision at pinned `e = 1`; the
`e = 1` arm is the unedited honest clip and every other arm re-keyframes the aftermath at
the edited `e` (`--edit-restitution`). Pre-contact positions are identical across arms
(max diff 0), momentum is identical, only energy moves. Contact normals span -179° to 151°,
so the per-scene physical e-direction `b = (-n/m_A, +n/m_B)` rotates between seeds. Plan:
`plan_energy_voe_set_register.py --mode ladder --output-dir energy_ladder_v1 --num-seeds 20
--seed-offset 9500`. **189 clips complete, 31 dropped** (all impossible arms, by the
relative in-frame rule — every seed keeps all 6 legal arms), 0 failed, all 49 frames.

**Realised e drifts in one seed.** `s09501`'s arms realise +0.014 to +0.026 above plan
(0.214 / 0.349 / 0.518 / 0.686 / 0.821, `restitution_realised_error` in `labels.json`);
three other seeds drift <= 0.015 on impossible arms only. Unexplained. It is why the
population test matches arms by rank within a seed and places each centroid at the median
realised `e` — matching by rounded `e` emptied the cross-seed intersection and silently
returned no population numbers on the first run.

**Capture.** CogVideoX-2b, `--inversion-method noise`: VAE encode, one forward noising at
each of `t` = 50, 100, 200, 300, 400, 500, 600, 700, 800 with ONE fixed `eps`
(`noise_seed=0`) shared by every clip, so arms of a seed differ only in pixels; one
unguided denoiser call per level, empty prompt. Sites `block.0-29.resid_post` plus
`vae.latent` (no noise axis); frame-wide pools `all.pre` (latent frames 0-5), `all.post`
(6-12), `all.all` — no masked pools, so no §20 leak. 8 s/clip.

**Test (`scripts/elasticity/elasticity_manifold.py`, `population_curve`).** Per site x pool
x noise level, in the full 1920 dims (no PCA — with 20 x 6 points a reduction only adds a
choice to defend): take each seed's 6 legal arms, subtract the seed's mean over them,
average across seeds per arm -> 6 centroids.
* **retention** = |centroid| / rms per-seed residual. ~1/sqrt(20) = 0.22 if the seeds'
  e-offsets point in unrelated directions; 1 if they share one.
* **leave-one-seed-out R2** of a held-out seed's residuals from the other 19 seeds'
  centroids, three ways: raw centroids, a line in `e`, a cubic in `e`.
* **null**: shuffle `e` within each seed, 100 draws (so p = 0.010 is the floor).

**Result — `all.post`, t = 400 (the strongest level), depth profile:**

| site | retention (null 0.20) | LOSO R2, line (null -0.03) |
|---|---|---|
| `vae.latent` | 0.57 | +0.36 |
| block 1 | 0.64 | +0.46 |
| **block 3** | **0.73** | **+0.59** |
| block 6 | 0.66 | +0.48 |
| block 10-12 | 0.54-0.56 | +0.30-0.32 |
| block 15 | 0.36 | +0.09 |
| blocks 18-29 | 0.29-0.35 | +0.00-0.07 |

Best block by noise level (`all.post`, LOSO R2 line): t50 0.31, t100 0.34, t200 0.45,
**t300 0.55, t400 0.59, t500 0.59**, t600 0.52, t700 0.50, t800 0.49 — always block 1-3.
`all.all` tracks `all.post` within 0.01-0.02.

1. **A shared curve exists, and it is early.** Blocks 1-6 keep 60-73% of the per-seed
   e-offset through the average and predict an unseen seed's ladder at R2 up to 0.59.
   Every cell is at the p = 0.010 floor; the margin over the null (-0.03) is ~0.6, so
   selection over 31 sites cannot produce it.
2. **The transformer adds to it, but the pixels already carry much of it.** The clean VAE
   latent alone scores 0.36; block 3 lifts it to 0.59. Most plausibly "the aftermath is
   faster / slower" — a scene-independent kinematic contrast — sharpened by the first blocks.
3. **From block ~15 on the e-direction is scene-specific.** Retention falls to within
   ~0.1 of its null and LOSO R2 to ~0. The per-seed geometry (§56 columns, same artifacts)
   still orders arms by `e` within every seed (Spearman(PC1, e) 0.96-1.00 at every block), so
   e is encoded there — but in a direction that differs between scenes, and the cross-seed
   average cancels it. The middle third, where the paper puts physics, is exactly where a
   population curve does not exist.
4. **The cubic adds nothing.** Raw centroids, line and cubic agree to +-0.01 in every cell
   of all 3 pools x 31 sites x 9 levels. The population manifold is a straight line in `e`.
5. **Pre-contact tokens carry e although their pixels do not.** Within-seed spread of
   `all.pre`: VAE 0.0011 (norm 7.2, i.e. identical inputs), block 3 0.050, block 15 0.51.
   `all.pre` LOSO R2 reaches 0.35-0.41 against the VAE's 0.07. This is the aftermath
   leaking backwards through bidirectional attention (§4), not anticipation — do not read
   it as prediction.

**Per-seed cross-seed alignment (the §56 Q2 column)** agrees: |cos| between seeds' PC1s is
0.3-0.75 against a random reference of 0.018, and its correlation with the cosine between
the scenes' physical legal directions is ~0 (-0.06 to +0.16) — the shared part is not
organised by contact geometry.

**What this means for steering.**
* A population direction exists only at blocks 1-6, and is a line. Steering along it is
  closer to "speed up / slow down the aftermath" than to a physics edit, but it is the
  one candidate the data supports.
* In blocks 15+ a single averaged curve cannot steer; the direction would have to be
  predicted per scene (e.g. conditioned on contact normal and masses) rather than averaged.
* **What more seeds would buy:** not a different verdict. The curve is already linear and
  the mid-block direction already cancels at 20 seeds; 200 seeds would only tighten the
  mid-block numbers. The mentor's full 4000-clip render is not needed for step 1.
* **CogVideoX-2b cannot show a steered clip** (§7). Pixel-level steering needs a
  generatively competent model — `CogVideoX1.5-5B-I2V` is downloaded to
  `/workspace/models/CogVideoX1.5-5B-I2V` (23 files, SHA-256 verified against the hub at
  revision 46c9052870). Order: (1) unsteered feasibility — first frame of 10-20 clips, does
  it produce a plausible two-ball collision; (2) capture this ladder on that model and redo
  this test there; (3) steer per Kneifl et al. 2026 (arXiv 2610.03154: one block, first
  quarter of denoising, full-resolution per-token direction — pooled ones did not steer
  there) and measure realised `e` from pixels with `video.py`'s tracker.

**Reproduce** (CPU, ~25 min for all nine levels in parallel — **cap BLAS threads**: nine
uncapped jobs open 64 threads each and ran ~10x slower):

```bash
OUT=outputs/ladder_cogvideox-2b
HF_HOME=/root/hf_cache HF_HUB_OFFLINE=1 /root/envs/physics_venv/bin/python \
    scripts/pipeline/run_experiment.py configs/momentum_probe.yaml --mode capture \
    --model cogvideox-2b --inversion-method noise \
    --dataset-dir datasets/energy_ladder_v1 -o $OUT --keep-going          # ~25 min GPU
export OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14
for k in 0 1 2 3 4 5 6 7 8; do
  /root/envs/physics_venv/bin/python scripts/elasticity/elasticity_manifold.py \
      --captures $OUT/momentum_probe/activations --step $k \
      --pools all.post all.pre all.all --out $OUT/manifold_step$k.json > $OUT/manifold_step$k.txt &
done; wait
```

### 56j. Object-local pooling does not reveal a shared e-direction — and the "scene-specific" mid blocks hold a shared code after all, which is separation speed

Run 2026-11-02 (RTX 4090 pod, CPU only). The question: is §56i/§66's "from block ~15 the
e-direction is scene-specific" an artifact of FRAME-WIDE pooling? Kneifl et al. (arXiv
2610.03154) find physical state "computed globally but stored locally" at the object's
tokens. **No GPU capture was needed**: the §56i capture (`outputs/ladder_cogvideox-2b`)
already holds the object-local pools `A`/`B` x `pre`/`post`/`all`; §56i analysed only `all.*`.
Artifacts: `docs/artifacts/ladder_manifold_cogvideox-2b/manifold_objpools_step{0..8}.{json,txt}`,
`manifold_objpools_summary.txt`, `manifold_maskfloor.*`, `probe_loso_*`, `maskfloor_{e,sep}.*`,
`kinematic_oracle.txt`, `oracle_compare.{json,txt}`.

**The §20 control, built for this test (`scripts/elasticity/ladder_mask_floor.py`).** After
contact the arms of a seed put their balls in different places at different speeds, so the
pooling mask alone carries e. Per clip, three mask-only "sites" rebuilt exactly as
`capture_scene` builds its weights: the normalised pooling weights (17550-d), the same
weights applied to CogVideoX-2b's fixed 3-D sin-cos positional embedding (`mask.posembed`,
1920-d — the position information the patch embedder adds to every token, pooled like the
activations), and per-latent-frame centroid + area (39-d; numerically unstable under the
ridge below, not used). `mask.posembed` is the binding floor everywhere. Pre-contact and
frame-wide masks are identical across a seed's arms, so their floor is 0 by construction.

**1. The §56i population test on object pools: no shared curve at any depth or noise level.**
Retention / LOSO R2 (line), t = 400, null retention ~0.22, null R2 ~-0.03:

| site | `all.post` (§56i) | `A.post` | `B.post` | `A.post+B.post` | `A.pre+B.pre` |
|---|---|---|---|---|---|
| `vae.latent` | 0.57 / +0.36 | 0.25 / -0.01 | 0.37 / +0.10 | 0.32 / +0.04 | 0.34 / +0.07 |
| block 3 | **0.73 / +0.59** | 0.24 / -0.02 | 0.27 / -0.01 | 0.25 / -0.02 | 0.37 / +0.07 |
| block 10 | 0.56 / +0.30 | 0.24 / -0.02 | 0.29 / +0.00 | 0.26 / -0.01 | 0.30 / +0.02 |
| block 15 | 0.36 / +0.09 | 0.24 / -0.02 | 0.27 / -0.00 | 0.25 / -0.02 | 0.24 / -0.02 |
| block 24 | 0.34 / +0.06 | 0.25 / -0.01 | 0.32 / +0.04 | 0.28 / +0.01 | 0.23 / -0.02 |
| **mask floor** | 0 | 0.32 / +0.02 | 0.35 / +0.03 | 0.33 / +0.02 | 0 |

Best block over all nine levels: object pools never exceed +0.07 (blocks) / +0.10 (VAE,
`B.post`), i.e. at their mask floor; `A.pre+B.pre` reaches +0.17-0.21 at blocks 0-3 only.
**Object-local pooling does not rescue a mid-block population curve — it removes even the
early-block one.** Within a seed the ball tokens' dominant variation is where the ball lands,
which differs per scene. Per §66's step-3 condition (object pools changing the mid-block
picture), `per_scene_direction.py` / `per_scene_speed.py` were not rerun.

**2. But the centroid test was the wrong instrument for "is there a shared code".** It is a
fixed decoder — the cross-seed mean of each arm — so a shared e-code that is not the dominant
within-seed direction is invisible to it. `scripts/elasticity/ladder_probe_loso.py` fits the
weakest thing that could still be called shared: centre each seed's 6 legal arms on the seed
mean (removing everything constant within a scene), fit ONE ridge readout of the centred
target on 19 seeds (kernel ridge, alpha by inner leave-one-seed-out), score pooled R2 on the
20th. Null: target permuted within every seed (20 draws at t = 300-500, 10 elsewhere — so p
floors at 0.048 / 0.091; read the null max). t = 400:

| site | `all.post` e | `all.pre` e | `A.post+B.post` e | `all.post` sep | `A.post+B.post` sep |
|---|---|---|---|---|---|
| `vae.latent` | 0.48 | 0.39 | 0.45 | 0.84 | 0.65 |
| block 3 | 0.65 | 0.61 | 0.26 | 0.90 | 0.59 |
| block 10 | 0.74 | 0.78 | 0.40 | 0.93 | 0.58 |
| block 15 | 0.66 | 0.72 | 0.43 | 0.90 | 0.54 |
| block 21 | 0.74 | 0.53 | 0.50 | 0.83 | 0.60 |
| block 29 | 0.54 | 0.40 | 0.50 | 0.72 | 0.66 |
| **null max (rows shown)** | <= 0.11 | <= 0.12 | <= 0.14 | <= 0.17 | <= 0.14 |
| **mask floor** | 0 | 0 | **0.34** | 0 | **0.73** |

("sep" = separation speed, e x approach speed.) Frame-wide, a readout fit on 19 scenes reads
a held-out scene's e at **0.64-0.79 (best of blocks 15-29) at every one of the nine noise
levels** — block 15 alone reads 0.66 at t = 400, where §56i's centroid scored 0.09. **The mid blocks are not
scene-specific in the sense §56i/§66 drew from it: a shared linear code exists there; it is
just not the direction the arms mostly differ along.**

**3. That shared code is separation speed, not restitution (§12).** Within a seed approach
speed is constant, so centred e and centred separation speed are exactly collinear; only the
cross-seed scale (approach speed, CV 0.59 here) separates them. Kinematic oracle
(`kinematic_oracle.txt`): a representation holding ONLY exact separation speed affords a
shared e-readout of 0.72 (0.65-0.69 at 0.90-0.95 fidelity); one holding only exact e affords
a separation-speed readout of at most **0.715**. Measured: separation speed reads **0.83-0.95**
frame-wide — above anything an e-code could give — and the e-readout sits on the oracle
computed at each block's own separation-speed fidelity (median gap +0.007 / +0.020 / +0.022 at
t = 300 / 400 / 500; single-block maxima +0.12-0.15, not consistent across levels). The
representation carries how fast the balls leave, at every depth; e is the readout's division.

**4. Object-local readouts are mostly their mask.** `A.post+B.post`: best e 0.42-0.61 over
the nine levels against a mask floor of 0.34 and its own VAE latent's 0.45 — a real but small
margin; separation speed 0.53-0.82 against a floor of **0.73** — below it in every block except
block 1. `A.post` / `B.post` alone are worse (e <= 0.50 against 0.13 /
0.25). In this set frame-wide pools beat object pools on both targets, at every level —
§38d / §44 again, not Kneifl's localisation. (Their targets are constant-in-time states of a
single moving object; ours is a two-ball aftermath relation, and the masks themselves encode
much of it.)

**5. A caveat for §56i point 5.** `all.pre` reads e at 0.72-0.79 in blocks 6-15 — and the
VAE latent's `all.pre` at **0.39**. The ridge here is scale-free (features are normalised to
unit rms after seed-centring), so it amplifies the tiny within-seed pre-contact pixel
differences §56i measured (0.0011 against a norm of 7.2) — and those differences are not noise:
they predict e across seeds. Something in frames 0-20 (most likely frame 20, rendered at
contact 20.03-20.07, or the denoiser) differs systematically by arm. The pre-contact "e"
in the blocks may be part leak-through-attention, part rendering. Untested.

**What this means.** For steering along an e-manifold on CogVideoX-2b: there is no
object-local population direction; the frame-wide shared direction that does exist at every
depth is a separation-speed direction (steering it = "make the aftermath faster/slower"), not
a restitution direction. Kneifl-style per-token localisation does not change that here.

**Reproduce** (CPU; the probe is the slow part, ~10 min per pool x level at 3 threads):

```bash
C=outputs/ladder_cogvideox-2b/momentum_probe/activations
/root/envs/physics_venv/bin/python scripts/elasticity/ladder_mask_floor.py --captures $C \
    --dataset-dir datasets/energy_ladder_v1 -o /root/ladder_roi/mask_floor          # --shard i/n
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for k in 0 1 2 3 4 5 6 7 8; do
  /root/envs/physics_venv/bin/python scripts/elasticity/elasticity_manifold.py --captures $C --step $k \
      --pools A.post B.post A.post+B.post A.all+B.all A.pre+B.pre --out .../manifold_objpools_step$k.json
done
/root/envs/physics_venv/bin/python scripts/elasticity/elasticity_manifold.py --captures /root/ladder_roi/mask_floor \
    --names mask_floor --step 0 --pools A.post B.post A.post+B.post A.all+B.all
for T in e sep; do
  /root/envs/physics_venv/bin/python scripts/elasticity/ladder_probe_loso.py --captures $C --step 4 \
      --pools all.post all.pre A.post B.post A.post+B.post A.pre+B.pre --target $T --null-draws 20
  /root/envs/physics_venv/bin/python scripts/elasticity/ladder_probe_loso.py --captures /root/ladder_roi/mask_floor \
      --names mask_floor --step 0 --pools A.post B.post A.post+B.post A.all+B.all --target $T --null-draws 20
done
```

### 65. CogVideoX1.5-5B-I2V cannot produce a measurable two-ball collision — steering not attempted

Run 2026-10-29 on the RTX 5090 pod. Step (1) of the §56i plan: before steering on the
1.5-5B-I2V model, check it can generate a collision whose `e` is readable from pixels.
**It cannot, in any setting tried, so steps (2) capture + nine-level manifold test and
(3) steering were not run** — steering a clip whose `e` cannot be measured has no readout.
Artifacts: `outputs/i2v_feasibility/feasibility_run1_default.json`,
`outputs/i2v_feasibility/feasibility_variants.json`, the mp4s beside them.

**Setup.** `cogvideox1.5-5b-i2v` adapter (`physics_interp/models/cogvideox_i2v.py`): 42 blocks
x 3072, image latent concatenated on channels, `ofs = 2`, clips upsampled 1.6x to 768x1152.
**45 frames, not 49**: with `patch_size_t = 2`, 49 frames pad one latent at the front and the
token at contact straddles pixel frames 17-24; at 45 (12 latents) token-frames 0-2 are
frames 0-20 and 3-5 are 21-44, cleanly split at contact. Seeds 9500-9502, `honest` arm
(e = 1). Modes: `free` = I2V from frame 0; `hold` = latents 0-5 (frames 0-20) held to the
real clip, 6-11 regenerated via `runner.inpaint` (the steering setting). 50 steps, ~250 s
per clip. Readout: `measure_momentum` pre/post velocities, `e = -(v_A'-v_B').n / (v_A-v_B).n`
with the label normal; measurable iff all four straight-line fits have residual <= 0.02.

**Result — fit residual (e from pixels); label e = 1.0:**

| scene | real | free, g6 | hold, g6 | hold strict+neg, g6 | hold strict+neg, g3 | hold empty, g1 |
|---|---|---|---|---|---|---|
| s09500 | 0.003 (1.005) | 3.27 | 0.45 | 0.32 (0.07) | 1.40 | 1.84 |
| s09501 | 0.001 (0.977) | 1.60 | 1.68 | 1.38 | **0.082 (0.82)** | 0.66 |
| s09502 | 0.002 (0.984) | 3.11 | 2.78 | 0.25 | 0.30 | 0.41 |

0 of 15 generated clips pass; the tracker reads all 3 real clips to within 0.023 of the label.

1. **Default prompt: object permanence fails at contact.** Balls duplicate, a blue ball
   appears, `free` also zooms the camera. Holding the approach does not prevent it.
2. **Strict prompt + negative prompt fixes the look, not the physics.** s09500 strict g6 is
   visually clean — two balls, fixed camera — but the struck ball never moves and the
   striker bounces back on a curved path. The residual gate rejects it correctly: with zero
   gravity a curve is wrong motion, not tracker noise.
3. **Caveat found in §67:** the strict prompt says "one red and one grey metal" ball, which is
   right only for s09500 — s09501 is rubber/rubber, s09502 metal/metal. Its s09501 and s09502
   cells asked for the wrong objects; the default-prompt cells are unaffected.
4. **No setting is reliably closest.** The best variant changes per scene (g6 on s09500, g3
   on s09501, g6 on s09502), and the one near-pass (0.082) is 4x the gate.

**What this means.** Like CogVideoX-2b (§7), this model does not generate the collision the
§56i curve would be steered along. Loosening the gate or shortening the post window would
measure the model's failure modes, not `e`. Untested before abandoning the model: more
steps, holding the image latent only (no pixel hold), 81-frame native length. The next
candidate is a different, physics-stronger video model — a choice, not a fix.

**Reproduce:**

```bash
/root/envs/physics_venv/bin/python scripts/elasticity/i2v_feasibility.py \
    --dataset-dir datasets/energy_ladder_v1 --seeds 9500-9502 \
    --modes real --variants strict_g6 strict_g3 empty_g1 \
    --result-name feasibility_variants.json -o outputs/i2v_feasibility     # ~40 min
```

### 66. The per-scene e-direction is not predicted by contact geometry — only its size, by approach speed

Run 2026-10-29, CPU, on the §56i captures (`outputs/ladder_cogvideox-2b`, `all.post`). §56i
left one route open: from block ~15 the e-direction is scene-specific, so predict it per
scene instead of averaging. **It cannot be predicted from anything fixed before contact.**
Artifacts: `docs/artifacts/ladder_manifold_cogvideox-2b/per_scene_direction_step{0..8}.{json,txt}`,
`per_scene_speed_step4.txt`.

**Test (`scripts/elasticity/per_scene_direction.py`).** Per seed (6 legal arms, centred), the
scene's e-direction is the least-squares slope `g_s` of its residuals on `e`. A held-out
seed's ladder is rebuilt as `(e - e_bar) g_hat` and scored by pooled LOSO R2, bracketed by the
population mean slope (floor, = §56i's line) and the seed's own slope (ceiling). Predictors,
fit on the other 19 seeds: ridge from `normal` (n), `legal_b` ((-n/m_A, n/m_B)), `physics`
(n, v_A, v_B, log m), `position` (contact pixel — a control, since the pools are frame-wide);
and a kernel average over seeds weighted by cos of the physical legal direction. Null:
scene features permuted across seeds, 200 draws.

**Result.** Ceiling 0.59-0.92, so a scene's own line holds most of its ladder. **No predictor
beats the floor:** 5 of 1395 site x predictor x level cells exceed the population line, by at
most +0.012; 15 cells have p <= 0.01, the ~14 expected by chance. Real scores sit on their
nulls throughout — the scene-to-direction pairing carries nothing these features capture.
`physics` overfits to below the floor at every mid block.

**The one thing that does predict: approach speed, and only the size**
(`scripts/elasticity/per_scene_speed.py`, t = 400). Principal axes of the per-seed slopes, one
feature at a time: approach speed predicts a top PC at every block tested (LOSO R2 0.32-0.54,
p <= 0.004); normal angle, mass ratio, contact position, v_A angle do not. The population line
scaled by the scene's approach speed:

| site | population | x approach speed | null | p | ceiling |
|---|---|---|---|---|---|
| `vae.latent` | 0.37 | **0.49** | 0.25 | 0.002 | 0.81 |
| block 3 | 0.59 | **0.69** | 0.42 | 0.004 | 0.92 |
| block 6 | 0.48 | **0.62** | 0.33 | 0.002 | 0.90 |
| block 15 | 0.09 | 0.12 | 0.06 | 0.002 | 0.71 |
| block 24 | 0.06 | 0.09 | 0.03 | 0.002 | 0.64 |

corr(approach speed, |g_s|) is 0.62-0.85 at every block.

1. **Early blocks read separation speed.** Separation speed is `e x approach speed`, so a
   representation of aftermath speed has an e-slope proportional to approach speed — which is
   what is seen, and seen equally in the VAE latent. The §56i curve is kinematics in pixels.
2. **The mid-block scene-specific part is unexplained.** Speed scaling recovers ~0.03 of a
   0.6-0.7 ceiling. What remains still agrees across seeds (|cos| 0.14-0.35 after removing
   the speed part, random ~0.02), but neither contact geometry, masses, velocities nor
   screen position predict it, linearly or by similarity of the physical direction.
3. **20 seeds is the binding limit for anything richer** — 19 training points against a
   1920-d target. A nonlinear predictor (e.g. from per-object masked pools, which these
   captures do not have) would need more seeds; this is where the mentor's 200 seeds would
   buy something, unlike for §56i.

**What this means for steering.** There is no per-scene direction to steer along that can be
computed from the scene. Steering on CogVideoX-2b would have to use each scene's *own*
measured ladder — which requires rendering that scene's ladder first, i.e. an oracle — and
2b cannot show the result anyway (§7, §65).

### 67. Wan2.2-TI2V-5B: coherent clips with the paste hold, but no physical collision — 0/18

Run 2026-10-29 on the RTX 5090 pod. The §65 gate on Wan2.2-TI2V-5B
(`/workspace/models/Wan2.2-TI2V-5B-Diffusers`, bf16, ~100 s per clip). Same three scenes,
readout and 0.02 residual gate. **No generated clip passes; nothing was captured or steered.**
Artifacts: `outputs/wan_feasibility/feasibility_run1.json`, `feasibility_scene.json`, mp4s.

**Setup (`scripts/elasticity/wan_feasibility.py`).** 49 frames at 8 fps (Wan trains at 24),
upsampled to 1056x704; 13 latents x 22 x 33 tokens, no temporal patching, so latents 0-5 are
exactly frames 0-20. Wan2.2-TI2V conditions by feeding frame 0's latent *clean* at timestep 0
(per-token timesteps). Holds: `mask` = latents 0-5 as that clean context; `paste` = frame 0 by
mask, latents 1-5 re-noised to each step's sigma and re-imposed (Euler). `*_scene` prompts are
built per scene from labels (materials, relative size, which ball moves). VAE round-trip of a
real clip reads e 1.005 vs 1.004: the VAE is not the problem.

**Fit residual (e from pixels); label e = 1.0; "lost" = a ball untracked after contact:**

| scene | free | hold | hold_strict | free_scene | hold_scene | **paste_scene** |
|---|---|---|---|---|---|---|
| s09500 metal->rubber | 0.42 | 0.16 | 0.10 | 0.24 | 0.16 | 0.074 (-0.15) |
| s09501 rubber/rubber | lost | lost | lost | lost | lost | 0.052 (0.46) |
| s09502 metal/metal | lost | 0.79 | 0.046 | lost | 0.038 | lost |

1. **Clean-context hold cuts at the seam.** With 6 clean latents (Wan is trained on 1) the
   first generated frame zooms ~1.5x and re-lays out the balls, or drops one.
2. **Paste fixes the seam** — constant scale, two balls, tracking gaps only at contact like the
   real clips — **but not the physics.** s09500: the struck ball never moves, the striker
   glances off. s09501: both move, roles reversed vs. the real outcome and the small ball
   decelerates on a curve. s09502: both metal balls vanish after contact.
3. **Wan's failures are of physics, CogVideoX1.5's (§65) also of objects.** Neither produces a
   collision whose e could be steered and read back.

**Untested:** clips interpolated to 24 fps (the 3x motion-per-frame mismatch); Wan2.1/2.2
14B (needs CPU offload on 32 GB). Neither addresses failure 2, which is the binding one.

**Environment notes.** UniPC's corrector calls `torch.linalg.solve` on GPU and cuSOLVER fails
to create a handle on this pod — the script sets `preferred_linalg_library("magma")`. The
untiled fp32 VAE needs most of the card: the transformer is moved to CPU around VAE calls;
diffusers 0.35.1's *tiled* Wan2.2 VAE path is broken (skips patchify: "expected 12 channels,
got 3"). Wan's T5 tokenizer needs `ftfy` (not in requirements).

**Reproduce (~20 min):**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /root/envs/physics_venv/bin/python \
    scripts/elasticity/wan_feasibility.py --dataset-dir datasets/energy_ladder_v1 \
    --seeds 9500-9502 --variants free_scene hold_scene paste_scene \
    --result-name feasibility_scene.json -o outputs/wan_feasibility
```

