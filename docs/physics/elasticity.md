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
