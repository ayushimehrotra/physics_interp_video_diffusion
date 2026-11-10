# Status (2026-11-05, end of H100 session; resume: docs/RESUME.md)

## Spec v2 adopted (2026-11-10)

`docs/SPEC.md` is now v2 (v1 kept as `docs/SPEC_v1.md`). Milestone numbers in the table below are v1's;
v2 §10 keeps 1, 2, 3, 4, 6 and 7 (part) as DONE and renumbers the rest (12 to 29).
Goal set by the user on 2026-11-10: build every part of v2 and give each one a small test run on this machine.
The full-scale runs will be done by the user's mentor on larger GPUs.

**§11 decisions so far (user, 2026-11-10).**
* **F1 stays at 64 groups.** Milestone 12 therefore renders nothing, and its re-runs (3, 4, 6, CogVideoX-2b L0/L2)
  are already done on these 64.
* **Disk.** Deleted `/root/models/Wan2.2-I2V-A14B-bf16` (a transformer_2-only copy that no repo code used) and the
  uv/pip caches. 79 GB is now free for the new models' test weights. The 14B models are tested only through their
  smaller siblings' code paths.
* **Handoff.** The code goes to a private GitHub remote. The repo was `git init`-ed on 2026-11-10; `.gitignore`
  excludes data, cache, logs, result media and the reference PDFs.
* Still open: the PC1 pass criterion, the PC2 base model, F9/F10, and the venue/deadline.

**Where v2 and the built code disagree.** The built behaviour is kept until the user decides (v2 rule 3):
1. **F1 size.** v2 targets 400 groups; built: 64, confirmed by the user on 2026-11-10 (v2 §11). Groups
   f1_100000-063 are a seed prefix of the 400-group set, so a larger render extends this one.
2. **Nuisance variants.** v1 and v2 both ask for 20% of seeds. Built: `nuisance_frac: 0.0`, and `nuisance.py` does
   not exist yet.
3. **Positive control.** On 2026-11-05 it was deferred by decision. v2 puts PC1 on the critical path (milestone 16).
4. **L1 S_momentum reference.** Built: a shuffled-group forecast is the reference (v1 deviation 2). v2 demotes
   S_momentum to "reported, not primary" and does not define its reference, so the built one stays.
5. **Git hash in every output (v2 §9).** `physlaw` is not a git repository, so outputs record only the config hash.
6. **Checkpoint sha256 (v2 §4.3).** Only `videomae_v2_giant.yaml` pins sha256 so far.
7. **CogVideoX1.5-5B-I2V at 400 groups.** Milestone 12 does not re-run it. Pass 1 costs about 24 min per group
   here, so it stays on the 64-group subset unless someone re-runs it.
8. **CogVideoX1.5 pass 2.** It writes `gen_seed0.mp4` by context-latent replacement, which v2 §4.2 lists as a failed
   route to track A. v2 uses pass 2 only for L1 forecast activations. The frame-0 I2V idea for CogVideoX1.5
   (discussed 2026-11-09) is not in v2: v2's track B generators are Wan2.2-I2V-A14B and Wan2.1-I2V-14B.
9. **Module names.** v2 §1 names `models/wan22_ti2v.py`; the repo has the unbuilt stub `models/wan.py`. Neither is
   implemented, so the v2 name will be used. `configs/data/f5_mass_independence.yaml` (unbuilt) is superseded by
   the F5 merge into F6 and F7.

## Milestones (spec §9)

| # | Milestone | State |
|---|---|---|
| 1 | F1 generator + twins + acceptance tests | **built**; F1 subset rendered: 64 groups / 320 clips in `data/f1`, validation 705/705 passed. **Decision 2026-11-09: the remaining 336 groups are not rendered; F1 = these 64 groups** (`docs/HANDOFF_F1.md` kept for reference) |
| 2 | V-JEPA 2 adapter + token projection + overlays | **built**: `models/vjepa2.py` (encode/forecast/surprise), `models/untrained.py`, `scripts/03_token_overlays.py` (20 overlays checked by eye: ROIs sit on the balls; masks match the label centres to <0.5 px), `scripts/04_cache_activations.py` |
| 3 | L0 + L2 on V-JEPA 2 + untrained control | **done on the 64-group subset**: `eval/l0_states.py`, `eval/l2_surprise.py`, `probes/grid.py` (batched protocol, tested equal to `fit_probe`/`permutation_null`), `scripts/05_surprise.py` (pixel baselines), `scripts/06_eval.py`. Results: `results/<model>/f1/l{0,2}.{json,md}` |
| 4 | L1 on V-JEPA 2 | **built + run on the 64-group subset**: `eval/l1_forecast_laws.py`, results `results/vjepa2-vitl/f1/l1.{json,md}` (see below for the deviations) |
| 5 | positive control | **deferred by decision (2026-11-05)**: not needed for L0–L2 numbers, but required before claiming that models fail L1/L2 |
| 6 | VideoMAE-v2 adapter; L0 to L2 | **done (giant, with its pre-training decoder)**: `models/videomae2.py`, `configs/models/videomae_v2_giant.yaml`; results `results/videomaev2-giant/f1/` |
| 7 | CogVideoX adapter; L0 to L2; output-level | **2B: L0, L1, L2, gate (FAILED 0/20) and output eval done**. 1.5-5B-I2V: adapter built + config fixed, not run. Stages 07 / 09, `eval/outputs.py` built |
| 8–11 | | not built; each has a config and a stage CLI that exits with "not implemented" |

### Milestone 3 results (F1 subset, 64 groups x 5 twins; full protocol: 12 fold seeds, 500 permutations)

**L0.** V-JEPA 2 decodes every per-object state from its own-ball ROI pool at R² 0.85–0.97
(best layers 8–14; corrected p = 0.002, the floor for 500 draws). It beats the pooling-mask floor
(≤ 0.40) and, on mass, both the untrained control and the size-only tracker. Mass is visible here:
in the legacy appearance arm density (1.5 / 6.0) is tied to the material.
The untrained control decodes post-contact velocity at R² 0.63–0.67, but that is sample size:
pre-contact targets repeat across a group's 5 twins (64 distinct values), post-contact ones do not
(320). Refit on one twin per group (n = 64): untrained −0.04, V-JEPA 2 0.84. The untrained control's
scores will rise with the full 400-group set.

**L2.** Unadjusted for motion, V-JEPA 2's surprise mostly follows the copy-last-frame baseline:
* `energy_create` vs `conserve_e015`: win rate 0.89 (cos_obj), but copy-last-frame reaches 0.97
  (mse_all) and the legal-vs-legal control `conserve_e1` vs `conserve_e015` already gives 0.75.
  More post-contact motion means more surprise, whatever the law.
* Momentum violations (KE-reducing split, so less motion): cos 0.25–0.41 (LESS surprised);
  l1/mse_obj 0.62–0.72. On the energy-physical subset, cos_obj (0.16) equals copy-last-frame (0.16).
* The untrained control is at chance (0.44–0.56) on every contrast.
So this subset does not show that V-JEPA 2 is surprised by law violations beyond motion
magnitude. **L2, motion-adjusted** (`l2.md` top; calibrated on the legal pairs, Δlog surprise = β·Δlog covariates,
leave-one-group-out for the control). Copy-last-frame alone explains little of the legal-pair surprise
(R² ≤ 0.27); post-contact speed + copy-last explains up to 0.51 (cos) / 0.22 (mse_obj) and brings the
legal control to chance (0.47–0.61). With that set:
* momentum violations stay MORE surprising on the object-region magnitude distances: l1_obj / mse_obj
  adjusted win 0.67–0.77, CIs exclude 0.5, p 1e-3 to 1e-5, under all four covariate sets. The violating
  balls move LESS, so motion cannot explain this. Cosine distances: about 0.5.
* energy_create: adjusted 0.53–0.66, CIs mostly include 0.5. The old "energy surprise" is mostly motion.
* untrained control: chance everywhere.

### L1 results (V-JEPA 2, F1 subset). Deviations from spec §5 L1
1. Probes are cross-fitted over groups (5-fold x 4 seeds); with 64 groups a split leaves ~9 test groups.
2. The spec's null (balls continue unchanged) conserves p and KE EXACTLY in F1, so it cannot anchor
   the law score. S_law uses a shuffled-group forecast as the reference; the spec null anchors an
   added **outcome** score E = ||v̂'−v'|| / ||v'−v_pre|| (null = 1).
3. Probes trained on encoder outputs do not transfer to predictor outputs: decoded forecasts have 0.12
   of the real spread. `*_mm` variants moment-match forecast features to the real ones (per-dim
   mean/sd from training groups, no labels) before applying the probe.
4. Forecast pooling: `roi` (twin's true post ROI, spec; leaks location) and `frame` (no leak).

Results, legal twins (conserve_e1 / conserve_e015):
* roi_mm: S_momentum 0.55 / 0.54; S_outcome +0.25 [0.17, 0.33] / −0.12. frame_mm: S_momentum 0.43 / 0.40,
  S_outcome +0.26 [−0.01, 0.47] / −1.
* Energy cannot be scored: decoded forecasts are shrunk (slow), so they never create energy, while
  probe noise alone makes the floor create some (S hits the clip at 1). It is uninformative, not passed.
* Momentum is degenerate as a test of "understanding the collision": a forecast that ignores the
  collision conserves momentum. Decoded forecasts correlate 0.54 with the true post velocities and 0.39
  with the pre velocities. The outcome score is the informative one: the forecast recovers about a
  quarter of the velocity change of the ELASTIC outcome, and is no better than "no collision" for e = 0.15.
  (Restitution is not visible in the context; the forecast leans elastic.)
* The information is in the forecast. A probe trained ON forecasts (ROI pool) gets post-velocity R² 0.74,
  against 0.18 from the ROI placement alone at the same n, and 0.29 for pre velocity.
  So the predictor does anticipate the outcome; the real-frame probes read it poorly.
* Open: a better transfer than moment matching, e.g. fitting probes on the predictor's own outputs for
  masked REAL post-contact frames (context = full clip). That keeps probes independent of the
  context-only forecast.

### Milestone 7 results so far (CogVideoX-2b, F1 subset)
* L0: R² 0.91–0.99 from own-ball ROIs (best mostly at SNR +8 dB, layers 9–15), above the mask floor
  (≤ 0.49; finer 30x45 grid leaks more position than the encoders' 16x16).
* L2 (training denoising loss, clean or noisy context): chance after motion adjustment on every
  contrast (all CIs include 0.5). The calibration explains ≤ 12% of legal-pair differences.
* Generation: the configs were checked against the model card, the official CogVideo repo
  (`inference/cli_demo.py`), and the diffusers pipeline. My sampler reproduces the official pipeline exactly
  (difference 0.0). The model-card example (panda) and a natural ball prompt render correctly. The VAE round trip
  of our clips is clean (43.6 dB PSNR, 33.9 in the balls).
  - From our context, CogVideoX-2b (T2V, latent replacement) never continues the scene. On the grey floor
    the balls vanish; with a "natural" wood floor + zoom + long prompt it CUTS to its own shot
    (`scripts/experiments/naturalness_cogvideox.py`: 0/16 genuine continuations). The generated part follows
    the text, not the context.
  - Native image-to-video (official CogVideoX1.5-5B-I2V pipeline, DPM + dynamic CFG, from the last
    pre-contact frame): the balls PERSIST, but framing jumps after frame 0 and identity/count flickers
    (`scripts/experiments/native_i2v_cogvideox15.py`, 4 videos).
  - Output eval now drops scene cuts (background change of the first 12 generated frames vs the context
    median: real clips ≤ 0.13, cuts 0.67–1.0, threshold 0.25).
* L1 (12 SNR/layer spaces x 4 variants): best S_outcome +0.60 [0.49, 0.70] (conserve_e1, SNR −20 dB, layer 6,
  ROI + moment matching); most spaces 0.2–0.35. Treat this as PRELIMINARY: it is the best of 48 variants
  (no selection correction), it is ROI-pooled at the TRUE post-contact location (leaks the outcome), and it
  comes from a sampling run whose decoded video loses the balls. Next: select the space on held-out groups
  and report the leak-free frame-pool variants.
* Output level: gate FAILED (0/20 measurable: lost after context 6, no post-contact track 14), so no
  extra seeds. Over 64 groups, 1 generated clip was measurable.
* CogVideoX1.5-5B-I2V config corrected to the official repo's recommendation for 5B models
  (CogVideoXDPMScheduler trailing + use_dynamic_cfg); the checkpoint's default was DDIM.

### Milestone 7, output level: minimal prompt + training-free continuation (2026-11-05/06, A100 pod)
Experiment script `scripts/experiments/minimal_prompt_cogvideox.py` (seed 0, first 5 F1 groups, official settings,
outputs in `cache/<model>/f1_minimal*/`). Prompt = `models.cogvideox.minimal_prompt` (the user's template: name the
two balls, "The two balls collide and then separate.", nothing else); `CogVideoXAdapter.set_prompt()` re-encodes per clip.
* **CogVideoX-2b, context replacement (stage-4 sampler):** 0/5. Every clip CUTS to a close-up at the first generated
  frame; balls static, sometimes the wrong material. Same failure as the fixed prompt: the prompt is not the problem.
* **CogVideoX1.5-5B-I2V, context replacement:** 0/2 (stopped at 2/5, ~17-28 min per clip): coherent for a few frames,
  then a zoom/cut and extra balls (3-4).
* **2b + held-frame start** (`_sample(mode="hold")`: post-context frames = copies of the last context frame, noised,
  SDEdit): strength 0.7 keeps the scene but the balls FREEZE; 0.75 and 0.85 already cut. No working window.
* **2b + reconstruction guidance** (`mode="guided"`, Ho et al. 2022; `guide_tmin` switches it off at low noise, where
  it otherwise turns into texture noise): w=2 cuts; w=5/w=8 for t >= 300 on 5 groups (10 clips): 1 measurable
  (f1_100003 w5: scene held exactly, balls touch then STICK AND STOP: a faithful continuation with wrong physics).
  f1_100000: the only clips with a visible collision and separation, but the balls grow 1.5-1.7x (rejected,
  size_change / bad_fit). f1_100001/2/4: cuts, an extra ball, or a material change.
  Verdict: guidance sometimes holds the scene, never reliably; not a fair gate. Real continuation needs fine-tuning
  (cf. arXiv 2506.18901; spec's optional CogVideoX positive control) or a natively multi-frame model.
* **Readout fix (`eval/outputs.py`, 2026-11-06).** The stage-7 tracker split ANY big blob into two "balls", so a single
  zoomed-in ball after a cut read as a measurable static pair (4 false positives found by eye). Now: split only blobs
  with elongation >= 1.35 and area <= 2x the two balls; reject `size_change` (generated ball area > 1.5x context) and
  `extra_objects` (> 2 ball-sized blobs in > 2 frames). Calibrated on 128 real legal clips per geometry (merged-blob
  elongation 1.45-2.9, area ratio <= 1.20, 0 extra-blob frames). Real-clip verdicts unchanged (0/256 changed:
  115/128 and 105/128 measurable before and after); all 4 false positives rejected. The 2b gate's earlier 0/20 is
  unaffected (it had no measurable clips); re-run `09_eval_outputs` before quoting any output-level number.
* Elsewhere (old project, `physics_interp_video_diffusion/outputs/collide/wan14b_iter/`): Wan2.2-I2V-A14B with the
  minimal prompt from frame 0 (Physics-IQ-style I2V, the model invents the approach) gave 2/5 measurable collisions on
  one scene; its 10-scene sweep is paused with 8 scenes queued.

### Output track B (spec §6b, self-consistency) — 2026-11-06
* Spec change (user decision): `docs/SPEC.md` §6b; §10's forecast-mode decision resolved as two output tracks.
* Readout `eval/outputs.readout_self` (ported from the predecessor's validated collision readout; thresholds in px).
  **Validated on all 320 real F1 clips (64 groups x 5 twins, resampled to 16 fps): 312/320 measurable.** Median
  |e - e_true| 0.006-0.022; momentum residual 0.03-0.04 on conserving twins vs 0.33 / 0.61 on the momentum-violating
  twins (true 0.33 / 0.57); KE ratio 1.32 on energy_create (true 1.33). The 8 drops are all e = 0.15 twins whose
  balls stay within shadow-merging distance (known limit). Two fixes found on the real clips: the opposite-impulse
  check only when both |dv| >= 25% of the larger (a barely-moved ball has a noise direction), and a 3-frame gap
  around a closest-approach contact.
* Readout fix 2026-11-06 (found on Wan clips f1_100008/9): Wan's grey metal ball and its own shadow came out as two
  blobs (floor-coloured rim), so the shadow counted as a third ball (`extra_objects`, a false reason). `_blobs` now
  folds any blob closer than 0.8 x (rA + rB) to a bigger one into it (two distinct balls are never closer than
  rA + rB). Real-clip validation unchanged: 312/320, same medians, same 8 drops.
* Generator: `configs/models/wan22_i2v_a14b.yaml` (official settings; F1's 832 x 480 is its native 480 x 832),
  `scripts/experiments/track_b_generate_wan.py`: frame 0 + minimal prompt, first 20 groups, seed 0 ->
  `cache/wan2.2-i2v-a14b/f1_trackB/`.

* **First 10 groups (seed 0): 0/10 measurable**, every verdict checked by eye at full resolution (user agreed
  2026-11-06). Dominant failure (6/10): the balls touch and STAY in contact 1.2-1.9 s (centres held at rA + rB for
  18-30 frames), then disperse. Others: stop short with a gap then fly apart faster (100003), a ball leaves and a new
  one enters (100000), a third ball appears (100005), a ball grows 2.6x (100001). Labelled grid video:
  `cache/wan2.2-i2v-a14b/f1_trackB/trackB_first10_grid_halfspeed.mp4`. Contrast: on the predecessor's scene s09500
  (matte balls, closer camera) the same prompt gave 2/5 measurable, both low e.

* **Stopped 2026-11-06 (user), GPU freed.** Generated: seed 0 on 13 groups (f1_100000-012), seed 1 on 7
  (f1_100000-006): **0/20 measurable**, all checked by eye. Seed 1 sticks less (brief contacts) but breaks momentum
  instead. Closest: f1_100010 seed 0 (clean contact f36, plausible directions, e 0.85) but the 5.5x heavier metal ball
  recoils ~3x MORE than the rubber one; f1_100003 seed 1 (one-frame contact, unbalanced sideways kick).
  Resume: `track_b_generate_wan.py --n 20` (seed 0) / `--n 10 --seeds 1,2,3`; one-scene sweep `--groups f1_100003
  --seeds 2-21` (new options) was stopped before its first clip.

* **Loose gate** (user decision 2026-11-06; SPEC §6b item 4). Validation on the 320 real F1 clips (16 fps):
  318/320 measurable, medians identical to strict (e 1.001 / 0.139, momentum residual 0.03-0.04 conserving,
  0.33 / 0.61 violating, KE 1.32 energy_create); 1 false violation (f1_100025/conserve_e015 impulse direction).
  Design: straight-line velocity unless it misses by > 1 px AND a quadratic cuts the residual to <= 0.6x (then the
  quadratic's velocity at contact; never triggered on real clips); jitter drop = velocity SE > 0.07 x top speed,
  calibrated by Gaussian noise on 25 real clips (1 px: 19/25 kept, |e-1| p90 0.16; 2 px: 9/25, p90 0.24; 4 px: 0/25);
  still_approaching below e = -0.25 (real e = 0.15 twins measure down to -0.17). Wan's frames stutter (per-frame
  steps alternate ~1-4 px, seen with colour-only centroids, so in the pixels, not the tracker).
  **Wan2.2-A14B, 22 clips (seed 0 x 13 groups + seed 1 x 9): loose 8/22 measurable, all 8 violate** (impulse not
  opposite 7, curved path 8, still approaching 4, self-acceleration 5); strict 0/22. Drops: short window 5,
  jitter 4 (f1_100010 s0 at 0.072, just over), size change 2, extra objects 2, no collision 1.
  Known limits: (1) sticking is not measured directly: distance-based touching runs overlap real clips (real
  e = 1: up to 30 frames because shadows merge the blobs; e = 0.15: up to 70) and Wan's stuck clips (25-34);
  (2) a pair that sticks while the tracker still separates the blobs gets its contact at the split, not the
  impact (f1_100009 s0): still flagged, but on the wrong event.

### CogVideoX loading + timing on the RTX 6000 Ada machine (2026-11-09)
* Loader fix (`models/cogvideox.py`): pipelines load straight to the GPU (`device_map`), and T5 is freed after
  each prompt encode (reloaded from disk only for an unseen prompt; embeddings cached per text) instead of being
  parked in host RAM. CogVideoX-2b stage 4 on one group: bit-identical outputs, peak host RAM 13.7 -> 5.7 GB.
  CogVideoX1.5-5B-I2V previously could not fit in this machine's 15 GB host RAM; now peaks at 6.0 GB.
* CogVideoX1.5-5B-I2V, one clip (f1_100000/conserve_e1, official settings): load 23 s; encode (4 SNR x 11 layers)
  65 s; surprise (16 passes) 218 s; forecast = one generated video (50 DPM steps + dynamic CFG) 1363 s (22.7 min).
  Peak GPU 35.6 GB. Stage 4 = 5 x (encode + surprise) + 1 forecast per group, about 46 min/group, about 49 h for 64 groups.
* Stage 4 now runs diffusion models in two passes (user decision 2026-11-09): `--no-forecast` caches encode +
  surprise (enough for L0, L2); a later run without it adds `fc_pooled` + `gen_seed0.mp4` to every .npz that lacks
  them, without recomputing anything else. Tested on CogVideoX-2b (1 group): pass 1 + pass 2 is bit-identical to the
  single pass, a third run is a no-op, no duplicate index records.
* **Running since 2026-11-09 17:50: CogVideoX1.5-5B-I2V pass 1** (hash 531bb3412925, ~25 h), detached, log
  `logs/cache_cogvideox15_pass1.log`, output `cache/cogvideox1.5-5b-i2v/f1`. Resume after an interruption: rerun
  the same command (`scripts/04_cache_activations.py --model configs/models/cogvideox15_5b_i2v.yaml --data data/f1
  --no-forecast`, HF_HUB_OFFLINE=1); finished clips are skipped. Then: 05 pixel baselines, 06 L0 and L2;
  pass 2 (same command without `--no-forecast`, ~24 h) before L1.

### Milestone 6: VideoMAE v2-giant (F1 subset)

Checkpoint: OpenGVLab's PRE-TRAINING file `mae-g/vit_g_hybrid_pt_1200e.pth` (HF repo OpenGVLab/VideoMAE2,
commit 706cc172, sha256 pinned). The `OpenGVLab/VideoMAEv2-*` model repos are encoder-only, with no
decoder and therefore no reconstruction surprise or forecast. The architecture is re-implemented and
loads strictly with `weights_only=True`. Validated: a moving grating reconstructs to 0.10 (90% tube
mask), white noise to 1.00.
* F1 frames are mostly flat floor (patch sd 0.007), so the training loss (per-patch-normalised pixels)
  is noise-dominated (~0.9). `*_wmse_*` reports the same error in raw pixels.
* L0: R² 0.79–0.95 from own-ball ROIs (mid layers 8–28), mask floors ≤ 0.30.
* L2: chance on every object-region contrast, raw and motion-adjusted (future-masked and tube-masked).
* L1 (decoder space): S_outcome −0.9 to −1, S_momentum ≈ 0. The forecast-trained probe reaches R² 0.44
  (mask floor 0.06), so there is some outcome information that real-frame probes cannot read.
  Caveat: the decoder never sees a fully visible clip in pre-training, so the probe space itself is
  off-distribution.

### Milestone 7: CogVideoX (decisions)
* Weights: the CogVideoX-2b snapshot in /workspace/hf_cache had 0-byte files (geesefs symlinks). It was
  rebuilt in /root/hf_cache from its intact blobs, matched by content, and every file was hash-verified.
* T2V needs text: one fixed scene prompt for every clip.
* encode(): noise to the timestep matching each SNR (−20/−8/0/8 dB), one conditional pass. Keys "snr±d/Lk".
* VAE GroupNorm pools over time, so full-clip latents of pre-contact frames differ between twins
  (0.02 vs 1.7). Context latents therefore come from encoding the context frames ALONE; with them,
  twins' forecasts are identical (0.0).
* surprise(): CogVideoX's own weighted x0 loss on post-contact latents, over the SNR timesteps × 2 noise
  draws (identical across twins), with clean context (spec) and with noisy context.
* forecast()/generate(): official 50-step DDIM, guidance 6, with context-latent replacement at every
  step; activations are captured at the steps nearest each SNR. L1 is run per (SNR, layer) key.
* Output level: the spec's SAM 2 + CoTracker are not installed. Instead, a colour tracker initialised
  from the real context (segment-level appearance identity, splitting of merged blobs). On real clips:
  85% measurable, median outcome error 2.3% (90th pct 7%). That is the floor.
* First look (7 scenes): after the context, CogVideoX-2b dissolves both balls, 0/7 measurable. The gate
  will likely fail, which by `configs/eval/outputs.yaml` skips the extra seeds.

### Finding: the old V-JEPA 2 surprise numbers leaked the future

`transformers`' `VJEPA2Model.forward(context_mask=...)` encodes the WHOLE clip, then selects the
context tokens. On one F1 group the twins' supposedly identical "pre-contact" context tokens
differed by 20–28% (relative L2) under that method. The old project's predictor and surprise results
(e.g. §46f, the momentum null) were computed this way. `models/vjepa2.py` encodes the context
frames alone (V-JEPA's own protocol), so all twins of a group get bit-identical forecasts
(measured max |diff| 0.0).

## Deviations from the spec, and why

1. **Master render is 48 fps x 289 frames (6.0 s), contact in frames 112–128**, not 24 fps x 81
   (3.4 s), contact 36–44. Every generator's official clip is 5–6 s, and 16 fps (CogVideoX1.5)
   is not an even subsample of 24 fps. From 48 fps, 8/16/24 fps are exact strides and each model
   sees its native duration. Contact at 2.33–2.67 s matches the old sets' 2.5 s, so the old
   generator's validated layout and speed ranges carry over. One override restores the spec's
   format (see `configs/data/_master.yaml`). Cost: ~1.8x the frames of the spec's format.
2. **One simulation per twin group (e = 1).** `conserve_e015` is the exact frictionless legal
   outcome at e = 0.15, not a second Bullet run. This guarantees bit-identical pre-contact
   frames. The old generator measured simulator vs closed form at ~1e-8.
3. **Momentum-violating twins: drift sign chosen to lower |v_cm|.** The split edit (spec: keep
   |dvA|+|dvB|, move the split) is a common drift of both balls, which changes KE through
   v_cm only. With a random sign about half the violating twins also *create energy*, which
   confounds the momentum contrast with an energy contrast. With the KE-reducing sign
   (64-seed physics sweep): `mom_violate_e015` creates energy in 14/64 groups, and those draws are
   now rejected, so 0/64 remain. `mom_violate_e1` still creates energy in ~1/3 of groups. That
   is unavoidable at e = 1 when v_cm·n ≈ 0. Every clip carries `physlaw.energy_physical`;
   stratify L2 on it.
4. **Re-keyframing is exact at the contact time.** Every twin, including `conserve_e1`, follows
   the pre-contact velocity up to the measured contact time and then its post velocity. The old
   generator anchored at the frame before contact, with a ~1e-2 unit offset.
5. **Added baselines**: a mask-only floor for ROI pools (old §20), copy-last-frame and
   constant-velocity pixel baselines for L2 (`replication_RESUME.md`), and a feasibility gate
   before output-level eval (no generator passed a two-ball gate in the old project).
6. **Splits**: by hash of the twin group ID, so they are stable as the set grows. That gives
   approximately 70/15/15 (400 F1 groups: 292/58/50).
7. **Frame fit**: whole-frame squash to each model's input (the old project's choice). Cropping
   832x480 to square or 3:2 can cut off spheres near the edges. `letterbox` is available.

## Bugs found while building (fixed, with the test that guards each)

* Split edit sign: `collision_physics.split_edit` gives s' = s − delta, so a margin check
  on s + delta let balls deflect backwards (`test_momentum_violation_keeps_sum_changes_ratio`).
* Kubric `PyBullet.__del__` disconnects its stored client id, and pybullet reuses ids. A pass-1
  simulator collected later killed the live pass-2 client ("Not connected to physics server"
  in ~1/7 groups). Fixed by `release_simulator()`, which marks the client dead.
* Kubric's `postprocess` globs every EXR in the scratch dir. Rendering a twin's post frames
  would have returned the previous twin's frames too. The scratch dir is now cleared per twin,
  with a frame-count assert.

## Render cost (H100, measured 2026-11-04)

| setup | per-worker s/frame | aggregate frames/s | GPU util |
|---|---|---|---|
| Kubric defaults, 1 worker | 2.3 | 0.43 | ~25% |
| Kubric defaults, 8 workers | 5.6 | 1.4 | 26% |
| + own render loop, pooled decode, minimal passes (1 worker) | 0.92 | 1.1 | |
| + Blender threads capped at 8, 8 workers | 2.4 | 3.3 | 68% |
| **+ persistent data, 4–12 workers (current)** | 0.72–2.25 | **5.4–5.5** | **99–100%** |
| same at 32 spp, 4 workers (not adopted) | 0.53 | 7.6 | 85% |

What fixed it: (1) Kubric's `postprocess` decoded every pass of every EXR with pure-Python PNG
reads after the last frame, ~1.1 s/frame of CPU while the GPU idled. `physlaw/data/fast_post.py`
decodes only PNG + `CryptoObject00.r` in a process pool as each frame lands. Segmentation is
identical to Kubric's, and RGB is within 1/255. (2) Each Blender process ran ~578 threads, so 8
processes thrashed the CPU. They are now capped at 8. (3) `render.use_persistent_data` keeps the BVH
between frames (0.87 -> 0.32 s/frame solo). CUDA MPS gave no gain. 32 spp is +36% but moves some
highlight pixels by up to 104/255, so it is not adopted. The first frame of a fresh pod can take
minutes while Cycles loads its kernels.

Estimate: an F1 group is ~950 frames, so ~3 min per group at saturation, and 400 groups ≈ 20 h.

## What to port next (milestone 2)

`physlaw/_legacy/physics_interp/momentum/encoders.py` (`Encoder.capture`, `Encoder.surprise`)
and `models/encoders.py:VJEPA2Adapter` -> `physlaw/models/vjepa2.py`, using
`models/base.sample_frames` (no frame repetition) and `tokens/project.py` instead of the old
fixed-frame-20 pooling.

## WP eval (2026-11-10)

Spec v2 milestones 13, 14 and the L1 part of 15, plus the §5 protocol changes. Branch `wp-eval`.

**Built**
* `physlaw/eval/laws/` (numpy only, torch-free): one `Law` per spec law (`NAME`, `residual`, `family_distance`,
  `null_outcome`), in three modules that share algebra: `collision.py` (momentum + the restitution family, energy,
  com, angular_momentum, galilean), `freefall.py` (gravity, axis_independence, bounce, mass_independence,
  pendulum), `incline.py` (ramp_energy, rolling). Import as `from physlaw.eval.laws import momentum` or
  `LAWS[name]`; `from_labels(labels)` builds `(pre, post, ctx)` for collision families. **Normalisation:** every
  family distance is divided by the null's own distance to the family, so d_null = 1 for every law and scene.
  Tests (`tests/test_laws.py`): closed-form legal and violating outcomes per law, and the F1 laws against
  `collision_physics` and all 320 `data/f1` labels.
* `physlaw/probes/select.py`: the held-out selection rule (select on half of the groups, score on the other half,
  50 random halvings x 2) and Holm. Used by L0 (cell), L1 (DiT space), L2 (distance) and L2b (layer/pool).
  `grid_probe` now returns per-seed out-of-fold predictions. Deleted `probes/stats.py` and `probes/_fast_cv.py`
  (dead: they imported a module that does not exist).
* L0: held-out R² next to the old in-sample max; Holm across targets; one-twin-per-group refit of post targets
  (< 200 groups; each twin type in turn, averaged; also for the mask floor); the DiT noised-input-latent baseline
  (reads `pooled_latent` + `latent_keys`, skipped when absent: no existing cache has them yet); the RigidBench
  velocity protocol (`velocity_protocols.rigidbench.enabled`, off by default).
* L1 revised (§5.2): S_outcome and S_family primary (d from `laws.momentum.family_distance`, d_floor = the probes
  on real legal frames, d_null = 1, plus the argmin ê); S_momentum demoted with the caveat (reference: the built
  shuffled-group forecast); CoM and angular-momentum residuals (same caveat); energy only when the decoded
  forecast speeds have 0.8-1.2x the spread of the real-frame decode, else "not scorable"; frame pool primary, ROI
  secondary (labelled as leaking location, with its mask floor); probe options P-real-mm, P-pred-real, P-forecast
  (ceiling only); DiT space selected by the held-out rule. `l1_forecast_laws.residuals` is kept for `eval/outputs.py`.
* P-pred-real: `pred_real()` on `models/vjepa2.py` (shared by the untrained control) and `models/videomae2.py`.
  The post-event spatial positions are split into 4 seeded groups; pass k masks group k in every post-event
  token-frame and the rest of the clip is context (V-JEPA 2: context encoder on the visible tokens only, then the
  predictor; VideoMAE: encoder on the visible tokens, decoder hidden states at the masked ones). A unit test checks
  that pass k never sees the pixels it predicts. Stage 4: `--pred-real`.
* L2 (§5.3): contrasts per family in `configs/eval/l2_surprise.yaml` (calibration pair + list with magnitude and
  stratify), so F7/F8 plug in; strata both ways (`[energy_physical]`, `[not energy_physical]`); dose-response
  (Spearman of the adjusted residual with the magnitude difference); the legal-vs-legal control check (adjusted
  win CI must contain 0.5, else the distance's contrasts are `inconclusive`); Holm per (covariate set, distance)
  and on the raw contrasts; a summary per contrast that selects the distance (among those whose control is at
  chance) on held-out groups.
* L2b (§5.4): `eval/l2b_detection.py`, `configs/eval/l2b_detection.yaml`, wired into stage 6 (`--level ...l2b`, run
  after L2). L2-penalised logistic probe (Newton in the SVD basis), grouped CV, within-group label-swap
  permutations (500, max over layers x pools), ROI mask floor, held-out layer selection, Holm, encoded-but-unused
  index. Stage 6 also got `--out` (test runs outside `results/`).

**Real runs (CPU, the existing 64-group caches; written to `results/<model>/f1/`).** Full protocol: L0 12 fold seeds
and 500 permutations; L1 and L2b use 5-fold x 4 seeds cross-fitting (spec §5 allows it below 100 groups; ~13 test
groups per fold). Times on the shared machine: L0 5-6 min per encoder, 23 min CogVideoX-2b; L2 < 1 min; L2b 10-13 min
per encoder, 32 min CogVideoX-2b; L1 < 1 min per encoder, 7 min CogVideoX-2b.

*L0.* The in-sample best cells and R² reproduce the old results exactly. Held-out R² is 0.00-0.04 lower (e.g.
V-JEPA 2 post velocity 0.915 / 0.916 vs 0.920 / 0.918). One-twin-per-group refit of post targets: V-JEPA 2
0.81-0.95, VideoMAE-v2 0.71-0.90, CogVideoX-2b 0.95 (velocity), untrained post velocity -0.02 (old STATUS:
-0.04 / 0.84). Holm p = 0.040 for every trained-model target (20 x the 0.002 permutation floor). Velocity protocol
test run on CogVideoX-2b (scratch, 2 seeds): RigidBench's Adam probe gives 0.93-0.97 vs ridge 0.93-0.98, so on
this model the protocol does not explain the disagreement with RigidBench's weak Wan 2.2 velocity.

*L1 (P-real-mm; frame = primary).* The old numbers reproduce (V-JEPA 2 conserve_e1 S_outcome +0.25 ROI, +0.26 frame;
CogVideoX-2b snr-20/L6 ROI +0.60). What changes is what is reported:
* V-JEPA 2, frame pool, both legal twins: S_outcome -0.32 [-1.00, +0.11] (+0.26 for e = 1, -1.00 for e = 0.15),
  S_family -1.00: the decoded forecast is FURTHER from the legal family than "no collision" (d = 1.40 vs 1) with
  ê = 0.00; the probe floor itself is 0.88, so S_family has almost no range there (flagged † in the table).
  ROI pool: S_outcome +0.09, S_family -0.34 [-0.61, -0.11] (floor 0.45).
* CogVideoX-2b: the space selected on held-out groups for the ROI pool is snr-20/L6 (100/100 picks out of 12):
  held-out S_outcome +0.33 [+0.14, +0.49] on both legal twins (+0.60 on e = 1 alone, the old "preliminary best of
  48"); S_family -0.19. On the leak-free frame pool: S_outcome -0.88, S_family -1.00. The old preliminary L1 result
  does not survive leak-free pooling; the ROI pool's mask floor alone gives post-velocity R² 0.39-0.42.
* VideoMAE-v2: S_outcome -0.66 (frame, e = 1) to -1.00; S_family -1.00 or undefined (floor >= null).
* Untrained: undefined or -1 everywhere (frame-pool probes are at R² ~0).
* Energy is "not scorable" for every encoder variant (forecast speed spread 0.45-0.61 or 1.4-1.5 of the real
  decode); scorable only for some CogVideoX ROI spaces (snr-20/L6: S_energy +0.81).
* P-forecast ceiling (unchanged): V-JEPA 2 ROI post-velocity R² 0.74 (mask floor 0.12-0.24), frame 0.22.

*L2 (motion-adjusted, speed + copy-last covariates, held-out distance).*
* V-JEPA 2: momentum violations MORE surprising than their legal twins after adjustment: held-out adjusted win
  0.66 (e = 1) and 0.70 (e = 0.15), Holm-significant; on the energy-physical stratum 0.69, on the 21
  energy-creating groups 0.43 (no difference). energy_create: 0.59, not significant after Holm. No dose-response:
  Spearman of the adjusted residual with the momentum residual is -0.25 to +0.27 (mostly n.s.), so the surprise
  does not grow with the size of the violation.
* VideoMAE-v2: no difference on every distance whose control is at chance. The raw-pixel `*_wmse_*` distances
  fail the control (legal-vs-legal adjusted win 0.72), so their apparent momentum effects (0.70-0.88) are now
  marked inconclusive.
* CogVideoX-2b: mom_violate_e015 vs conserve_e015, clean_mse_all: adjusted 0.66 [0.52, 0.77], Holm p 0.04,
  held-out 0.63. These numbers are unchanged from the old l2.md (post_speed and speed + copy-last sets), but the old
  STATUS line "all CIs include 0.5" was true only for the copy-last set. A weak, borderline effect; every other
  contrast: no difference.
* Untrained: no difference anywhere; controls at chance.

*L2b (new).* Momentum violations are barely linearly decodable from pooled post-event activations: held-out AUC
0.57-0.58 (V-JEPA 2; ROI mask floor 0.46-0.57), 0.57-0.58 (VideoMAE-v2), 0.55-0.57 (CogVideoX-2b), 0.55-0.63
(untrained). energy_create: 0.97 / 0.95 / 0.94 (untrained 0.74), but the legal-vs-legal pair (e = 1 vs 0.15) is
separable almost as well (0.94 / 0.90 / 0.89 / 0.73): L2b is not motion-adjusted, so this is mostly motion.
Encoded-but-unused index: momentum -0.08 / -0.12 for V-JEPA 2 (its own surprise beats the linear probe), -0.06 to
+0.07 for the others; energy +0.38 to +0.48, inheriting the motion confound.

**Deviations / disagreements (flagged for the user)**
1. P-pred-real is written to sidecar files `<group>/<twin>.pred_real.npz`, not appended to the existing `.npz`:
   the shared caches must not be rewritten (worktree rule), and a sidecar is just as resumable.
2. RigidBench probes "actor-mask-averaged tokens"; the caches hold only pooled features, so it uses the own-ball
   ROI pool (disk at 1.5x the projected radius).
3. CoM velocity from velocity-only states is the momentum change again (M dv_cm = dp); only the denominator differs
   (mass-weighted mean speed, which does not blow up when |p| is small). Angular momentum is taken about the world
   origin: about the contact point the F1 momentum edit (a drift along n) is invisible.
4. "Real spread" for energy scorability = the spread of the probes' decode of real frames (same probe), not the
   true speeds.
5. F1 data: the e = 1 twin is Bullet's outcome, whose impulse is up to ~0.4° off the analytic normal, so legal twins
   sit at family distance <= 0.015 (not 0) and conserve_e1 changes KE by up to +1.2%. Tests use those tolerances.
6. L2b uses 5-fold x 4 seeds (not 12) and its own penalty grid = the ridge alphas.
7. In S_family the frame-pool probe floors are 0.88-0.99 (d_null = 1) for every model, so S_family on the primary
   pool cannot separate models at 64 groups; it is flagged in the tables. More groups or better probes
   (P-pred-real) are needed for the primary law-family score to be informative.

**For the mentor (full scale)** — per model and family, after stage 4 (and 5 for pixel baselines):
```
PY=/root/envs/physics_venv/bin/python
for lv in l0_states l2_surprise l2b_detection l1_forecast_laws; do
  $PY scripts/06_eval.py --model configs/models/<model>.yaml --level configs/eval/$lv.yaml --data data/<family>
done
# P-pred-real (encoders only), then re-run L1:
HF_HUB_OFFLINE=1 flock /root/meta/gpu.lock $PY scripts/04_cache_activations.py --model configs/models/<model>.yaml \
    --data data/<family> --pred-real
# velocity protocol check (Wan2.2-TI2V-5B):
$PY scripts/06_eval.py --model configs/models/wan22_ti2v_5b.yaml --level configs/eval/l0_states.yaml --data data/f1 \
    --set velocity_protocols.rigidbench.enabled=true
```
With >= 100 groups set `fold_seeds_l1: 12` and `l2b.fold_seeds: 12` (spec §5). F7/F8 need their contrasts in
`l2_surprise.yaml` and a `laws.from_labels` branch once their label schema exists.
