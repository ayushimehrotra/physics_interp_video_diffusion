# Model smoke matrix (2026-11-21)

Does every model run through the benchmark pipeline? `scripts/queues/run_models.py --smoke 2` on the first 2 groups
of the frozen F1 render (`data/f1`, 10 clips), one A100 80GB PCIe shared with other jobs (`--max-gb 40` for the
first pass). No data was generated for this. Smoke results are pipeline checks only (2 groups: L2b has no scorable
folds, L4 / gates are uninformative); never read them as model results.

Rule (user): one size per architecture; a bigger sibling with the same adapter path is not re-tested. CogVideoX
(2b, 5b, 1.5-5B-I2V) was already run on the pipeline and was not re-tested.

| model config | adapter path | stages run (all exit 0) | stage 04 / 07 min | peak GPU GB | covers |
|---|---|---|---|---|---|
| vjepa2_vitl | VJEPA2Adapter (continuation) | 04, 04 --pred-real, 05, L0, L2, L2b, L1, L4 | 2.8 / - | 3.6 | vjepa2_vitg |
| untrained_vitl | UntrainedVJEPA2Adapter | 04, 04 --pred-real, 05, L0, L2, L2b, L1, L4 | 1.2 / - | 3.6 | |
| videomae_v2_giant | VideoMAEv2Adapter (pre-train ckpt) | 04, 05, L0, L2, L2b | 1.5 / - | 5.0 | |
| videomae_v2_base | VideoMAEv2Adapter (HF encoder-only) | 04, L0, L2b (+05, L2 before the fix) | 0.3 / - | 1.1 | |
| videorepa_5b | CogVideoXAdapter + LoRA | 04, 05, L0, L2, L2b | 9.7 / - | 24.8 | cogvideox_5b |
| wan21_t2v_1_3b | WanAdapter (Wan2.1 VAE) | 04, 05, L0, L2, L2b | 11.9 / - | 17.7 | wan21_t2v_14b |
| wan22_ti2v_5b | WanAdapter (Wan2.2 VAE) | 04, 05, L0, L2, L2b | 13.5 / - | 35.5 | |
| physrvg | WanAdapter + DiT + LoRA, native_* surprise | 04, 05, L0, L2, L2b, track A 07/08/09 | 8.1 / 6.0 | 27.6 | |
| wan22_ti2v_5b_physrvg | WanAdapter, PhysRVG settings | 04, 05, L0, L2, L2b | 5.6 / - | 23.1 | |
| wan21_vace_1_3b | WanVACEAdapter (track A variant) | track A 07/08/09 | - / 24.4 | 27.3 | wan21_vace_14b |
| wan22_i2v_a14b | WanI2VAdapter (2 experts) | track B 07/08 (+ --real)/09 | - / 65.2 (~35 of it loading weights) | 65.0 | |
| wan21_i2v_14b | WanI2VAdapter (CLIP image encoder) | track B 07/08 (+ --real)/09 | - / 46.8 | 43.4 | |
| ltx_video_13b_distilled | LTXConditionAdapter (track A native) | track A 07/08/09 | - / 22.5 | 42.7 | |

Not runnable here (skipped by the planner with the reason): **cosmos_predict25_2b** (gated weights not downloaded,
needs `/root/envs/cosmos_venv`), **positive_control** (PC1) and **pc2_wan22_ti2v_5b** (train with stage 10 first).

Found and fixed:
* VideoMAE-v2 base has encoder-only weights (`predicts` False): its cache has empty `surprise`, so stage 05 / L2
  produced an L2 with no model rows. The config now says `predicts: false`; the planner runs L0 / L2b / L3 only.
* `scripts/00_setup_envs.sh`: the kubric build fails at random (its version is a build-time timestamp); now retried.

Notes for the full run:
* The 14B weights are fp32 safetensors on /workspace (Wan2.2-I2V-A14B: 108 GB): loading alone takes ~35 min per
  stage-07 call on this pod. Peaks: A14B needs an 80 GB GPU; everything else fits in 48 GB.
* Smoke mode limits track A only through the gate (`track_a.gate.n_clips`): when the gate passes (LTX did), stage 07
  continues to every group (seeds_per_clip 1): ~25 min for F1 with LTX.
* Not smoke-run (bigger siblings, same adapter path): wan21_t2v_14b, wan21_vace_14b, vjepa2_vitg, cogvideox_5b.

Track A gate on 2 clips: PhysRVG 0/2 measurable, VACE-1.3B see `results/wan2.1-vace-1.3b/f1/outputs_trackA.md`
(pipeline runs; the gate needs the full 20 clips to mean anything).

Raw: `logs/_status.tsv` (task, exit code, minutes), `logs/<task>.log`, `results/<model>/f1/`.
