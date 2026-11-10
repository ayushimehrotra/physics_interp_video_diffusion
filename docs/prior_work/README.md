# Prior work: physics_interp_video_diffusion (copied 2026-11-04)

These files are copies for reference. **Do not cite their numbers as PhysLaw results.**

Why they are not carried over:

1. **Diffusion models were run off their official configurations.** Known cases:
   CogVideoX1.5-5B-I2V at 45 frames, outside its supported 16N+1 lengths (§65, §68);
   CogVideoX-2b probed along an inverted trajectory with project-specific inversion settings
   (old `configs/momentum_probe.yaml`: unguided, empty prompt, 50-step Euler); Wan2.2 driven by a custom re-noise-and-paste loop rather than its pipeline (§67).
   Only §70 (CogVideoX-5b-I2V) states that it used official settings.
2. **Encoders were fed off-spec clips too**: V-JEPA2 saw every frame twice (`"5:37:1x2"`),
   halving apparent speed.
3. **The data format changes**: every old set is 49 frames at 8 fps with contact fixed at
   frame 20; PhysLaw renders 289 frames at 48 fps with contact jittered (configs/data/_master.yaml).
4. Several headline results were already revised or contradicted inside the old project
   (e.g. the §46f energy "dissipation prior" did not survive the full ladder in
   `replication_RESUME.md`, where a copy-last-frame baseline reproduced it).

What IS worth keeping from these files is **methodology**: the identity trap (§12), the
winner's curse / grid-max nulls (§2), the pooling-mask leak (§20), mass-free shortcuts in
violation designs (§38, §44), and the Kubric/Bullet physics fixes (DATASETS_README_old_generator.md).
Those are built into PhysLaw's configs and tests.
