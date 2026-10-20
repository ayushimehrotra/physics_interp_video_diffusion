# Running the momentum protocols

All analysis is CPU-only numpy and reads capture directories; only
`capture_encoder.py` and rendering need a GPU. Every script writes a JSON
artifact under `docs/artifacts/` (or `--out`).

## 0. Smoke test without a GPU

```bash
python tests/test_momentum.py                    # physics identities, stats, generator edits
python - <<'EOF'
from pathlib import Path
from physics_interp.momentum import synthetic as syn
root = Path("outputs/synthetic")
syn.write_synthetic_captures(syn.make_conservation_set(150), root / "conservation_set_v1/synthetic/activations")
syn.write_synthetic_captures(syn.make_ladder_set(20),        root / "energy_ladder_v1/synthetic/activations")
syn.write_synthetic_captures(syn.make_three_arm_set(150),    root / "energy_voe_450/synthetic/activations")
syn.write_synthetic_captures(syn.make_ladder_set(150, seed0=101, arms=syn.LADDER_LEGAL + syn.LADDER_IMPOSSIBLE,
                             with_violating=True, pinned=(1.0, 0.15)), root / "scene_ladder_v1/synthetic/activations")
