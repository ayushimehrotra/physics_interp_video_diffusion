#!/usr/bin/env python3
"""Contact sheet: one row per scene, one column per key frame, as a single PNG.

    python3 contact_sheet.py --dataset-dir pilot --num-scenes 5 --seed 0

The point is to eyeball whether the collision reads clearly at 720x480, so frames
either side of contact are included and the contact frame is called out. Each row is
annotated with the parameters that determine what the collision should look like.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FRAMES = (0, 10, 18, 20, 22, 30, 48)
# A friction scene has no contact event, so the collision layout wastes four of
# its seven columns on the same instant. Spread them over the whole slide.
FRAMES_FRICTION = (0, 8, 16, 24, 32, 40, 48)
CONTACT_FRAME = 20
THUMB_WIDTH = 360          # each 720x480 frame is halved
LABEL_HEIGHT = 22          # column header strip
ROW_LABEL_HEIGHT = 40      # per-scene caption strip
PAD = 6
BG = (252, 252, 251)
INK = (11, 11, 11)
INK_MUTED = (82, 81, 78)
ACCENT = (208, 59, 59)     # marks the contact frame column


def load_font(size, bold=False):
  candidates = [
      f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
      f"/usr/share/fonts/truetype/liberation/LiberationSans-{'Bold' if bold else 'Regular'}.ttf",
  ]
  for path in candidates:
    if Path(path).exists():
      return ImageFont.truetype(path, size)
  return ImageFont.load_default()


def layout(labels):
  """-> (frames, contact_frame). Friction scenes have no contact to mark."""
  if "sampled_parameters" in labels:
    return FRAMES, CONTACT_FRAME
  return FRAMES_FRICTION, None


def friction_caption(labels):
  """The incline and slide sets: one object, and mu is the variable of interest."""
  f, a = labels["friction"], labels["objects"]["A"]
  return (
      f"{labels['scene_id']}  (seed {labels['seed']})   "
      f"scenario {labels['scenario']}   arm {labels['arm']}   "
      f"cube size {a['size']:.2f}   albedo {labels['objects']['B'].get('albedo', float('nan')):.2f}",
      f"mu {f['mu']:.3f}   theta {f['theta_deg']:.1f}°   "
      f"accel {f['accel']:.3f} (measured {f['accel_measured']:.3f}, "
      f"rel err {f['accel_rel_error']:.1e})   "
      f"mu from kinematics {f['mu_from_kinematics']:.3f}   "
      f"travel {f['travel']:.2f}   speed_final {f['speed_final']:.3f}")


def caption(labels):
  if "sampled_parameters" not in labels:
    return friction_caption(labels)
  objects, collision = labels["objects"], labels["collision"]
  params = labels["sampled_parameters"]
  return (
      f"{labels['scene_id']}  (seed {labels['seed']})   "
      f"A: {objects['A']['material']} r={objects['A']['radius']:.2f} m={objects['A']['mass']:.2f}   "
      f"B: {objects['B']['material']} r={objects['B']['radius']:.2f} m={objects['B']['mass']:.2f}   "
      f"mass ratio {labels['mass_ratio']:.2f}",
      f"speed {params['speed_a']:.3f}   direction {params['direction_a_rad']*57.2958:.0f}°   "
      f"|b|/(rA+rB) {params['impact_parameter_frac']:.2f}   e={labels['restitution']:g} "
      f"(measured {collision['restitution_measured']:.3f})   "
      f"contact frame {collision['actual_frame']:.3f}   "
      f"KE_post/KE_pre {collision['kinetic_energy_ratio']:.3f}   "
      f"|ΔP|/|P| {collision['momentum_rel_error']:.1e}")


def build(scene_dirs, out_path: Path):
  font_small = load_font(12)
  font_head = load_font(13, bold=True)
  font_cap = load_font(13, bold=True)

  frames, contact = layout(json.loads((scene_dirs[0] / "labels.json").read_text()))

  probe = Image.open(scene_dirs[0] / "frames" / "frame_0000.png")
  scale = THUMB_WIDTH / probe.width
  thumb_h = int(round(probe.height * scale))

  cols, rows = len(frames), len(scene_dirs)
  width = PAD + cols * (THUMB_WIDTH + PAD)
  height = PAD + LABEL_HEIGHT + rows * (ROW_LABEL_HEIGHT + thumb_h + PAD)

  sheet = Image.new("RGB", (width, height), BG)
  draw = ImageDraw.Draw(sheet)

  for c, frame in enumerate(frames):
    x = PAD + c * (THUMB_WIDTH + PAD)
    is_contact = frame == contact
    text = f"frame {frame}" + ("  ← contact" if is_contact else "")
    draw.text((x, PAD + 3), text, font=font_head, fill=ACCENT if is_contact else INK_MUTED)

  y = PAD + LABEL_HEIGHT
  for scene_dir in scene_dirs:
    labels = json.loads((scene_dir / "labels.json").read_text())
    line1, line2 = caption(labels)
    draw.text((PAD, y + 2), line1, font=font_cap, fill=INK)
    draw.text((PAD, y + 20), line2, font=font_small, fill=INK_MUTED)
    y += ROW_LABEL_HEIGHT

    for c, frame in enumerate(frames):
      x = PAD + c * (THUMB_WIDTH + PAD)
      image = Image.open(scene_dir / "frames" / f"frame_{frame:04d}.png").convert("RGB")
      sheet.paste(image.resize((THUMB_WIDTH, thumb_h), Image.LANCZOS), (x, y))
      if frame == contact:
        draw.rectangle([x, y, x + THUMB_WIDTH - 1, y + thumb_h - 1], outline=ACCENT, width=2)
    y += thumb_h + PAD

  sheet.save(out_path)
  return out_path, sheet.size


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--dataset-dir", type=str, required=True)
  parser.add_argument("--num-scenes", type=int, default=5)
  parser.add_argument("--seed", type=int, default=0, help="seed for choosing which scenes")
  parser.add_argument("--output", type=str, default=None)
  parser.add_argument("--scenes", type=str, nargs="*", default=None,
                      help="explicit scene directory names instead of a random choice")
  args = parser.parse_args()

  dataset_dir = Path(args.dataset_dir)
  available = sorted(d for d in dataset_dir.glob("scene_*") if (d / "labels.json").exists())
  if not available:
    print(f"no scenes found in {dataset_dir}", file=sys.stderr)
    return 2

  if args.scenes:
    chosen = [dataset_dir / name for name in args.scenes]
  else:
    chosen = random.Random(args.seed).sample(available, min(args.num_scenes, len(available)))
    chosen.sort()

  out_path = Path(args.output) if args.output else dataset_dir / "_validation" / "contact_sheet.png"
  out_path.parent.mkdir(parents=True, exist_ok=True)
  path, size = build(chosen, out_path)
  print(f"wrote {path} ({size[0]}x{size[1]}) from {', '.join(d.name for d in chosen)}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
