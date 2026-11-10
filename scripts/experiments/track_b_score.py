#!/usr/bin/env python3
"""Score output-track-B clips (spec §6b) with eval/outputs.readout_self and draw a review sheet per clip:
tracked centres on frame 0 (cyan A, magenta B; bright before contact, dark after) + frames + verdict + laws.

    /root/envs/physics_venv/bin/python scripts/experiments/track_b_score.py \\
        --model configs/models/wan22_i2v_a14b.yaml --data data/f1
Writes cache/<model>/<family>_trackB/_readout.jsonl (rewritten each run) and <group>/review_seed<s>.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from physlaw.config import load_config, resolve_path  # noqa: E402


def review(frames, r, labels, path, title):
    import numpy as np
    from PIL import Image, ImageDraw
    from physlaw.data.io import load_clip  # noqa: F401
    H, W = frames.shape[1:3]
    T = len(frames)
    bg = Image.fromarray(frames[0]).convert("RGB")
    d = ImageDraw.Draw(bg)
    c0 = r.get("contact_frame")
    for k, col in ((0, (0, 200, 255)), (1, (255, 0, 200))):
        for t, p in enumerate(r["_pix"][:, k]):
            if np.isfinite(p).all():
                fill = col if (c0 is None or t < c0) else tuple(int(x * 0.5) for x in col)
                d.ellipse((p[1] - 3, p[0] - 3, p[1] + 3, p[0] + 3), fill=fill)
    tw, th = 300, round(300 * H / W)
    idx = np.linspace(0, T - 1, 8).round().astype(int)
    im = Image.new("RGB", (tw * 3 + tw * 4, th * 2 + 40), (0, 0, 0))
    D = ImageDraw.Draw(im)
    D.text((4, 3), title[:200], fill=(255, 255, 255))
    ok = r["measurable"]
    laws = (f"e={r['e']:.2f}  momentum residual={r['r_momentum']:.2f}  KE ratio={r['ke_ratio']:.2f}"
            if r.get("e") is not None and np.isfinite(r.get("e", np.nan)) else "laws: n/a")
    D.text((4, 18), f"{'MEASURABLE' if ok else 'REJECTED'} reason={r.get('reason')} violations={r.get('violations')} contact f{c0}->f{r.get('separate_frame')} | "
                    f"{laws} | size x{r.get('size_ratio', float('nan')):.2f}", fill=(120, 255, 120) if ok else (255, 150, 150))
    im.paste(bg.resize((tw * 3, th * 2)), (0, 40))
    for j, i in enumerate(idx):
        t = Image.fromarray(frames[i]).resize((tw, th))
        ImageDraw.Draw(t).text((4, 4), f"f{i}", fill=(255, 255, 0))
        im.paste(t, (tw * 3 + (j % 4) * tw, 40 + (j // 4) * th))
    im.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--gate", default="strict", choices=["strict", "loose"],
                    help="loose: only measurement failures drop a clip; law breaks are listed as violations")
    a = ap.parse_args()
    import numpy as np
    from physlaw.data.io import load_clip
    from physlaw.eval import outputs as O
    cfg = load_config(a.model)
    data = resolve_path(a.data)
    out = REPO / "cache" / cfg.name / f"{data.name}_trackB"
    fps = float(cfg["official"]["fps"])
    rows = []
    for mp4 in sorted(out.glob("*/gen_seed*.mp4")):
        gid, seed = mp4.parent.name, int(mp4.stem.split("seed")[1])
        c = load_clip(data / gid / "conserve_e1")
        v = O._read_mp4(mp4)
        r = O.readout_self(v, c, fps, gate=a.gate)
        _, sig, start = O.clip_inputs(c, [0], v.shape[1], v.shape[2], "squash", 1)
        r["_pix"] = O.track(v, sig, start)["pixel"]
        sfx = "" if a.gate == "strict" else f"_{a.gate}"
        review(v, r, c.labels, mp4.parent / f"review_seed{seed}{sfx}.png", f"{cfg.name} track B {gid} seed{seed} gate={a.gate}")
        r.pop("_pix")
        rows.append({"group": gid, "seed": seed, **r})
        e = r.get("e")
        print(f"{gid} seed{seed}: measurable={r['measurable']} reason={r.get('reason')} "
              f"e={e if e is None else round(e, 2)} r_p={r.get('r_momentum') and round(r['r_momentum'], 2)} "
              f"KE={r.get('ke_ratio') and round(r['ke_ratio'], 2)} size={r['size_ratio']:.2f} violations={r.get('violations')}", flush=True)
    (out / f"_readout{sfx}.jsonl").write_text("\n".join(json.dumps(x, default=float) for x in rows) + "\n")
    m = [x for x in rows if x["measurable"]]
    print(f"[trackB] {len(m)}/{len(rows)} measurable")


if __name__ == "__main__":
    main()
