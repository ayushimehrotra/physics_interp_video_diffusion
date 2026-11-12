#!/usr/bin/env python3
"""Stage 8 - track generated videos and read out the event (spec §6.1): one row per video in a readout file.

    /root/envs/physics_venv/bin/python scripts/08_track_outputs.py --track a --model configs/models/cogvideox_2b.yaml --data data/f1
    /root/envs/physics_venv/bin/python scripts/08_track_outputs.py --track b --model configs/models/wan22_i2v_a14b.yaml \\
        --data data/f1 [--real] [--review]

Tracker and readouts: physlaw/eval/outputs.py (colour tracker initialised from the real context; it replaces the
spec's SAM 2 + CoTracker, see its docstring). Rows hold the measurement and the pre / post states; stage 9 scores
them with physlaw.eval.laws.
Track A reads cache/<model>/<family>/<group>/gen_seed*.mp4 and writes _readout_trackA.jsonl (generated) and
  _readout_trackA_real.jsonl (the real legal twins at the model's geometry: the tracker floor).
Track B reads cache/<model>/<family>_trackB/<group>/gen_seed*.mp4 and writes _readout.jsonl (strict gate) and
  _readout_loose.jsonl. --real reads the REAL clips (every twin, resampled to the model's fps and size) and writes
  _readout_real[_loose].jsonl: the per-family readout validation of §6.1. --review draws a sheet per clip.
--gen / --out override the input / output directory (default: both the generation directory).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from physlaw.config import load_config, resolve_path  # noqa: E402
from physlaw.eval import outputs as O  # noqa: E402


def review(frames, r, pix, path, title):
    """Tracked centres on frame 0 (cyan A, magenta B; bright before contact, dark after) + 8 frames + verdict."""
    import numpy as np
    from PIL import Image, ImageDraw
    H, W = frames.shape[1:3]
    bg = Image.fromarray(frames[0]).convert("RGB")
    d = ImageDraw.Draw(bg)
    c0 = r.get("contact_frame")
    for k, col in ((0, (0, 200, 255)), (1, (255, 0, 200))):
        for t, p in enumerate(pix[:, k]):
            if np.isfinite(p).all():
                fill = col if (c0 is None or t < c0) else tuple(int(x * 0.5) for x in col)
                d.ellipse((p[1] - 3, p[0] - 3, p[1] + 3, p[0] + 3), fill=fill)
    tw, th = 300, round(300 * H / W)
    im = Image.new("RGB", (tw * 7, th * 2 + 40), (0, 0, 0))
    D = ImageDraw.Draw(im)
    D.text((4, 3), title[:200], fill=(255, 255, 255))
    ok = r["measurable"]
    laws = "  ".join(f"{k}={r[k]:.2f}" for k in O.STAT_KEYS
                     if isinstance(r.get(k), (int, float)) and np.isfinite(r[k])) or "laws: n/a"
    D.text((4, 18), f"{'MEASURABLE' if ok else 'REJECTED'} reason={r.get('reason')} violations={r.get('violations')} "
                    f"contact f{c0}->f{r.get('separate_frame')} | {laws} | size x{r.get('size_ratio', float('nan')):.2f}",
           fill=(120, 255, 120) if ok else (255, 150, 150))
    im.paste(bg.resize((tw * 3, th * 2)), (0, 40))
    for j, i in enumerate(np.linspace(0, len(frames) - 1, 8).round().astype(int)):
        t = Image.fromarray(frames[i]).resize((tw, th))
        ImageDraw.Draw(t).text((4, 4), f"f{i}", fill=(255, 255, 0))
        im.paste(t, (tw * 3 + (j % 4) * tw, 40 + (j // 4) * th))
    im.save(path)


def _track_b_job(args):
    """One clip x all gates (worker): generated (mp4 path) or real (clip dir)."""
    src, clip_dir, fps, hw, gates, review_dir, tag = args
    from physlaw.data.io import load_clip
    c = load_clip(clip_dir)
    v = O._read_mp4(src) if src else O.real_video(c, fps, hw)
    fn = O.READOUT_SELF[c.pl["family"]]
    out = {}
    for g in gates:
        r = fn(v, c, fps, gate=g)
        if review_dir:
            _, sig, start = O.clip_inputs(c, [0], v.shape[1], v.shape[2], "squash", 1)
            pix = O.track(v, sig, start)["pixel"]
            sfx = "" if g == "strict" else f"_{g}"
            review(v, r, pix, Path(review_dir) / f"review_{tag}{sfx}.png", f"{tag} gate={g}")
        out[g] = O._row(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--track", choices=["a", "b"], required=True)
    ap.add_argument("--gen", default=None, help="generated videos (default cache/<model>/<family>[_trackB])")
    ap.add_argument("--out", default=None, help="readout directory (default: --gen)")
    ap.add_argument("--real", action="store_true", help="track B: read out the real clips instead (§6.1 validation)")
    ap.add_argument("--gates", default=",".join(O.GATES))
    ap.add_argument("--review", action="store_true", help="track B: review sheet per clip")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    mcfg = load_config(a.model)
    data = resolve_path(a.data)
    gen = Path(a.gen) if a.gen else REPO / "cache" / mcfg.name / (data.name if a.track == "a" else f"{data.name}_trackB")
    out = Path(a.out) if a.out else gen
    out.mkdir(parents=True, exist_ok=True)
    dump = lambda name, rows: (out / name).write_text("".join(json.dumps(r, default=float) + "\n" for r in rows))

    if a.track == "a":
        gen_rows, real_rows = O.track_a_rows(gen, data, O.Geometry(mcfg))
        dump("_readout_trackA.jsonl", gen_rows)
        dump("_readout_trackA_real.jsonl", real_rows)
        print(f"[08] track A: {sum(r['measurable'] for r in gen_rows)}/{len(gen_rows)} generated measurable, "
              f"real {sum(r['measurable'] for r in real_rows)}/{len(real_rows)} -> {out}")
        return

    off = mcfg["official"]
    fps, hw = float(off["fps"]), (int(off["height"]), int(off["width"]))
    gates = a.gates.split(",")
    if a.real:
        from physlaw.data.io import list_clips
        jobs = [(None, str(p), fps, hw, gates, str(out) if a.review else None, f"{p.parent.name}_{p.name}")
                for p in list_clips(data)]
        keys = [{"group": Path(j[1]).parent.name, "twin": Path(j[1]).name} for j in jobs]
    else:
        mp4s = sorted(gen.glob("*/gen_seed*.mp4"))
        jobs = [(str(p), str(data / p.parent.name / O.legal_twins(data / p.parent.name)[0]), fps, hw, gates,
                 str(p.parent) if a.review and out == gen else (str(out) if a.review else None),
                 f"{p.parent.name}_seed{p.stem.split('seed')[1]}") for p in mp4s]
        keys = [{"group": p.parent.name, "seed": int(p.stem.split("seed")[1])} for p in mp4s]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(_track_b_job, jobs))
    for g in gates:
        rows = [{**k, **r[g]} for k, r in zip(keys, res)]
        dump(O.readout_file(g, a.real), rows)
        print(f"[08] track B {'real' if a.real else 'generated'} gate={g}: "
              f"{sum(r['measurable'] for r in rows)}/{len(rows)} measurable -> {out / O.readout_file(g, a.real)}")


if __name__ == "__main__":
    main()
