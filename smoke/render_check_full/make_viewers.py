#!/usr/bin/env python3
"""Full-resolution render-check viewers (2026-11-12): per-twin mp4s at 832 px, a comparison grid, and 2-ups.

    python smoke/render_check_full/make_viewers.py smoke/render_check_full/f4_camboost \
        --pairs usmall_e1:cam_usmall_e1 ularge_e1:cam_ularge_e1 --cols 5 --tile 384
    python smoke/render_check_full/make_viewers.py smoke/render_check_full/f8_ramp_v2 --pairs eps1.0:eps1.5

Reuses smoke/render_check/make_videos.py (labels summary, annotation: twin name, red EVENT border, frame counter).
Writes <dir>/viz/<group>__<twin>.mp4 (832 px, crf 20), <group>__compare.mp4 (grid, frame-synced; crf raised until
<= 4 MB), <group>__2up__<a>__<b>.mp4 (two 832 px tiles side by side) and summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "render_check"))
import make_videos as MV  # noqa: E402


def enc(inputs, fc, dst: Path, fps: int, crf: int, cap: int = 4 << 20) -> int:
    while True:
        MV.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", fc, "-map", "[o]", "-r", str(fps),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "medium",
                "-movflags", "+faststart", str(dst)])
        if dst.stat().st_size <= cap or crf >= 40:
            return crf
        crf += 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fam_dir", type=Path)
    ap.add_argument("--pairs", nargs="*", default=[], help="2-ups, <twin_a>:<twin_b>")
    ap.add_argument("--cols", type=int, default=0)
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--crf", type=int, default=24)
    a = ap.parse_args()
    viz = a.fam_dir / "viz"
    viz.mkdir(exist_ok=True)
    out = {}
    for g in sorted(p for p in a.fam_dir.iterdir() if p.is_dir() and p.name.startswith("f") and "nuisance" not in p.name):
        twins = [MV.twin_info(t) for t in sorted(g.iterdir()) if (t / "video.mp4").exists()]
        if not twins:
            continue
        idx = {}
        for t in twins:
            L = json.loads((Path(t["dir"]) / "labels.json").read_text())
            idx[t["twin"]] = L["physlaw"].get("twin_index", 0)
        twins.sort(key=lambda i: idx[i["twin"]])
        fps = twins[0]["fps"]
        for i in twins:
            dst = viz / f"{g.name}__{i['twin']}.mp4"
            MV.run(["ffmpeg", "-y", "-v", "error", "-i", str(Path(i["dir"]) / "video.mp4"), "-vf", MV.annotate(i, 832, 18),
                    "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
                    "-movflags", "+faststart", str(dst)])
            i["viz"] = str(dst)
        n = len(twins)
        cols = a.cols or math.ceil(math.sqrt(n))
        tw = a.tile or max(208, min(416, (1664 // cols) // 2 * 2))
        th = int(round(tw * 480 / 832 / 2)) * 2
        fs = max(10, tw // 26)
        inputs, chains = [], []
        for k, i in enumerate(twins):
            inputs += ["-i", str(Path(i["dir"]) / "video.mp4")]
            chains.append(f"[{k}:v]{MV.annotate(i, tw, fs)},scale={tw}:{th},setsar=1[v{k}]")
        layout = "|".join(f"{(k % cols) * tw}_{(k // cols) * th}" for k in range(n))
        fc = ";".join(chains) + ";" + "".join(f"[v{k}]" for k in range(n)) + f"xstack=inputs={n}:layout={layout}:fill=black[o]"
        dst = viz / f"{g.name}__compare.mp4"
        crf = enc(inputs, fc, dst, fps, a.crf)
        rec = {"twins": twins, "compare": str(dst), "compare_bytes": dst.stat().st_size, "compare_crf": crf,
               "grid": [cols, math.ceil(n / cols), tw, th], "two_up": {}}
        by = {i["twin"]: i for i in twins}
        for pr in a.pairs:
            x, y = pr.split(":")
            if x not in by or y not in by:          # a pair for another sub-family's twins
                continue
            d2 = viz / f"{g.name}__2up__{x}__{y}.mp4"
            fc2 = (f"[0:v]{MV.annotate(by[x], 832, 18)},setsar=1[a];[1:v]{MV.annotate(by[y], 832, 18)},setsar=1[b];"
                   f"[a][b]hstack=inputs=2[o]")
            c2 = enc(["-i", str(Path(by[x]["dir"]) / "video.mp4"), "-i", str(Path(by[y]["dir"]) / "video.mp4")], fc2, d2,
                     fps, 20)
            rec["two_up"][pr] = {"path": str(d2), "bytes": d2.stat().st_size, "crf": c2}
        out[g.name] = rec
    (viz / "summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({g: {"n": len(v["twins"]), "compare": v["compare_bytes"],
                          "two_up": {k: x["bytes"] for k, x in v["two_up"].items()}} for g, v in out.items()}))


if __name__ == "__main__":
    main()
