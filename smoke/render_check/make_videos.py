#!/usr/bin/env python3
"""Render-check viewers: per-twin annotated mp4s and one tiled comparison mp4 per family dir.

    python smoke/render_check/make_videos.py smoke/render_check/f12_spin [--crf 28]

For every <family_dir>/<group>/<twin>/video.mp4 (nuisance groups skipped) writes
<family_dir>/viz/<group>__<twin>.mp4 (<= 640 px wide, native fps, H.264 yuv420p, twin name top-left,
red border + EVENT tag for 6 frames from the event frame) and <family_dir>/viz/<group>__compare.mp4
(grid of all twins, frame-synced). Prints a JSON summary of the twins' labels.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def twin_info(tdir: Path) -> dict:
    L = json.loads((tdir / "labels.json").read_text())
    P = L.get("physlaw", {})
    ev = P.get("event_frame", P.get("contact_frame"))
    legal, law = P.get("legal"), P.get("violated_law")
    if legal is None and "momentum_conserved" in P:   # F1 (labels-1 schema): derive from the edit flags
        e = float(P.get("e_target", 1.0))
        legal = bool(P["momentum_conserved"]) and e <= 1.0 + 1e-9
        law = None if legal else ("momentum" if not P["momentum_conserved"] else "energy")
    mf = P.get("master_format", {})
    return {"twin": tdir.name, "dir": str(tdir), "legal": legal,
            "violated_law": law, "magnitude": P.get("violation_magnitude"),
            "epsilon": P.get("epsilon", P.get("e_target")), "event_kind": P.get("event_kind"),
            "clean_until_s": P.get("clean_until_s"), "event_frame": ev, "event_time_s": P.get("event_time_s"),
            "fps": int(mf.get("fps", 48)), "resolution": mf.get("resolution"), "subfamily": P.get("subfamily"),
            "render_device": P.get("render_device"), "pre_segment": P.get("pre_segment"),
            "constants": {k: v for k, v in (P.get("constants") or {}).items() if k in ("overhang_x", "mu", "ratio")}}


def short(i: dict) -> str:
    s = i["twin"]
    if i["legal"] is False or i["violated_law"]:
        s += f"  [{i['violated_law'] or 'viol'}]"
    elif i["legal"]:
        s += "  [legal]"
    return s


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "").replace("%", "\\%").replace(",", "\\,")


def annotate(i: dict, w: int, fs: int) -> str:
    ev = int(i["event_frame"]) if i["event_frame"] is not None else -10
    b = max(2, w // 120)
    return (f"scale={w}:-2:flags=area,"
            f"drawbox=x=0:y=0:w=iw:h=ih:color=red:t={b}:enable='between(n,{ev},{ev + 5})',"
            f"drawtext=fontfile={FONT}:text='{esc(short(i))}':x=4:y=4:fontsize={fs}:fontcolor=white:"
            f"box=1:boxcolor=black@0.55:boxborderw=2,"
            f"drawtext=fontfile={FONT}:text='EVENT f{ev}':x=4:y=h-th-4:fontsize={fs}:fontcolor=yellow:"
            f"box=1:boxcolor=black@0.55:boxborderw=2:enable='between(n,{ev},{ev + 5})',"
            f"drawtext=fontfile={FONT}:text='f%{{n}}':x=w-tw-4:y=h-th-4:fontsize={fs}:fontcolor=white:"
            f"box=1:boxcolor=black@0.4:boxborderw=2")


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])


def sort_key(i):
    import re
    nat = [(0, float(t), "") if re.fullmatch(r"\d+(\.\d+)?", t) else (1, 0.0, t)
           for t in re.findall(r"\d+(?:\.\d+)?|[^\d.]+|\.", i["twin"])]
    return (0 if i["legal"] else 1, nat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fam_dir", type=Path)
    ap.add_argument("--crf", type=int, default=28)
    ap.add_argument("--src", type=Path, nargs="*", default=None, help="group dirs (default: groups in fam_dir)")
    ap.add_argument("--tile", type=int, default=0, help="comparison tile width (default: auto)")
    a = ap.parse_args()
    viz = a.fam_dir / "viz"
    viz.mkdir(exist_ok=True)
    out = {}
    groups = a.src or sorted(p for p in a.fam_dir.iterdir() if p.is_dir() and p.name.startswith("f") and "nuisance" not in p.name)
    for g in groups:
        twins = [twin_info(t) for t in sorted(g.iterdir()) if (t / "video.mp4").exists() and (t / "labels.json").exists()]
        twins.sort(key=sort_key)
        if not twins:
            continue
        for i in twins:
            dst = viz / f"{g.name}__{i['twin']}.mp4"
            run(["ffmpeg", "-y", "-v", "error", "-i", str(Path(i["dir"]) / "video.mp4"), "-vf", annotate(i, 640, 15),
                 "-r", str(i["fps"]), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
                 "-movflags", "+faststart", str(dst)])
            i["viz"] = str(dst)
        n = len(twins)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        tw = a.tile or max(208, min(416, (1664 // cols) // 2 * 2))
        th = int(round(tw * 480 / 832 / 2)) * 2
        fs = max(10, tw // 26)
        inputs, chains, labels = [], [], []
        for k, i in enumerate(twins):
            inputs += ["-i", str(Path(i["dir"]) / "video.mp4")]
            chains.append(f"[{k}:v]{annotate(i, tw, fs)},scale={tw}:{th},setsar=1[v{k}]")
            labels.append(f"[v{k}]")
        layout = "|".join(f"{(k % cols) * tw}_{(k // cols) * th}" for k in range(n))
        fc = ";".join(chains) + ";" + "".join(labels) + (
            f"xstack=inputs={n}:layout={layout}:fill=black[o]" if n > 1 else "null[o]")
        dst = viz / f"{g.name}__compare.mp4"
        crf = a.crf
        while True:
            run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", fc, "-map", "[o]", "-r", str(twins[0]["fps"]),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "medium",
                 "-movflags", "+faststart", str(dst)])
            if dst.stat().st_size <= 4 * 1024 * 1024 or crf >= 40:
                break
            crf += 3
        out[g.name] = {"twins": twins, "compare": str(dst), "compare_bytes": dst.stat().st_size, "compare_crf": crf,
                       "grid": [cols, rows, tw, th]}
    (viz / "summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({g: {"n": len(v["twins"]), "compare": v["compare"], "bytes": v["compare_bytes"]} for g, v in out.items()}))


if __name__ == "__main__":
    main()
