#!/usr/bin/env python3
"""Stage 2 — data acceptance tests (spec §2.4) + a validation report and contact sheet, any family.

    /root/envs/kubric_venv/bin/python scripts/02_validate_data.py --data data/f1
    ... --data data/f7          # every family dir, nuisance variants included

Runs tests/test_data.py against the directory, then writes
<data>/_validation/report.json (per-twin statistics) and contact_sheet.png
(for each of the first groups: last pre-contact frame and two post-contact
frames per twin, so an edit that looks wrong is visible at a glance).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def stats(data: Path) -> dict:
    """Per-twin statistics; collision families add the F1 summary metrics. Render cost in s/frame."""
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    groups = sorted(p for p in data.iterdir() if (p / "_GROUP_COMPLETE").exists())
    rej = collections.Counter()
    for g in groups:
        gr = json.loads((g / "_GROUP_COMPLETE").read_text())
        for t in g.iterdir():
            if not (t / "_COMPLETE").exists():
                continue
            L = json.loads((t / "labels.json").read_text())
            pl = L["physlaw"]
            key = (pl.get("subfamily") + "/" if pl.get("subfamily") else "") + pl["twin"]
            for k in ("momentum_residual", "ke_ratio", "restitution_realised", "deflection_ratio_b_over_a"):
                if k in pl.get("summary", {}):
                    per[key][k].append(pl["summary"][k])
            per[key]["energy_physical"].append(float(pl["energy_physical"]))
            per[key]["violation_magnitude"].append(float(pl.get("violation_magnitude", 0.0)))
            per[key]["render_seconds"].append(pl["render_seconds"])
            if pl.get("frame_sha256_16") and pl["render_seconds"] > 0:
                full = pl["twin_index"] == 0 or pl.get("pre_segment", "0") not in per["_seen"][g.name]
                n = len(L["per_frame"]) - (0 if full else pl["first_post_contact_frame"])
                per["_render"]["seconds_per_frame"].append(pl["render_seconds"] / max(n, 1))
            per["_seen"][g.name].append(pl.get("pre_segment", "0"))
            if pl["twin_index"] == 0:
                for r in L["sampling_attempts"]["rejections"]:
                    rej[r["reason"].split("_at_")[0][:70]] += 1
        per["_group"]["seconds"].append(gr.get("seconds", 0.0))
        per["_group"]["first_post_contact_frame"].append(gr["first_post_contact_frame"])
        per["_group"]["split_" + gr["split"]].append(1)
        if (gr.get("physics_check") or {}).get("max_err_m") is not None:
            per["_group"]["physics_check_max_err_m"].append(gr["physics_check"]["max_err_m"])
    per.pop("_seen", None)
    q = lambda v: {"n": len(v), "min": float(np.min(v)), "median": float(np.median(v)),
                   "max": float(np.max(v)), "mean": float(np.mean(v))}
    return {"n_groups": len(groups), "rejections": dict(rej),
            "twins": {t: {k: q(v) for k, v in d.items() if v} for t, d in per.items()}}


def contact_sheet(data: Path, out: Path, n_groups: int = 4) -> None:
    import imageio.v3 as iio
    groups = sorted(p for p in data.iterdir() if (p / "_GROUP_COMPLETE").exists())[:n_groups]
    rows = []
    for g in groups:
        ts = sorted((t for t in g.iterdir() if (t / "video.mp4").exists()),
                    key=lambda t: json.loads((t / "labels.json").read_text())["physlaw"]["twin_index"])
        if not ts:
            continue
        L = json.loads((ts[0] / "labels.json").read_text())
        k, fps = L["physlaw"]["first_post_contact_frame"], L["scene_constants"]["frame_rate"]
        idx = [k - 1, min(k + fps // 2, L["scene_constants"]["num_frames"] - 1),
               L["scene_constants"]["num_frames"] - 1]
        row = []
        for t in ts:
            v = iio.imread(t / "video.mp4", plugin="FFMPEG")
            # overlay: average of the three frames shows the trajectory as a trail
            trail = np.mean([v[i].astype(float) for i in range(k - 1, v.shape[0], max(1, fps // 4))], axis=0)
            tile = np.concatenate([v[idx[0]], trail.astype(np.uint8)], axis=0)[::2, ::2]
            row.append(tile)
        rows.append(np.concatenate(row, axis=1))
    if rows:
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0))) for r in rows]
        iio.imwrite(out, np.concatenate(rows, axis=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--no-tests", action="store_true")
    a = ap.parse_args()
    data = a.data if a.data.is_absolute() else REPO / a.data
    vdir = data / "_validation"
    vdir.mkdir(exist_ok=True)
    rc = 0
    if not a.no_tests:
        env = {**os.environ, "PHYSLAW_DATA": str(data)}
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                               str(REPO / "tests" / "test_data.py")], cwd=REPO, env=env,
                              capture_output=True, text=True)
        (vdir / "pytest.txt").write_text(proc.stdout + proc.stderr)
        print(proc.stdout[-3000:])
        rc = proc.returncode
    rep = stats(data)
    rep["tests_passed"] = rc == 0
    (vdir / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("n_groups", "rejections", "tests_passed")}, indent=1))
    for t, d in rep["twins"].items():
        if t.startswith("_"):
            print(f"  {t:24s} " + "  ".join(f"{k} med {v['median']:.3g}" for k, v in d.items()))
            continue
        extra = (f"|dp|/|p| med {d['momentum_residual']['median']:.3g}  KE ratio med {d['ke_ratio']['median']:.3f}"
                 f"  e {d['restitution_realised']['median']:.3f}  ") if "momentum_residual" in d else ""
        print(f"  {t:24s} {extra}violation {d['violation_magnitude']['median']:.3g}  "
              f"energy_physical {d['energy_physical']['mean']:.2f}  render {d['render_seconds']['median']:.0f}s")
    try:
        contact_sheet(data, vdir / "contact_sheet.png")
        print(f"contact sheet: {vdir / 'contact_sheet.png'}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"contact sheet skipped: {exc!r}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
