#!/usr/bin/env python3
"""Write smoke/render_check/MANIFEST.md from every <family>/viz/summary.json plus the notes below."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
NOTES = json.loads((ROOT / "notes.json").read_text())
ORDER = ["f1", "f2", "f3", "f4", "f6_flat", "f6_incline", "f6_slides", "f6_glide", "f7_projectile", "f7_drop",
         "f7_bounce", "f8_ramp", "f9_pair", "f9_swing", "f10_rolling", "f11_support_x0.65", "f11_support_x1.35",
         "f12_spin"]


def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def main():
    out = [NOTES["header"].rstrip(), ""]
    for fam in ORDER:
        sp = ROOT / fam / "viz" / "summary.json"
        out.append(f"## {fam}")
        out.append("")
        if not sp.exists():
            out += [f"NOT RENDERED. {NOTES['families'].get(fam, '')}", ""]
            continue
        for gid, g in json.loads(sp.read_text()).items():
            tw = g["twins"]
            src = Path(tw[0]["dir"]).parent
            dev = sorted({t["render_device"] for t in tw if t.get("render_device")})
            res = tw[0].get("resolution")
            out.append(f"* group `{gid}`, {len(tw)} twins, source `{src.relative_to(REPO) if src.is_relative_to(REPO) else src}`, "
                       f"render {res[0]}x{res[1]} @ {tw[0]['fps']} fps" + (f", {', '.join(dev)}" if dev else "")
                       if res else f"* group `{gid}`, {len(tw)} twins")
            out.append(f"* comparison: `{Path(g['compare']).relative_to(ROOT.parent.parent) if Path(g['compare']).is_absolute() else 'smoke/render_check/' + g['compare']}` "
                       f"({g['compare_bytes'] / 1e6:.2f} MB, grid {g['grid'][0]}x{g['grid'][1]}, tile {g['grid'][2]}x{g['grid'][3]}); "
                       f"per-twin mp4s next to it (`<group>__<twin>.mp4`)")
            out.append("")
            out.append("| twin | legal | violated_law | magnitude | epsilon | event_kind | event frame | clean_until_s |")
            out.append("|---|---|---|---|---|---|---|---|")
            for t in tw:
                out.append(f"| {t['twin']} | {fmt(t['legal'])} | {fmt(t['violated_law'])} | {fmt(t['magnitude'])} | "
                           f"{fmt(t['epsilon'])} | {fmt(t['event_kind'])} | {fmt(t['event_frame'])} | {fmt(t['clean_until_s'])} |")
            out.append("")
        note = NOTES["families"].get(fam)
        if note:
            out += [note.rstrip(), ""]
    (ROOT / "MANIFEST.md").write_text("\n".join(out) + "\n")
    print("wrote", ROOT / "MANIFEST.md")


if __name__ == "__main__":
    main()
