"""Per family: are twins' decoded frames < first_post identical across twins of one pre_segment? Longest run of
identical consecutive frames after the event per twin (frozen frames)."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, imageio.v3 as iio
for fam in sys.argv[1:]:
    s = json.loads(Path(fam, "viz", "summary.json").read_text())
    for gid, g in s.items():
        segs = defaultdict(list)
        frozen = {}
        for t in g["twins"]:
            L = json.loads(Path(t["dir"], "labels.json").read_text())["physlaw"]
            fp = L.get("first_post_contact_frame", (t["event_frame"] or 0) + 1)
            v = iio.imread(Path(t["dir"], "video.mp4"))
            segs[str(L.get("pre_segment", "0"))].append((t["twin"], fp, v))
            d = np.abs(np.diff(v[fp:].astype(np.int16), axis=0)).max(axis=(1, 2, 3))
            run = best = 0
            for x in d:
                run = run + 1 if x <= 2 else 0
                best = max(best, run)
            frozen[t["twin"]] = best
        bad = []
        for seg, lst in segs.items():
            n0, fp0, v0 = lst[0]
            for n, fp, v in lst[1:]:
                k = min(fp, fp0)
                diff = int(np.abs(v[:k].astype(np.int16) - v0[:k]).max())
                if diff > 0:
                    bad.append(f"{n} vs {n0} pre-event maxdiff {diff}")
        print(fam, gid, "segments", {k: len(v) for k, v in segs.items()}, "pre-event identical" if not bad else bad,
              "| longest static run after event (frames):", {k: v for k, v in frozen.items() if v >= 24} or "none >= 24")
