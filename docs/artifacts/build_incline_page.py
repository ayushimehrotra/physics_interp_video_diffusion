import base64, json, os, pathlib
SP = "/tmp/claude-0/-workspace/a174e055-0555-41ae-a752-85214cf89308/scratchpad"
DS = "/workspace/physics_interp_video_diffusion/momentum_dataset/incline_set_v2"
rows = json.load(open(f"{SP}/v2_rows.json"))
clips = ["scene_00082","scene_00136","scene_00008","scene_00106","scene_00076","scene_00025"]
vid = {}
for s in clips:
    b = pathlib.Path(f"{DS}/{s}/video.mp4").read_bytes()
    vid[s] = "data:video/mp4;base64," + base64.b64encode(b).decode()
data_js = json.dumps([{"mu": round(r["mu"],5), "theta": round(r["th"],5), "a": round(r["a"],5),
                       "travel": round(r["travel"],5), "vf": round(r["v0"],5), "size": round(r["size"],5),
                       "id": r["id"][6:]} for r in rows], separators=(",",":"))
tpl = pathlib.Path(f"{SP}/incline_template.html").read_text()
for s in clips:
    tpl = tpl.replace("{{VID_%s}}" % s, vid[s])
tpl = tpl.replace("{{SCENES_JSON}}", data_js)
tpl = tpl.replace("{{LEAK_V2}}", json.dumps(json.load(open(f"{SP}/leak_curve_v2.json")), separators=(",",":")))
out = f"{SP}/incline_set_v1.html"
pathlib.Path(out).write_text(tpl)
print(out, os.path.getsize(out)/1e6, "MB")
