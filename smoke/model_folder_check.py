"""Check /workspace/models folders against their pinned HF revision: missing / unreadable / vanishing files, bad json.
Re-fetch only small json/txt/tokenizer files (never weights). Usage: model_check.py [--fix]"""
import fnmatch, json, os, sys, time
from huggingface_hub import HfApi, hf_hub_download
R = "/workspace/models"; FIX = "--fix" in sys.argv
PINS = {  # folder: (repo, revision, include globs or None=all, exclude globs)
 "CogVideoX-2b": ("zai-org/CogVideoX-2b", "1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01", None, []),
 "CogVideoX-5b": ("zai-org/CogVideoX-5b", "8fc5b281006c82b82d34fd2543d2f0ebb4e7e321", None, ["text_encoder/*"]),
 "CogVideoX1.5-5B-I2V": ("zai-org/CogVideoX1.5-5B-I2V", "46c90528707aebbe69066390b4fe7e7d24c9c2a4", None, []),
 "vjepa2-vitl-fpc64-256": ("facebook/vjepa2-vitl-fpc64-256", "b3c1679b7c34d3255ef3547f27c7b226aefab26f", None, []),
 "vjepa2-vitg-fpc64-256": ("facebook/vjepa2-vitg-fpc64-256", "875c192b7b704b87d1e1d99345769632dd5f739a", None, []),
 "VideoMAEv2-Base": ("OpenGVLab/VideoMAEv2-Base", "78c337a418cc4adaf7f1ff6c7ba343418393966b", None, []),
 "videomae2": ("OpenGVLab/VideoMAE2", "706cc172d65ebd4dedbee3f9c0183a93df9fa125", ["*vit_g_hybrid_pt_1200e.pth"], []),
 "PhysRVG": ("HappyP4nda/PhysRVG", "d8caf2dd7db30d7470d474875c84e6a4eb21b9c6", None, []),
 "VideoREPA": ("aHapBean/VideoREPA", "5438a0208c973fa5bec7d74486694f03e72ec939", None, []),
 "Wan2.2-TI2V-5B-Diffusers": ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "b8fff7315c768468a5333511427288870b2e9635", None, []),
 "Wan2.1-T2V-1.3B-Diffusers": ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "0fad780a534b6463e45facd96134c9f345acfa5b", None, ["text_encoder/*"]),
 "Wan2.1-T2V-14B-Diffusers": ("Wan-AI/Wan2.1-T2V-14B-Diffusers", "38ec498cb3208fb688890f8cc7e94ede2cbd7f68", None, []),
 "Wan2.1-VACE-1.3B-diffusers": ("Wan-AI/Wan2.1-VACE-1.3B-diffusers", "ec4d2cb062b548996b179d493fdd05340de702a1", None, []),
 "Wan2.1-VACE-14B-diffusers": ("Wan-AI/Wan2.1-VACE-14B-diffusers", "db79b90c60bbb45ceec9e41b9d5a4df934538ac4", None, []),
 "LTX-Video-0.9.8-13B-distilled": ("Lightricks/LTX-Video-0.9.8-13B-distilled", "7c64400e1861cc0d7b98d570a1926d5408ec60cd", None,
                                   ["vae/text_encoder/*", "vae/transformer/*", "vae/tokenizer/*", "vae/scheduler/*", "vae/model_index.json"]),
 "LTX-Video-0.9.8-2B-distilled": ("Lightricks/LTX-Video", "8984fa2500", ["ltxv-2b-0.9.8-distilled.safetensors", "ltxv-spatial-upscaler-0.9.8.safetensors", "README.md", "LTX-Video-Open-Weights-License-0.X.txt"], []),
 "Wan2.2-I2V-A14B-Diffusers": ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "596658fd9ca6b7b71d5057529bbf319ecbc61d74", None, []),
 "Wan2.1-I2V-14B-480P-Diffusers": ("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "b184e23a8a16b20f108f727c902e769e873ffc73", None, []),
}
SKIP = ["media/*", "assets/*", "examples/*", ".gitattributes", "*.gif", "*.png", "*.jpg", "*.mp4", "*.webp"]
SMALL = ("*.json", "*.txt", "*.model", "*.md", "*.py", "*.yaml", "*.tiktoken", "*.jinja")
api = HfApi()
def match(f, globs): return any(fnmatch.fnmatch(f, g) for g in globs)
def local_files(d): return sorted(os.path.relpath(os.path.join(r, f), d) for r, _, fs in os.walk(d) if "/.cache" not in r + "/" and ".cache" not in os.path.relpath(r, d).split(os.sep)[:1] for f in fs)
def probe(p):
    try:
        with open(p, "rb") as h:
            b = h.read(1) if not p.endswith(".json") else h.read()
        if p.endswith(".json"): json.loads(b)
        return None
    except Exception as e: return type(e).__name__
report = {}
for folder in sorted(os.listdir(R)):
    d = os.path.join(R, folder)
    if not os.path.isdir(d): continue
    loc = [f for f in local_files(d) if not f.endswith(".sha256")]
    bad1 = {f: e for f in loc if (e := probe(os.path.join(d, f)))}
    exp, pin = None, PINS.get(folder)
    if pin:
        repo, rev, inc, exc = pin
        try:
            rf = api.list_repo_files(repo, revision=rev)
            exp = [f for f in rf if (inc is None or match(f, inc)) and not match(f, exc) and not match(f, SKIP)]
        except Exception as e: exp = f"ERR {type(e).__name__}"
    report[folder] = {"pin": pin, "local": loc, "bad1": bad1, "exp": exp}
time.sleep(10)
fixes = []
for folder, r in report.items():
    d = os.path.join(R, folder)
    loc2 = set(f for f in local_files(d) if not f.endswith(".sha256"))
    bad2 = {f: e for f in r["local"] if (e := probe(os.path.join(d, f)))}
    vanished = sorted(set(r["local"]) - loc2)
    missing = sorted(set(r["exp"]) - loc2) if isinstance(r["exp"], list) else r["exp"]
    broken = sorted(set(r["bad1"]) | set(bad2))
    flaky = sorted(set(r["bad1"]) ^ set(bad2))
    status = "OK" if not (missing or broken or vanished) else "PROBLEM"
    if r["pin"] is None: status += " (no pin: readability only)"
    print(f"{folder}: {status}; {len(loc2)} files" + (f"; MISSING {missing}" if missing else "") +
          (f"; UNREADABLE/BAD JSON {[(f, (r['bad1'] | bad2)[f]) for f in broken]}" if broken else "") +
          (f"; FLAKY {flaky}" if flaky else "") + (f"; VANISHED {vanished}" if vanished else ""))
    if FIX and r["pin"] and isinstance(missing, list):
        for f in sorted(set(missing) | set(broken)):
            if match(f, SMALL) and (r["exp"] is None or f in r["exp"]):
                repo, rev, *_ = r["pin"]
                hf_hub_download(repo, f, revision=rev, local_dir=d, force_download=True)
                ok = probe(os.path.join(d, f)) is None
                fixes.append((folder, f, "refetched" + ("" if ok else " BUT STILL BAD")))
            else:
                fixes.append((folder, f, "NOT refetched (weight or not in pinned repo)"))
for x in fixes: print("FIX", *x)
