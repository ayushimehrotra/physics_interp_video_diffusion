"""Per pool x probe option (selected space): the contact-pool question table. Usage: python contact_table.py <l1.json>"""
import json, math, sys

r = json.load(open(sys.argv[1]))
f = lambda x, d=3: "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"
print(f"### {r.get('model')} (n_groups {r.get('n_groups')}, spaces {r['protocol']['n_spaces']})\n")
print("| pool | option | space | probe R² floor | η floor | η blind | margin (blind−floor) | η forecast | S_mom [95% CI] "
      "| interaction rate fc/floor | mask-only R² (A, B) | S_mom informative |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for key, sel in r["selected"].items():
    opt, pool = key.split("|")
    space = sel["momentum"]["selected"].split("|")[0]
    v = r["variants"][f"{space}|{opt}|{pool}"]
    m = v["twins"]["legal"]["momentum"]
    e = m["eta"]
    c = r["ceiling"].get(f"{space}|{pool}", {})
    mo = f"{f(c.get('mask_floor_post_A'), 2)}, {f(c.get('mask_floor_post_B'), 2)}" if "mask_floor_post_A" in c else "— (frame)"
    print(f"| {pool} | {opt} | {space} | {f(v['probe_r2_floor'])} | {f(e['floor'])} | {f(e['mass_blind'])} | "
          f"{f(e['mass_blind'] - e['floor'])} | {f(e['forecast'])} | {f(m.get('S'), 2)} [{f(m['S_ci95'][0], 2)}, {f(m['S_ci95'][1], 2)}] | "
          f"{m['interaction_rate']:.2f} / {m['interaction_rate_floor']:.2f} | {mo} | {m.get('S_mom_informative')} |")
