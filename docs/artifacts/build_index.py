import json, pathlib, html
# (part, id, title, lines, status, summary, cut)
R = [
 ("Handoff banners","","STATE AS OF HANDOFF (2026-08-20)",17,"live","§32b voids incline_set_v1; the current top-of-file state.",""),
 ("Handoff banners","","STATE AS OF HANDOFF (2026-08-17b)",36,"superseded","§34 as news: Delta-p is a positive, two negatives withdrawn. Now duplicated by §34 itself.","Fold into one current banner"),
 ("Handoff banners","","STATE AS OF HANDOFF (2026-08-17)",27,"superseded","§32 as news: the friction line was an acceleration result. Duplicated by §32.","Fold into one current banner"),
 ("Handoff banners","","STATE AS OF HANDOFF (2026-08-16)",33,"superseded","§26 as news: products are not stored. Duplicated by §26.","Fold into one current banner"),
 ("Handoff banners","","STATE AS OF HANDOFF (2026-08-13)",115,"superseded","The oldest banner: 'part 1 answered, part 2 open'. Longest of the five and the most out of date.","Fold into one current banner"),

 ("Orientation","","Version control state — READ BEFORE ANY git OPERATION",21,"live","Everything since commit 87e9fad is uncommitted and unbacked-up. Still true, and now includes today's fix.",""),
 ("Orientation","","What happened on the migration",26,"live","A git clone lost the entire dataset once. Cautionary and still relevant to the RunPod move.",""),
 ("Orientation","","This box",10,"reference","Hardware table for the current machine.",""),
 ("Orientation","","setup.sh: the wheel-speed warning does NOT apply here",16,"stale","Notes that pip was slow on a machine that no longer exists.","Trivia about a dead host"),
 ("Orientation","","Reproducing what has been run",170,"reference","Every command that produced the existing results. Long, but it is the reproduction record.",""),
 ("Orientation","","How to read the results",45,"reference","Scoring conventions: accuracy vs R², what the nulls mean.",""),

 ("Findings","1","Probing — physical variables are decodable, conservation is not",31,"revised","The original result table. Its momentum claim is revised by §26 and its conservation null by §34a.",""),
 ("Findings","2","The winner's curse, and the shuf* fix",21,"live","Selecting the best cell on test over 279 cells inflates results; the correct null is the grid max. Core methodology.",""),
 ("Findings","3","Momentum is not an independently steerable factor",34,"revised","p = m·v is an identity, so there is no residual to steer. The 'probe beat the oracle' half is withdrawn by §26.",""),
 ("Findings","4","The probes were reading, not predicting",22,"live","CogVideoX attends bidirectionally, so a 'pre' token has already seen the post frames.",""),
 ("Findings","5","Steering — negative, and consistent with (3)",51,"live","Seven direction estimators, none beat its control.",""),
 ("Findings","6","The measurement harness, and its floors",33,"reference","How physics is recovered from pixels alone, and what its noise floors are.",""),
 ("Findings","7","Generative conservation test — not answerable on this checkpoint",45,"live","Inpainting post-contact frames does not produce usable physics. Closes the generative route.",""),
 ("Findings","8","Captured but not analysed",18,"stale","Describes outputs_heads/, which was deleted in a disk cleanup. Nothing depends on it.","The data it describes no longer exists"),
 ("Findings","9","Encoders replicate the diffusion result exactly (v1)",24,"live","VideoMAE and V-JEPA2 give the same answers as CogVideoX, with two fewer axes.",""),
 ("Findings","10","The v3 latent_mass arm — what it is, and that it worked",38,"live","Equal radii + identical albedo makes mass invisible; mass_A falls 0.93 → 0.28. The control that proves the pipeline measures what it claims.",""),
 ("Findings","11","probe_set_v2_latent is superseded — the antiparallel leak",36,"stale","v2's violation mechanism let a probe separate classes at AUC 1.000 from pure geometry. The set is mostly deleted; the lesson is live.","Keep ~8 lines of lesson, drop the set autopsy"),
 ("Findings","12","THE IDENTITY TRAP",48,"live","The project's most reusable idea: a target that is an exact function of the visible trajectory tests arithmetic, not the model. Core.",""),
 ("Findings","13","v3 results, and the oracle that explains them",61,"revised","The mass-ratio table. Its oracle argument is withdrawn by §34c as basis-dependent and unfalsifiable.",""),
 ("Findings","14","Conservation on v3 is unidentifiable by construction",49,"live","A theorem about the dataset, not a measurement of the models: no observer can beat chance there.",""),
 ("Findings","15","The prediction route is CLOSED on both encoders",112,"superseded","Rebuilt and re-measured in §21, which reaches the same verdict with better instruments.","Replace with a 3-line pointer to §21"),
 ("Findings","16","Object addressability — how the two objects are encoded",70,"live","Cross-decoding matrix between object pools. Identity-trap-free. Its prediction is refuted by §18.",""),
 ("Findings","17","Reproducing the one-off analyses (§13, §15, §16)",151,"reference","Scripts and commands for analyses not wired into the CLI.",""),
 ("Findings","18","Per-object velocity steering — POSITIVE",212,"live","The second real positive: velocity directions steer per object with matched controls. Longest finding section.",""),
 ("Findings","19","Polar vs Cartesian factorisation — correlational only",46,"live","Whether velocity is stored as (vx,vy) or (speed,heading). Correlational, no causal test.",""),
 ("Findings","20","THE POOLING-MASK LEAK — a standing control",70,"live","Object-local pooling uses the ground-truth mask, so mask geometry leaks into the pooled vector. A standing control every result must clear.",""),
 ("Findings","21","The predictor test rebuilt — §15's verdict survives",87,"live","Supersedes §15. Same conclusion, honest measurement.",""),
 ("Findings","22","Joseph et al. — independent corroboration and a method to borrow",47,"live","The closest related work (FAIR/Mila, Feb 2026), same simulator. Related-work anchor for the paper.",""),
 ("Findings","23","The four-velocity concatenated probe — changes nothing",99,"live","Hand the probe all four velocity vectors and ask for the mass ratio: no gain. A clean null.",""),
 ("Findings","24","The sliding-friction dataset — a second physical domain",122,"superseded","The first friction build: two balls, top-down. Redesigned into §25 on request.","Superseded by §25's redesign"),
 ("Findings","25","slide_set_v2 — one cube, a friction floor, a SIDE view",168,"revised","The flat-floor friction set. Correct as measured, but §32 shows it measured acceleration, not friction.",""),
 ("Findings","26","PRODUCTS ARE NOT STORED — and this revises §3",104,"live","The strongest claim in the project: factors are stored separately and linearly; every product is composed by the readout. Mass-ratio rows withdrawn by §34b.",""),
 ("Findings","27","The pool list was incomplete, and it cost a published number",46,"live","Three pools were never probed and one result was reported from the wrong object.",""),
 ("Findings","28","Conservation through the delta vector",55,"superseded","Explicitly resolved by §34, which is 296 lines away and says so at the top.","Replace with a 3-line pointer to §34"),
 ("Findings","29","How to diagnose whether a momentum variable exists — the protocol",38,"live","Ten tests from 'present' to 'used', with status. Three are still unrun. The planning spine.",""),
 ("Findings","30","The friction line, run end to end",208,"revised","Probe, oracle and steering for μ on the flat floor. Correctly measured, wrongly named — it is acceleration. Second longest section.",""),
 ("Findings","31","Is mu bound to the SURFACE or the OBJECT?",75,"revised","Causal patching says the object. True, but §32 renames the variable: it is acceleration that is object-bound.",""),
 ("Findings","32","THE INCLINE SET — separating mu from acceleration",81,"live","Why the flat floor could not distinguish friction from acceleration, and the geometry that fixes it.",""),
 ("Findings","32b","THE SINGLE-FRAME LEAK — incline_set_v1 is void",89,"live","Today's finding: μ reads off one frame's cube position at R² 0.88–0.99. The fix for §32a caused it. Generator and validator now fixed.",""),
 ("Findings","33","A taxonomy of physical variables",74,"live","Five levels of physical variable and where the field has looked. Planning, not results.",""),
 ("Findings","34","§28 RESOLVED — Delta-p IS represented",296,"live","The largest section. Delta-p decodes with p = 0.000 in three models; the binary label was the wrong instrument; withdraws the oracle, the mass-ratio subspace result, and two §26 rows.",""),
 ("Findings","","What to do next, given all of the above",153,"live","The resume block. Updated today to lead with the incline re-render.",""),

 ("Reference","","Layout",125,"reference","Directory and module map of the repo.",""),
 ("Reference","","Environment — the traps",89,"reference","Two Pythons that cannot be merged, and the ways the install breaks.",""),
 ("Reference","","Moving to another machine",56,"reference","The transfer unit is the directory, not the git repo. Directly relevant to the RunPod migration.",""),
 ("Reference","","Regenerating probe_set instead of copying it",75,"reference","Scenes are a pure function of their seed, so the set can be rebuilt rather than moved.",""),

 ("Design notes","","Inversion is DDIM, not the paper's flow-matching Euler",20,"reference","Why the reference paper's update rule does not transfer to CogVideoX-2b.",""),
 ("Design notes","","Full inversion does not round-trip — truncate to edit",33,"reference","Probing and editing need different inversion depths.",""),
 ("Design notes","","The pre/post collision split is exact",8,"reference","Latent frames 0–5 are strictly pre-contact, 6–12 post. Re-check if the contact frame ever moves.",""),
 ("Design notes","","Pooling is by segmentation mask, not frame-wide mean",7,"reference","Why nine pools exist and what they cost in storage.",""),
 ("Design notes","","The momentum/mass confound is why the dataset was rebuilt",23,"live","The pilot had corr(|p|, mass) = 0.97. This is the section that answers your mentor's objection directly.",""),
 ("Design notes","","Conservation needs both classes",13,"live","Violating scenes are authored, not simulated. Explains the mechanism your grid inherits.",""),
 ("Design notes","","momentum_dataset/pilot/ is stale — do not probe it",8,"live","The 50-scene pilot with the mass confound. Short and load-bearing.",""),
 ("Design notes","","Physics facts baked into the labels",17,"reference","Zero gravity, planar motion, exact contact normals.",""),
 ("Design notes","","Conventions",15,"reference","Linear probes, permutation controls, how the ridge penalty is chosen.",""),
]
STATUS = {
 "live":("live","Current and load-bearing"),
 "revised":("revised","Still cited, but a later section changes what it means"),
 "superseded":("superseded","A later section replaces it outright"),
 "stale":("stale","Describes something that no longer exists"),
 "reference":("reference","How-to, not a finding"),
}
parts = []
for part in ["Handoff banners","Orientation","Findings","Reference","Design notes"]:
    rows = [r for r in R if r[0]==part]
    parts.append((part, sum(r[3] for r in rows), rows))
total = sum(r[3] for r in R)
cuts = [r for r in R if r[6]]
cut_lines = sum(r[3] for r in cuts)
counts = {}
for r in R: counts[r[4]] = counts.get(r[4],0)+1
maxlines = max(r[3] for r in R)

def rowhtml(r):
    part,rid,title,lines,status,summary,cut = r
    num = f'<span class="rid">§{html.escape(rid)}</span>' if rid else '<span class="rid dash">—</span>'
    cutb = f'<div class="cut">✂ {html.escape(cut)}</div>' if cut else ''
    w = max(2, round(100*lines/maxlines))
    return f'''<li class="row" data-status="{status}" data-lines="{lines}">
      <div class="rid-cell">{num}</div>
      <div class="main">
        <div class="title">{html.escape(title)}</div>
        <div class="sum">{html.escape(summary)}</div>
        {cutb}
      </div>
      <div class="meta">
        <span class="chip {status}">{status}</span>
        <span class="lines"><span class="bar" style="width:{w}%"></span><b>{lines}</b></span>
      </div>
    </li>'''

body = []
for part, n, rows in parts:
    body.append(f'<section class="grp"><header class="grph"><h2>{html.escape(part)}</h2>'
                f'<span class="grpn">{len(rows)} sections · {n} lines</span></header><ul class="rows">')
    body.extend(rowhtml(r) for r in rows)
    body.append('</ul></section>')

tpl = pathlib.Path('/tmp/claude-0/-workspace/a174e055-0555-41ae-a752-85214cf89308/scratchpad/index_template.html').read_text()
out = (tpl.replace("{{ROWS}}", "\n".join(body))
          .replace("{{TOTAL}}", f"{total:,}")
          .replace("{{NSEC}}", str(len(R)))
          .replace("{{CUTN}}", str(len(cuts)))
          .replace("{{CUTL}}", f"{cut_lines:,}")
          .replace("{{CUTPCT}}", f"{100*cut_lines/total:.0f}")
          .replace("{{LIVE}}", str(counts.get('live',0)))
          .replace("{{REV}}", str(counts.get('revised',0)))
          .replace("{{SUP}}", str(counts.get('superseded',0)))
          .replace("{{STALE}}", str(counts.get('stale',0)))
          .replace("{{REF}}", str(counts.get('reference',0))))
p = pathlib.Path('/tmp/claude-0/-workspace/a174e055-0555-41ae-a752-85214cf89308/scratchpad/claude_md_index.html')
p.write_text(out); print(p, total, len(R), cut_lines)
