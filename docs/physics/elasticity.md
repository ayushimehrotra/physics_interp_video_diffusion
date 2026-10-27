# Elasticity (coefficient of restitution)

Split out of `CLAUDE.md` on 2026-10-27, verbatim apart from paths updated to the new layout. Section numbers are unchanged; the index in
`CLAUDE.md` says which file each § lives in. The rest of the elasticity story is told
from the energy side in [`kinetic_energy.md`](kinetic_energy.md) §44, and the restitution
readout before contact in [`momentum.md`](momentum.md) §62.

---

### 36. Elastic vs inelastic on the existing sets — measurements

Run 2026-08-21 on existing labels and captures. No new renders. Recorded because
the four-way elastic/inelastic x conserved/violated grid was proposed and these are
the numbers it would be built on.

**Realised restitution vs the sampled label.** `e_real = -(v_rel_post . n) / (v_rel_pre . n)`,
with `n` the line of centres at `collision.actual_frame`, velocities at frames 18
and 22:

| set | scenes | corr(e_label, e_real) | median abs diff |
|---|---|---|---|
| `probe_set` v1, conserving | 75 | +1.0000 | 0.0001 |
| `probe_set` v1, violating | 75 | +0.9999 | 0.0000 |
| v3, conserving | 225 | +0.9999 | 0.0001 |
| v3, violating | 75 | **+0.6333** | **0.6754** |

v1's violation adds a common `dv` to both balls, which cancels in `v_A - v_B`, so
restitution is untouched. v3's scales the two impulse magnitudes, which does not
cancel.

**`cos(dv_A, dv_B)` on violating scenes.** v1: -0.9999 to **+0.9993**. v3: -1.0000
to +0.2404. Conserving is -1.0000 in both.

**Kinetic energy, conserving scenes only.** `KE_post/KE_pre`, frames 18 -> 22:

| | `probe_set` v1 (n=75) | v3 (n=225) |
|---|---|---|
| median | 0.839 | 0.940 |
| range | 0.180 - 1.000 | 0.119 - 1.000 |
| >= 0.98 | 9 (12%) | 70 (31%) |
| 0.90-0.98 | 18 (24%) | 69 (31%) |
| 0.70-0.90 | 20 (27%) | 52 (23%) |
| < 0.70 | 28 (37%) | 34 (15%) |

`KE_loss = 0.5 * mu_red * (1 - e^2) * u_n^2` reproduces the observed loss at
corr **+1.00000**, median relative error 1e-4, in both sets.

**KE ratio is not restitution.** corr(e, KE ratio): v1 conserving **+0.548**, v1
violating **+0.058**; v3 conserving +0.307. On violating scenes the aftermath is
overwritten, so KE ratio runs to **2.63** — energy is created. By impact parameter
`b = |b|/(r_A+r_B)`, v1 conserving median KE ratio: b<0.3 **0.69** (n=23),
0.3-0.6 0.81 (n=31), b>0.6 **0.92** (n=21).

**Probes for restitution and energy, VideoMAE on `probe_set` v1.** Mean held-out R2
over 20 random 60/40 splits, standardised features, fixed alpha 10 (NOT the
`fit_probe` protocol — a separate one-off):

| cell | restitution | KE ratio |
|---|---|---|
| blk 5 all.pre | -0.040 | **0.459** |
| blk 5 all.post | 0.014 | 0.417 |
| blk 9 all.pre | 0.235 | 0.334 |
| blk 11 all.pre | **0.248** | 0.222 |
| blk 9 all.all | -0.112 | 0.025 |

**Retrospective 2x2 on `probe_set`, split at the median restitution 0.53.**
Violation AUC via the §34a route (regress `momentum_delta`, take the norm of the
prediction, score AUC against the conserved/violated label), VideoMAE all.pre
block 9, mean over 40 splits:

| cell | n | violating | AUC |
|---|---|---|---|
| all 150 | 150 | 75 | 0.675 |
| e > 0.53 | 75 | 42 | 0.531 |
| e <= 0.53 | 75 | 33 | 0.579 |

Cell sizes in the existing set: elastic/conserved 33, elastic/violated 42,
inelastic/conserved 42, inelastic/violated 33.
