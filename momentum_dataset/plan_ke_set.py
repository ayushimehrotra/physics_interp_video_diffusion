#!/usr/bin/env python3
"""Plan `ke_factor_set`: kinetic energy stepped through SPEED alone or MASS alone.

    /root/envs/kubric_venv/bin/python momentum_dataset/plan_ke_set.py --out momentum_dataset/ke_factor_set

THE QUESTION. Reviewers: the e > 1 VoE result shows sensitivity to an implausible
collision, not a representation of kinetic energy -- a model can see the balls
separate too fast without holding mass or energy. And under conservation the SIGN of
dKE = 1/2 mu v_rel^2 (e^2 - 1) is set by e alone, so no impossible-vs-legal contrast
can ever need mass. The mass-dependent part of KE is its MAGNITUDE, and the only way
to show a model holds it is to move KE along two physically different routes and ask
whether one readout follows both:

    family S  (speed)  masses fixed, both velocities scaled by s   -> KE x s^2
    family M  (mass)   velocities fixed, both masses scaled by c   -> KE x c

Each GROUP is one scene rendered at 5 levels, KE x {1/2, 1/sqrt2, 1, sqrt2, 2} of the
group's centre. Within a group every other parameter is bit-identical. Scaling BOTH
masses keeps the mass ratio, so mu, the total KE, the relative KE and each ball's KE
all scale by the same factor -- there is one KE, not four that disagree. Scaling BOTH
velocities keeps every direction, so the geometry is shared too. In family M the
post-contact velocities are therefore identical at every level (they depend on the
mass RATIO and e only): the motion is the same video, only the balls' mass differs.

e is pinned at 1.0 (elastic), so KE_pre == KE_post and the clip has ONE kinetic
energy. That also removes restitution -- the reviewers' channel -- from the set: it
is the same constant in every clip.

LEAK CONTROLS, all by construction and all re-checked by validate_ke_set.py:

  * Same population. Every group is drawn from ONE sampler and must be feasible on the
    full 5 x 5 grid of speed and mass levels before it is assigned to a family. So the
    two families are identically distributed by construction. The fixed level of each
    group (the mass level an S group sits at, the speed level an M group sits at) is
    balanced, 20 groups per level, so the clip-level marginals of mass, speed and KE
    are identical between families too.
  * Size is not mass. There are two materials (rubber 1.5, metal 6.0, 4x). Each mass
    level picks rubber or metal and the radius that realises the mass, and the
    per-group assignment is chosen so radius does NOT step with mass: corr(log radius,
    mass level) within family M is ~0. The mass cue the model must use is material
    and size TOGETHER, as in every legacy-arm set.
  * Nothing else varies with level. Directions, impact parameter, contact point,
    b_moving and contact frame (20) are group constants. Restitution is a set constant.

WHAT CANNOT BE REMOVED, and how to read it. |p| also rises with KE along both axes
(p ~ c in M, p ~ s in S), so a binary or rank readout cannot tell a KE code from a
momentum code. A CALIBRATED one can. Within a group dlog|p| = dlogKE/2 in S (p ~ v, KE ~ v^2) but
dlog|p| = dlogKE in M (p ~ m, KE ~ m). So train a logKE readout on S and apply it to
M: the slope of predicted on true dlogKE is 1 for an energy code, 2 for a momentum
code, 0 for a speed code. Train on M, apply to S: 1 for energy, 0.5 for momentum,
0 for a mass code. The level structure (KE factor 2^(k/2) in both families) makes
that slope estimable within group.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_scene as gs  # noqa: E402

LEVELS = (-2, -1, 0, 1, 2)
RESTITUTION = 1.0
SEED0 = 30000                    # disjoint from conservation (101+), ladders (9000+, 9500+)
KE_FACTOR = {k: 2.0 ** (k / 2) for k in LEVELS}        # 0.5 .. 2, same in both families
SPEED_SCALE = {k: KE_FACTOR[k] ** 0.5 for k in LEVELS}  # KE ~ v^2
MASS_SCALE = KE_FACTOR                                  # KE ~ m
SMAX = SPEED_SCALE[2]
# Centre speeds must leave room for the whole ladder inside the generator's legal range.
CENTRE_SPEED = (gs.SPEED_RANGE[0] * SMAX, gs.SPEED_RANGE[1] / SMAX)
DENSITY = {m: gs.MATERIALS[m]["density"] for m in ("rubber", "metal")}
APPEARANCE_KEYS = ("color", "metallic", "roughness", "specular", "ior")
MAX_SIZE_SLOPE = 0.03            # |slope of log g on level| allowed per group; 0.023 is the
                                 # structural minimum with two materials. Pooled corr ~0.01


def _mass_ladder(rng, r_a0, r_b0, rho0):
  """Per level: material and radius scale g realising mass x MASS_SCALE[k], radius-flat.

  Returns None if no material assignment keeps every radius in range.
  """
  lo, hi = gs.RADIUS_RANGE
  opts = []
  for k in LEVELS:
    ok = []
    for mat, rho in DENSITY.items():
      g = (MASS_SCALE[k] * rho0 / rho) ** (1 / 3)
      if lo <= r_a0 * g <= hi and lo <= r_b0 * g <= hi:
        ok.append((mat, g))
    if not ok:
      return None
    opts.append(ok)
  ks = np.array(LEVELS, dtype=float)
  best = []
  for idx in np.ndindex(*[len(o) for o in opts]):
    logg = np.log([opts[i][j][1] for i, j in enumerate(idx)])
    slope = float(np.polyfit(ks, logg, 1)[0])
    best.append((abs(slope), idx))
  flat = [idx for s, idx in best if s <= MAX_SIZE_SLOPE]
  if not flat:
    return None
  idx = flat[rng.integers(len(flat))]
  return [{"material": opts[i][j][0], "g": float(opts[i][j][1])} for i, j in enumerate(idx)]


def clip_params(grp: dict, speed_level: int, mass_level: int) -> dict:
  m = grp["mass_ladder"][LEVELS.index(mass_level)]
  s = SPEED_SCALE[speed_level]
  mat = m["material"]
  app = {k: (list(v) if isinstance(v, tuple) else v)
         for k, v in gs.MATERIALS[mat].items() if k in APPEARANCE_KEYS}
  return {
      "arm": "legacy",
      "radius_a": grp["r_a0"] * m["g"], "radius_b": grp["r_b0"] * m["g"],
      "material_a": mat, "material_b": mat,
      "density_a": DENSITY[mat], "density_b": DENSITY[mat],
      "appearance_a": app, "appearance_b": dict(app),
      "speed_a": grp["v_a0"] * s, "speed_b": grp["v_b0"] * s,
      "direction_a_rad": grp["dir_a"], "direction_b_rad": grp["dir_b"],
      "b_moving": grp["b_moving"],
      "impact_parameter_frac": grp["b_frac"], "impact_parameter_sign": grp["b_sign"],
      "contact_offset": grp["contact_offset"],
      "restitution": RESTITUTION,
      "violates_momentum": False, "violation_scale": 0.0, "violation_direction_rad": 0.0,
      "stratum": "ke_factor_set", "target_momentum_a": None,
  }


def sample_group(seed: int) -> dict:
  """One scene, feasible at every one of the 25 (speed, mass) level pairs."""
  for attempt in range(10000):
    rng = np.random.default_rng([seed, attempt])
    r_a0, r_b0 = (float(x) for x in rng.uniform(*gs.RADIUS_RANGE, size=2))
    rho0 = float(np.exp(rng.uniform(np.log(DENSITY["rubber"]), np.log(DENSITY["metal"]))))
    ladder = _mass_ladder(rng, r_a0, r_b0, rho0)
    if ladder is None:
      continue
    b_moving = bool(rng.uniform() < gs.B_MOVING_PROB)
    lv = np.log(CENTRE_SPEED)
    jr = gs.CONTACT_JITTER_RADIUS * np.sqrt(rng.uniform())
    jt = rng.uniform(0, 2 * np.pi)
    grp = {
        "seed": seed, "attempt": attempt, "r_a0": r_a0, "r_b0": r_b0, "rho0": rho0,
        "mass_ladder": ladder, "b_moving": b_moving,
        "v_a0": float(np.exp(rng.uniform(*lv))),
        "v_b0": float(np.exp(rng.uniform(*lv))) if b_moving else 0.0,
        "dir_a": float(rng.uniform(0, 2 * np.pi)),
        "dir_b": float(rng.uniform(0, 2 * np.pi)) if b_moving else 0.0,
        "b_frac": float(rng.uniform(0, gs.IMPACT_PARAM_MAX_FRAC)),
        "b_sign": int(rng.choice([-1, 1])),
        "contact_offset": [float(jr * np.cos(jt)), float(jr * np.sin(jt))],
    }
    if all(gs.check_frame0(p, gs.solve_geometry(p)) is None
           for p in (clip_params(grp, ks, km) for ks in LEVELS for km in LEVELS)):
      return grp
  raise RuntimeError(f"seed {seed}: no feasible group")


def make_clips(grp: dict, family: str, gid: int, fixed: int) -> list[dict]:
  clips = []
  for k in LEVELS:
    ks, km = (k, fixed) if family == "S" else (fixed, k)
    clips.append({
        "name": f"{family}{gid:03d}_k{k:+d}", "seed": grp["seed"], "family": family,
        "group": f"{family}{gid:03d}", "level": k, "ke_factor": KE_FACTOR[k],
        "speed_scale": SPEED_SCALE[ks], "mass_scale": MASS_SCALE[km], "fixed_level": fixed,
        "params": clip_params(grp, ks, km),
    })
  return clips


def _check(clip):
  import generate_ke_scene
  return clip["name"], generate_ke_scene.physics_check(clip["params"])


def physics_ok(r: dict) -> bool:
  return (r["ok"] and abs(r["ke_post"] / r["ke_pre"] - 1) < 1e-3
          and abs(r["e"] - RESTITUTION) < 1e-3 and r["momentum_rel_error"] < 1e-6)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--out", default="momentum_dataset/ke_factor_set")
  ap.add_argument("--groups-per-family", type=int, default=100)
  ap.add_argument("--workers", type=int, default=32)
  args = ap.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  n = args.groups_per_family
  assert n % len(LEVELS) == 0, "fixed levels are balanced, so groups/family must divide by 5"

  t0 = time.time()
  next_seed = SEED0
  # family and fixed level per slot, balanced and shuffled once, independent of any draw
  rng = np.random.default_rng(SEED0)
  slots = [(fam, lvl) for fam in "SM" for lvl in LEVELS for _ in range(n // len(LEVELS))]
  rng.shuffle(slots)
  gid = {"S": 0, "M": 0}
  todo = []
  for fam, lvl in slots:
    todo.append((fam, gid[fam], lvl))
    gid[fam] += 1

  groups, clips, replaced = {}, {}, []
  # spawn, not fork: each worker imports kubric (and with it TensorFlow), which does
  # not survive being forked from a parent that has already imported it
  pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"))
  while todo:
    batch = []
    for fam, g, lvl in todo:
      grp = sample_group(next_seed)
      next_seed += 1
      groups[f"{fam}{g:03d}"] = {**grp, "family": fam, "fixed_level": lvl}
      batch += make_clips(grp, fam, g, lvl)
    res = dict(pool.map(_check, batch, chunksize=4))
    bad = sorted({c["group"] for c in batch if not physics_ok(res[c["name"]])})
    for c in batch:
      c["physics"] = res[c["name"]]
      if c["group"] not in bad:
        clips[c["name"]] = c
    for gname in bad:
      replaced.append({"group": gname, "seed": groups[gname]["seed"],
                       "reasons": sorted({res[c["name"]].get("reason", "tolerance")
                                          for c in batch if c["group"] == gname})})
    todo = [(g[0], int(g[1:]), groups[g]["fixed_level"]) for g in bad]
    print(f"  checked {len(batch)} clips, {len(bad)} groups to resample "
          f"({time.time() - t0:.0f}s)", flush=True)
  pool.shutdown()

  order = sorted(clips.values(), key=lambda c: (c["group"], c["level"]))
  plan = {"design": {"levels": LEVELS, "ke_factor": KE_FACTOR, "speed_scale": SPEED_SCALE,
                     "mass_scale": MASS_SCALE, "restitution": RESTITUTION, "seed0": SEED0,
                     "centre_speed": CENTRE_SPEED, "max_size_slope": MAX_SIZE_SLOPE},
          "groups": groups, "replaced_groups": replaced, "clips": order}
  (out / "plan.json").write_text(json.dumps(plan, indent=1))
  print(f"{len(order)} clips in {len(groups)} groups, "
        f"{len(replaced)} groups resampled, {time.time() - t0:.0f}s -> {out / 'plan.json'}")


if __name__ == "__main__":
  main()
