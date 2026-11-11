"""Twin construction (spec §2.2): every member of a twin group shares one simulated
pre-contact segment and differs only in the post-contact velocities.

Pure numpy, no Kubric. ``generate.py`` calls :func:`f1_twins` / :func:`f2_twins` on
the honest simulation and re-keyframes the spheres along each result; the unit
tests in ``tests/test_twins.py`` check the spec's identities on these functions
directly.

Frictionless contact along the unit normal ``n`` (A -> B), approach speed
``u_n = (v_A - v_B) . n > 0``. Write the deflection magnitudes along ``n`` as

    a = -(dv_A . n) >= 0,      b = (dv_B . n) >= 0

The two laws are then two independent constraints on (a, b):

    a + b = (1 + e) u_n        restitution sets the SUM
    b / a = m_A / m_B          momentum conservation sets the SPLIT

so the F1 twins are the four corners of {legal, illegal split} x {sum at e=1,
sum at e=0.15} plus a fifth twin that keeps the legal split and moves the sum
past e = 1 (energy created, momentum still exact):

    conserve_e1        legal split, e = 1           (the simulator's own outcome)
    conserve_e015      legal split, e = 0.15        restitution edit, factor (1+0.15)/(1+1)
    mom_violate_e1     split moved, e = 1           common drift -delta*n on both balls
    mom_violate_e015   split moved, e = 0.15        same delta_split as the e1 twin
    energy_create      legal split, e = 1.5         factor lambda = (1+1.5)/(1+1)

Every twin group is built from ONE simulation at e_sim = 1. ``conserve_e015`` is
the exact frictionless legal outcome at e = 0.15, which the old generator showed
matches a Bullet run at that restitution to ~1e-8 (``restitution_realised_error``).
Simulating it separately would add nothing physically and would risk a different
pre-contact segment.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _load_collision_physics():
    """Load collision_physics.py by path so this works in the Kubric venv (no package install)."""
    name = "physlaw_collision_physics"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / "collision_physics.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CP = _load_collision_physics()
PreState = CP.PreState

F1_TWINS = ("conserve_e1", "conserve_e015", "mom_violate_e1", "mom_violate_e015", "energy_create")
F2_LEVELS = tuple(round(0.1 * i, 1) for i in range(11))
F3_RATIOS = (0.33, 0.5, 1.0, 2.0, 3.0)
F4_BOOSTS = ("u0", "usmall", "ularge")


@dataclass
class Twin:
    name: str
    v_a: np.ndarray            # [2] post-contact velocity of A
    v_b: np.ndarray            # [2] post-contact velocity of B
    e_target: float            # restitution the aftermath realises (separation / approach)
    momentum_conserved: bool
    energy_physical: bool      # KE_post <= KE_pre
    record: dict[str, Any] = field(default_factory=dict)
    masses: tuple | None = None    # (m_A, m_B) when the twin's masses differ from the simulation's (F3)
    boost: np.ndarray | None = None  # common velocity added to the whole clip (F4)
    pre_segment: str = "0"         # twins sharing it are identical before contact

    def pre(self, pre: PreState) -> PreState:
        """The pre-contact state this twin's clip shows (its masses, its boost)."""
        m_a, m_b = self.masses or (pre.m_a, pre.m_b)
        u = np.zeros(2) if self.boost is None else np.asarray(self.boost, float)[:2]
        return PreState(m_a=m_a, m_b=m_b, v_a=pre.v_a + u, v_b=pre.v_b + u, n=pre.n)


# --------------------------------------------------------------------------- #
# diagnostics, shared by the generator, the validator and the tests
# --------------------------------------------------------------------------- #


def deflections(pre: PreState, v_a, v_b) -> tuple[float, float]:
    """(a, b): speed changes along n, both positive for a real collision."""
    dv_a = np.asarray(v_a, float)[:2] - pre.v_a
    dv_b = np.asarray(v_b, float)[:2] - pre.v_b
    return float(-np.dot(dv_a, pre.n)), float(np.dot(dv_b, pre.n))


def momentum_residual(pre: PreState, v_a, v_b) -> float:
    """|p_post - p_pre| / |p_pre| (spec L1's r_p, with true post velocities)."""
    p_pre = pre.m_a * pre.v_a + pre.m_b * pre.v_b
    p_post = pre.m_a * np.asarray(v_a, float)[:2] + pre.m_b * np.asarray(v_b, float)[:2]
    return float(np.linalg.norm(p_post - p_pre) / max(np.linalg.norm(p_pre), 1e-12))


def kinetic_energy(pre: PreState, v_a, v_b) -> float:
    v_a, v_b = np.asarray(v_a, float)[:2], np.asarray(v_b, float)[:2]
    return float(0.5 * pre.m_a * v_a @ v_a + 0.5 * pre.m_b * v_b @ v_b)


def ke_pre(pre: PreState) -> float:
    return kinetic_energy(pre, pre.v_a, pre.v_b)


def realised_restitution(pre: PreState, v_a, v_b) -> float:
    sep = float(np.dot(np.asarray(v_b, float)[:2] - np.asarray(v_a, float)[:2], pre.n))
    return sep / pre.approach_speed


def summarize(pre: PreState, twin: Twin) -> dict[str, float]:
    a, b = deflections(pre, twin.v_a, twin.v_b)
    ke0 = ke_pre(pre)
    ke1 = kinetic_energy(pre, twin.v_a, twin.v_b)
    return {
        "deflection_a": a, "deflection_b": b, "deflection_sum": a + b,
        "deflection_ratio_b_over_a": b / a if a > 0 else float("inf"),
        "mass_ratio_a_over_b": pre.m_a / pre.m_b,
        "momentum_residual": momentum_residual(pre, twin.v_a, twin.v_b),
        "ke_pre": ke0, "ke_post": ke1, "ke_ratio": ke1 / ke0,
        "restitution_realised": realised_restitution(pre, twin.v_a, twin.v_b),
        "approach_speed": pre.approach_speed,
    }


# --------------------------------------------------------------------------- #
# the momentum-violating split
# --------------------------------------------------------------------------- #


def ke_reducing_sign(pre: PreState) -> float:
    """Sign of ``delta_split`` whose drift lowers |v_cm|, hence lowers kinetic energy.

    The split edit adds the same velocity ``-delta * (a+b) * n`` to both balls, so it
    changes only the centre-of-mass velocity: KE_post = KE_cm + (M/2)|v_cm'|^2 with
    KE_cm untouched. KE falls iff the drift points against v_cm, i.e.
    ``delta * (v_cm . n) > 0``. With a random sign, about half of all
    momentum-violating twins also CREATE energy (and at e = 1 every one does when
    v_cm . n ~ 0), which confounds the momentum contrast with an energy contrast.
    """
    v_cm = (pre.m_a * pre.v_a + pre.m_b * pre.v_b) / (pre.m_a + pre.m_b)
    return 1.0 if float(np.dot(v_cm, pre.n)) >= 0 else -1.0


def choose_split_delta(rng: np.random.Generator, split_honest: float,
                       magnitude_range=(0.10, 0.35), margin: float = 0.05,
                       sign: float | None = None) -> float | None:
    """Draw ``delta_split`` so the edited split fraction s' = s - delta stays in [margin, 1-margin].

    s = b/(a+b) = m_A/(m_A+m_B). If s' left (0, 1) one ball would be deflected
    *backwards* through the other, which is a visual giveaway rather than a
    momentum violation. ``sign`` is preferred if given (see :func:`ke_reducing_sign`),
    else random; if it does not fit, the other is tried; if neither fits the
    magnitude is shrunk to the largest that does, and None is returned when that
    falls below the range's minimum.
    """
    mag = float(rng.uniform(*magnitude_range))
    drawn = float(rng.choice([-1.0, 1.0]))
    sign = drawn if sign is None else float(sign)
    # collision_physics.split_edit moves b -> b - delta*(a+b), so s' = s - delta_split.
    lo, hi = split_honest - (1.0 - margin), split_honest - margin   # allowed delta interval
    for s in (sign, -sign):
        if lo <= s * mag <= hi:
            return s * mag
    room = max(hi, -lo)
    if room < magnitude_range[0]:
        return None
    return room if hi >= -lo else -room


# --------------------------------------------------------------------------- #
# twin families
# --------------------------------------------------------------------------- #


def _restitution(pre, post_a, post_b, e_sim, e_new):
    va, vb, rec = CP.restitution_edit(pre, post_a, post_b, e_sim, e_new)
    return va, vb, rec


def f1_twins(pre: PreState, post_a, post_b, e_sim: float, delta_split: float, *,
             e_low: float = 0.15, e_create: float = 1.5) -> dict[str, Twin]:
    """The five F1 twins from one honest outcome at restitution ``e_sim`` (normally 1)."""
    pre.check()
    post_a, post_b = CP._xy(post_a), CP._xy(post_b)
    out: dict[str, Twin] = {}

    for e, tag in ((1.0, "e1"), (e_low, "e015")):
        if abs(e - e_sim) < 1e-12:
            va, vb, rec = post_a.copy(), post_b.copy(), {"kind": "simulator", "e_sim": float(e_sim)}
        else:
            va, vb, rec = _restitution(pre, post_a, post_b, e_sim, e)
            rec = {"kind": "restitution", **rec}
        out[f"conserve_{tag}"] = Twin(f"conserve_{tag}", va, vb, e, True, True, rec)

        va2, vb2, srec = CP.split_edit(pre, va, vb, delta_split)
        out[f"mom_violate_{tag}"] = Twin(f"mom_violate_{tag}", va2, vb2, e, False,
                                         kinetic_energy(pre, va2, vb2) <= ke_pre(pre) + 1e-12,
                                         {"kind": "split", "base": f"conserve_{tag}", **srec})

    va, vb, rec = _restitution(pre, post_a, post_b, e_sim, e_create)
    lam = (1.0 + e_create) / (1.0 + e_sim)
    out["energy_create"] = Twin("energy_create", va, vb, e_create, True, False,
                                {"kind": "energy_create", "lambda": lam, **rec})
    return {k: out[k] for k in F1_TWINS}


def f2_twins(pre: PreState, post_a, post_b, e_sim: float, levels=F2_LEVELS) -> dict[str, Twin]:
    """F2: the legal outcome at each restitution level, identical pre-contact segments."""
    pre.check()
    out = {}
    for e in levels:
        name = f"e{e:.1f}"
        if abs(e - e_sim) < 1e-12:
            va, vb, rec = CP._xy(post_a), CP._xy(post_b), {"kind": "simulator", "e_sim": float(e_sim)}
        else:
            va, vb, rec = _restitution(pre, post_a, post_b, e_sim, e)
            rec = {"kind": "restitution", **rec}
        out[name] = Twin(name, va, vb, float(e), True, True, rec)
    return out


def f3_twins(pre: PreState, e: float, ratios=F3_RATIOS, m_total: float | None = None) -> dict[str, Twin]:
    """F3: legal outcomes at restitution ``e`` for each mass ratio m_A/m_B (masses via density only).

    The geometric mean of the two masses is held at that of the simulated pair, so the
    ratio is the only thing that changes; pre-contact frames do not depend on mass.
    """
    pre.check()
    gm = math.sqrt(pre.m_a * pre.m_b)
    out = {}
    for r in ratios:
        m_a, m_b = gm * math.sqrt(r), gm / math.sqrt(r)
        p2 = PreState(m_a=m_a, m_b=m_b, v_a=pre.v_a, v_b=pre.v_b, n=pre.n)
        dv = CP.legal_outcome(p2, e)
        name = f"ratio_{r:g}"
        out[name] = Twin(name, pre.v_a + dv[:2], pre.v_b + dv[2:], float(e), True, True,
                         {"kind": "mass_ratio", "mass_ratio": float(r), "m_A": m_a, "m_B": m_b}, masses=(m_a, m_b))
    return out


def f4_twins(pre: PreState, post_a, post_b, e_sim: float, boosts: dict, e_low: float = 0.15) -> dict[str, Twin]:
    """F4: the F1 legal twins (e = 1 and e_low) under Galilean boosts ``{name: u (2-vector)}``.

    A boost adds u to every velocity of the clip (positions gain u (t - t_contact)), so
    restitution and the centre-of-mass-frame scattering angle are invariant exactly.
    """
    pre.check()
    out = {}
    for bname, u in boosts.items():
        u = np.asarray(u, float)[:2]
        for e, tag in ((1.0, "e1"), (e_low, "e015")):
            if abs(e - e_sim) < 1e-12:
                va, vb = CP._xy(post_a), CP._xy(post_b)
            else:
                va, vb, _ = _restitution(pre, post_a, post_b, e_sim, e)
            name = f"{bname}_{tag}"
            out[name] = Twin(name, va + u, vb + u, e, True, True,
                             {"kind": "boost", "boost": u.tolist(), "boost_level": bname, "e": e},
                             boost=u, pre_segment=bname)
    return out


def com_frame_angle(pre: PreState, v_a, v_b) -> float:
    """Scattering angle in the centre-of-mass frame: angle between relative velocities before and after."""
    r0 = pre.v_a - pre.v_b
    r1 = np.asarray(v_a, float)[:2] - np.asarray(v_b, float)[:2]
    c = float(np.dot(r0, r1) / (np.linalg.norm(r0) * np.linalg.norm(r1)))
    return math.acos(max(-1.0, min(1.0, c)))
