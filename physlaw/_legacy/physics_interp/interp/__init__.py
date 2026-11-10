"""Interpretability layer: hook substrate plus the linear-probe readout.

:mod:`~physics_interp.interp.hooks` stays opinion-free — it decides only *how*
to tap a model, never what to record. This module is the other half: given
activations that a capture run already pooled and stored, it fits the linear
probes that turn "is this quantity present in the representation" into a
number.

Probes are deliberately linear and deliberately weak. The claim a probe can
support is that a quantity is *linearly decodable* from a representation, which
is a statement about how accessible the information is, not about whether the
model uses it. Two habits keep that claim honest and both are enforced here:
the ridge penalty is selected on a validation split carved out of the training
half (never the test half), and every fit reports a permutation control so
"0.6 R-squared" can be read against what the same pipeline scores on shuffled
labels.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .hooks import HookContext, HookHandle, HookManager, StepFilter

__all__ = [
    "HookContext",
    "HookHandle",
    "HookManager",
    "StepFilter",
    "extract_label",
    "ridge_fit",
    "fit_probe",
    "fit_attentive_probe",
    "probe_grid",
    "select_scenes",
    "ActivationSteer",
    "build_directions",
]


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #

#: Targets that are not stored in labels.json but derived from what is. Keeping
#: them here rather than in the generator means a new question can be asked of
#: an already-rendered dataset.
DERIVED: dict[str, Any] = {
    "log_mass_ratio": lambda l: float(np.log(l["mass_ratio"])),
    "momentum_pre": lambda l: l["collision"]["pre"]["total_momentum"][:2],
    "momentum_post": lambda l: l["collision"]["post"]["total_momentum"][:2],
    "momentum_magnitude_pre": lambda l: float(
        np.linalg.norm(l["collision"]["pre"]["total_momentum"])
    ),
    "momentum_magnitude_post": lambda l: float(
        np.linalg.norm(l["collision"]["post"]["total_momentum"])
    ),
    "momentum_A_pre": lambda l: l["collision"]["pre"]["momentum_A"][:2],
    "momentum_B_pre": lambda l: l["collision"]["pre"]["momentum_B"][:2],
    "momentum_A_post": lambda l: l["collision"]["post"]["momentum_A"][:2],
    "momentum_B_post": lambda l: l["collision"]["post"]["momentum_B"][:2],
    "momentum_delta": lambda l: (
        np.asarray(l["collision"]["post"]["total_momentum"][:2])
        - np.asarray(l["collision"]["pre"]["total_momentum"][:2])
    ),
    # |Delta p|, the size of the conservation violation: exactly 0 on honest clips.
    # A scalar NORM, so a linear readout of it is not the same question as reading
    # the vector and taking the norm outside (§34a's "norm route").
    "momentum_delta_norm": lambda l: float(np.linalg.norm(
        np.asarray(l["collision"]["post"]["total_momentum"][:2])
        - np.asarray(l["collision"]["pre"]["total_momentum"][:2])
    )),
    "impulse": lambda l: l["collision"]["impulse_vector"][:2],
    # Energies. The ratio is the elasticity of the collision; the absolute values
    # are quadratic in speed and so a genuinely different readout from momentum,
    # which is linear in it — a representation could carry one and not the other.
    "ke_ratio": lambda l: float(l["collision"]["kinetic_energy_ratio"]),
    "kinetic_energy_pre": lambda l: float(l["collision"]["pre"]["kinetic_energy"]),
    "kinetic_energy_post": lambda l: float(l["collision"]["post"]["kinetic_energy"]),
    "log_kinetic_energy_pre": lambda l: float(
        np.log(max(l["collision"]["pre"]["kinetic_energy"], 1e-12))
    ),
    # Velocities: the purely kinematic part of momentum, visible in the pixels
    # without any knowledge of mass. A probe that gets velocity but not momentum
    # is reading motion; one that gets both is combining motion with mass.
    "velocity_A_pre": lambda l: l["collision"]["pre"]["velocity_A"][:2],
    "velocity_B_pre": lambda l: l["collision"]["pre"]["velocity_B"][:2],
    "velocity_A_post": lambda l: l["collision"]["post"]["velocity_A"][:2],
    "velocity_B_post": lambda l: l["collision"]["post"]["velocity_B"][:2],
    "speed_B_pre": lambda l: float(
        np.linalg.norm(l["collision"]["pre"]["velocity_B"])
    ),
    # Velocity CHANGE across the collision, per ball. These are the quantities the
    # mass ratio is actually built from: momentum conservation says
    # m_A*dv_A = -m_B*dv_B, so m_A/m_B = |dv_B|/|dv_A| exactly (frictionless, so
    # both changes lie along the contact normal — verified at 1.9e-16 on the
    # rendered set). A probe reaching these is reading kinematics, which is
    # visible; whether it can then reach the *ratio* is a separate question,
    # because that needs a quotient of magnitudes rather than a linear map.
    "delta_v_A": lambda l: (
        np.asarray(l["collision"]["post"]["velocity_A"][:2])
        - np.asarray(l["collision"]["pre"]["velocity_A"][:2])
    ),
    "delta_v_B": lambda l: (
        np.asarray(l["collision"]["post"]["velocity_B"][:2])
        - np.asarray(l["collision"]["pre"]["velocity_B"][:2])
    ),
    # The log-magnitudes, which are the *intermediates* the identity decomposes
    # into: log(m_A/m_B) = log|dv_B| - log|dv_A|. If these two decode but the
    # ratio does not, the model holds the pieces without combining them; if these
    # do not decode either, the chain breaks earlier and the ratio was never
    # reachable. Probing them localises the failure instead of leaving one null.
    "log_speed_change_A": lambda l: float(np.log(max(np.linalg.norm(
        np.asarray(l["collision"]["post"]["velocity_A"][:2])
        - np.asarray(l["collision"]["pre"]["velocity_A"][:2])), 1e-12))),
    "log_speed_change_B": lambda l: float(np.log(max(np.linalg.norm(
        np.asarray(l["collision"]["post"]["velocity_B"][:2])
        - np.asarray(l["collision"]["pre"]["velocity_B"][:2])), 1e-12))),
    # The mass SCALE, as distinct from the ratio. A collision cannot reveal it:
    # scaling both masses by any factor leaves every trajectory identical, since
    # dv = J/m and J scales with m. So this is a free negative control — a probe
    # returning it from the latent-mass set means something is leaking. Note that
    # raw mass_A is NOT the right control here: log m_A = log(scale) +
    # 0.5*log(ratio), so it inherits most of its variance from a quantity the
    # collision genuinely does give you, and reads ~0.68 even under a clean oracle.
    "log_mass_scale": lambda l: float(0.5 * np.log(
        l["objects"]["A"]["mass"] * l["objects"]["B"]["mass"])),
    # ...and the version that is actually invisible. log(mass scale) =
    # log((4/3)pi) + 3*log(radius) + log(density scale), and radius is plainly
    # visible (it decodes at 0.98), so mass scale is NOT a clean leak detector —
    # it legitimately carries a geometric term. Density scale is the quantity no
    # observer can reach: it survives neither appearance (both spheres are
    # rendered identically) nor the collision (uniform mass scaling leaves every
    # trajectory unchanged). This is the control that should read ~0.
    "log_density_scale": lambda l: float(0.5 * np.log(
        l["objects"]["A"]["density"] * l["objects"]["B"]["density"])),
    # ---- sliding-friction set (`generate_friction_scene.py`) -------------------
    #
    # Two balls coast onto per-lane friction patches at the entry frame, so the
    # pre/post split and every velocity target above apply unchanged. What is new
    # is that the quantity under test belongs to the FLOOR, not to a ball.
    #
    # Read these the way §13 reads the mass ratio. `mu = a / g` and `a` is plainly
    # visible in the trajectory, so a post-entry probe on `mu_A` is measuring
    # kinematics plus a division by a constant — the §12 identity again, and the
    # number to beat is a kinematic oracle, not zero. What is NOT an identity is
    # the same target read from a PRE-entry pool: before the ball reaches the
    # patch its motion carries nothing about the coefficient, so a pre-entry read
    # must come from the surface's appearance. That makes the arms a matched
    # pair — `visible_friction` a positive control, `latent_friction` a null that
    # is a theorem about the dataset rather than a fact about the model.
    #
    # Deceleration is mass-independent (a = mu*g for any mass), which is why
    # `radius_A`/`mass_A` are cleaner controls here than in the momentum set:
    # there they enter the target algebraically, here they cannot.
    # ---- sliding-cube set (`generate_slide_scene.py`, §25) ---------------------
    #
    # One cube on a floor whose mu varies, side view. `A` is the cube, `B` is the
    # FLOOR — the floor is the second object here and it owns the coefficient, so
    # `mu` from a B pool asks whether the model attributed the property to the
    # surface, and in the latent arm that must read zero.
    #
    # TWO coefficients, and the difference matters. `mu` is what the sampler drew;
    # `mu_effective` is what the rendered motion actually realises, decel_measured/g.
    # They differ by a SYSTEMATIC +1.4% (median; +0.5% at mu 0.15-0.25 rising to
    # +3.3% at 0.60-0.80, above mu*g in 100% of scenes) because the banded floor
    # perturbs the slide at tile seams and high-mu scenes have few moving frames to
    # average over. They correlate at r = 0.9998, so rankings barely move — but
    # `mu_effective` is the label of what the pixels show, and it is the one to
    # probe when the question is "what does the video contain".
    "mu": lambda l: float(l["friction"]["mu"]),
    "log_mu": lambda l: float(np.log(l["friction"]["mu"])),
    "mu_effective": lambda l: float(
        l["friction"]["decel_measured"] / l["friction"]["gravity"]),
    "log_mu_effective": lambda l: float(np.log(
        l["friction"]["decel_measured"] / l["friction"]["gravity"])),
    "decel": lambda l: float(l["friction"]["decel"]),
    "decel_measured": lambda l: float(l["friction"]["decel_measured"]),
    "stop_frame": lambda l: float(l["friction"]["stop_frame"]),
    "travel": lambda l: float(l["friction"]["travel"]),
    # Controls for the slide set. a = mu*g is independent of mass AND size, so
    # unlike the momentum set these cannot enter the target algebraically — a probe
    # that reads them has demonstrably not read friction.
    "cube_size": lambda l: float(l["objects"]["A"]["size"]),
    "speed_initial": lambda l: float(l["objects"]["A"]["speed_initial"]),
    # Constant in the latent arm (probe_grid drops it); the cue in the visible arm.
    "floor_albedo": lambda l: float(l["objects"]["B"]["albedo"]),
    # ---- inclined plane (`generate_incline_scene.py`, §32) ---------------------
    #
    # THE SET THAT SEPARATES mu FROM ACCELERATION. On a level floor a = mu*g, and
    # with g fixed the two are the same number — measured on slide_set_v2,
    # corr(mu, decel) = 0.99976. On a slope
    #
    #     a = g*(sin th - mu*cos th)     so     mu = (sin th - a/g) / cos th
    #
    # and theta is sampled independently of mu, so corr(mu, a) drops to ~ -0.65.
    # A linear readout of acceleration ALONE therefore tops out near R2 0.43 on
    # mu; anything above that is not "reading acceleration".
    #
    # `theta` is the control that makes the composition test possible: it is a
    # large visual feature and should decode easily. §26/§30 predict that the
    # DIRECT probe on mu now falls short of composing decoded a with decoded
    # theta, because mu here is a quotient rather than a linear functional — the
    # opposite of the flat-floor case.
    "theta": lambda l: float(l["friction"]["theta"]),
    "theta_deg": lambda l: float(l["friction"]["theta_deg"]),
    "sin_theta": lambda l: float(l["friction"]["sin_theta"]),
    "cos_theta": lambda l: float(l["friction"]["cos_theta"]),
    "tan_theta": lambda l: float(l["friction"]["tan_theta"]),
    "accel": lambda l: float(l["friction"]["accel"]),
    "accel_measured": lambda l: float(l["friction"]["accel_measured"]),
    # mu as recovered from the visible kinematics by the exact identity — equals
    # the sampled mu to solver precision. It is the label to probe when the
    # question is "what do the pixels contain", the same role mu_effective plays
    # on the slide set.
    "mu_from_kinematics": lambda l: float(l["friction"]["mu_from_kinematics"]),
    "speed_final": lambda l: float(l["friction"]["speed_final"]),
    # ---- two-ball friction set (`generate_friction_scene.py`, §24) -------------
    "mu_A": lambda l: float(l["friction"]["mu_A"]),
    "mu_B": lambda l: float(l["friction"]["mu_B"]),
    "log_mu_A": lambda l: float(np.log(l["friction"]["mu_A"])),
    "log_mu_B": lambda l: float(np.log(l["friction"]["mu_B"])),
    # The relational one: needs both lanes, so no single object-local pool holds
    # its ingredients — the same structural reason §20 found `log_mass_ratio`
    # clean of the pooling-mask leak.
    "log_mu_ratio": lambda l: float(l["friction"]["log_mu_ratio"]),
    # The scale, as distinct from the ratio. Unlike `log_density_scale` in the
    # momentum set this is NOT an unrecoverable control: both decelerations are
    # visible, so their geometric mean is too. It is here to be compared against
    # the ratio, not to read zero.
    "log_mu_scale": lambda l: float(0.5 * (np.log(l["friction"]["mu_A"])
                                           + np.log(l["friction"]["mu_B"]))),
    "decel_A": lambda l: float(l["friction"]["decel_A"]),
    "decel_B": lambda l: float(l["friction"]["decel_B"]),
    "log_decel_A": lambda l: float(np.log(l["friction"]["decel_A"])),
    "stop_frame_A": lambda l: float(l["friction"]["stop_frame_A"]),
    "stop_frame_B": lambda l: float(l["friction"]["stop_frame_B"]),
    # Appearance of the surface. Constant in the latent arm (probe_grid drops a
    # constant target), and in the visible arm it is what a pre-entry read of
    # `mu_A` would have to be going through — so probing both separates "reads
    # the surface" from "reads the motion".
    "patch_albedo_A": lambda l: float(l["floor"]["patch_albedo_A"]),
    "patch_albedo_B": lambda l: float(l["floor"]["patch_albedo_B"]),
    "boundary_x": lambda l: float(l["floor"]["boundary_x"]),
    # Static/appearance variables. These are the controls: they are readable
    # straight off a single frame, so a probe scoring well on them proves only
    # that the capture pipeline works, and sets the bar a dynamic quantity has
    # to clear to be interesting.
    "mass_A": lambda l: float(l["objects"]["A"]["mass"]),
    "mass_B": lambda l: float(l["objects"]["B"]["mass"]),
    "radius_A": lambda l: float(l["objects"]["A"]["radius"]),
    "radius_B": lambda l: float(l["objects"]["B"]["radius"]),
    "restitution": lambda l: float(l["restitution"]),
    "speed_A_pre": lambda l: float(
        np.linalg.norm(l["collision"]["pre"]["velocity_A"])
    ),
    "speed_A_post": lambda l: float(
        np.linalg.norm(l["collision"]["post"]["velocity_A"])
    ),
    "speed_B_post": lambda l: float(
        np.linalg.norm(l["collision"]["post"]["velocity_B"])
    ),
    # DIRECTION, as the unit vector (cos h, sin h). The polar half of the
    # velocity factorisation, and the reason it is a 2-vector rather than the
    # angle itself: heading is circular, so regressing the raw angle puts a
    # discontinuity at +-pi into the target and a linear probe pays for it
    # twice. The unit vector is the standard fix and is what §19 measures.
    #
    # NOT an identity trap and NOT a §34b collapse: h = v/|v| is a *quotient*,
    # so the heading weights are not forced to lie in the span of the velocity
    # weights the way a linear combination would be. How far from linear it is
    # in THIS dataset is an empirical question and is measured, not assumed --
    # if speed varied little, h would be near-linear in v and the comparison
    # would go vacuous. See the label-algebra audit in scripts/geometry_v1.py.
    #
    # Undefined for a stationary ball. B starts at rest in 24 of v1's 75
    # conserving scenes, so anything reading heading_B must filter on
    # `b_moving` -- the target returns (0, 0) there, which is not a heading.
    "heading_A_pre": lambda l: (
        np.asarray(l["collision"]["pre"]["velocity_A"][:2])
        / max(float(np.linalg.norm(l["collision"]["pre"]["velocity_A"][:2])), 1e-12)
    ),
    "heading_B_pre": lambda l: (
        np.asarray(l["collision"]["pre"]["velocity_B"][:2])
        / max(float(np.linalg.norm(l["collision"]["pre"]["velocity_B"][:2])), 1e-12)
    ),
    "b_moving": lambda l: float(l.get("b_moving", True)),
    # Defaults to conserving when the key is absent, so datasets generated before
    # violations existed (the original pilot) still resolve this target — and
    # resolve it *correctly*, since every scene in them does conserve momentum.
    # A probe on such a set gets a constant label and scores nan, which is the
    # honest answer rather than a crash.
    "momentum_conserved": lambda l: float(l.get("momentum_conserved", True)),
    "violates_momentum": lambda l: float(l.get("violates_momentum", False)),
}


def extract_label(labels: dict, target: str) -> np.ndarray:
    """Pull one probe target out of a scene's ``labels.json``.

    Accepts a name from :data:`DERIVED` or a dotted path into the label tree.
    Always returns a 1-D array, so scalar and vector targets are handled by the
    same downstream code — a 2-vector momentum is fit component-wise and scored
    jointly, which is what makes "is momentum decodable" a single number rather
    than two.

    The z components are dropped by the derived momentum targets on purpose:
    the scenes are exactly planar by construction, so z is identically zero and
    including it would inflate every multi-output score with a component that
    is trivially predictable from a constant.
    """
    if target in DERIVED:
        value = DERIVED[target](labels)
    else:
        value = labels
        for part in target.split("."):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(
                    f"label path {target!r} failed at {part!r}. "
                    f"Available derived targets: {sorted(DERIVED)}"
                )
            value = value[part]
    return np.atleast_1d(np.asarray(value, dtype=np.float64))


# --------------------------------------------------------------------------- #
# ridge
# --------------------------------------------------------------------------- #


def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ridge. Returns ``(W [d, k], b [k])``.

    Solves in whichever space is smaller. With a few hundred scenes against
    1920 activation dimensions the dual form is not an optimisation but the
    difference between a 1920x1920 solve and an n-by-n one, per alpha, per
    block, per step — and the probing grid does thousands of these.
    """
    n, d = X.shape
    x_mean, y_mean = X.mean(0), Y.mean(0)
    Xc, Yc = X - x_mean, Y - y_mean

    if n >= d:
        A = Xc.T @ Xc + alpha * np.eye(d)
        W = np.linalg.solve(A, Xc.T @ Yc)
    else:
        # W = X^T (X X^T + aI)^-1 Y
        G = Xc @ Xc.T + alpha * np.eye(n)
        W = Xc.T @ np.linalg.solve(G, Yc)
    return W, y_mean - x_mean @ W


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Multi-output R-squared, pooled over components.

    Pooled rather than averaged per component: for a vector target like a 2-D
    momentum, a probe that nails x and fails y should not be rescued by
    averaging, and pooling weights each component by how much variance it
    actually carries.
    """
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_probe(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    alphas: Sequence[float] = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4),
    train_fraction: float = 0.6,
    split_seed: int = 0,
    standardize: bool = True,
    permutation_control: bool = True,
) -> dict[str, Any]:
    """Fit one linear probe and score it on held-out scenes.

    ``X`` is ``[n_scenes, dim]`` and ``Y`` is ``[n_scenes, k]``. The split is
    by scene, which is the only unit that is actually independent here — two
    pooled vectors from the same clip share everything except which tokens they
    averaged, so splitting on anything finer would leak.
    """
    n = X.shape[0]
    if n < 8:
        raise ValueError(f"need at least 8 scenes to fit and score a probe, got {n}")

    rng = np.random.default_rng(split_seed)
    order = rng.permutation(n)
    n_train = int(round(train_fraction * n))
    tr, te = order[:n_train], order[n_train:]
    # Carve the alpha-selection split out of train only. Selecting on test is
    # the standard way to report a probe accuracy that does not replicate.
    n_inner = max(4, int(round(0.75 * len(tr))))
    inner_tr, inner_val = tr[:n_inner], tr[n_inner:]
    if len(inner_val) < 2:
        inner_tr, inner_val = tr, tr  # too few scenes to hold out; report honestly

    def prep(fit_idx):
        mu = X[fit_idx].mean(0)
        sd = X[fit_idx].std(0) + 1e-6 if standardize else np.ones(X.shape[1])
        return mu, sd

    mu, sd = prep(inner_tr)
    scores = []
    for a in alphas:
        W, b = ridge_fit((X[inner_tr] - mu) / sd, Y[inner_tr], a)
        scores.append(_r2(Y[inner_val], (X[inner_val] - mu) / sd @ W + b))
    best_alpha = float(alphas[int(np.nanargmax(scores))])

    mu, sd = prep(tr)
    W, b = ridge_fit((X[tr] - mu) / sd, Y[tr], best_alpha)
    test_r2 = _r2(Y[te], (X[te] - mu) / sd @ W + b)
    train_r2 = _r2(Y[tr], (X[tr] - mu) / sd @ W + b)

    out = {
        "r2": test_r2,
        "train_r2": train_r2,
        "alpha": best_alpha,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "dim": int(X.shape[1]),
    }

    # A binary target (plausible vs implausible) is a classification question, and
    # R-squared is the wrong way to report it — a ridge fit can have a mediocre
    # R-squared while separating the classes almost perfectly. Threshold the same
    # fit instead. Balanced accuracy is reported alongside because a split that
    # happens to be class-imbalanced would flatter plain accuracy.
    if Y.shape[1] == 1 and set(np.unique(Y)) <= {0.0, 1.0}:
        pred = ((X[te] - mu) / sd @ W + b).ravel() > 0.5
        truth = Y[te].ravel() > 0.5
        out["accuracy"] = float((pred == truth).mean())
        per_class = [
            float((pred[truth == c] == c).mean()) for c in (False, True)
            if (truth == c).any()
        ]
        out["balanced_accuracy"] = float(np.mean(per_class))
        out["positive_rate"] = float(truth.mean())

    if permutation_control:
        # Same pipeline, labels shuffled. With d >> n a ridge probe can look
        # impressive on pure noise, so the number that matters is the gap
        # between r2 and this, not r2 itself.
        perm = rng.permutation(n)
        Wp, bp = ridge_fit((X[tr] - mu) / sd, Y[perm][tr], best_alpha)
        out["r2_shuffled"] = _r2(Y[perm][te], (X[te] - mu) / sd @ Wp + bp)
        if "accuracy" in out:
            pred = ((X[te] - mu) / sd @ Wp + bp).ravel() > 0.5
            out["accuracy_shuffled"] = float(
                (pred == (Y[perm][te].ravel() > 0.5)).mean()
            )
    return out


def fit_attentive_probe(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    train_fraction: float = 0.6,
    split_seed: int = 0,
    hidden: int = 256,
    heads: int = 4,
    epochs: int = 120,
    lr: float = 1e-3,
    weight_decay: float = 0.1,
    permutation_control: bool = True,
    device: str = "cuda",
) -> dict[str, Any]:
    """Patch-preserving attentive-MLP probe: a learned query pools, an MLP reads.

    ``X`` is ``[n_scenes, tokens, dim]`` — the raw tokens, not pooled. A single
    learned query attends over them and the pooled vector goes through a small
    MLP, so the probe chooses *which patches to look at* instead of being handed
    a mean. Joseph et al. run this alongside the linear probe because pooling can
    hide spatial and temporal structure, and read the two together.

    **This probe is non-linear, and that changes what its number means.** A
    linear probe bounds what is *linearly accessible*; this bounds what is
    *present in the patches at all*. A gap between them says the information is
    there but not linearly formatted — which is a claim about the representation,
    not about the model's competence. Do not quote this figure where a linear
    probe result is expected: the two answer different questions, and the whole
    reason the probes here are linear is that "decodable by an arbitrary
    non-linear head" is nearly unfalsifiable.

    Split is by scene, like :func:`fit_probe`, and a permutation control is
    reported for the same reason — with a flexible head on few scenes, fitting
    noise is easy.
    """
    import torch
    import torch.nn as nn

    n = X.shape[0]
    if n < 8:
        raise ValueError(f"need at least 8 scenes, got {n}")
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(split_seed)
    order = rng.permutation(n)
    n_train = int(round(train_fraction * n))
    tr, te = order[:n_train], order[n_train:]

    class Attentive(nn.Module):
        def __init__(self, dim, out_dim):
            super().__init__()
            self.q = nn.Parameter(torch.randn(1, 1, dim) * dim ** -0.5)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.head = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                nn.Linear(hidden, out_dim))

        def forward(self, x):
            q = self.q.expand(x.shape[0], -1, -1)
            pooled, _ = self.attn(q, x, x, need_weights=False)
            return self.head(pooled[:, 0])

    def run(Yv):
        torch.manual_seed(split_seed)
        model = Attentive(X.shape[2], Yv.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
        yt = torch.as_tensor(Yv, dtype=torch.float32, device=dev)
        mu, sd = yt[tr].mean(0), yt[tr].std(0) + 1e-6
        itr = torch.as_tensor(tr, device=dev)
        for _ in range(epochs):
            model.train()
            perm = itr[torch.randperm(len(itr), device=dev)]
            for k in range(0, len(perm), 32):
                b = perm[k:k + 32]
                loss = ((model(xt[b]) - (yt[b] - mu) / sd) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = (model(xt[te]) * sd + mu).cpu().numpy()
        return pred

    pred = run(Y)
    out = {"r2": _r2(Y[te], pred), "n_train": int(len(tr)), "n_test": int(len(te)),
           "dim": int(X.shape[2]), "tokens": int(X.shape[1]), "probe": "attentive_mlp"}
    if Y.shape[1] == 1 and set(np.unique(Y)) <= {0.0, 1.0}:
        p, t = pred.ravel() > 0.5, Y[te].ravel() > 0.5
        out["accuracy"] = float((p == t).mean())
        out["balanced_accuracy"] = float(np.mean(
            [float((p[t == c] == c).mean()) for c in (False, True) if (t == c).any()]))
    if permutation_control:
        Yp = Y[rng.permutation(n)]
        pp = run(Yp)
        out["r2_shuffled"] = _r2(Yp[te], pp)
        if "accuracy" in out:
            out["accuracy_shuffled"] = float(
                ((pp.ravel() > 0.5) == (Yp[te].ravel() > 0.5)).mean())
    return out


def select_scenes(labels: list[dict], where: dict[str, Any] | None) -> np.ndarray:
    """Boolean mask over scenes matching every ``label_path == value`` in ``where``.

    Exists so the matched-momentum stratum can be probed without regenerating or
    re-capturing anything: the stratum is the subset where momentum and mass are
    decorrelated, and therefore the only subset on which a momentum probe is
    distinguishable from a mass probe.
    """
    mask = np.ones(len(labels), dtype=bool)
    for path, wanted in (where or {}).items():
        for i, scene in enumerate(labels):
            value = scene
            for part in path.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            mask[i] &= (value == wanted)
    return mask


def probe_grid(
    activations: dict[str, np.ndarray],
    labels: list[dict],
    *,
    targets: Sequence[str],
    pools: Sequence[str],
    pool_names: Sequence[str],
    steps: Sequence[int] | None = None,
    where: dict[str, Any] | None = None,
    **fit_kwargs,
) -> list[dict[str, Any]]:
    """Fit a probe for every (site, capture step, pool, target) combination.

    ``activations`` maps site name to ``[n_scenes, steps, pools, dim]``. The
    result is a flat list of records rather than a nested structure because the
    interesting views — accuracy against depth, against noise level, against
    which object was pooled — are all different groupings of the same table.

    ``where`` restricts the fit to a subset of scenes, e.g.
    ``{"stratum": "matched_momentum"}``.
    """
    pool_idx = {name: i for i, name in enumerate(pool_names)}
    missing = sorted({part for p in pools for part in p.split("+")} - set(pool_idx))
    if missing:
        raise KeyError(f"unknown pool(s) {missing}; have {list(pool_names)}")

    subset = select_scenes(labels, where)
    if subset.sum() < 8:
        raise ValueError(
            f"`where`={where} selects {subset.sum()} scene(s); need at least 8"
        )
    labels = [l for l, keep in zip(labels, subset) if keep]
    activations = {s: a[subset] for s, a in activations.items()}

    Y_by_target = {}
    for t in targets:
        Y = np.stack([extract_label(l, t) for l in labels])
        if np.ptp(Y, axis=0).max() == 0:
            # A constant target cannot be regressed and would report nan for every
            # fit; say so once here rather than emitting a grid of nans.
            continue
        Y_by_target[t] = Y

    rows = []
    for site, arr in sorted(activations.items()):
        n_steps = arr.shape[1]
        step_range = range(n_steps) if steps is None else steps
        for si in step_range:
            for pool in pools:
                # A '+'-joined spec concatenates pools along the FEATURE axis
                # instead of naming one. This is not a convenience: a relation
                # *between* two objects cannot survive a mean over both of them.
                # Pooling A and B together maps them onto the same coordinates,
                # so their ratio is gone before the probe sees anything —
                # measured on v1, log_mass_ratio reads 0.70-0.81 from an
                # object-local pool and 0.02 from any frame-wide one. Giving the
                # probe [A ; B] side by side lets a linear readout form a
                # difference between them, which in log space is the ratio.
                X = np.concatenate(
                    [arr[:, si, pool_idx[p], :] for p in pool.split("+")], axis=-1
                ).astype(np.float64)
                # A pool can be empty when its object left the frame; those
                # rows are exactly zero by construction in build_pool_weights.
                keep = np.abs(X).sum(1) > 0
                if keep.sum() < 8:
                    continue
                for target, Y in Y_by_target.items():
                    rec = fit_probe(X[keep], Y[keep], **fit_kwargs)
                    rec.update(
                        site=site, step_index=int(si), pool=pool, target=target,
                        n_dropped=int((~keep).sum()),
                    )
                    rows.append(rec)
    return rows


# --------------------------------------------------------------------------- #
# steering
# --------------------------------------------------------------------------- #


class ActivationSteer:
    """Write a fixed direction into the video-token stream at every hooked site.

    The write-side counterpart to :class:`~physics_interp.ActivationRecorder`:
    same hook substrate, same model-independent site addressing, but it returns
    a modified tensor instead of storing one. Probing asks whether a quantity is
    *present*; this asks whether it is *used*, and those come apart — a probe can
    decode a variable from a direction the model never reads.

    Three choices here are not incidental.

    ``scale="sigma"`` multiplies the unit direction by each token's own standard
    deviation before adding it, so a given ``alpha`` means the same perturbation
    size at every depth. Activation norms grow substantially through a residual
    stream, so a fixed absolute ``alpha`` would be a negligible nudge at block 2
    and a wrecking ball at block 28, and any depth sweep run that way measures
    the norm profile rather than where the model is sensitive.

    ``tokens`` restricts the edit to a subset of video tokens — the cells where
    one sphere actually is, or only the pre-collision latent frames. This is the
    localization that separates "the model has a momentum variable" from "adding
    this vector anywhere perturbs the video": a direction that only works when
    written to the tokens the object occupies is far better evidence than one
    that needs the whole frame. The mask is the same ``[tokens]`` layout the
    pooling weights are built in, so a pool used for probing can be reused
    directly as a steering region.

    ``branch`` matters only under CFG. Inversion and resampling here run
    unguided, so the batch is a single row and this is trivially ``[0]`` — but
    naming it keeps the class honest if a guided trajectory is ever steered.
    """

    def __init__(
        self,
        direction,
        alpha: float = 1.0,
        *,
        branch: str = "cond",
        scale: str = "sigma",
        tokens=None,
    ):
        import torch

        d = torch.as_tensor(direction, dtype=torch.float32)
        if d.ndim != 1:
            raise ValueError(f"direction must be 1-D [dim], got shape {tuple(d.shape)}")
        norm = torch.linalg.norm(d)
        if norm == 0:
            raise ValueError("direction has zero norm")
        # Stored unit-length so alpha is the only magnitude knob. A direction
        # fitted by ridge carries an arbitrary scale that has nothing to do with
        # how far the model should be pushed.
        self.direction = d / norm
        self.alpha = float(alpha)
        self.branch = branch
        if scale not in ("sigma", "absolute"):
            raise ValueError(f"scale must be 'sigma' or 'absolute', got {scale!r}")
        self.scale = scale
        self.tokens = None if tokens is None else torch.as_tensor(tokens, dtype=torch.float32)
        self.calls = 0

    def __call__(self, x, ctx):
        # Imported per call rather than at module scope: everything else in this
        # module is numpy-only so probes can be re-fit on a machine with no
        # torch, and steering is the one part that necessarily has the model.
        import torch

        rows = ctx.rows(self.branch) or [0]
        sl = ctx.layout.video_slice()
        vid = x[rows[0], sl, :]

        d = self.direction.to(device=vid.device, dtype=torch.float32)
        if self.scale == "sigma":
            gain = vid.float().std(dim=-1, keepdim=True)     # [tokens, 1]
        else:
            gain = 1.0
        delta = self.alpha * gain * d                        # [tokens, dim]

        if self.tokens is not None:
            mask = self.tokens.to(device=vid.device, dtype=torch.float32)
            if mask.shape[0] != vid.shape[0]:
                raise ValueError(
                    f"token mask has {mask.shape[0]} entries for {vid.shape[0]} video tokens"
                )
            delta = delta * mask[:, None]

        out = x.clone()
        out[rows[0], sl, :] = vid + delta.to(vid.dtype)
        self.calls += 1
        return out


def build_directions(
    X: np.ndarray,
    y: np.ndarray,
    *,
    controls: dict[str, np.ndarray] | None = None,
    temporal_mean: np.ndarray | None = None,
    subset: np.ndarray | None = None,
    alpha: float = 100.0,
    quantile: float = 0.33,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Candidate steering directions for one scalar target, all unit-length.

    ``X`` is ``[n_scenes, dim]`` pooled activations at one site/step/pool and
    ``y`` is ``[n_scenes]``. Pooling is a weighted mean over tokens, so it
    leaves the feature axis untouched and a direction fitted here lives in the
    same space a token occupies — which is what makes it addable back into the
    residual stream at inference.

    Several estimators rather than one, because probe weights and causal
    directions are not the same object: a ridge solution picks whatever
    predicts, and under ``d >> n`` (1920 dimensions against 150 scenes) that
    need not be a direction the model reads. Which one actually steers is an
    empirical question, so they are all returned and swept.

    * ``probe`` — ridge weights, rescaled out of standardized space. The
      convention in the CAV steering literature.
    * ``diff_means`` — top minus bottom ``quantile`` of the target. The standard
      alternative, and usually more robust when ``d >> n``.
    * ``diff_means_subset`` — the same on ``subset`` only. Here that is the
      matched-momentum stratum, the 41 scenes where momentum and mass are
      decorrelated by construction and so the only place a momentum direction is
      distinguishable from a mass direction.
    * ``temporal`` — fitted after subtracting ``temporal_mean``, the per-scene
      mean across the phase pools. Motion-control work isolates dynamics from
      appearance exactly this way: the static content is shared across frames,
      so removing the frame-wise mean leaves what moves.
    * ``residual`` — ``probe`` with every control direction projected out, so it
      is orthogonal to the mass and radius readouts and cannot be either of
      them. The direct answer to this dataset's central confound.
    * ``random`` / ``shuffled`` — negative controls. ``random`` is an isotropic
      unit vector; ``shuffled`` is the same probe pipeline on permuted labels.
      A real effect has to beat both *at the same alpha*, which matters because
      a large enough perturbation in any direction eventually degrades a video.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    d = X.shape[1]

    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def probe_dir(Xa, ya):
        mu, sd = Xa.mean(0), Xa.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        W, _ = ridge_fit((Xa - mu) / sd, ya.reshape(-1, 1), alpha)
        # Back out of standardized space: d(y)/d(x) = W / sd. Skipping this
        # returns a direction that is only correct where every feature happens
        # to have the same scale, which is never.
        return unit((W[:, 0] / sd))

    def dim_dir(Xa, ya):
        lo, hi = np.quantile(ya, quantile), np.quantile(ya, 1 - quantile)
        a, b = Xa[ya >= hi], Xa[ya <= lo]
        if len(a) < 3 or len(b) < 3:
            return np.full(d, np.nan)
        return unit(a.mean(0) - b.mean(0))

    out = {"probe": probe_dir(X, y), "diff_means": dim_dir(X, y)}

    if subset is not None and subset.sum() >= 12:
        out["diff_means_subset"] = dim_dir(X[subset], y[subset])

    if temporal_mean is not None:
        out["temporal"] = probe_dir(X - temporal_mean, y)

    if controls:
        v = out["probe"].copy()
        for c in controls.values():
            c = unit(np.asarray(c, dtype=np.float64))
            v = v - (v @ c) * c
        out["residual"] = unit(v)

    out["random"] = unit(rng.standard_normal(d))
    out["shuffled"] = probe_dir(X, rng.permutation(y))
    return out
