"""Grouping helpers for the arm-structured sets (§54b, §56, §57, §59, §60)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import physics
from .captures import ClipSet

__all__ = ["Ladder", "ladders"]


@dataclass
class Ladder:
    """One seed's arms: indices into the clip set, sorted by realised restitution."""

    seed: int
    idx: np.ndarray            # all non-violating arms, sorted by e
    e: np.ndarray              # their restitution_realised
    honest: int | None         # index of the honest arm (the simulated one), if tagged
    violating: int | None      # index of the violating twin, if present
    pre: physics.PreState

    @property
    def legal(self) -> np.ndarray:
        return self.idx[self.e <= 1.0 + 1e-9]

    @property
    def legal_e(self) -> np.ndarray:
        return self.e[self.e <= 1.0 + 1e-9]

    @property
    def impossible(self) -> np.ndarray:
        return self.idx[self.e > 1.0 + 1e-9]

    def knots_without_honest(self) -> tuple[np.ndarray, np.ndarray]:
        m = (self.e <= 1.0 + 1e-9) & (self.idx != self.honest)
        return self.idx[m], self.e[m]


def ladders(cs: ClipSet, *, min_arms: int = 2) -> list[Ladder]:
    out = []
    for seed, idx in cs.by_seed().items():
        arms = np.array([i for i in idx if cs.arm[i] != "violating"])
        if len(arms) < min_arms:
            continue
        e = cs.restitution[arms]
        order = np.argsort(e, kind="mergesort")
        arms, e = arms[order], e[order]
        honest = [i for i in arms if cs.arm[i] in ("honest", "anchor")]
        viol = [i for i in idx if cs.arm[i] == "violating"]
        out.append(Ladder(seed=int(seed), idx=arms, e=e,
                          honest=honest[0] if honest else None,
                          violating=viol[0] if viol else None,
                          pre=physics.pre_state_from_labels(cs.labels[arms[0]])))
    return out
