"""Kinematic-observer baselines (spec §5): fitted through the identical probe pipeline,
from ground-truth VISIBLE quantities only (no masses).

* obs_vel     pre-contact velocity components of both objects          [4]
* obs_speed   obs_vel + both speeds                                     [6]
* obs_quad    obs_speed + all pairwise products / squares               [27]
* obs_normal  pre velocities projected on the contact normal and tangent [4]

An internal-representation result is only interesting where it beats these.
"""
from __future__ import annotations

import itertools

import numpy as np

OBSERVERS = ("obs_vel", "obs_speed", "obs_quad", "obs_normal")


def _pre(labels: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pl = labels["physlaw"]
    return (np.asarray(pl["pre_velocity"]["A"], float), np.asarray(pl["pre_velocity"]["B"], float),
            np.asarray(pl["contact_normal"], float))


def observer_features(labels: dict, kind: str) -> np.ndarray:
    va, vb, n = _pre(labels)
    vel = np.concatenate([va, vb])
    speed = np.concatenate([vel, [np.linalg.norm(va), np.linalg.norm(vb)]])
    if kind == "obs_vel":
        return vel
    if kind == "obs_speed":
        return speed
    if kind == "obs_quad":
        quad = [a * b for a, b in itertools.combinations_with_replacement(speed, 2)]
        return np.concatenate([speed, quad])
    if kind == "obs_normal":
        t = np.array([-n[1], n[0]])
        return np.array([va @ n, va @ t, vb @ n, vb @ t])
    raise KeyError(kind)


def observer_matrix(all_labels: list[dict], kind: str) -> np.ndarray:
    return np.stack([observer_features(l, kind) for l in all_labels])
