"""Split assignment by twin group ID (spec §2.1): every twin of a scene lands in one split.

A hash of the group ID, not a position in a list, so a group's split never changes
when the set grows or is generated in a different order.
"""
from __future__ import annotations

import hashlib


def split_for_group(group_id: str, train: float = 0.70, val: float = 0.15) -> str:
    u = int(hashlib.sha256(group_id.encode()).hexdigest()[:12], 16) / float(16 ** 12)
    return "train" if u < train else ("val" if u < train + val else "test")
