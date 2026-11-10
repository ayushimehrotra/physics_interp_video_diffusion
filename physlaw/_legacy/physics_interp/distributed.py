"""Multi-GPU work sharding.

Design decision: **no process group, no collectives.** For a 2B denoiser that
fits comfortably on one card, the throughput-optimal and by far the most robust
arrangement is one full model replica per GPU, each chewing through its own
slice of the run grid and writing its own output files. Nothing is ever reduced
across ranks, so there is no NCCL, no rendezvous, and no way for one slow or
crashed rank to hang the others — the failure mode of a sweep is "some runs
missing", which a re-launch fixes because completed runs are skipped.

Launch with ``torchrun``, which sets ``RANK``/``WORLD_SIZE``/``LOCAL_RANK``, or
with nothing at all for a single-process run. ``scripts/pipeline/run_experiment.sh``
picks the GPU count automatically.

If a model ever needs to be *split* across GPUs (a 14B denoiser on 24GB cards),
that is a different axis — set ``model.cpu_offload`` or add a device_map to the
adapter — and it composes with this one rather than replacing it.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Sequence, TypeVar

import torch

__all__ = ["DistInfo", "get_dist_info", "shard"]

T = TypeVar("T")


@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    device: str
    hostname: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def log(self, msg: str) -> None:
        print(f"[rank {self.rank}/{self.world_size} {self.device}] {msg}", flush=True)

    def describe(self) -> dict:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "device": self.device,
            "hostname": self.hostname,
        }


def get_dist_info(device_override: str | None = None) -> DistInfo:
    """Read the launcher's environment. Works unlaunched (world size 1)."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))

    if device_override:
        device = device_override
    elif torch.cuda.is_available():
        # Respect CUDA_VISIBLE_DEVICES: if the launcher already narrowed the
        # view to one card, local_rank indexes past the end.
        n = torch.cuda.device_count()
        device = f"cuda:{local_rank % n}" if n else "cpu"
    else:
        device = "cpu"

    if device.startswith("cuda"):
        torch.cuda.set_device(device)

    return DistInfo(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        hostname=socket.gethostname(),
    )


def shard(items: Sequence[T], info: DistInfo) -> list[T]:
    """This rank's slice of ``items``, round-robin.

    Round-robin rather than contiguous chunks: runs of the same prompt differ
    only by seed and cost the same, so interleaving keeps every rank's workload
    balanced and spreads each prompt's seeds across GPUs.
    """
    return list(items[info.rank :: info.world_size])
