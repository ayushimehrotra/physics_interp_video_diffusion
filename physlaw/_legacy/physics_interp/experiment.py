"""Run specs: turning one config into a list of independent units of work.

An experiment config describes a *grid* (prompts x seeds). Expanding it up-front
into concrete :class:`RunSpec` objects is what makes the whole thing shardable
across GPUs and resumable: each spec has a deterministic ``run_id`` derived from
a hash of everything that affects its output, so any process can decide on its
own whether a spec is already done.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, PromptConfig, stable_hash

__all__ = ["RunSpec", "expand_grid", "DONE_MARKER"]

#: Written into a run directory once its outputs are complete and fsync'd.
DONE_MARKER = "DONE"


@dataclass
class RunSpec:
    """One video to generate: a prompt, a seed, and everything they imply."""

    experiment: str
    prompt: PromptConfig
    seed: int
    config: ExperimentConfig = field(repr=False)

    @property
    def signature(self) -> dict[str, Any]:
        """Everything that determines this run's pixels.

        Deliberately excludes ``output_dir``, ``notes``, and the memory knobs in
        ``ModelConfig`` (tiling/offload change speed, not output), so moving an
        experiment between machines does not invalidate finished work.
        """
        m = self.config.model
        return {
            "model": {"id": m.id, "repo": m.repo, "dtype": m.dtype},
            "generation": {
                k: v
                for k, v in self.config.generation.__dict__.items()
                if not k.startswith("save_")
            },
            "prompt": self.prompt.text,
            "seed": self.seed,
            "hooks": [h.__dict__ for h in self.config.hooks],
        }

    @property
    def run_id(self) -> str:
        return stable_hash(self.signature)

    @property
    def name(self) -> str:
        return f"{self.prompt.id}__seed{self.seed}__{self.run_id}"

    def output_dir(self, root: str | Path | None = None) -> Path:
        root = Path(root or self.config.output_dir)
        return root / self.experiment / self.name

    def is_done(self, root: str | Path | None = None) -> bool:
        return (self.output_dir(root) / DONE_MARKER).exists()


def expand_grid(cfg: ExperimentConfig) -> list[RunSpec]:
    """All runs described by ``cfg``, in a stable order.

    Prompt-major ordering: consecutive specs differ by seed. Combined with the
    round-robin shard in :mod:`physics_interp.distributed`, that spreads each
    prompt's seeds across GPUs, so a partially finished sweep still has
    balanced coverage of every prompt rather than the first few complete.
    """
    specs: list[RunSpec] = []
    for prompt in cfg.prompts:
        for seed in cfg.seeds:
            specs.append(
                RunSpec(experiment=cfg.name, prompt=prompt, seed=int(seed), config=cfg)
            )
    ids = [s.run_id for s in specs]
    if len(set(ids)) != len(ids):
        raise ValueError(
            "grid contains duplicate runs — check for repeated seeds or "
            "prompts with identical text"
        )
    return specs


def pending(specs: list[RunSpec], root: str | Path | None = None) -> list[RunSpec]:
    """Specs that have not yet written a completion marker."""
    return [s for s in specs if not s.is_done(root)]
