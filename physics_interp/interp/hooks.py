"""The hook substrate.

This module deliberately contains **no opinion about what to record or how to
intervene**. It provides only the four things that a video diffusion model makes
awkward to get at, and that you would otherwise have to retrofit into every
analysis you write:

1. **Which denoising step you are in.** The same block runs `num_inference_steps`
   times per video. A pre-hook on the denoiser root counts calls and reads the
   timestep off the forward's kwargs, so any hook can ask ``ctx.step`` /
   ``ctx.timestep`` / ``ctx.progress``.

2. **What each row of the batch is.** Classifier-free guidance stacks
   ``[uncond, cond]`` into one forward. ``ctx.branches`` labels every batch row,
   and ``ctx.rows("cond")`` gives the indices — so a hook can look at one branch,
   at both, or at the *difference* between them (the guidance direction itself,
   which has no analogue in LLM interp).

3. **Where a token is in space and time.** ``ctx.layout`` maps token index to
   (latent frame, y, x) — see :class:`~physics_interp.models.base.TokenLayout`.

4. **A read/write tap on any module in/out, addressed by a model-independent
   name.** ``block.12.resid_post`` means the same thing whether the denoiser is
   CogVideoX, Wan, or LTX.

The entire API is one method::

    def fn(x, ctx):        # x is [batch, tokens, dim]
        ...
        return None        # observe only
        return new_x       # replace

    with runner.hooks.hook("block.*.resid_post", fn):
        runner.generate(prompt=...)

Everything else — what to store, how to reduce it, what edits to apply — belongs
on top of this and is intentionally not decided here.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch

from ..models.base import ModelAdapter, SiteSpec, TokenLayout

__all__ = ["StepFilter", "HookContext", "HookHandle", "HookManager"]


class StepFilter:
    """Which denoising steps a hook fires on.

    Specs: ``"all"`` / ``None``; an int or list of ints; or a dict with any of
    ``start`` / ``end`` (half-open) / ``every``.
    """

    def __init__(self, spec: Any = "all"):
        self.spec = spec
        self._explicit: set[int] | None = None
        self._start, self._end, self._every = 0, None, 1

        if spec is None or spec == "all":
            pass
        elif isinstance(spec, bool):
            raise TypeError("step filter must not be a bool")
        elif isinstance(spec, int):
            self._explicit = {spec}
        elif isinstance(spec, (list, tuple, set)):
            self._explicit = {int(s) for s in spec}
        elif isinstance(spec, dict):
            unknown = set(spec) - {"start", "end", "every"}
            if unknown:
                raise ValueError(f"unknown step-filter key(s): {sorted(unknown)}")
            self._start = int(spec.get("start", 0))
            self._end = None if spec.get("end") is None else int(spec["end"])
            self._every = max(1, int(spec.get("every", 1)))
        else:
            raise TypeError(f"cannot interpret step filter {spec!r}")

    def __call__(self, step: int) -> bool:
        if self._explicit is not None:
            return step in self._explicit
        if step < self._start or (self._end is not None and step >= self._end):
            return False
        return (step - self._start) % self._every == 0

    def __repr__(self) -> str:
        return f"StepFilter({self.spec!r})"


@dataclass
class HookContext:
    """Everything a hook function needs to know about where it fired."""

    site: SiteSpec
    layout: TokenLayout
    step: int
    timestep: float
    call_idx: int
    branches: list[str]
    num_steps: int | None = None
    adapter: ModelAdapter | None = None

    def rows(self, branch: str) -> list[int]:
        """Batch-row indices belonging to a CFG branch (``"cond"``/``"uncond"``)."""
        if branch == "both" or branch == "all":
            return list(range(len(self.branches)))
        return [i for i, b in enumerate(self.branches) if b == branch]

    def select(self, x: torch.Tensor, branch: str) -> torch.Tensor:
        """The slice of ``x`` belonging to ``branch``, as ``[n, tokens, dim]``."""
        return x[self.rows(branch)]

    @property
    def progress(self) -> float:
        """Fraction of the denoising trajectory completed, in ``[0, 1)``.

        Step 0 is pure noise, step ``num_steps-1`` is nearly the final video, so
        this is the axis along which "when does the model commit to a physical
        outcome" is measured.
        """
        return self.step / self.num_steps if self.num_steps else 0.0


@dataclass
class HookHandle:
    """A registered hook. Pass to :meth:`HookManager.remove` to detach it."""

    sites: set[str]
    fn: Callable[[torch.Tensor, HookContext], torch.Tensor | None]
    steps: StepFilter
    name: str = "hook"
    calls: int = 0


class HookManager:
    """Owns all taps on one loaded model, plus the denoising bookkeeping.

    Attached once for the lifetime of a runner; individual hooks come and go, so
    a grid of experiments reuses one set of loaded weights.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        layout: TokenLayout,
        *,
        guidance_scale: float = 6.0,
        num_steps: int | None = None,
    ):
        self.adapter = adapter
        self.layout = layout
        self.guidance_scale = guidance_scale
        self.num_steps = num_steps

        self._hooks: list[HookHandle] = []
        self._handles: list[Any] = []

        self.call_idx = 0
        self.step = 0
        self.timestep = float("nan")
        self.timesteps_seen: list[float] = []
        self._checked_layout: set[str] = set()

    # ---- registration ----------------------------------------------------- #

    def add(
        self,
        sites: str | Iterable[str],
        fn: Callable[[torch.Tensor, HookContext], torch.Tensor | None],
        *,
        steps: Any = "all",
        name: str = "hook",
    ) -> HookHandle:
        """Tap ``sites`` with ``fn``.

        ``fn(x, ctx)`` receives the whole batch as ``[batch, tokens, dim]``.
        Return ``None`` to observe, or a same-shaped tensor to replace. Do not
        mutate ``x`` in place — other hooks on the same site see it too.
        """
        resolved = self.adapter.resolve_sites(sites, self.layout)
        handle = HookHandle(
            sites={s.name for s in resolved}, fn=fn, steps=StepFilter(steps), name=name
        )
        self._hooks.append(handle)
        self._refresh()
        return handle

    def remove(self, handle: HookHandle) -> None:
        if handle in self._hooks:
            self._hooks.remove(handle)
            self._refresh()

    def clear(self) -> None:
        self._hooks.clear()
        self._refresh()

    @contextlib.contextmanager
    def hook(self, sites, fn, **kw):
        """``with hooks.hook(sites, fn):`` — register, then auto-remove."""
        handle = self.add(sites, fn, **kw)
        try:
            yield handle
        finally:
            self.remove(handle)

    def sites_in_use(self) -> set[str]:
        out: set[str] = set()
        for h in self._hooks:
            out |= h.sites
        return out

    # ---- lifecycle -------------------------------------------------------- #

    def attach(self) -> None:
        """Install the root step-counter hook. Idempotent."""
        if self._handles:
            return
        self._handles.append(
            self.adapter.denoiser.register_forward_pre_hook(
                self._root_pre_hook, with_kwargs=True
            )
        )
        self._refresh()

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        """Clear per-generation state. The runner calls this before sampling."""
        self.call_idx = 0
        self.step = 0
        self.timestep = float("nan")
        self.timesteps_seen = []
        for h in self._hooks:
            h.calls = 0

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *exc):
        self.detach()
        return False

    # ---- plumbing --------------------------------------------------------- #

    def _refresh(self) -> None:
        """Re-install per-site hooks to match the current set. Root hook kept."""
        if not self._handles:
            return  # attach() will call us
        for h in self._handles[1:]:
            h.remove()
        del self._handles[1:]

        wanted = self.sites_in_use()
        if not wanted:
            return
        all_sites = self.adapter.sites(self.layout)
        grouped: dict[tuple[str, str], list[SiteSpec]] = {}
        for nm in sorted(wanted):
            spec = all_sites[nm]
            grouped.setdefault((spec.module_path, spec.hook), []).append(spec)

        for (module_path, hook_type), specs in grouped.items():
            module = self.adapter.get_module(module_path)
            if hook_type == "output":
                h = module.register_forward_hook(self._make_output_hook(specs))
            elif hook_type == "input":
                h = module.register_forward_pre_hook(
                    self._make_input_hook(specs), with_kwargs=True
                )
            else:
                raise ValueError(f"unknown hook type {hook_type!r}")
            self._handles.append(h)

    def _root_pre_hook(self, module, args, kwargs):
        ts = kwargs.get("timestep", kwargs.get("timesteps"))
        if ts is None:
            for a in args[1:]:
                if torch.is_tensor(a) and a.dim() <= 1 and a.numel():
                    ts = a
                    break
        if torch.is_tensor(ts):
            ts = ts.flatten()[0].item() if ts.numel() else float("nan")
        self.timestep = float(ts) if ts is not None else float("nan")
        self.step = self.call_idx // max(1, self.adapter.calls_per_step)
        self.timesteps_seen.append(self.timestep)
        self.call_idx += 1
        return None

    def _dispatch(self, spec: SiteSpec, x: Any) -> torch.Tensor | None:
        if not torch.is_tensor(x) or x.dim() != 3:
            raise TypeError(
                f"site {spec.name!r} tapped a "
                f"{tuple(x.shape) if torch.is_tensor(x) else type(x).__name__}; "
                f"expected a [batch, tokens, dim] tensor. The adapter's site map "
                f"disagrees with the model."
            )
        self._verify_layout(spec, x)

        ctx = HookContext(
            site=spec,
            layout=spec.layout,
            step=self.step,
            timestep=self.timestep,
            call_idx=self.call_idx - 1,
            branches=self.adapter.branch_labels(x.shape[0], self.guidance_scale),
            num_steps=self.num_steps,
            adapter=self.adapter,
        )

        out: torch.Tensor | None = None
        for h in self._hooks:
            if spec.name not in h.sites or not h.steps(self.step):
                continue
            h.calls += 1
            new = h.fn(out if out is not None else x, ctx)
            if new is None:
                continue
            if new.shape != x.shape:
                raise ValueError(
                    f"hook {h.name!r} at {spec.name!r} changed shape "
                    f"{tuple(x.shape)} -> {tuple(new.shape)}"
                )
            out = new
        return out

    def _verify_layout(self, spec: SiteSpec, x: torch.Tensor) -> None:
        """Check the adapter's token bookkeeping against reality, once per site.

        Cheap insurance: a silently wrong latent grid would not crash, it would
        make every spatial claim wrong. Checked per site because different sites
        carry different token groups, and a model version can change one
        without changing the others.
        """
        if spec.name in self._checked_layout:
            return
        self._checked_layout.add(spec.name)
        expected, got = spec.layout.num_tokens, x.shape[1]
        if expected != got:
            raise RuntimeError(
                f"layout mismatch at {spec.name!r}: adapter expects {expected} "
                f"tokens ({spec.layout}), model produced {got}. Fix "
                f"{type(self.adapter).__name__}.latent_layout() or its site map."
            )

    def _make_output_hook(self, specs: Sequence[SiteSpec]):
        def hook(module, args, output):
            listed = list(output) if isinstance(output, (tuple, list)) else None
            changed = False
            result = output
            for spec in specs:
                if spec.index is None:
                    value = result
                elif listed is None or spec.index >= len(listed):
                    continue
                else:
                    value = listed[spec.index]
                new = self._dispatch(spec, value)
                if new is None:
                    continue
                changed = True
                if spec.index is None:
                    result = new
                else:
                    listed[spec.index] = new
            if not changed:
                return None
            if listed is None:
                return result
            return tuple(listed) if isinstance(output, tuple) else listed

        return hook

    def _make_input_hook(self, specs: Sequence[SiteSpec]):
        def hook(module, args, kwargs):
            args = list(args)
            changed = False
            for spec in specs:
                key = spec.index
                if isinstance(key, str):
                    if key not in kwargs:
                        continue
                    value = kwargs[key]
                elif isinstance(key, int) and key < len(args):
                    value = args[key]
                else:
                    continue
                new = self._dispatch(spec, value)
                if new is None:
                    continue
                changed = True
                if isinstance(key, str):
                    kwargs[key] = new
                else:
                    args[key] = new
            return (tuple(args), kwargs) if changed else None

        return hook
