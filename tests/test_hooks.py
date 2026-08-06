"""Tests for the hook substrate, against a synthetic model.

These run in a second on CPU with no weights, and they cover the parts that are
easy to get subtly wrong and expensive to discover later: step indexing, CFG
branch labelling, write-back through tuple outputs and keyword inputs, and the
token-index <-> (frame, y, x) mapping.

    .venv/bin/python -m pytest tests/ -q
    .venv/bin/python tests/test_hooks.py      # no pytest needed
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physics_interp.interp.hooks import HookManager, StepFilter  # noqa: E402
from physics_interp.models.base import ModelAdapter, SiteSpec, TokenLayout  # noqa: E402

DIM = 8
TEXT_LEN = 5
FRAMES, H, W = 2, 3, 4
N_VIDEO = FRAMES * H * W


class ToyBlock(nn.Module):
    """Mimics CogVideoXBlock: two streams in, two streams out, as a tuple."""

    def forward(self, hidden_states, encoder_hidden_states, temb=None):
        return hidden_states + 1.0, encoder_hidden_states + 1.0


class ToyDenoiser(nn.Module):
    def __init__(self, n_blocks: int = 3):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(ToyBlock() for _ in range(n_blocks))

    def forward(self, hidden_states, timestep=None, encoder_hidden_states=None):
        h, e = hidden_states, encoder_hidden_states
        for block in self.transformer_blocks:
            h, e = block(hidden_states=h, encoder_hidden_states=e)
        return h


class ToyAdapter(ModelAdapter):
    key = "toy"
    calls_per_step = 1

    def __init__(self):
        self._denoiser = ToyDenoiser()
        super().__init__(pipe=None, repo="toy", device=torch.device("cpu"), dtype=torch.float32)

    @classmethod
    def load(cls, cfg, device):
        return cls()

    @property
    def denoiser(self):
        return self._denoiser

    @property
    def blocks(self):
        return self._denoiser.transformer_blocks

    @property
    def hidden_dim(self):
        return DIM

    def latent_layout(self, num_frames, height, width):
        return TokenLayout(kind="video", frames=FRAMES, height=H, width=W, text_len=0)

    def build_sites(self, layout):
        text = TokenLayout("text", FRAMES, H, W, TEXT_LEN)
        sites = {}
        for i in range(len(self.blocks)):
            p = f"transformer_blocks.{i}"
            sites[f"block.{i}.resid_pre"] = SiteSpec(
                f"block.{i}.resid_pre", p, "input", "hidden_states", layout, True, i
            )
            sites[f"block.{i}.resid_post"] = SiteSpec(
                f"block.{i}.resid_post", p, "output", 0, layout, True, i
            )
            sites[f"block.{i}.text_resid_post"] = SiteSpec(
                f"block.{i}.text_resid_post", p, "output", 1, text, True, i
            )
        return sites

    def generate(self, **kw):
        raise NotImplementedError


def make(batch: int = 2, guidance: float = 6.0):
    adapter = ToyAdapter()
    layout = adapter.latent_layout(0, 0, 0)
    hooks = HookManager(adapter, layout, guidance_scale=guidance, num_steps=4)
    hooks.attach()
    h = torch.zeros(batch, N_VIDEO, DIM)
    e = torch.zeros(batch, TEXT_LEN, DIM)
    return adapter, hooks, h, e


def run_steps(adapter, h, e, n: int = 4):
    for step in range(n):
        adapter.denoiser(hidden_states=h, timestep=torch.tensor(1000 - step * 100),
                         encoder_hidden_states=e)


# --------------------------------------------------------------------------- #


def test_step_filter():
    assert [s for s in range(6) if StepFilter("all")(s)] == [0, 1, 2, 3, 4, 5]
    assert [s for s in range(6) if StepFilter([1, 3])(s)] == [1, 3]
    assert [s for s in range(6) if StepFilter({"every": 2})(s)] == [0, 2, 4]
    assert [s for s in range(6) if StepFilter({"start": 1, "end": 5, "every": 2})(s)] == [1, 3]
    assert [s for s in range(6) if StepFilter(2)(s)] == [2]


def test_step_and_timestep_tracking():
    adapter, hooks, h, e = make()
    seen = []
    hooks.add("block.1.resid_post", lambda x, ctx: seen.append((ctx.step, ctx.timestep)))
    run_steps(adapter, h, e, n=4)
    assert [s for s, _ in seen] == [0, 1, 2, 3], seen
    assert [t for _, t in seen] == [1000, 900, 800, 700], seen
    assert hooks.timesteps_seen == [1000, 900, 800, 700]


def test_step_filtering_applies():
    adapter, hooks, h, e = make()
    seen = []
    hooks.add("block.0.resid_post", lambda x, ctx: seen.append(ctx.step),
              steps={"every": 2})
    run_steps(adapter, h, e, n=4)
    assert seen == [0, 2]


def test_cfg_branch_labels():
    adapter, hooks, h, e = make(batch=2, guidance=6.0)
    got = {}
    hooks.add("block.0.resid_post",
              lambda x, ctx: got.update(branches=ctx.branches,
                                        cond=ctx.rows("cond"),
                                        uncond=ctx.rows("uncond")))
    run_steps(adapter, h, e, n=1)
    assert got["branches"] == ["uncond", "cond"]
    assert got["cond"] == [1] and got["uncond"] == [0]

    # guidance <= 1 means no CFG doubling: every row is the conditional branch.
    adapter, hooks, h, e = make(batch=1, guidance=1.0)
    got.clear()
    hooks.add("block.0.resid_post", lambda x, ctx: got.update(branches=ctx.branches))
    run_steps(adapter, h, e, n=1)
    assert got["branches"] == ["cond"]


def test_observe_does_not_modify():
    adapter, hooks, h, e = make()
    hooks.add("block.*.resid_post", lambda x, ctx: None)
    out = adapter.denoiser(hidden_states=h, timestep=torch.tensor(1.0),
                           encoder_hidden_states=e)
    assert torch.allclose(out, torch.full_like(out, 3.0))  # 3 blocks, +1 each


def test_write_through_tuple_output():
    adapter, hooks, h, e = make()
    hooks.add("block.0.resid_post", lambda x, ctx: x * 0 + 10.0)
    out = adapter.denoiser(hidden_states=h, timestep=torch.tensor(1.0),
                           encoder_hidden_states=e)
    # block0 output forced to 10, then blocks 1 and 2 each add 1
    assert torch.allclose(out, torch.full_like(out, 12.0)), out.unique()


def test_write_through_kwarg_input():
    adapter, hooks, h, e = make()
    hooks.add("block.2.resid_pre", lambda x, ctx: x * 0 + 100.0)
    out = adapter.denoiser(hidden_states=h, timestep=torch.tensor(1.0),
                           encoder_hidden_states=e)
    assert torch.allclose(out, torch.full_like(out, 101.0)), out.unique()


def test_edit_only_one_cfg_branch():
    adapter, hooks, h, e = make(batch=2)

    def bump_cond(x, ctx):
        y = x.clone()
        y[ctx.rows("cond")] += 50.0
        return y

    hooks.add("block.0.resid_post", bump_cond)
    out = adapter.denoiser(hidden_states=h, timestep=torch.tensor(1.0),
                           encoder_hidden_states=e)
    assert torch.allclose(out[0], torch.full_like(out[0], 3.0))    # uncond untouched
    assert torch.allclose(out[1], torch.full_like(out[1], 53.0))   # cond bumped


def test_hooks_compose_in_order():
    adapter, hooks, h, e = make()
    hooks.add("block.0.resid_post", lambda x, ctx: x + 1.0, name="first")
    hooks.add("block.0.resid_post", lambda x, ctx: x * 2.0, name="second")
    out = adapter.denoiser(hidden_states=h, timestep=torch.tensor(1.0),
                           encoder_hidden_states=e)
    # block0 -> 1; +1 -> 2; *2 -> 4; then two more blocks add 1 each
    assert torch.allclose(out, torch.full_like(out, 6.0)), out.unique()


def test_context_manager_removes_hook():
    adapter, hooks, h, e = make()
    calls = []
    with hooks.hook("block.0.resid_post", lambda x, ctx: calls.append(1)):
        run_steps(adapter, h, e, n=1)
    assert len(calls) == 1
    run_steps(adapter, h, e, n=1)
    assert len(calls) == 1  # not called again
    assert hooks.sites_in_use() == set()


def test_text_site_has_text_layout():
    adapter, hooks, h, e = make()
    shapes = []
    hooks.add("block.0.text_resid_post",
              lambda x, ctx: shapes.append((tuple(x.shape), ctx.layout.kind)))
    run_steps(adapter, h, e, n=1)
    assert shapes == [((2, TEXT_LEN, DIM), "text")]


def test_layout_mismatch_raises():
    adapter, hooks, h, e = make()
    hooks.add("block.0.resid_post", lambda x, ctx: None)
    bad = torch.zeros(2, N_VIDEO + 1, DIM)  # wrong token count
    try:
        adapter.denoiser(hidden_states=bad, timestep=torch.tensor(1.0),
                         encoder_hidden_states=e)
    except RuntimeError as exc:
        assert "layout mismatch" in str(exc)
    else:
        raise AssertionError("expected a layout mismatch error")


def test_grid_indexing_roundtrip():
    layout = TokenLayout("video", FRAMES, H, W, 0)
    assert layout.num_video_tokens == N_VIDEO
    # a single cell maps to the frame-major flat index
    idx = layout.grid_indices(frames=(1, 2), y=(2, 3), x=(3, 4))
    assert idx.tolist() == [1 * (H * W) + 2 * W + 3]
    # as_grid inverts the flattening
    x = torch.arange(N_VIDEO, dtype=torch.float32).view(1, N_VIDEO, 1)
    grid = layout.as_grid(x)
    assert grid.shape == (1, FRAMES, H, W, 1)
    assert grid[0, 1, 2, 3, 0].item() == 1 * (H * W) + 2 * W + 3


def test_grid_indexing_with_text_offset():
    layout = TokenLayout("text+video", FRAMES, H, W, TEXT_LEN)
    assert layout.num_tokens == TEXT_LEN + N_VIDEO
    assert layout.text_slice() == slice(0, TEXT_LEN)
    assert layout.video_slice() == slice(TEXT_LEN, TEXT_LEN + N_VIDEO)
    # video indices are offset past the text block
    idx = layout.grid_indices(frames=(0, 1), y=(0, 1), x=(0, 1))
    assert idx.tolist() == [TEXT_LEN]


def test_unknown_site_pattern_raises():
    adapter, hooks, h, e = make()
    try:
        hooks.add("block.99.nonexistent", lambda x, ctx: None)
    except ValueError as exc:
        assert "matched nothing" in str(exc)
    else:
        raise AssertionError("expected a ValueError for an unmatched pattern")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
