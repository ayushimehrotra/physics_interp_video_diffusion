"""Tests for the hook substrate and the inversion path, against a toy model.

These run in a second on CPU with no weights, and they cover the parts that are
easy to get subtly wrong and expensive to discover later: step indexing, CFG
branch labelling, write-back through tuple outputs and keyword inputs, the
token-index <-> (frame, y, x) mapping, and the schedule algebra that inversion
rests on.

The inversion tests use an *oracle* denoiser — one that returns the exact noise
a latent was built from. Against an oracle the recovered trajectory has a
closed form, so the test can assert the loop reproduces it to floating-point
precision. That isolates the integrator and the parameterisation algebra from
any question about whether a real checkpoint is well-behaved.

    .venv/bin/python -m pytest tests/ -q
    .venv/bin/python tests/test_hooks.py      # no pytest needed
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physics_interp.config import InversionConfig  # noqa: E402
from physics_interp.interp import fit_probe  # noqa: E402
from physics_interp.interp.hooks import HookManager, StepFilter  # noqa: E402
from physics_interp.models.base import ModelAdapter, SiteSpec, TokenLayout  # noqa: E402
from physics_interp.runner import (  # noqa: E402
    ActivationRecorder,
    VideoDiffusionRunner,
    build_pool_weights,
)
from physics_interp.video import masks_to_latent_grid, pixel_frame_to_latent  # noqa: E402

DIM = 8
TEXT_LEN = 5
FRAMES, H, W = 2, 3, 4
N_VIDEO = FRAMES * H * W
N_TIMESTEPS = 1000


class ToyScheduler:
    """A DDPM-style schedule with a_bar[0] == 1 exactly.

    The exact 1.0 at the clean end matters: it makes the first inversion step's
    ``(x0, eps)`` split the identity, so an oracle trajectory has a closed form
    the test can check against rather than approximate.
    """

    def __init__(self, prediction_type: str = "epsilon"):
        betas = torch.linspace(0.0, 0.02, N_TIMESTEPS, dtype=torch.float64)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.config = SimpleNamespace(prediction_type=prediction_type)
        self.timesteps = torch.arange(N_TIMESTEPS - 1, -1, -1)

    def set_timesteps(self, num_steps: int, device=None):
        stride = N_TIMESTEPS // num_steps
        self.timesteps = torch.arange(0, N_TIMESTEPS, stride)[:num_steps].flip(0)


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

    def __init__(self, prediction_type: str = "epsilon"):
        self._denoiser = ToyDenoiser()
        self._scheduler = ToyScheduler(prediction_type)
        #: When set, denoise() returns this instead of the block stack's output,
        #: making the adapter an oracle for a latent built from this noise. The
        #: blocks still run, so hooks fire exactly as they would for real.
        self.oracle_eps: torch.Tensor | None = None
        self.latent: torch.Tensor | None = None
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

    # ---- inversion primitives (latent is [1, F, C, H, W], as for CogVideoX) -- #

    @property
    def scheduler(self):
        return self._scheduler

    def encode_video(self, video):
        return self.latent

    def decode_latents(self, latents):
        return latents

    def embed_prompt(self, prompt):
        return torch.zeros(1, TEXT_LEN, DIM)

    def denoise(self, latents, timestep, prompt_embeds):
        b, f, c, hh, ww = latents.shape
        tokens = latents.permute(0, 1, 3, 4, 2).reshape(b, f * hh * ww, c)
        out = self._denoiser(
            hidden_states=tokens, timestep=timestep, encoder_hidden_states=prompt_embeds
        )
        if self.oracle_eps is not None:
            return self.oracle_eps
        return out.reshape(b, f, hh, ww, c).permute(0, 1, 4, 2, 3)


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


# --------------------------------------------------------------------------- #
# schedule algebra and inversion
# --------------------------------------------------------------------------- #


def test_split_prediction_roundtrips_every_parameterisation():
    """x0/eps recovered from a model output must rebuild the sample exactly.

    Each parameterisation encodes the same information differently; getting one
    of the sign conventions wrong still produces a plausible-looking latent, so
    the check is algebraic rather than visual.
    """
    torch.manual_seed(0)
    x0 = torch.randn(4)
    eps = torch.randn(4)
    alpha, sigma = 0.6, 0.8
    sample = alpha * x0 + sigma * eps

    for kind, model_out in [
        ("epsilon", eps),
        ("v_prediction", alpha * eps - sigma * x0),
        ("sample", x0),
        ("flow_prediction", eps - x0),
    ]:
        adapter = ToyAdapter(prediction_type=kind)
        if kind == "flow_prediction":
            # Rectified flow uses x_t = (1-s) x0 + s eps, so alpha + sigma == 1.
            a, s = 0.4, 0.6
            smp = a * x0 + s * eps
            got_x0, got_eps = adapter.split_prediction(model_out, smp, a, s)
        else:
            got_x0, got_eps = adapter.split_prediction(model_out, sample, alpha, sigma)
        assert torch.allclose(got_x0, x0, atol=1e-5), (kind, got_x0, x0)
        assert torch.allclose(got_eps, eps, atol=1e-5), (kind, got_eps, eps)


def test_alpha_sigma_is_a_unit_decomposition():
    adapter = ToyAdapter()
    for t in (0, 100, 999):
        alpha, sigma = adapter.alpha_sigma(t)
        assert abs(alpha**2 + sigma**2 - 1.0) < 1e-6, (t, alpha, sigma)
    assert adapter.alpha_sigma(0)[0] == 1.0        # clean end is exact


def make_runner(num_frames=FRAMES, prediction_type="epsilon"):
    adapter = ToyAdapter(prediction_type)
    from physics_interp.config import GenerationConfig

    runner = VideoDiffusionRunner(adapter, GenerationConfig(num_frames=num_frames))
    return adapter, runner


def test_inversion_recovers_the_oracle_trajectory():
    """Against an oracle, every visited latent must equal alpha*x0 + sigma*eps.

    This is the whole correctness claim of the inversion loop: that it lands on
    the noise levels the scheduler defines, carrying the right x0 and eps. If
    the timestep the model is conditioned on were off by one step — the easiest
    mistake to make here — the reconstruction would still look fine while every
    recorded activation came from the wrong point on the trajectory.
    """
    torch.manual_seed(0)
    adapter, runner = make_runner()

    x0 = torch.randn(1, FRAMES, DIM, H, W)
    eps = torch.randn(1, FRAMES, DIM, H, W)
    adapter.latent, adapter.oracle_eps = x0, eps

    cfg = InversionConfig(num_steps=10, num_captures=3)
    result = runner.invert(torch.zeros(FRAMES, 4, 4, 3), inversion=cfg)

    final_t = result.timesteps[-1]
    alpha, sigma = adapter.alpha_sigma(final_t)
    expected = alpha * x0 + sigma * eps
    assert torch.allclose(result.noise, expected, atol=1e-5), (
        (result.noise - expected).abs().max()
    )
    # Ascending noise: the ladder must run clean -> noise, not the other way.
    assert result.timesteps[0] < result.timesteps[-1]
    assert len(result.timesteps) == 10


def test_heun_matches_euler_on_a_linear_oracle():
    """A second-order correction must not move an already-exact trajectory."""
    torch.manual_seed(0)
    adapter, runner = make_runner()
    adapter.latent = torch.randn(1, FRAMES, DIM, H, W)
    adapter.oracle_eps = torch.randn(1, FRAMES, DIM, H, W)

    euler = runner.invert(
        torch.zeros(1), inversion=InversionConfig(num_steps=8, solver="euler")
    )
    heun = runner.invert(
        torch.zeros(1), inversion=InversionConfig(num_steps=8, solver="heun")
    )
    assert torch.allclose(euler.noise, heun.noise, atol=1e-5)


def test_capture_steps_are_spread_and_recorded_once():
    """Recording must happen at the requested steps, exactly once each.

    The Heun correction calls the denoiser a second time within the same step;
    if that call were not excluded, it would silently overwrite the step's
    activations with states from a point the trajectory only tentatively
    visited.
    """
    torch.manual_seed(0)
    adapter, runner = make_runner()
    adapter.latent = torch.randn(1, FRAMES, DIM, H, W)

    layout = runner.layout
    regions = {"all": np.ones((layout.frames, layout.height, layout.width), np.float32)}
    weights, names = build_pool_weights(layout, regions, {"all": (0, layout.frames)})

    for solver in ("euler", "heun"):
        recorder = ActivationRecorder(weights, names)
        cfg = InversionConfig(num_steps=9, num_captures=3, solver=solver)
        result = runner.invert(
            torch.zeros(1), inversion=cfg, recorder=recorder,
            sites="block.*.resid_post",
        )
        assert result.captured_steps == [0, 4, 8], result.captured_steps
        acts = result.activations
        assert set(acts) == {f"block.{i}.resid_post" for i in range(3)}
        for site, arr in acts.items():
            assert arr.shape == (3, 1, DIM), (site, arr.shape)
        # Each toy block adds exactly 1 to the stream, so consecutive blocks
        # must differ by 1 at every captured step. That pins the site-to-block
        # mapping and the per-step bookkeeping together: an off-by-one in
        # either would break this while leaving shapes intact.
        for i in range(2):
            delta = (
                acts[f"block.{i + 1}.resid_post"].astype(np.float64)
                - acts[f"block.{i}.resid_post"].astype(np.float64)
            )
            assert np.allclose(delta, 1.0, atol=1e-2), (solver, i, delta)
        assert result.noise_levels == [0.0, 0.5, 1.0]


def test_recorder_never_perturbs_the_trajectory():
    torch.manual_seed(0)
    adapter, runner = make_runner()
    adapter.latent = torch.randn(1, FRAMES, DIM, H, W)
    adapter.oracle_eps = torch.randn(1, FRAMES, DIM, H, W)
    cfg = InversionConfig(num_steps=6, num_captures=6)

    clean = runner.invert(torch.zeros(1), inversion=cfg)

    layout = runner.layout
    regions = {"all": np.ones((layout.frames, layout.height, layout.width), np.float32)}
    weights, names = build_pool_weights(layout, regions, {"all": (0, layout.frames)})
    watched = runner.invert(
        torch.zeros(1), inversion=cfg,
        recorder=ActivationRecorder(weights, names), sites="block.*.resid_post",
    )
    assert torch.allclose(clean.noise, watched.noise)


def test_step_override_restored_after_inversion():
    """A leaked override would silently mislabel every later generation step."""
    torch.manual_seed(0)
    adapter, runner = make_runner()
    adapter.latent = torch.randn(1, FRAMES, DIM, H, W)
    runner.invert(torch.zeros(1), inversion=InversionConfig(num_steps=4))
    assert runner.hooks.step_override is None


# --------------------------------------------------------------------------- #
# pooling and mask alignment
# --------------------------------------------------------------------------- #


def test_pool_weights_are_normalised_and_disjoint_by_phase():
    layout = TokenLayout("video", FRAMES, H, W, 0)
    occ = np.zeros((FRAMES, H, W), np.float32)
    occ[:, 0, 0] = 1.0
    weights, names = build_pool_weights(
        layout, {"A": occ}, {"pre": (0, 1), "post": (1, FRAMES)}
    )
    assert names == ["A.pre", "A.post"]
    assert np.allclose(weights.sum(1).numpy(), 1.0)
    # A phase split must not let one phase's tokens bleed into the other.
    assert float((weights[0] * weights[1]).sum()) == 0.0


def test_empty_pool_is_zero_not_nan():
    """An object outside the frame must yield a droppable row, not NaNs."""
    layout = TokenLayout("video", FRAMES, H, W, 0)
    empty = np.zeros((FRAMES, H, W), np.float32)
    weights, _ = build_pool_weights(layout, {"gone": empty}, {"all": (0, FRAMES)})
    assert np.isfinite(weights.numpy()).all()
    assert float(weights.abs().sum()) == 0.0


def test_pixel_frame_to_latent_matches_the_vae_grouping():
    # First frame alone, then groups of four: the causal VAE's actual layout.
    assert [pixel_frame_to_latent(f) for f in range(9)] == [0, 1, 1, 1, 1, 2, 2, 2, 2]
    # The dataset's contact frame is the last pixel frame of its latent cell,
    # so pre- and post-collision phases split cleanly with no straddling cell.
    assert pixel_frame_to_latent(20) == 5
    assert pixel_frame_to_latent(21) == 6
    assert pixel_frame_to_latent(48) == 12


def test_masks_to_latent_grid_measures_coverage():
    layout = TokenLayout("video", 13, 30, 45, 0)
    masks = np.zeros((49, 480, 720), np.uint8)
    masks[:, :16, :16] = 1          # exactly one latent cell, fully covered
    occ = masks_to_latent_grid(masks, layout, ids=(1,))
    assert occ.shape == (1, 13, 30, 45)
    assert occ[0, :, 0, 0].min() == 1.0
    assert occ[0, :, 0, 1].max() == 0.0
    # Half a cell reads as half occupancy, which is what makes soft weights
    # meaningful for objects smaller than a 16x16 block.
    masks[:] = 0
    masks[:, :8, :16] = 1
    occ = masks_to_latent_grid(masks, layout, ids=(1,))
    assert abs(occ[0, 0, 0, 0] - 0.5) < 1e-6


def test_probe_separates_signal_from_shuffled_control():
    rng = np.random.default_rng(0)
    n, d = 80, 64
    Y = rng.normal(size=(n, 2))
    X = Y @ rng.normal(size=(2, d)) + rng.normal(size=(n, d)) * 0.5
    good = fit_probe(X, Y, split_seed=0)
    assert good["r2"] > 0.8, good
    assert good["r2_shuffled"] < 0.3, good

    noise_only = fit_probe(rng.normal(size=(n, d)), Y, split_seed=0)
    assert noise_only["r2"] < 0.3, noise_only


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
