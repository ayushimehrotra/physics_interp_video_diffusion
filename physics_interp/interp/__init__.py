"""Interpretability substrate.

Currently just the hook layer: denoising-step and CFG-branch aware read/write
taps addressed by canonical site name. Analysis and steering primitives are
built on top of this and live outside it, so that what gets recorded and what
gets edited stays a decision for the experiment rather than for the framework.
"""

from .hooks import HookContext, HookHandle, HookManager, StepFilter

__all__ = ["HookContext", "HookHandle", "HookManager", "StepFilter"]
