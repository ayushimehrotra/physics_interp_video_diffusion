"""Momentum-conservation protocols (register §2–§60).

Everything the protocol scripts under ``scripts/`` share lives here, so a number
quoted in the register is produced by exactly one implementation:

* :mod:`.physics`  — the two-body collision algebra. The outcome space, the
  valid-collision line, the {normal, tangential, drift} basis, the two edits
  that build the datasets (the split-changing momentum violation and the
  restitution rescale), and the label accessors that avoid the ``restitution``
  label trap.
* :mod:`.stats`    — raw AUC, label-permutation nulls, grouped folds by seed,
  the ridge readout with its penalty chosen inside the training split, PCA
  helpers. No paired accuracy anywhere (§55).
* :mod:`.captures` — the on-disk capture format (one ``.npz`` per clip, pooled
  activations per site) and the clip-set index that groups clips into seeds,
  twins and ladder arms.
* :mod:`.synthetic` — a synthetic encoder + dataset so every protocol script
  can be exercised end to end without a GPU or a renderer.
"""

from . import physics, stats, captures  # noqa: F401
