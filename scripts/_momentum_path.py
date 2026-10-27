"""Make ``physics_interp`` and every ``scripts/<category>/`` folder importable.

Scripts import shared code from each other (``from probe_energy import ...``) across
category folders. Each script puts ``scripts/`` on ``sys.path`` and then imports this.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DIRS = [_ROOT] + sorted(p for p in (_ROOT / "scripts").iterdir() if p.is_dir() and not p.name.startswith("_"))
for _d in reversed(_DIRS):
    if str(_d) not in sys.path:
        sys.path.insert(1, str(_d))
