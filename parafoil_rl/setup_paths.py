# parafoil_rl/setup_paths.py
# =============================================================================
# Import this at the top of every script before any parafoil imports.
# Handles: Windows DLL registration, sys.path setup for all submodules.
#
# Works correctly in both:
#   - Normal script execution
#   - SubprocVecEnv worker processes (spawned by multiprocessing)
# =============================================================================

import os
import sys
from pathlib import Path

# ── 1. Windows DLL registration (must be before any .pyd import) ──────────
_dll_candidates = [
    r"C:\Strawberry\c\bin",
    r"C:\Strawberry\c\x86_64-w64-mingw32\bin",
    r"C:\Strawberry\perl\bin",
    r"C:\msys64\mingw64\bin",
    r"C:\msys64\ucrt64\bin",
    r"C:\mingw64\bin",
    r"C:\Windows\System32",
]
for _entry in os.environ.get("PATH", "").split(os.pathsep):
    if any(k in _entry.lower() for k in ["strawberry", "mingw", "msys"]):
        if os.path.isdir(_entry):
            _dll_candidates.append(_entry)

for _d in _dll_candidates:
    if os.path.isdir(_d):
        try:
            os.add_dll_directory(_d)
        except (OSError, AttributeError):
            pass

# ── 2. Locate parafoil_rl/ root reliably ─────────────────────────────────
# setup_paths.py lives directly in parafoil_rl/ so its parent IS parafoil_rl/
PARAFOIL_RL_ROOT = Path(__file__).resolve().parent
PYTHON_ENV_DIR   = PARAFOIL_RL_ROOT / "python_env"

# The simulation root (nineDOF_Parafoil-Payload_Dynamic_Simulation/) is one
# level above parafoil_rl/
SIM_ROOT = PARAFOIL_RL_ROOT.parent

# ── 3. Add all needed dirs to sys.path ───────────────────────────────────
# Priority order: python_env first (has .pyd and bridge modules),
# then sim root (has nineDOF_*.py files), then parafoil_rl root itself.
for _p in [str(PYTHON_ENV_DIR), str(SIM_ROOT), str(PARAFOIL_RL_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)