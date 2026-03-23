# python_env/atmosphere_bridge.py
# =============================================================================
# Bridges the Python dynamicAtmosphere (Dryden, layered wind) to the C++ env.
#
# The C++ plant's AtmosphereParameters only stores static wind values
# (VXWIND, VYWIND, VZWIND, DEN). This module drives those values by calling
# the Python dynamicAtmosphere.update() each RL step, then pushing the results
# into the C++ env via set_wind().
#
# Usage (see ParafoilGymEnv - atmosphere_mode arg):
#   The gym wrapper instantiates one of these per environment and calls
#   bridge.step(t, altitude, airspeed) after each physics step.
# =============================================================================

# ── Path setup ───────────────────────────────────────────────────────────
import sys, os
from pathlib import Path
# atmosphere_bridge.py lives in python_env/ -> parent = parafoil_rl/
_rl_root = Path(__file__).resolve().parent.parent
if str(_rl_root) not in sys.path:
    sys.path.insert(0, str(_rl_root))
import setup_paths  # registers DLLs + adds python_env/ and sim root
# ─────────────────────────────────────────────────────────────────────────

import numpy as np
from typing import Optional


try:
    from nineDOF_Atmosphere import dynamicAtmosphere, TurbulenceMode, staticAtmosphere
    _HAS_ATMOSPHERE = True
except ImportError:
    _HAS_ATMOSPHERE = False


# =============================================================================
# Atmosphere mode constants  (used by gym wrapper and training scripts)
# =============================================================================
ATMO_STATIC  = "static"    # No wind, no turbulence
ATMO_DRYDEN  = "dryden"    # Full MIL-F-8785C Dryden turbulence
ATMO_SIMPLE  = "simple"    # Sinusoidal gust model
ATMO_WIND    = "wind"      # Layered mean wind only, no turbulence


class AtmosphereBridge:
    """
    Owns a dynamicAtmosphere and drives the C++ env wind state each step.

    Call reset(seed) at episode start to re-randomize the wind profile.
    Call step(cpp_env, t, altitude, airspeed) after each physics sub-step.
    """

    def __init__(
        self,
        mode: str = ATMO_STATIC,
        intensity: str = "moderate",   # "light", "moderate", "severe"
    ):
        self.mode      = mode
        self.intensity = intensity
        self._atm      = None   # created fresh on each reset()

        if mode != ATMO_STATIC and not _HAS_ATMOSPHERE:
            raise ImportError(
                "nineDOF_Atmosphere.py not found. "
                "Make sure it is in the project root folder."
            )

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None):
        """Re-create atmosphere with a new random seed (new wind profile)."""
        if self.mode == ATMO_STATIC:
            self._atm = None
            return

        turbulence_map = {
            ATMO_DRYDEN: TurbulenceMode.DRYDEN,
            ATMO_SIMPLE: TurbulenceMode.SIMPLE,
            ATMO_WIND:   TurbulenceMode.NONE,
        }
        turb_mode = turbulence_map.get(self.mode, TurbulenceMode.NONE)

        self._atm = dynamicAtmosphere(
            turbulence_mode      = turb_mode,
            turbulence_intensity = self.intensity,
            seed                 = seed,
        )

    # ------------------------------------------------------------------
    def step(self, cpp_env, t: float, altitude: float, airspeed: float):
        """
        Update Python atmosphere and push wind state into C++ env.
        Call once per RL step (not per physics sub-step) for efficiency.
        """
        if self._atm is None:
            return   # Static atmosphere - C++ env already has 0 wind

        self._atm.update(t, altitude, airspeed)
        cpp_env.set_wind(
            self._atm.VXWIND,
            self._atm.VYWIND,
            self._atm.VZWIND,
        )

    # ------------------------------------------------------------------
    @property
    def is_static(self) -> bool:
        return self.mode == ATMO_STATIC

    def get_current_wind(self):
        """Return (vx, vy, vz) - useful for logging."""
        if self._atm is None:
            return (0.0, 0.0, 0.0)
        return (self._atm.VXWIND, self._atm.VYWIND, self._atm.VZWIND)

    def get_turbulence_info(self) -> dict:
        if self._atm is None:
            return {"mode": "static"}
        try:
            return self._atm.get_turbulence_info()
        except Exception:
            return {"mode": self.mode}