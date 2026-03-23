"""
python_env/parafoil_gym_env.py
===============================
Gymnasium-compatible wrapper around the C++ parafoil RL environment.

Atmosphere modes (pass as atmosphere_mode=):
    "static"  - No wind, no turbulence (default, fastest training)
    "dryden"  - MIL-F-8785C Dryden turbulence + layered wind
    "simple"  - Sinusoidal gust model + layered wind
    "wind"    - Layered mean wind only, no turbulence

Usage:
    # Static atmosphere (original behaviour)
    env = ParafoilGymEnv(target=(0.0, 0.0))

    # Dryden turbulence
    env = ParafoilGymEnv(target=(0.0, 0.0), atmosphere_mode="dryden",
                         turbulence_intensity="moderate")

    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(action)
"""

import sys
# ── Path setup (must be first) ────────────────────────────────────────────
import sys, os
from pathlib import Path
# parafoil_gym_env.py lives in python_env/ -> parent = parafoil_rl/
_rl_root = Path(__file__).resolve().parent.parent
if str(_rl_root) not in sys.path:
    sys.path.insert(0, str(_rl_root))
import setup_paths  # registers DLLs + adds python_env/ and sim root
# ─────────────────────────────────────────────────────────────────────────

try:
    from load_parafoil_cpp import parafoil_cpp
except ImportError:
    try:
        import parafoil_cpp
    except ImportError as e:
        raise ImportError(
            "Could not import parafoil_cpp.pyd.\n"
            "1. Make sure the module is built (see README.md)\n"
            "2. Run diagnose_dll.py to find missing DLLs\n"
            f"Original error: {e}"
        ) from e

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict

from atmosphere_bridge import (
    AtmosphereBridge,
    ATMO_STATIC, ATMO_DRYDEN, ATMO_SIMPLE, ATMO_WIND,
)


# Physical action limits
MAX_BRAKE = 0.94
MIN_BRAKE = 0.0


class ParafoilGymEnv(gym.Env):
    """
    Gymnasium wrapper for the 9DOF parafoil C++ environment.

    atmosphere_mode options:
        "static"  - No wind (fastest, good for initial training)
        "dryden"  - Dryden turbulence + layered wind (most realistic)
        "simple"  - Sinusoidal gusts + layered wind
        "wind"    - Layered mean wind, no turbulence
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        target: Tuple[float, float] = (0.0, 0.0),
        dt_physics: float = 0.01,
        dt_action:  float = 0.1,
        max_episode_time: float = 1200.0,
        domain_random: bool = True,
        atmosphere_mode: str = ATMO_STATIC,
        turbulence_intensity: str = "moderate",
        wind_x: float = 0.0,
        wind_y: float = 0.0,
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            target:               (x, y) landing target in inertial frame (m).
            dt_physics:           Physics integration timestep (s).
            dt_action:            Time between RL decisions (s).
            max_episode_time:     Hard episode timeout (s).
            domain_random:        Randomize initial conditions on reset().
            atmosphere_mode:      "static" | "dryden" | "simple" | "wind"
            turbulence_intensity: "light" | "moderate" | "severe"
            wind_x:               Constant wind X (m/s). Only used when atmosphere_mode="static".
            wind_y:               Constant wind Y (m/s). Only used when atmosphere_mode="static".
            render_mode:          Unused.
        """
        super().__init__()

        self.target               = target
        self.domain_random        = domain_random
        self.render_mode          = render_mode
        self.atmosphere_mode      = atmosphere_mode
        self.turbulence_intensity = turbulence_intensity
        self._dt_action           = dt_action

        # C++ environment
        self._params = parafoil_cpp.SystemParameters()
        self._env = parafoil_cpp.ParafoilEnv(
            self._params,
            target[0], target[1],
            dt_physics, dt_action,
        )
        self._env.set_max_episode_time(max_episode_time)

        if not domain_random:
            cfg = parafoil_cpp.DomainRandomConfig()
            cfg.pos_x_range   = 0.0
            cfg.pos_y_range   = 0.0
            cfg.alt_min       = 500.0
            cfg.alt_max       = 500.0
            cfg.heading_range = 0.0
            cfg.u_noise       = 0.0
            cfg.w_noise       = 0.0
            cfg.theta_noise   = 0.0
            cfg.phi_noise     = 0.0
            self._env.set_domain_random_config(cfg)

        # Atmosphere bridge (handles Dryden / static / wind)
        self._atm_bridge = AtmosphereBridge(
            mode      = atmosphere_mode,
            intensity = turbulence_intensity,
        )

        # Apply static wind immediately if requested
        if atmosphere_mode == ATMO_STATIC and (wind_x != 0.0 or wind_y != 0.0):
            self._env.set_wind(wind_x, wind_y)
        self._static_wind_x = wind_x
        self._static_wind_y = wind_y

        # Spaces
        self.action_space = spaces.Box(
            low=np.full(2, -1.0, dtype=np.float32),
            high=np.full(2,  1.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.full(parafoil_cpp.OBS_SIZE, -1.0, dtype=np.float32),
            high=np.full(parafoil_cpp.OBS_SIZE,  1.0, dtype=np.float32),
            dtype=np.float32,
        )

    # --------------------------------------------------------------------------
    # Gymnasium interface
    # --------------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        cpp_seed = seed if seed is not None else -1

        # Re-seed atmosphere with a new wind profile each episode
        # Use a deterministic but different seed per episode
        atm_seed = (cpp_seed * 31337 + 13) % (2**31) if cpp_seed >= 0 else None
        self._atm_bridge.reset(seed=atm_seed)

        obs  = self._env.reset(seed=cpp_seed)
        obs  = np.array(obs, dtype=np.float32)
        info = self._env.get_info()

        # Re-apply static wind (reset() clears it inside the C++ env)
        if self.atmosphere_mode == ATMO_STATIC:
            if self._static_wind_x != 0.0 or self._static_wind_y != 0.0:
                self._env.set_wind(self._static_wind_x, self._static_wind_y)

        # Inject initial atmosphere state for dynamic modes
        self._atm_bridge.step(self._env, 0.0,
                               info.get("altitude", 500.0),
                               info.get("airspeed", 10.0))
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        phys_action = self._denorm_action(action)
        cpp_action  = np.array(phys_action, dtype=np.float64)

        obs, reward, done, info = self._env.step(cpp_action)

        # Update atmosphere after each RL step
        if not self._atm_bridge.is_static:
            self._atm_bridge.step(
                self._env,
                self._env.get_time(),
                info.get("altitude", 0.0),
                info.get("airspeed", 10.0),
            )

        obs        = np.array(obs, dtype=np.float32)
        reward     = float(reward)
        terminated = bool(info.get("hit_ground", False) or info.get("diverged", False))
        truncated  = done and not terminated

        # Add wind info to the info dict for logging
        if not self._atm_bridge.is_static:
            vx, vy, vz = self._atm_bridge.get_current_wind()
            info["wind_x"] = vx
            info["wind_y"] = vy
            info["wind_speed"] = float(np.sqrt(vx**2 + vy**2))

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    # --------------------------------------------------------------------------
    # Action remapping
    # --------------------------------------------------------------------------

    def _denorm_action(self, action: np.ndarray) -> np.ndarray:
        """Map policy output [-1, 1] to physical brake range [0, 0.94]."""
        normalized = (np.clip(action, -1.0, 1.0) + 1.0) / 2.0
        return normalized * (MAX_BRAKE - MIN_BRAKE) + MIN_BRAKE

    # --------------------------------------------------------------------------
    # Accessors
    # --------------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        return np.array(self._env.get_state(), dtype=np.float64)

    def get_time(self) -> float:
        return self._env.get_time()

    def set_target(self, x: float, y: float):
        self.target = (x, y)
        self._env.set_target(x, y)

    def set_domain_random_config(self, cfg):
        self._env.set_domain_random_config(cfg)

    @property
    def params(self):
        return self._params


# =============================================================================
# Factory function for SubprocVecEnv
# =============================================================================

def make_env(
    target: Tuple[float, float] = (0.0, 0.0),
    dt_physics: float = 0.01,
    dt_action:  float = 0.1,
    max_episode_time: float = 1200.0,
    seed: int = 0,
    domain_random: bool = True,
    atmosphere_mode: str = ATMO_STATIC,
    turbulence_intensity: str = "moderate",
    wind_x: float = 0.0,
    wind_y: float = 0.0,
):
    """
    Factory for SubprocVecEnv. Each parallel env gets its own atmosphere instance.
    For Dryden/dynamic modes, each env gets a different random wind profile because
    AtmosphereBridge.reset() is re-seeded from the gym seed on each episode reset.
    """
    def _init():
        env = ParafoilGymEnv(
            target               = target,
            dt_physics           = dt_physics,
            dt_action            = dt_action,
            max_episode_time     = max_episode_time,
            domain_random        = domain_random,
            atmosphere_mode      = atmosphere_mode,
            turbulence_intensity = turbulence_intensity,
            wind_x               = wind_x,
            wind_y               = wind_y,
        )
        env.reset(seed=seed)
        return env
    return _init