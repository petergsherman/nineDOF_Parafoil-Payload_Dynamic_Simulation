"""
python_env/parafoil_gym_env.py
===============================
Gymnasium-compatible wrapper around the C++ parafoil RL environment.

Usage:
    from parafoil_gym_env import ParafoilGymEnv
    env = ParafoilGymEnv(target=(0.0, 0.0))
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(action)

Observation space (12 dims, all in [-1, 1]):
    See ParafoilEnv.getObservation() in ParafoilEnv.h for full description.

Action space (2 dims, in [0, 1] normalized -> remapped to [0, 0.94]):
    [0] Left brake command
    [1] Right brake command

    Two-brake action space is used (not a single differential) because:
    1. The plant already takes deltaL/deltaR separately.
    2. A single differential command would hide the symmetric-brake degree
       of freedom that the policy might need to control descent rate.
    3. PPO handles multi-dimensional continuous action spaces well.
"""

import sys
from pathlib import Path

# Ensure the directory containing parafoil_cpp.pyd is on sys.path.
# This works whether the script is run from the project root, python_env/,
# training/, or anywhere else.
_python_env_dir = Path(__file__).parent.resolve()
if str(_python_env_dir) not in sys.path:
    sys.path.insert(0, str(_python_env_dir))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

# Import the compiled C++ backend.
# Build instructions: see cpp_env/CMakeLists.txt
try:
    import parafoil_cpp
except ImportError as e:
    raise ImportError(
        "Could not import parafoil_cpp. "
        "Build the pybind11 module first:\n"
        "  cd cpp_env && mkdir build && cd build\n"
        "  cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) && make install\n"
        f"Original error: {e}"
    )


# Physical action limits (match the plant)
MAX_BRAKE = 0.94
MIN_BRAKE = 0.0


class ParafoilGymEnv(gym.Env):
    """
    Gymnasium wrapper for the 9DOF parafoil C++ environment.

    This class:
      - Defines observation_space and action_space for SB3.
      - Translates normalized actions [-1, 1] -> physical [0, 0.94].
      - Calls the C++ backend for all physics.
      - Supports parallel environments via SB3's SubprocVecEnv.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        target: Tuple[float, float] = (0.0, 0.0),
        dt_physics: float = 0.01,
        dt_action:  float = 0.1,
        max_episode_time: float = 1200.0,
        domain_random: bool = True,
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            target:           (x, y) landing target in inertial frame (m).
            dt_physics:       Physics integration timestep (s).
            dt_action:        Time between RL decisions (s).
            max_episode_time: Hard episode timeout (s).
            domain_random:    If True, randomize initial conditions on reset().
            render_mode:      Currently unused ('human' mode not implemented).
        """
        super().__init__()

        self.target = target
        self.domain_random = domain_random
        self.render_mode = render_mode

        # Build default SystemParameters (matches nineDOF_Parameters.py defaults)
        self._params = parafoil_cpp.SystemParameters()

        # Create C++ env
        self._env = parafoil_cpp.ParafoilEnv(
            self._params,
            target[0], target[1],
            dt_physics, dt_action,
        )
        self._env.set_max_episode_time(max_episode_time)

        if not domain_random:
            # Zero-noise config for deterministic evaluation
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

        # -----------------------------------------------------------------------
        # Action space: two normalized brake commands in [-1, 1]
        # Internally remapped to [0, 0.94] (see _denorm_action).
        # Using [-1, 1] is standard for PPO with tanh policy output.
        # -----------------------------------------------------------------------
        self.action_space = spaces.Box(
            low   = np.full(2, -1.0, dtype=np.float32),
            high  = np.full(2,  1.0, dtype=np.float32),
            dtype = np.float32,
        )

        # -----------------------------------------------------------------------
        # Observation space: 12 normalized dims, all in [-1, 1]
        # -----------------------------------------------------------------------
        self.observation_space = spaces.Box(
            low   = np.full(parafoil_cpp.OBS_SIZE, -1.0, dtype=np.float32),
            high  = np.full(parafoil_cpp.OBS_SIZE,  1.0, dtype=np.float32),
            dtype = np.float32,
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
        obs = self._env.reset(seed=cpp_seed)
        obs = np.array(obs, dtype=np.float32)
        info = self._env.get_info()
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Remap action from [-1, 1] -> [MIN_BRAKE, MAX_BRAKE]
        phys_action = self._denorm_action(action)

        cpp_action = np.array(phys_action, dtype=np.float64)
        obs, reward, done, info = self._env.step(cpp_action)

        obs = np.array(obs, dtype=np.float32)
        reward = float(reward)

        # Gymnasium uses terminated (natural end) vs truncated (timeout)
        terminated = bool(info.get("hit_ground", False) or info.get("diverged", False))
        truncated  = done and not terminated  # Timeout

        return obs, reward, terminated, truncated, info

    def render(self):
        pass  # Visualization is handled separately by nineDOF_Visualization.py

    def close(self):
        pass

    # --------------------------------------------------------------------------
    # Action remapping
    # --------------------------------------------------------------------------

    def _denorm_action(self, action: np.ndarray) -> np.ndarray:
        """Map policy output [-1, 1] to physical brake range [0, 0.94]."""
        # [-1, 1] -> [0, 1] -> [MIN_BRAKE, MAX_BRAKE]
        normalized = (np.clip(action, -1.0, 1.0) + 1.0) / 2.0
        return normalized * (MAX_BRAKE - MIN_BRAKE) + MIN_BRAKE

    def _norm_action(self, phys_action: np.ndarray) -> np.ndarray:
        """Map physical brake [0, 0.94] back to [-1, 1] (for logging)."""
        normalized = (phys_action - MIN_BRAKE) / (MAX_BRAKE - MIN_BRAKE)
        return 2.0 * normalized - 1.0

    # --------------------------------------------------------------------------
    # Convenience accessors
    # --------------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Return the full 18-element state vector."""
        return np.array(self._env.get_state(), dtype=np.float64)

    def get_time(self) -> float:
        return self._env.get_time()

    def set_target(self, x: float, y: float):
        """Change landing target (takes effect on next reset)."""
        self.target = (x, y)
        self._env.set_target(x, y)

    def set_domain_random_config(self, cfg: parafoil_cpp.DomainRandomConfig):
        self._env.set_domain_random_config(cfg)

    @property
    def params(self) -> parafoil_cpp.SystemParameters:
        return self._params


def make_env(
    target: Tuple[float, float] = (0.0, 0.0),
    dt_physics: float = 0.01,
    dt_action:  float = 0.1,
    max_episode_time: float = 1200.0,
    seed: int = 0,
    domain_random: bool = True,
):
    """
    Factory function for creating a single environment.
    Designed for use with SB3's make_vec_env / SubprocVecEnv.

    Usage:
        from stable_baselines3.common.env_util import make_vec_env
        vec_env = make_vec_env(make_env, n_envs=8, env_kwargs=dict(target=(0,0)))
    """
    def _init():
        env = ParafoilGymEnv(
            target=target,
            dt_physics=dt_physics,
            dt_action=dt_action,
            max_episode_time=max_episode_time,
            domain_random=domain_random,
        )
        env.reset(seed=seed)
        return env
    return _init