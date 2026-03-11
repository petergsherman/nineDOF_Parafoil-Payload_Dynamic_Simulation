"""
config/rl_config.py
====================
Central configuration for parafoil RL experiments.
Edit this file to set up training sweeps, reward tuning, curriculum, etc.

Import in training/evaluation scripts:
    from config.rl_config import ENV_CONFIG, TRAINING_CONFIG, SWEEP_CONFIG
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional


# ==============================================================================
# Environment Configuration
# ==============================================================================

@dataclass
class EnvConfig:
    """Environment parameters shared across training and evaluation."""

    # Landing target (inertial frame, meters)
    target: Tuple[float, float] = (0.0, 0.0)

    # Timesteps
    dt_physics:  float = 0.01    # Simulation integration timestep (s)
    dt_action:   float = 0.1     # RL decision frequency (s)
                                  # steps_per_action = dt_action / dt_physics = 10

    # Episode limits
    max_episode_time: float = 1200.0   # Hard timeout (s)

    # Action limits (physical brake deflection)
    max_brake: float = 0.94
    min_brake: float = 0.0


# ==============================================================================
# Domain Randomization Configuration
# ==============================================================================

@dataclass
class DomainRandomizationConfig:
    """
    Initial condition randomization ranges.
    All ranges are +/- unless noted (min/max).

    Tune these as curriculum progresses:
      - Start narrow (easy) and gradually widen (hard)
      - Or start wide and rely on PPO to learn through difficulty
    """

    # Initial horizontal position spread around target (m)
    pos_x_range: float = 600.0
    pos_y_range: float = 600.0

    # Initial altitude (m AGL)
    alt_min: float = 400.0
    alt_max: float = 700.0

    # Initial heading - full 360 deg by default (radians)
    heading_range: float = 3.14159   # pi

    # Velocity noise around trim
    u_nominal: float = 10.0   # Forward speed (m/s)
    u_noise:   float = 1.0
    w_nominal: float = -0.5   # Descent rate (m/s)
    w_noise:   float = 0.2

    # Attitude noise (rad)
    theta_noise: float = 0.05
    phi_noise:   float = 0.02

    # Wind (m/s) - keep zero for static atmosphere training
    wind_x_range: float = 0.0
    wind_y_range: float = 0.0


# ==============================================================================
# Reward Configuration
# Adjust weights here to reshape learning incentives.
# ==============================================================================

@dataclass
class RewardConfig:
    """
    Reward shaping weights.
    These match the computeReward() implementation in ParafoilEnv.cpp.
    If you change these, update ParafoilEnv.cpp to match.
    """
    # Dense shaping: reward for reducing horizontal distance per step
    progress_weight:      float = 2.0

    # Per-step alive bonus
    alive_bonus:          float = 0.01

    # Penalty per unit of differential brake (asymmetric yaw input)
    control_penalty:      float = 0.05

    # Terminal: landing reward = terminal_base - terminal_dist_scale * distance
    terminal_base:        float = 200.0
    terminal_dist_scale:  float = 0.6

    # Extra accuracy bonuses
    bonus_25m:            float = 100.0
    bonus_10m:            float = 200.0

    # Divergence penalty
    divergence_penalty:   float = 200.0


# ==============================================================================
# PPO Training Configuration
# ==============================================================================

@dataclass
class TrainingConfig:
    """PPO hyperparameters."""

    # Network
    policy_size:    str   = "medium"    # "tiny", "small", "medium", "large", "xlarge"

    # PPO core
    learning_rate:  float = 3e-4
    n_steps:        int   = 2048        # Steps per rollout per env
    batch_size:     int   = 256
    n_epochs:       int   = 10
    gamma:          float = 0.995
    gae_lambda:     float = 0.95
    clip_range:     float = 0.2
    ent_coef:       float = 0.005
    vf_coef:        float = 0.5
    max_grad_norm:  float = 0.5

    # Training length
    total_steps:    int   = 2_000_000

    # Parallelism
    n_envs:         int   = 8           # Number of parallel envs (SubprocVecEnv)

    # Logging
    checkpoint_freq:    int = 100_000
    n_eval_episodes:    int = 20
    seed:               int = 42


# ==============================================================================
# Sweep Configuration
# For thesis: 5 model sizes x 5 training budgets = 25 runs
# ==============================================================================

POLICY_SIZE_SWEEP: List[str] = [
    "tiny",     # ~8k params
    "small",    # ~33k params
    "medium",   # ~130k params
    "large",    # ~400k params
    "xlarge",   # ~800k params
]

TRAINING_BUDGET_SWEEP: List[int] = [
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
]

WIND_SWEEP_CONDITIONS: List[Tuple[float, float, str]] = [
    (0.0,  0.0,  "no_wind"),
    (3.0,  0.0,  "wind_3ms_x"),
    (-3.0, 0.0,  "wind_3ms_xneg"),
    (0.0,  3.0,  "wind_3ms_y"),
    (5.0,  5.0,  "wind_5ms_diag"),
    (8.0,  0.0,  "wind_8ms_x"),
    (0.0,  8.0,  "wind_8ms_y"),
]

# Success threshold for paper reporting
SUCCESS_RADIUS_M: float = 50.0  # Commonly cited in parafoil literature


# ==============================================================================
# Default instances (import these directly in scripts)
# ==============================================================================

ENV_CONFIG      = EnvConfig()
DR_CONFIG       = DomainRandomizationConfig()
REWARD_CONFIG   = RewardConfig()
TRAINING_CONFIG = TrainingConfig()
