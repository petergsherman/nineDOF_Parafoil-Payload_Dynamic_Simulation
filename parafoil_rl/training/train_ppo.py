"""
training/train_ppo.py
======================
PPO training script for the autonomous parafoil landing controller.

Supports:
  - Single-env and parallel-env training (via SubprocVecEnv)
  - Configurable model size and training budget for sweeps
  - Checkpoint saving and TensorBoard logging
  - Easy to extend for curriculum, wind sweeps, etc.

Usage:
    # Basic training (8 parallel envs, 2M steps):
    python train_ppo.py

    # Custom run:
    python train_ppo.py --n_envs 16 --total_steps 5_000_000 --run_name thesis_run1

    # Model size sweep:
    python train_ppo.py --policy_size small --total_steps 2_000_000
    python train_ppo.py --policy_size medium --total_steps 2_000_000
    python train_ppo.py --policy_size large --total_steps 2_000_000
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory to path so we can import python_env
_python_env = Path(__file__).parent.parent.resolve() / "python_env"
if str(_python_env) not in sys.path:
    sys.path.insert(0, str(_python_env))

from parafoil_gym_env import ParafoilGymEnv, make_env

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

# ============================================================================
# Policy network size presets
# These define the hidden layer sizes for actor and critic.
# Sweep these to study model capacity vs. training efficiency.
# ============================================================================
POLICY_SIZES = {
    "tiny":   dict(net_arch=[64, 64]),
    "small":  dict(net_arch=[128, 128]),
    "medium": dict(net_arch=[256, 256]),
    "large":  dict(net_arch=[512, 256, 128]),
    "xlarge": dict(net_arch=[512, 512, 256, 128]),
}

# ============================================================================
# PPO hyperparameters
# Tuned for the parafoil task: medium episode length, continuous action space.
# ============================================================================
PPO_DEFAULTS = dict(
    learning_rate   = 3e-4,
    n_steps         = 2048,     # Steps per env per rollout collection
    batch_size      = 256,      # Mini-batch size for gradient updates
    n_epochs        = 10,       # PPO epochs per rollout
    gamma           = 0.995,    # Discount - high because episodes are long
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.005,    # Entropy bonus - encourages exploration
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    verbose         = 1,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on parafoil landing task")

    # Environment
    parser.add_argument("--target_x",        type=float, default=0.0,
                        help="Landing target X (m)")
    parser.add_argument("--target_y",        type=float, default=0.0,
                        help="Landing target Y (m)")
    parser.add_argument("--dt_action",       type=float, default=0.1,
                        help="RL action timestep (s)")
    parser.add_argument("--dt_physics",      type=float, default=0.01,
                        help="Physics integration timestep (s)")
    parser.add_argument("--max_episode_time",type=float, default=1200.0,
                        help="Max episode duration (s)")

    # Training
    parser.add_argument("--n_envs",         type=int,   default=8,
                        help="Number of parallel environments")
    parser.add_argument("--total_steps",    type=int,   default=2_000_000,
                        help="Total environment steps for training")
    parser.add_argument("--policy_size",    type=str,   default="medium",
                        choices=list(POLICY_SIZES.keys()),
                        help="Policy network size preset")
    parser.add_argument("--seed",           type=int,   default=42,
                        help="Global random seed")

    # PPO hyperparams (override defaults)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--n_steps",        type=int,   default=2048)
    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--gamma",          type=float, default=0.995)

    # I/O
    parser.add_argument("--run_name",       type=str,   default="",
                        help="Run name (auto-generated if empty)")
    parser.add_argument("--log_dir",        type=str,   default="runs/",
                        help="TensorBoard log directory")
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints/",
                        help="Directory for periodic checkpoints")
    parser.add_argument("--checkpoint_freq",type=int,   default=100_000,
                        help="Checkpoint every N steps (per-env steps)")
    parser.add_argument("--n_eval_episodes",type=int,   default=20,
                        help="Episodes for eval callback")

    return parser.parse_args()


def make_run_name(args) -> str:
    if args.run_name:
        return args.run_name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"ppo_{args.policy_size}_{args.total_steps // 1_000}k_{timestamp}"


def build_vec_env(args, training: bool = True):
    """Build a vectorized training or evaluation environment."""
    n_envs = args.n_envs if training else 1
    use_subprocess = (n_envs > 1)

    env_kwargs = dict(
        target         = (args.target_x, args.target_y),
        dt_physics     = args.dt_physics,
        dt_action      = args.dt_action,
        max_episode_time = args.max_episode_time,
        domain_random  = training,   # No randomization for eval
    )

    if use_subprocess:
        # SubprocVecEnv: each env runs in its own process for true parallelism.
        # Each env gets a different seed to ensure diverse rollouts.
        env_fns = [
            make_env(**env_kwargs, seed=args.seed + i)
            for i in range(n_envs)
        ]
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = make_vec_env(
            ParafoilGymEnv,
            n_envs    = 1,
            seed      = args.seed,
            env_kwargs = env_kwargs,
        )

    # VecMonitor wraps for episode-level logging (reward, length)
    vec_env = VecMonitor(vec_env)
    return vec_env


def build_callbacks(args, run_name: str, eval_env):
    """Build SB3 training callbacks."""
    checkpoint_dir = Path(args.checkpoint_dir) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq       = args.checkpoint_freq,
        save_path       = str(checkpoint_dir),
        name_prefix     = "parafoil_ppo",
        save_replay_buffer = False,
        verbose         = 1,
    )

    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes = args.n_eval_episodes,
        eval_freq       = max(10_000, args.checkpoint_freq // 2),
        log_path        = str(Path(args.log_dir) / run_name / "eval"),
        best_model_save_path = str(checkpoint_dir),
        deterministic   = True,
        verbose         = 1,
    )

    return CallbackList([checkpoint_cb, eval_cb])


def main():
    args = parse_args()
    run_name = make_run_name(args)
    print(f"\n{'='*60}")
    print(f"  Parafoil PPO Training")
    print(f"  Run:          {run_name}")
    print(f"  Policy size:  {args.policy_size}  {POLICY_SIZES[args.policy_size]}")
    print(f"  Total steps:  {args.total_steps:,}")
    print(f"  Parallel envs:{args.n_envs}")
    print(f"  Target:       ({args.target_x}, {args.target_y})")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Build environments
    # ------------------------------------------------------------------
    print("Building training environments...")
    train_env = build_vec_env(args, training=True)

    print("Building evaluation environment...")
    eval_env = build_vec_env(args, training=False)

    # ------------------------------------------------------------------
    # Build PPO model
    # ------------------------------------------------------------------
    policy_kwargs = POLICY_SIZES[args.policy_size].copy()

    # Override PPO defaults with any command-line args
    ppo_kwargs = PPO_DEFAULTS.copy()
    ppo_kwargs.update(dict(
        learning_rate = args.lr,
        n_steps       = args.n_steps,
        batch_size    = args.batch_size,
        gamma         = args.gamma,
        tensorboard_log = args.log_dir,
    ))

    model = PPO(
        policy         = "MlpPolicy",
        env            = train_env,
        policy_kwargs  = policy_kwargs,
        seed           = args.seed,
        **ppo_kwargs,
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Policy parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    callbacks = build_callbacks(args, run_name, eval_env)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\nStarting training for {args.total_steps:,} steps...")
    t0 = time.time()

    model.learn(
        total_timesteps = args.total_steps,
        callback        = callbacks,
        tb_log_name     = run_name,
        reset_num_timesteps = True,
        progress_bar    = True,
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} minutes.")

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    save_dir = Path(args.checkpoint_dir) / run_name
    final_path = save_dir / "parafoil_ppo_final"
    model.save(str(final_path))
    print(f"Final model saved to: {final_path}.zip")

    # Save training config alongside the model for reproducibility
    config_path = save_dir / "training_config.txt"
    with open(config_path, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nrun_name: {run_name}\n")
        f.write(f"policy_params: {total_params}\n")
        f.write(f"training_time_min: {elapsed/60:.1f}\n")
    print(f"Config saved to: {config_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()