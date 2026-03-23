"""
training/train_ppo.py
======================
PPO training script for the autonomous parafoil landing controller.

Atmosphere options:
    static   - No wind (fastest, good for initial training)
    dryden   - MIL-F-8785C Dryden turbulence + layered wind (most realistic)
    simple   - Sinusoidal gusts + layered wind
    wind     - Layered mean wind only

Static wind override (--atmosphere static only):
    --wind_x / --wind_y set a constant headwind during training.

Usage:
    # Static, no wind:
    python train_ppo.py --n_envs 8 --total_steps 2_000_000

    # Static with constant headwind:
    python train_ppo.py --n_envs 8 --total_steps 2_000_000 --atmosphere static --wind_x 3.0

    # Dryden turbulence, moderate:
    python train_ppo.py --n_envs 8 --total_steps 2_000_000 --atmosphere dryden --turbulence moderate

    # Train with Dryden, evaluate callback uses static (clean comparison):
    python train_ppo.py --atmosphere dryden --eval_atmosphere static

    # Model size sweep:
    python train_ppo.py --policy_size small  --total_steps 2_000_000
    python train_ppo.py --policy_size medium --total_steps 2_000_000
    python train_ppo.py --policy_size large  --total_steps 2_000_000
"""

# ── Path setup (must be first) ────────────────────────────────────────────
import sys, os
from pathlib import Path
_rl_root = Path(__file__).resolve().parent.parent  # parafoil_rl/
if str(_rl_root) not in sys.path:
    sys.path.insert(0, str(_rl_root))
import setup_paths  # registers DLLs + adds python_env/ and sim root
# ─────────────────────────────────────────────────────────────────────────

import argparse
import time
import numpy as np

from parafoil_gym_env import ParafoilGymEnv, make_env
from atmosphere_bridge import ATMO_STATIC, ATMO_DRYDEN, ATMO_SIMPLE, ATMO_WIND

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList,
)

# ============================================================================
# Policy network size presets
# ============================================================================
POLICY_SIZES = {
    "tiny":   dict(net_arch=[64, 64]),
    "small":  dict(net_arch=[128, 128]),
    "medium": dict(net_arch=[256, 256]),
    "large":  dict(net_arch=[512, 256, 128]),
    "xlarge": dict(net_arch=[512, 512, 256, 128]),
}

PPO_DEFAULTS = dict(
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 256,
    n_epochs      = 10,
    gamma         = 0.995,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.005,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    verbose       = 1,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on parafoil landing task")

    # Environment
    parser.add_argument("--target_x",         type=float, default=0.0)
    parser.add_argument("--target_y",         type=float, default=0.0)
    parser.add_argument("--dt_action",        type=float, default=0.1)
    parser.add_argument("--dt_physics",       type=float, default=0.01)
    parser.add_argument("--max_episode_time", type=float, default=1200.0)

    # Atmosphere
    parser.add_argument("--atmosphere",       type=str,   default="static",
                        choices=["static", "dryden", "simple", "wind"],
                        help="Atmosphere model for training environments.")
    parser.add_argument("--turbulence",       type=str,   default="moderate",
                        choices=["light", "moderate", "severe"],
                        help="Turbulence intensity (dryden/simple only).")
    # Static wind - only applied when --atmosphere static
    parser.add_argument("--wind_x",           type=float, default=0.0,
                        help="Constant wind X (m/s). Only used with --atmosphere static.")
    parser.add_argument("--wind_y",           type=float, default=0.0,
                        help="Constant wind Y (m/s). Only used with --atmosphere static.")
    # Eval callback atmosphere (can differ from training atmosphere)
    parser.add_argument("--eval_atmosphere",  type=str,   default=None,
                        help="Atmosphere for eval callback. Defaults to same as --atmosphere.")
    parser.add_argument("--eval_wind_x",      type=float, default=0.0,
                        help="Constant wind X for eval callback (static only).")
    parser.add_argument("--eval_wind_y",      type=float, default=0.0,
                        help="Constant wind Y for eval callback (static only).")

    # Training
    parser.add_argument("--n_envs",           type=int,   default=8)
    parser.add_argument("--total_steps",      type=int,   default=2_000_000)
    parser.add_argument("--policy_size",      type=str,   default="medium",
                        choices=list(POLICY_SIZES.keys()))
    parser.add_argument("--seed",             type=int,   default=42)

    # PPO hyperparams
    parser.add_argument("--lr",               type=float, default=3e-4)
    parser.add_argument("--n_steps",          type=int,   default=2048)
    parser.add_argument("--batch_size",       type=int,   default=256)
    parser.add_argument("--gamma",            type=float, default=0.995)

    # I/O
    parser.add_argument("--run_name",         type=str,   default="")
    parser.add_argument("--log_dir",          type=str,   default="runs/")
    parser.add_argument("--checkpoint_dir",   type=str,   default="checkpoints/")
    parser.add_argument("--checkpoint_freq",  type=int,   default=100_000)
    parser.add_argument("--n_eval_episodes",  type=int,   default=20)

    return parser.parse_args()


def make_run_name(args) -> str:
    if args.run_name:
        return args.run_name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    atm_tag   = "" if args.atmosphere == "static" else f"_{args.atmosphere}"
    turb_tag  = f"_{args.turbulence}" if args.atmosphere in ("dryden", "simple") else ""
    wind_tag  = (f"_wx{args.wind_x:.0f}wy{args.wind_y:.0f}"
                 if args.atmosphere == "static" and (args.wind_x or args.wind_y) else "")
    return f"ppo_{args.policy_size}_{args.total_steps // 1_000}k{atm_tag}{turb_tag}{wind_tag}_{timestamp}"


def atm_description(mode, turbulence, wind_x=0.0, wind_y=0.0) -> str:
    """Human-readable atmosphere description for printing."""
    if mode == ATMO_STATIC:
        if wind_x != 0.0 or wind_y != 0.0:
            return f"static wind ({wind_x:.1f}, {wind_y:.1f}) m/s"
        return "static (no wind)"
    return f"{mode} ({turbulence})"


def build_vec_env(args, training: bool = True):
    """Build vectorized training or evaluation environment."""
    n_envs   = args.n_envs if training else 1

    if training:
        atm_mode = args.atmosphere
        wind_x   = args.wind_x
        wind_y   = args.wind_y
    else:
        atm_mode = args.eval_atmosphere if args.eval_atmosphere else args.atmosphere
        wind_x   = args.eval_wind_x
        wind_y   = args.eval_wind_y

    env_kwargs = dict(
        target               = (args.target_x, args.target_y),
        dt_physics           = args.dt_physics,
        dt_action            = args.dt_action,
        max_episode_time     = args.max_episode_time,
        domain_random        = training,
        atmosphere_mode      = atm_mode,
        turbulence_intensity = args.turbulence,
    )

    if n_envs > 1:
        # Each subprocess env gets a unique seed and its own atmosphere instance.
        # For dynamic atmospheres (Dryden), each env gets a different wind profile
        # because AtmosphereBridge.reset() is seeded from the gym seed inside reset().
        env_fns = [make_env(**env_kwargs, seed=args.seed + i,
                            wind_x=wind_x, wind_y=wind_y)
                   for i in range(n_envs)]
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = make_vec_env(ParafoilGymEnv, n_envs=1, seed=args.seed,
                               env_kwargs={**env_kwargs,
                                           "wind_x": wind_x, "wind_y": wind_y})

    return VecMonitor(vec_env)


def build_callbacks(args, run_name: str, eval_env):
    checkpoint_dir = Path(args.checkpoint_dir) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq   = args.checkpoint_freq,
        save_path   = str(checkpoint_dir),
        name_prefix = "parafoil_ppo",
        verbose     = 1,
    )
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes      = args.n_eval_episodes,
        eval_freq            = max(10_000, args.checkpoint_freq // 2),
        log_path             = str(Path(args.log_dir) / run_name / "eval"),
        best_model_save_path = str(checkpoint_dir),
        deterministic        = True,
        verbose              = 1,
    )
    return CallbackList([checkpoint_cb, eval_cb])


def main():
    args     = parse_args()
    run_name = make_run_name(args)

    eval_atm_mode = args.eval_atmosphere if args.eval_atmosphere else args.atmosphere
    train_desc    = atm_description(args.atmosphere, args.turbulence, args.wind_x, args.wind_y)
    eval_desc     = atm_description(eval_atm_mode, args.turbulence, args.eval_wind_x, args.eval_wind_y)

    print(f"\n{'='*60}")
    print(f"  Parafoil PPO Training")
    print(f"  Run:            {run_name}")
    print(f"  Policy size:    {args.policy_size}  {POLICY_SIZES[args.policy_size]}")
    print(f"  Total steps:    {args.total_steps:,}")
    print(f"  Parallel envs:  {args.n_envs}")
    print(f"  Target:         ({args.target_x}, {args.target_y})")
    print(f"  Train atm:      {train_desc}")
    print(f"  Eval atm:       {eval_desc}")
    print(f"{'='*60}\n")

    print("Building training environments...")
    train_env = build_vec_env(args, training=True)

    print("Building evaluation environment...")
    eval_env = build_vec_env(args, training=False)

    policy_kwargs = POLICY_SIZES[args.policy_size].copy()
    ppo_kwargs    = PPO_DEFAULTS.copy()
    ppo_kwargs.update(dict(
        learning_rate   = args.lr,
        n_steps         = args.n_steps,
        batch_size      = args.batch_size,
        gamma           = args.gamma,
        tensorboard_log = args.log_dir,
    ))

    model = PPO("MlpPolicy", train_env,
                policy_kwargs=policy_kwargs, seed=args.seed, **ppo_kwargs)

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Policy parameters: {total_params:,}")

    callbacks = build_callbacks(args, run_name, eval_env)

    print(f"\nStarting training for {args.total_steps:,} steps...")
    t0 = time.time()
    model.learn(total_timesteps=args.total_steps, callback=callbacks,
                tb_log_name=run_name, reset_num_timesteps=True, progress_bar=True)
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} minutes.")

    save_dir   = Path(args.checkpoint_dir) / run_name
    final_path = save_dir / "parafoil_ppo_final"
    model.save(str(final_path))
    print(f"Final model saved to: {final_path}.zip")

    config_path = save_dir / "training_config.txt"
    with open(config_path, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nrun_name: {run_name}\n")
        f.write(f"policy_params: {total_params}\n")
        f.write(f"training_time_min: {elapsed/60:.1f}\n")
        f.write(f"train_atmosphere: {train_desc}\n")
        f.write(f"eval_atmosphere: {eval_desc}\n")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()