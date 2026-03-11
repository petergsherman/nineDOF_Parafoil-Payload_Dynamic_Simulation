"""
evaluation/evaluate_ppo.py
===========================
Evaluation script for trained parafoil PPO models.

Loads a trained model, runs deterministic evaluation episodes, and computes:
  - Mean/std landing error (m)
  - Success rate (within configurable radius)
  - Failure rate (diverged episodes)
  - Control effort statistics
  - Episode duration statistics

Optionally saves trajectories for post-hoc visualization.

Usage:
    python evaluate_ppo.py --model checkpoints/my_run/parafoil_ppo_final

    # Custom radius and more episodes:
    python evaluate_ppo.py --model checkpoints/run1/best_model --n_episodes 100 --success_radius 50

    # Evaluate across multiple wind conditions:
    python evaluate_ppo.py --model checkpoints/run1/best_model --wind_sweep
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np

_python_env = Path(__file__).parent.parent.resolve() / "python_env"
if str(_python_env) not in sys.path:
    sys.path.insert(0, str(_python_env))
from parafoil_gym_env import ParafoilGymEnv

try:
    from stable_baselines3 import PPO
except ImportError:
    raise ImportError("Install stable-baselines3: pip install stable-baselines3")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained parafoil PPO policy")

    parser.add_argument("--model",          type=str,   required=True,
                        help="Path to saved model (without .zip)")
    parser.add_argument("--n_episodes",     type=int,   default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--success_radius", type=float, default=50.0,
                        help="Landing success radius (m)")
    parser.add_argument("--target_x",      type=float, default=0.0)
    parser.add_argument("--target_y",      type=float, default=0.0)
    parser.add_argument("--dt_action",     type=float, default=0.1)
    parser.add_argument("--dt_physics",    type=float, default=0.01)
    parser.add_argument("--max_episode_time", type=float, default=1200.0)
    parser.add_argument("--seed",          type=int,   default=0)

    # Starting position randomization for evaluation
    # By default each episode starts at a random offset from target,
    # matching training conditions. Override with --start_x/y for fixed starts.
    parser.add_argument("--start_x",       type=float, default=None,
                        help="Fixed start X (m). Random if not set.")
    parser.add_argument("--start_y",       type=float, default=None,
                        help="Fixed start Y (m). Random if not set.")
    parser.add_argument("--start_alt",     type=float, default=None,
                        help="Fixed start altitude AGL (m). Random 400-700 if not set.")
    parser.add_argument("--start_radius",  type=float, default=600.0,
                        help="Max random start distance from target (m). Default 600.")

    # Domain randomization for eval (default: OFF for reproducibility)
    parser.add_argument("--domain_random", action="store_true",
                        help="Use domain randomization during evaluation")

    # Wind sweep evaluation
    parser.add_argument("--wind_sweep",    action="store_true",
                        help="Run evaluation under multiple wind conditions")

    # Output
    parser.add_argument("--save_dir",      type=str,   default="eval_results/",
                        help="Directory to save evaluation results")
    parser.add_argument("--save_trajectories", action="store_true",
                        help="Save full state trajectories for each episode")
    parser.add_argument("--plot",          action="store_true",
                        help="Plot trajectories after evaluation")

    return parser.parse_args()


def run_episode(
    model,
    env: ParafoilGymEnv,
    deterministic: bool = True,
    save_trajectory: bool = False,
    seed: int = 0,
    fixed_start: np.ndarray = None,
) -> Dict:
    """
    Run a single evaluation episode.
    fixed_start: optional 18-element state vector. If None, env resets normally.
    Returns a dict with episode metrics.
    """
    if fixed_start is not None:
        import parafoil_cpp
        obs = env._env.reset_fixed(fixed_start)
        obs = np.array(obs, dtype=np.float32)
        info = env._env.get_info()
    else:
        obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    step = 0
    trajectory = [] if save_trajectory else None

    # Track control effort
    actions = []

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step += 1
        actions.append(action.copy())

        if save_trajectory:
            state = env.get_state()
            trajectory.append(state.copy())

        done = terminated or truncated

    actions = np.array(actions)

    result = {
        "landing_error":   info.get("distance_to_target", float("nan")),
        "total_reward":    total_reward,
        "episode_steps":   step,
        "episode_time":    env.get_time(),
        "hit_ground":      info.get("hit_ground", False),
        "diverged":        info.get("diverged", False),
        "mean_action_l":   float(np.mean(actions[:, 0])),
        "mean_action_r":   float(np.mean(actions[:, 1])),
        "std_action_l":    float(np.std(actions[:, 0])),
        "std_action_r":    float(np.std(actions[:, 1])),
        "mean_differential": float(np.mean(np.abs(actions[:, 1] - actions[:, 0]))),
        "trajectory":      trajectory,
    }
    return result


def compute_summary(results: List[Dict], success_radius: float) -> Dict:
    """Compute aggregate metrics from episode results."""
    errors      = [r["landing_error"] for r in results if not r["diverged"]]
    rewards     = [r["total_reward"]  for r in results]
    times       = [r["episode_time"]  for r in results]
    n_diverged  = sum(1 for r in results if r["diverged"])
    n_success   = sum(1 for r in results if not r["diverged"] and r["landing_error"] <= success_radius)
    n_total     = len(results)

    summary = {
        "n_episodes":        n_total,
        "success_radius_m":  success_radius,
        "n_success":         n_success,
        "n_diverged":        n_diverged,
        "success_rate":      n_success / n_total if n_total > 0 else 0.0,
        "diverge_rate":      n_diverged / n_total if n_total > 0 else 0.0,
        "mean_landing_error_m": float(np.mean(errors)) if errors else float("nan"),
        "std_landing_error_m":  float(np.std(errors))  if errors else float("nan"),
        "median_landing_error_m": float(np.median(errors)) if errors else float("nan"),
        "p90_landing_error_m":    float(np.percentile(errors, 90)) if errors else float("nan"),
        "mean_episode_reward":    float(np.mean(rewards)),
        "std_episode_reward":     float(np.std(rewards)),
        "mean_episode_time_s":    float(np.mean(times)),
        "mean_differential_brake": float(np.mean([r["mean_differential"] for r in results])),
    }
    return summary


def print_summary(summary: Dict, label: str = ""):
    header = f"  Evaluation Results {label}"
    print(f"\n{'='*60}")
    print(header)
    print(f"{'='*60}")
    print(f"  Episodes:          {summary['n_episodes']}")
    print(f"  Success radius:    {summary['success_radius_m']:.0f} m")
    print(f"  Success rate:      {summary['success_rate']*100:.1f}%  ({summary['n_success']}/{summary['n_episodes']})")
    print(f"  Diverge rate:      {summary['diverge_rate']*100:.1f}%  ({summary['n_diverged']}/{summary['n_episodes']})")
    print(f"  Landing error:")
    print(f"    Mean:            {summary['mean_landing_error_m']:.1f} m")
    print(f"    Std:             {summary['std_landing_error_m']:.1f} m")
    print(f"    Median:          {summary['median_landing_error_m']:.1f} m")
    print(f"    P90:             {summary['p90_landing_error_m']:.1f} m")
    print(f"  Mean episode time: {summary['mean_episode_time_s']:.1f} s")
    print(f"  Mean diff brake:   {summary['mean_differential_brake']:.3f}")
    print(f"{'='*60}\n")


def make_start_state(args, rng: np.random.Generator, ep: int) -> np.ndarray:
    """
    Build an 18-element initial state for one evaluation episode.
    - If --start_x/y provided: use those exact coordinates.
    - Otherwise: random position within --start_radius of target,
      random heading, nominal velocity.
    """
    state = np.zeros(18)

    if args.start_x is not None and args.start_y is not None:
        # Fully fixed start
        x, y = args.start_x, args.start_y
    else:
        # Random position within start_radius of target, but NOT at origin.
        # Sample angle and distance so starts are spread around the target.
        angle    = rng.uniform(0, 2 * np.pi)
        distance = rng.uniform(args.start_radius * 0.3, args.start_radius)
        x = args.target_x + distance * np.cos(angle)
        y = args.target_y + distance * np.sin(angle)

    alt = args.start_alt if args.start_alt is not None else rng.uniform(400.0, 700.0)

    # Position
    state[0] = x
    state[1] = y
    state[2] = -alt   # NED: z negative = above ground

    # Random heading for both bodies
    heading = rng.uniform(-np.pi, np.pi)
    state[3] = rng.uniform(-0.02, 0.02)   # phi_p
    state[4] = rng.uniform(-0.05, 0.05)   # theta_p
    state[5] = heading                     # psi_p
    state[6] = rng.uniform(-0.02, 0.02)   # phi_c
    state[7] = rng.uniform(-0.05, 0.05)   # theta_c
    state[8] = heading                     # psi_c

    # Nominal trim velocity
    state[9]  = 10.0 + rng.uniform(-0.5, 0.5)   # u (forward)
    state[10] = 0.0
    state[11] = -0.5 + rng.uniform(-0.1, 0.1)   # w (descent)

    return state


def run_evaluation(
    model,
    args,
    wind_x: float = 0.0,
    wind_y: float = 0.0,
    label: str = "",
) -> Dict:
    """Run a full evaluation with given wind conditions."""

    # Create evaluation env (domain_random=False so we control starts explicitly)
    env = ParafoilGymEnv(
        target           = (args.target_x, args.target_y),
        dt_physics       = args.dt_physics,
        dt_action        = args.dt_action,
        max_episode_time = args.max_episode_time,
        domain_random    = False,
    )
    env._env.set_wind(wind_x, wind_y)

    rng = np.random.default_rng(args.seed)

    results = []
    print(f"\nRunning {args.n_episodes} evaluation episodes {label}...")
    if args.start_x is None:
        print(f"  Start positions: random within {args.start_radius:.0f}m of target ({args.target_x:.0f}, {args.target_y:.0f})")
    else:
        print(f"  Start position: fixed ({args.start_x:.0f}, {args.start_y:.0f})")

    for ep in range(args.n_episodes):
        start_state = make_start_state(args, rng, ep)
        start_dist  = np.sqrt((start_state[0]-args.target_x)**2 + (start_state[1]-args.target_y)**2)
        result = run_episode(
            model,
            env,
            deterministic   = True,
            save_trajectory = args.save_trajectories,
            seed            = args.seed + ep,
            fixed_start     = start_state,
        )
        result["start_x"]    = float(start_state[0])
        result["start_y"]    = float(start_state[1])
        result["start_alt"]  = float(-start_state[2])
        result["start_dist"] = float(start_dist)
        results.append(result)

        status = "LAND" if result["hit_ground"] else ("DIV" if result["diverged"] else "TIME")
        print(f"  Ep {ep+1:3d}: {status}  error={result['landing_error']:6.1f}m  "
              f"reward={result['total_reward']:8.1f}  t={result['episode_time']:.0f}s")

    env.close()

    summary = compute_summary(results, args.success_radius)
    print_summary(summary, label=label)
    return summary, results


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model_path = args.model
    if not model_path.endswith(".zip"):
        model_path_check = Path(model_path + ".zip")
    else:
        model_path_check = Path(model_path)

    if not model_path_check.exists():
        raise FileNotFoundError(f"Model not found: {model_path_check}")

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_stem = Path(args.model).stem

    # ------------------------------------------------------------------
    # Standard evaluation (no wind)
    # ------------------------------------------------------------------
    summary_no_wind, results_no_wind = run_evaluation(
        model, args, wind_x=0.0, wind_y=0.0, label="(no wind)"
    )

    all_summaries = {"no_wind": summary_no_wind}

    # ------------------------------------------------------------------
    # Wind sweep evaluation (optional)
    # ------------------------------------------------------------------
    if args.wind_sweep:
        wind_conditions = [
            (3.0, 0.0,  "wind_3ms_x"),
            (-3.0, 0.0, "wind_3ms_xneg"),
            (0.0, 3.0,  "wind_3ms_y"),
            (5.0, 5.0,  "wind_5ms_diag"),
            (8.0, 0.0,  "wind_8ms_x"),
        ]
        for wx, wy, label in wind_conditions:
            summary, _ = run_evaluation(
                model, args, wind_x=wx, wind_y=wy, label=f"({label})"
            )
            all_summaries[label] = summary

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_path = save_dir / f"{model_stem}_summary.json"
    with open(results_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary saved to: {results_path}")

    if args.save_trajectories:
        traj_path = save_dir / f"{model_stem}_trajectories.npy"
        trajs = [np.array(r["trajectory"]) for r in results_no_wind if r["trajectory"]]
        #np.save(str(traj_path), np.array(trajs, dtype=object))
        print(f"Trajectories saved to: {traj_path}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            import matplotlib.cm as cm

            trajs_valid = [r for r in results_no_wind
                           if r["trajectory"] and not r["diverged"]]
            trajs_diverged = [r for r in results_no_wind
                              if r["trajectory"] and r["diverged"]]

            errors = [r["landing_error"] for r in trajs_valid]
            max_err = max(errors) if errors else 1.0
            cmap = cm.RdYlGn_r

            # ------------------------------------------------------------------
            # Figure 1: 3D trajectory plot
            # ------------------------------------------------------------------
            fig = plt.figure(figsize=(14, 10))
            ax3d = fig.add_subplot(111, projection='3d')

            for r in trajs_valid:
                traj = np.array(r["trajectory"])
                x   = traj[:, 0]
                y   = traj[:, 1]
                alt = -traj[:, 2]
                color = cmap(r["landing_error"] / max_err)
                ax3d.plot(x, y, alt, color=color, alpha=0.5, linewidth=0.9)
                # Start marker (circle at initial altitude)
                ax3d.scatter(x[0], y[0], alt[0], color=color, s=40,
                             marker='o', alpha=0.9, edgecolors='k', linewidths=0.5)
                # Landing marker (triangle on ground)
                ax3d.scatter(x[-1], y[-1], 0, color=color, s=50,
                             marker='v', alpha=0.9, edgecolors='k', linewidths=0.5)

            for r in trajs_diverged:
                traj = np.array(r["trajectory"])
                ax3d.plot(traj[:, 0], traj[:, 1], -traj[:, 2],
                          color='grey', alpha=0.3, linewidth=0.6, linestyle='--')

            ax3d.scatter([args.target_x], [args.target_y], [0],
                         c='red', s=300, marker='*', zorder=10,
                         label="Target (0, 0)", depthshade=False)

            theta_c = np.linspace(0, 2 * np.pi, 120)
            ax3d.plot(args.target_x + args.success_radius * np.cos(theta_c),
                      args.target_y + args.success_radius * np.sin(theta_c),
                      np.zeros(120), 'r--', linewidth=1.5, alpha=0.7,
                      label=f"{args.success_radius:.0f}m radius")

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_err))
            sm.set_array([])
            fig.colorbar(sm, ax=ax3d, shrink=0.5, pad=0.1, label="Landing Error (m)")

            ax3d.set_xlabel("X (m)")
            ax3d.set_ylabel("Y (m)")
            ax3d.set_zlabel("Altitude (m)")
            n_success = sum(e <= args.success_radius for e in errors)
            start_dists = [r.get("start_dist", 0.0) for r in trajs_valid]
            ax3d.set_title(
                f"3D Parafoil Trajectories  (target: ({args.target_x:.0f}, {args.target_y:.0f}))\n"
                f"n={len(trajs_valid)} landed  |  mean error={np.mean(errors):.1f}m  |  "
                f"success={n_success}/{len(trajs_valid)}  |  "
                f"avg start dist={np.mean(start_dists):.0f}m"
            )
            from matplotlib.lines import Line2D
            ax3d.legend(handles=[
                Line2D([0],[0], marker="o", color="w", markerfacecolor="grey",
                       markersize=8, markeredgecolor="k", label="Start"),
                Line2D([0],[0], marker="v", color="w", markerfacecolor="grey",
                       markersize=8, markeredgecolor="k", label="Landing"),
                Line2D([0],[0], marker="*", color="w", markerfacecolor="red",
                       markersize=12, label="Target"),
            ], loc="upper left")
            ax3d.view_init(elev=25, azim=-60)
            plt.tight_layout()
            plot_path_3d = save_dir / f"{model_stem}_3d_trajectories.png"
            #plt.savefig(str(plot_path_3d), dpi=150)
            print(f"3D plot saved to: {plot_path_3d}")

            # ------------------------------------------------------------------
            # Figure 2: Landing scatter + error histogram
            # ------------------------------------------------------------------
            fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

            land_x = [np.array(r["trajectory"])[-1, 0] for r in trajs_valid]
            land_y = [np.array(r["trajectory"])[-1, 1] for r in trajs_valid]
            sc = axes2[0].scatter(land_x, land_y, c=errors, cmap='RdYlGn_r',
                                  s=60, alpha=0.8, vmin=0, vmax=max_err, zorder=3)
            fig2.colorbar(sc, ax=axes2[0], label="Landing Error (m)")
            axes2[0].scatter([args.target_x], [args.target_y],
                             c='red', s=300, marker='*', zorder=5, label="Target")
            axes2[0].add_patch(plt.Circle(
                (args.target_x, args.target_y), args.success_radius,
                fill=False, color='red', linestyle='--',
                label=f"{args.success_radius:.0f}m radius"))
            axes2[0].set_xlabel("X (m)"); axes2[0].set_ylabel("Y (m)")
            axes2[0].set_title("Landing Scatter (top-down)")
            axes2[0].set_aspect("equal"); axes2[0].legend(); axes2[0].grid(True, alpha=0.4)

            axes2[1].hist(errors, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
            axes2[1].axvline(np.mean(errors), color='red', linewidth=2,
                             label=f"Mean {np.mean(errors):.1f}m")
            axes2[1].axvline(np.median(errors), color='orange', linewidth=2,
                             linestyle='--', label=f"Median {np.median(errors):.1f}m")
            axes2[1].axvline(args.success_radius, color='green', linewidth=2,
                             linestyle=':', label=f"Success {args.success_radius:.0f}m")
            axes2[1].set_xlabel("Landing Error (m)"); axes2[1].set_ylabel("Count")
            axes2[1].set_title("Landing Error Distribution")
            axes2[1].legend(); axes2[1].grid(True, alpha=0.4)

            fig2.suptitle(f"Parafoil PPO Evaluation: {model_stem}", fontsize=13)
            plt.tight_layout()
            plot_path_2d = save_dir / f"{model_stem}_landing_scatter.png"
            #plt.savefig(str(plot_path_2d), dpi=150)
            print(f"Scatter plot saved to: {plot_path_2d}")
            plt.show()

        except ImportError:
            print("matplotlib not available - skipping plots")


if __name__ == "__main__":
    main()