"""
evaluation/evaluate_ppo.py
===========================
Evaluation script for trained parafoil PPO models.

Usage:
    # Static atmosphere (no wind)
    python evaluate_ppo.py --model ../checkpoints/run/best_model --n_episodes 50 --plot

    # Dryden turbulence
    python evaluate_ppo.py --model ../checkpoints/run/best_model --n_episodes 50 --plot --atmosphere dryden --turbulence moderate

    # Fixed static wind
    python evaluate_ppo.py --model ../checkpoints/run/best_model --n_episodes 50 --plot --atmosphere static --wind_x 5.0 --wind_y 3.0

    # Wind sweep (static wind at multiple headings)
    python evaluate_ppo.py --model ../checkpoints/run/best_model --n_episodes 30 --plot --wind_sweep
"""

# ── Path setup (must be first) ────────────────────────────────────────────
import sys, os, argparse, json
from pathlib import Path
_rl_root = Path(__file__).resolve().parent.parent
if str(_rl_root) not in sys.path:
    sys.path.insert(0, str(_rl_root))
import setup_paths
# ─────────────────────────────────────────────────────────────────────────

import numpy as np
from typing import List, Dict

from parafoil_gym_env import ParafoilGymEnv
from atmosphere_bridge import AtmosphereBridge, ATMO_STATIC, ATMO_DRYDEN, ATMO_SIMPLE, ATMO_WIND

try:
    from stable_baselines3 import PPO
except ImportError:
    raise ImportError("Install stable-baselines3: pip install stable-baselines3")


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained parafoil PPO policy")

    parser.add_argument("--model",           type=str,   required=True)
    parser.add_argument("--n_episodes",      type=int,   default=50)
    parser.add_argument("--success_radius",  type=float, default=50.0)
    parser.add_argument("--target_x",        type=float, default=0.0)
    parser.add_argument("--target_y",        type=float, default=0.0)
    parser.add_argument("--dt_action",       type=float, default=0.1)
    parser.add_argument("--dt_physics",      type=float, default=0.01)
    parser.add_argument("--max_episode_time",type=float, default=1200.0)
    parser.add_argument("--seed",            type=int,   default=0)

    # Atmosphere
    parser.add_argument("--atmosphere",      type=str,   default="static",
                        choices=["static", "dryden", "simple", "wind"],
                        help="Atmosphere model. 'static' = no wind unless --wind_x/y set.")
    parser.add_argument("--turbulence",      type=str,   default="moderate",
                        choices=["light", "moderate", "severe"])
    # Static wind components (only used when --atmosphere static)
    parser.add_argument("--wind_x",          type=float, default=0.0,
                        help="Constant wind X (m/s). Only used with --atmosphere static.")
    parser.add_argument("--wind_y",          type=float, default=0.0,
                        help="Constant wind Y (m/s). Only used with --atmosphere static.")

    # Start positions
    parser.add_argument("--start_x",         type=float, default=None)
    parser.add_argument("--start_y",         type=float, default=None)
    parser.add_argument("--start_alt",       type=float, default=None)
    parser.add_argument("--start_radius",    type=float, default=600.0)

    # Wind sweep (static wind only)
    parser.add_argument("--wind_sweep",      action="store_true",
                        help="Run multiple static wind conditions (overrides --atmosphere for sweep)")

    # Output
    parser.add_argument("--save_dir",        type=str,   default="eval_results/")
    parser.add_argument("--save_trajectories", action="store_true")
    parser.add_argument("--plot",            action="store_true")

    return parser.parse_args()


# =============================================================================
# Start state generator
# =============================================================================

def make_start_state(args, rng: np.random.Generator, ep: int) -> np.ndarray:
    state = np.zeros(18)

    if args.start_x is not None and args.start_y is not None:
        x, y = args.start_x, args.start_y
    else:
        angle    = rng.uniform(0, 2 * np.pi)
        distance = rng.uniform(args.start_radius * 0.3, args.start_radius)
        x = args.target_x + distance * np.cos(angle)
        y = args.target_y + distance * np.sin(angle)

    alt = args.start_alt if args.start_alt is not None else rng.uniform(400.0, 700.0)

    state[0]  = x
    state[1]  = y
    state[2]  = -alt
    heading   = rng.uniform(-np.pi, np.pi)
    state[3]  = rng.uniform(-0.02, 0.02)
    state[4]  = rng.uniform(-0.05, 0.05)
    state[5]  = heading
    state[6]  = rng.uniform(-0.02, 0.02)
    state[7]  = rng.uniform(-0.05, 0.05)
    state[8]  = heading
    state[9]  = 10.0 + rng.uniform(-0.5, 0.5)
    state[10] = 0.0
    state[11] = -0.5 + rng.uniform(-0.1, 0.1)
    return state


# =============================================================================
# Single episode runner
# =============================================================================

def run_episode(
    model,
    env: ParafoilGymEnv,
    atm_bridge: AtmosphereBridge,
    start_state: np.ndarray,
    ep_seed: int,
    save_trajectory: bool = False,
) -> Dict:
    """
    Run one evaluation episode.

    Uses env.reset(seed=) so the gym wrapper's atmosphere bridge is properly
    re-seeded, then immediately overrides the C++ state with start_state.
    This ensures dynamic atmospheres (Dryden etc.) are active from step 1.
    """
    # Reset via gym wrapper so AtmosphereBridge.reset() is called
    obs, info = env.reset(seed=ep_seed)

    # Now override the C++ state with our chosen start position
    obs_arr = env._env.reset_fixed(start_state)
    obs = np.array(obs_arr, dtype=np.float32)

    # Push initial atmosphere state into C++ env
    alt      = float(-start_state[2])
    airspeed = float(np.linalg.norm(start_state[9:12]))
    atm_bridge.step(env._env, 0.0, alt, airspeed)

    # Capture initial state for trajectory
    trajectory = [env.get_state().copy()] if save_trajectory else None

    total_reward = 0.0
    step         = 0
    done         = False
    actions      = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step         += 1
        actions.append(action.copy())

        if save_trajectory:
            trajectory.append(env.get_state().copy())

        done = terminated or truncated

    actions = np.array(actions)
    return {
        "landing_error":     info.get("distance_to_target", float("nan")),
        "total_reward":      total_reward,
        "episode_steps":     step,
        "episode_time":      env.get_time(),
        "hit_ground":        info.get("hit_ground", False),
        "diverged":          info.get("diverged",   False),
        "mean_action_l":     float(np.mean(actions[:, 0])),
        "mean_action_r":     float(np.mean(actions[:, 1])),
        "std_action_l":      float(np.std(actions[:, 0])),
        "std_action_r":      float(np.std(actions[:, 1])),
        "mean_differential": float(np.mean(np.abs(actions[:, 1] - actions[:, 0]))),
        "trajectory":        trajectory,
        "start_x":           float(start_state[0]),
        "start_y":           float(start_state[1]),
        "start_alt":         float(-start_state[2]),
        "start_dist":        float(np.sqrt((start_state[0]-0)**2 + (start_state[1]-0)**2)),
    }


# =============================================================================
# Summary statistics
# =============================================================================

def compute_summary(results: List[Dict], success_radius: float) -> Dict:
    errors     = [r["landing_error"] for r in results if not r["diverged"]]
    rewards    = [r["total_reward"]  for r in results]
    times      = [r["episode_time"]  for r in results]
    n_diverged = sum(1 for r in results if r["diverged"])
    n_success  = sum(1 for r in results
                     if not r["diverged"] and r["landing_error"] <= success_radius)
    n_total    = len(results)
    return {
        "n_episodes":              n_total,
        "success_radius_m":        success_radius,
        "n_success":               n_success,
        "n_diverged":              n_diverged,
        "success_rate":            n_success / n_total if n_total else 0.0,
        "diverge_rate":            n_diverged / n_total if n_total else 0.0,
        "mean_landing_error_m":    float(np.mean(errors))           if errors else float("nan"),
        "std_landing_error_m":     float(np.std(errors))            if errors else float("nan"),
        "median_landing_error_m":  float(np.median(errors))         if errors else float("nan"),
        "p90_landing_error_m":     float(np.percentile(errors, 90)) if errors else float("nan"),
        "mean_episode_reward":     float(np.mean(rewards)),
        "std_episode_reward":      float(np.std(rewards)),
        "mean_episode_time_s":     float(np.mean(times)),
        "mean_differential_brake": float(np.mean([r["mean_differential"] for r in results])),
    }


def print_summary(summary: Dict, label: str = ""):
    print(f"\n{'='*60}")
    print(f"  Evaluation Results {label}")
    print(f"{'='*60}")
    print(f"  Episodes:          {summary['n_episodes']}")
    print(f"  Success radius:    {summary['success_radius_m']:.0f} m")
    print(f"  Success rate:      {summary['success_rate']*100:.1f}%"
          f"  ({summary['n_success']}/{summary['n_episodes']})")
    print(f"  Diverge rate:      {summary['diverge_rate']*100:.1f}%"
          f"  ({summary['n_diverged']}/{summary['n_episodes']})")
    print(f"  Landing error:")
    print(f"    Mean:            {summary['mean_landing_error_m']:.1f} m")
    print(f"    Std:             {summary['std_landing_error_m']:.1f} m")
    print(f"    Median:          {summary['median_landing_error_m']:.1f} m")
    print(f"    P90:             {summary['p90_landing_error_m']:.1f} m")
    print(f"  Mean episode time: {summary['mean_episode_time_s']:.1f} s")
    print(f"  Mean diff brake:   {summary['mean_differential_brake']:.3f}")
    print(f"{'='*60}\n")


# =============================================================================
# Full evaluation run
# =============================================================================

def run_evaluation(
    model,
    args,
    atm_mode:  str   = None,
    wind_x:    float = 0.0,
    wind_y:    float = 0.0,
    label:     str   = "",
) -> Dict:
    """
    Run a full evaluation with a given atmosphere and optional static wind.
    atm_mode defaults to args.atmosphere if not specified.
    """
    if atm_mode is None:
        atm_mode = args.atmosphere

    # Build the atmosphere label for printing
    if atm_mode == ATMO_STATIC:
        if wind_x != 0.0 or wind_y != 0.0:
            atm_desc = f"static wind ({wind_x:.1f}, {wind_y:.1f}) m/s"
        else:
            atm_desc = "static (no wind)"
    else:
        atm_desc = f"{atm_mode} ({args.turbulence})"

    print(f"\nRunning {args.n_episodes} episodes  |  atmosphere: {atm_desc}  {label}")

    # Create gym env with the correct atmosphere mode
    env = ParafoilGymEnv(
        target               = (args.target_x, args.target_y),
        dt_physics           = args.dt_physics,
        dt_action            = args.dt_action,
        max_episode_time     = args.max_episode_time,
        domain_random        = False,
        atmosphere_mode      = atm_mode,
        turbulence_intensity = args.turbulence,
    )

    # For static atmosphere, apply fixed wind now
    if atm_mode == ATMO_STATIC:
        env._env.set_wind(wind_x, wind_y)

    # Create a standalone atmosphere bridge for per-episode reset
    # (separate from the one inside env so we can control seeding precisely)
    atm_bridge = AtmosphereBridge(mode=atm_mode, intensity=args.turbulence)

    if args.start_x is None:
        print(f"  Start: random within {args.start_radius:.0f}m of "
              f"({args.target_x:.0f}, {args.target_y:.0f})")
    else:
        print(f"  Start: fixed ({args.start_x:.0f}, {args.start_y:.0f})")

    rng     = np.random.default_rng(args.seed)
    results = []

    for ep in range(args.n_episodes):
        # Fresh atmosphere seed each episode (new wind profile for dynamic modes)
        atm_bridge.reset(seed=args.seed * 1000 + ep)

        start_state = make_start_state(args, rng, ep)
        start_dist  = float(np.sqrt(
            (start_state[0] - args.target_x)**2 +
            (start_state[1] - args.target_y)**2))

        result = run_episode(
            model       = model,
            env         = env,
            atm_bridge  = atm_bridge,
            start_state = start_state,
            ep_seed     = args.seed + ep,
            save_trajectory = args.save_trajectories or args.plot,
        )
        result["start_dist"] = start_dist

        # Log wind for dynamic modes
        if atm_mode != ATMO_STATIC:
            vx, vy, _ = atm_bridge.get_current_wind()
            result["wind_speed"] = float(np.sqrt(vx**2 + vy**2))

        status = ("LAND" if result["hit_ground"]
                  else "DIV" if result["diverged"] else "TIME")
        wind_str = (f"  wind={result.get('wind_speed', 0.0):.1f}m/s"
                    if atm_mode != ATMO_STATIC else "")
        print(f"  Ep {ep+1:3d}: {status}  "
              f"start=({start_state[0]:6.0f},{start_state[1]:6.0f})  "
              f"error={result['landing_error']:6.1f}m  "
              f"t={result['episode_time']:.0f}s{wind_str}")
        results.append(result)

    env.close()
    summary = compute_summary(results, args.success_radius)
    print_summary(summary, label=f"{label}  [{atm_desc}]")
    return summary, results


# =============================================================================
# Plotting (unchanged from before)
# =============================================================================

def plot_results(results_no_wind, args, model_stem, save_dir):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
        import matplotlib.cm as cm
        from matplotlib.lines import Line2D

        trajs_valid    = [r for r in results_no_wind if r["trajectory"] and not r["diverged"]]
        trajs_diverged = [r for r in results_no_wind if r["trajectory"] and r["diverged"]]

        if not trajs_valid:
            print("No valid trajectories to plot.")
            return

        errors  = [r["landing_error"] for r in trajs_valid]
        max_err = max(errors) if errors else 1.0
        cmap    = cm.RdYlGn_r

        # ── Figure 1: 3D trajectories ──────────────────────────────────────
        fig = plt.figure(figsize=(14, 10))
        ax3d = fig.add_subplot(111, projection="3d")

        for r in trajs_valid:
            traj  = np.array(r["trajectory"])
            color = cmap(r["landing_error"] / max_err)
            ax3d.plot(traj[:, 0], traj[:, 1], -traj[:, 2],
                      color=color, alpha=0.5, linewidth=0.9)
            ax3d.scatter(traj[0, 0], traj[0, 1], -traj[0, 2],
                         color=color, s=40, marker="o", alpha=0.9,
                         edgecolors="k", linewidths=0.5)
            ax3d.scatter(traj[-1, 0], traj[-1, 1], 0,
                         color=color, s=50, marker="v", alpha=0.9,
                         edgecolors="k", linewidths=0.5)

        for r in trajs_diverged:
            traj = np.array(r["trajectory"])
            ax3d.plot(traj[:, 0], traj[:, 1], -traj[:, 2],
                      color="grey", alpha=0.3, linewidth=0.6, linestyle="--")

        ax3d.scatter([args.target_x], [args.target_y], [0],
                     c="red", s=300, marker="*", zorder=10, depthshade=False)
        theta_c = np.linspace(0, 2 * np.pi, 120)
        ax3d.plot(args.target_x + args.success_radius * np.cos(theta_c),
                  args.target_y + args.success_radius * np.sin(theta_c),
                  np.zeros(120), "r--", linewidth=1.5, alpha=0.7)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_err))
        sm.set_array([])
        fig.colorbar(sm, ax=ax3d, shrink=0.5, pad=0.1, label="Landing Error (m)")

        n_success   = sum(e <= args.success_radius for e in errors)
        start_dists = [r.get("start_dist", 0.0) for r in trajs_valid]

        # Build atmosphere description for title
        if args.atmosphere == "static":
            atm_title = (f"static wind ({args.wind_x:.1f},{args.wind_y:.1f})m/s"
                         if (args.wind_x or args.wind_y) else "static (no wind)")
        else:
            atm_title = f"{args.atmosphere} ({args.turbulence})"

        ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Altitude (m)")
        ax3d.set_title(
            f"PPO 3D Trajectories  |  atm: {atm_title}\n"
            f"n={len(trajs_valid)} landed  |  mean error={np.mean(errors):.1f}m  |  "
            f"success={n_success}/{len(trajs_valid)}  |  "
            f"avg start dist={np.mean(start_dists):.0f}m"
        )
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
        p3d = save_dir / f"{model_stem}_3d_trajectories.png"
        plt.savefig(str(p3d), dpi=150)
        print(f"3D plot saved to: {p3d}")

        # ── Figure 2: scatter + histogram ─────────────────────────────────
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
        land_x = [np.array(r["trajectory"])[-1, 0] for r in trajs_valid]
        land_y = [np.array(r["trajectory"])[-1, 1] for r in trajs_valid]
        sc = axes2[0].scatter(land_x, land_y, c=errors, cmap="RdYlGn_r",
                              s=60, alpha=0.8, vmin=0, vmax=max_err, zorder=3)
        fig2.colorbar(sc, ax=axes2[0], label="Landing Error (m)")
        axes2[0].scatter([args.target_x], [args.target_y],
                         c="red", s=300, marker="*", zorder=5, label="Target")
        axes2[0].add_patch(plt.Circle(
            (args.target_x, args.target_y), args.success_radius,
            fill=False, color="red", linestyle="--",
            label=f"{args.success_radius:.0f}m radius"))
        axes2[0].set_xlabel("X (m)"); axes2[0].set_ylabel("Y (m)")
        axes2[0].set_title("Landing Scatter"); axes2[0].set_aspect("equal")
        axes2[0].legend(); axes2[0].grid(True, alpha=0.4)

        axes2[1].hist(errors, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
        axes2[1].axvline(np.mean(errors), color="red", linewidth=2,
                         label=f"Mean {np.mean(errors):.1f}m")
        axes2[1].axvline(np.median(errors), color="orange", linewidth=2,
                         linestyle="--", label=f"Median {np.median(errors):.1f}m")
        axes2[1].axvline(args.success_radius, color="green", linewidth=2,
                         linestyle=":", label=f"Success {args.success_radius:.0f}m")
        axes2[1].set_xlabel("Landing Error (m)"); axes2[1].set_ylabel("Count")
        axes2[1].set_title("Landing Error Distribution")
        axes2[1].legend(); axes2[1].grid(True, alpha=0.4)

        fig2.suptitle(f"PPO Evaluation  |  {atm_title}", fontsize=13)
        plt.tight_layout()
        p2d = save_dir / f"{model_stem}_landing_scatter.png"
        plt.savefig(str(p2d), dpi=150)
        print(f"Scatter plot saved to: {p2d}")
        plt.show()

    except ImportError:
        print("matplotlib not available - skipping plots")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    model_path = args.model
    if not model_path.endswith(".zip"):
        model_path_check = Path(model_path + ".zip")
    else:
        model_path_check = Path(model_path)
    if not model_path_check.exists():
        raise FileNotFoundError(f"Model not found: {model_path_check}")

    print(f"\nLoading model: {model_path}")
    print(f"Atmosphere:    {args.atmosphere}  (intensity: {args.turbulence})")
    model = PPO.load(model_path)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_stem = Path(args.model).stem

    all_summaries = {}

    # ── Primary evaluation using --atmosphere ────────────────────────────
    summary_primary, results_primary = run_evaluation(
        model,
        args,
        atm_mode = args.atmosphere,
        wind_x   = args.wind_x,
        wind_y   = args.wind_y,
        label    = "(primary)",
    )
    all_summaries["primary"] = summary_primary

    # ── Optional static wind sweep ───────────────────────────────────────
    if args.wind_sweep:
        wind_conditions = [
            (0.0,  0.0,  "no_wind"),
            (3.0,  0.0,  "wind_3ms_x"),
            (-3.0, 0.0,  "wind_3ms_xneg"),
            (0.0,  3.0,  "wind_3ms_y"),
            (5.0,  5.0,  "wind_5ms_diag"),
            (8.0,  0.0,  "wind_8ms_x"),
        ]
        for wx, wy, wlabel in wind_conditions:
            wsummary, _ = run_evaluation(
                model, args,
                atm_mode = ATMO_STATIC,
                wind_x   = wx,
                wind_y   = wy,
                label    = f"({wlabel})",
            )
            all_summaries[wlabel] = wsummary

    # ── Save summary ─────────────────────────────────────────────────────
    results_path = save_dir / f"{model_stem}_summary.json"
    with open(results_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary saved to: {results_path}")

    if args.save_trajectories:
        traj_path = save_dir / f"{model_stem}_trajectories.npy"
        trajs = [np.array(r["trajectory"]) for r in results_primary if r["trajectory"]]
        np.save(str(traj_path), np.array(trajs, dtype=object))
        print(f"Trajectories saved to: {traj_path}")

    if args.plot:
        plot_results(results_primary, args, model_stem, save_dir)


if __name__ == "__main__":
    main()