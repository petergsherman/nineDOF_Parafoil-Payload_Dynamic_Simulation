"""
plot_heading_error_ppo.py
=========================
Plot PPO heading-error convergence and control history using the
same graphing structure as plot_heading_error_control.py, while
loading and stepping neural-network policies the same way as
in evaluate_ppo.py.

Example usage:
    python plot_heading_error_ppo.py --model ../checkpoints/run/best_model
    python plot_heading_error_ppo.py --model ../checkpoints/run/best_model --turbulence severe
    python plot_heading_error_ppo.py --model ../checkpoints/run/best_model --save_name my_plot.png
"""

# ── Path setup (keep first, matching evaluate_ppo.py style) ──────────────────
import sys
from pathlib import Path

_rl_root = Path(__file__).resolve().parent.parent
if str(_rl_root) not in sys.path:
    sys.path.insert(0, str(_rl_root))

import setup_paths  # noqa: F401

# ── Standard imports ───────────────────────────────────────────────────────────
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from parafoil_gym_env import ParafoilGymEnv
from atmosphere_bridge import AtmosphereBridge, ATMO_STATIC, ATMO_DRYDEN

try:
    from stable_baselines3 import PPO
except ImportError as exc:
    raise ImportError("Install stable-baselines3: pip install stable-baselines3") from exc


# -----------------------------------------------------------------------------
# Defaults chosen to mirror the structure/style of plot_heading_error_control.py
# -----------------------------------------------------------------------------
T_FINAL_DEFAULT = 1000.0
DT_ACTION_DEFAULT = 0.1
DT_PHYSICS_DEFAULT = 0.01
TARGET_LANDING_POINT_DEFAULT = np.array([-500.0, 500.0], dtype=float)
WIND_INTENSITY_DEFAULT = "moderate"
SAVE_FIGURE_DEFAULT = True
FIGURE_NAME_DEFAULT = "heading_error_control_history_ppo.png"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot PPO heading error and control history with the same structure as plot_heading_error_control.py"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to PPO model (with or without .zip)")
    parser.add_argument("--target_x", type=float, default=float(TARGET_LANDING_POINT_DEFAULT[0]))
    parser.add_argument("--target_y", type=float, default=float(TARGET_LANDING_POINT_DEFAULT[1]))
    parser.add_argument("--t_final", type=float, default=T_FINAL_DEFAULT)
    parser.add_argument("--dt_action", type=float, default=DT_ACTION_DEFAULT)
    parser.add_argument("--dt_physics", type=float, default=DT_PHYSICS_DEFAULT)
    parser.add_argument("--turbulence", type=str, default=WIND_INTENSITY_DEFAULT,
                        choices=["light", "moderate", "severe"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_figure", action="store_true", default=SAVE_FIGURE_DEFAULT)
    parser.add_argument("--save_name", type=str, default=FIGURE_NAME_DEFAULT)
    parser.add_argument("--plot_3d", action="store_true",
                        help="Also make a simple 3D trajectory plot matching nineDOF_Visualization.py style")
    parser.add_argument("--plot_3d_case", type=str, default="with_wind",
                        choices=["no_wind", "with_wind"],
                        help="Which case to use for the 3D plot")
    parser.add_argument("--save_3d_name", type=str, default="trajectory_3d_ppo.png")
    return parser.parse_args()


def make_initial_state() -> np.ndarray:
    """Match the initial condition used in plot_heading_error_control.py."""
    state0 = np.zeros(18, dtype=float)
    state0[0:3] = [0.0, 0.0, -500.0]
    state0[3:6] = [0.0, 0.1, 0.0]
    state0[6:9] = [0.0, 0.1, 0.0]
    state0[9:12] = [10.0, 0.0, -0.5]
    return state0


def wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def align_yaxis_zeros(ax_left, ax_right):
    """Expand both y-axes symmetrically so their zeros are vertically aligned."""
    def symmetric_limits(ax):
        lo, hi = ax.get_ylim()
        mag = max(abs(lo), abs(hi))
        if mag == 0:
            mag = 1.0
        ax.set_ylim(-mag, mag)

    symmetric_limits(ax_left)
    symmetric_limits(ax_right)


# -----------------------------------------------------------------------------
# PPO episode runner
# -----------------------------------------------------------------------------
def run_case(model, target_xy, use_wind: bool, turbulence: str,
             t_final: float, dt_action: float, dt_physics: float, seed: int):
    """
    Run one PPO-controlled case and return histories in the exact spirit of
    plot_heading_error_control.py.

    - No-wind case  -> static atmosphere
    - With-wind case -> Dryden atmosphere
    """
    if use_wind:
        atm_mode = ATMO_DRYDEN
        title = f"With wind ({turbulence})"
    else:
        atm_mode = ATMO_STATIC
        title = "No wind"

    env = ParafoilGymEnv(
        target=(float(target_xy[0]), float(target_xy[1])),
        dt_physics=dt_physics,
        dt_action=dt_action,
        max_episode_time=t_final,
        domain_random=False,
        atmosphere_mode=atm_mode,
        turbulence_intensity=turbulence,
    )

    atm_bridge = AtmosphereBridge(mode=atm_mode, intensity=turbulence)
    start_state = make_initial_state()

    # Match evaluate_ppo.py reset style so atmosphere + env are initialized correctly.
    obs, _ = env.reset(seed=seed)
    obs_arr = env._env.reset_fixed(start_state)
    obs = np.array(obs_arr, dtype=np.float32)

    alt = float(-start_state[2])
    airspeed = float(np.linalg.norm(start_state[9:12]))
    atm_bridge.reset(seed=seed)
    atm_bridge.step(env._env, 0.0, alt, airspeed)

    times = [0.0]
    states = [env.get_state().copy()]
    control_history = [0.0]

    done = False
    step_idx = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=float).reshape(-1)

        # Same single control-history signal concept as the LQR plot:
        # dA = deltaR - deltaL
        if action.size >= 2:
            dA = float(action[1] - action[0])
        elif action.size == 1:
            dA = float(action[0])
        else:
            dA = 0.0

        obs, _, terminated, truncated, _ = env.step(action)
        step_idx += 1

        times.append(step_idx * dt_action)
        states.append(env.get_state().copy())
        control_history.append(dA)

        done = terminated or truncated

    env.close()

    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)
    control_history = np.asarray(control_history, dtype=float)

    pos_xy = states[:, 0:2]
    to_target = np.asarray(target_xy, dtype=float) - pos_xy
    desired_heading = np.arctan2(to_target[:, 1], to_target[:, 0])

    # Keep the exact heading convention from plot_heading_error_control.py.
    actual_heading = states[:, 5]
    heading_error = wrap_to_pi(actual_heading - desired_heading)

    return {
        "title": title,
        "times": times,
        "heading_error": heading_error,
        "control_history": control_history,
        "states": states,
    }


def plot_trajectory_3d(history: np.ndarray, title: str = "Parafoil Trajectory",
                       save_path: str | None = None):
    """
    Simple 3D trajectory plot matching the style of nineDOF_Visualization.py.

    history columns are assumed to be:
        x = North, y = East, z = Down
    and altitude is plotted as -Down.
    """
    north = history[:, 0]
    east = history[:, 1]
    down = history[:, 2]
    altitude = -down

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(east, north, altitude, label='Trajectory', linewidth=3)
    ax.scatter(east[0], north[0], altitude[0], c='green', s=100, marker='.', label='Start')

    ground_impact_idx = np.where(altitude <= 0)[0]
    if len(ground_impact_idx) > 0:
        impact_idx = ground_impact_idx[0]
    else:
        impact_idx = int(np.argmin(np.abs(altitude)))

    ax.scatter(east[impact_idx], north[impact_idx], altitude[impact_idx],
               c='red', s=100, marker='x', label='Ground Impact')

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title(title)

    max_range = np.array([
        east.max() - east.min(),
        north.max() - north.min(),
        altitude.max() - altitude.min(),
    ]).max() / 2.0
    if max_range == 0:
        max_range = 1.0

    mid_x = (east.max() + east.min()) * 0.5
    mid_y = (north.max() + north.min()) * 0.5
    mid_z = (altitude.max() + altitude.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D figure to {save_path}")

    plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    model_path = args.model
    if not model_path.endswith(".zip"):
        model_path_check = Path(model_path + ".zip")
    else:
        model_path_check = Path(model_path)

    if not model_path_check.exists():
        raise FileNotFoundError(f"Model not found: {model_path_check}")

    print(f"Loading PPO model: {model_path}")
    model = PPO.load(model_path)

    target_xy = np.array([args.target_x, args.target_y], dtype=float)

    no_wind = run_case(
        model=model,
        target_xy=target_xy,
        use_wind=False,
        turbulence=args.turbulence,
        t_final=args.t_final,
        dt_action=args.dt_action,
        dt_physics=args.dt_physics,
        seed=args.seed,
    )

    with_wind = run_case(
        model=model,
        target_xy=target_xy,
        use_wind=True,
        turbulence=args.turbulence,
        t_final=args.t_final,
        dt_action=args.dt_action,
        dt_physics=args.dt_physics,
        seed=args.seed + 1,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    ORANGE = "#E07B00"

    for ax, result in zip(axes, [no_wind, with_wind]):
        t = result["times"]
        heading_error_deg = np.degrees(result["heading_error"])
        control_hist = result["control_history"]

        ax2 = ax.twinx()

        ax.axhline(0.0, linestyle="--", linewidth=1.5)
        ax.plot(t, heading_error_deg, linewidth=2.0)
        ax2.plot(t, control_hist, linewidth=1.8, color=ORANGE)

        align_yaxis_zeros(ax, ax2)

        ax.set_ylabel("Heading error [deg]")
        ax2.set_ylabel("Control history, dA", color=ORANGE)
        ax2.tick_params(axis="y", colors=ORANGE)
        ax2.spines["right"].set_color(ORANGE)

        ax.set_title(result["title"])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Heading-error convergence and control history", fontsize=14)
    fig.tight_layout()

    if args.save_figure:
        out_path = Path(args.save_name)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {out_path}")

    plt.show()

    if args.plot_3d:
        result_3d = no_wind if args.plot_3d_case == "no_wind" else with_wind
        plot_trajectory_3d(
            result_3d["states"],
            title=f"PPO Trajectory - {result_3d['title']}",
            save_path=args.save_3d_name if args.save_figure else None,
        )


if __name__ == "__main__":
    main()
