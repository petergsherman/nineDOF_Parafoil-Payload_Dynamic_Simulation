import numpy as np
import matplotlib.pyplot as plt

from nineDOF_Plant import plant
from python.nineDOF_Control import LQRHeadingController, make_control_function
from nineDOF_Parameters import systemParameters
from nineDOF_Atmosphere import TurbulenceMode, dynamicAtmosphere, staticAtmosphere


# -----------------------------------------------------------------------------
# User-tunable settings
# -----------------------------------------------------------------------------
T_FINAL = 1000.0
DT = 0.01
TARGET_LANDING_POINT = np.array([-1000.0, 500.0], dtype=float)
WIND_INTENSITY = "moderate"
SAVE_FIGURE = True
FIGURE_NAME = "heading_error_control_history.png"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def make_initial_state() -> np.ndarray:
    """Match the initial condition used in nineDOF_Main.py."""
    state0 = np.zeros(18)
    state0[0:3] = [0.0, 0.0, -1000.0]
    state0[3:6] = [0.0, 0.1, 0.0]
    state0[6:9] = [0.0, 0.1, 0.0]
    state0[9:12] = [10.0, 0.0, -0.5]
    return state0


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def run_case(use_wind: bool):
    """
    Run one simulation case and return the truncated histories needed for plotting.
    """
    params = systemParameters()
    if use_wind:
        atm = dynamicAtmosphere(
            turbulence_mode=TurbulenceMode.DRYDEN,
            turbulence_intensity=WIND_INTENSITY,
        )
        title = f"With wind ({WIND_INTENSITY})"
    else:
        atm = staticAtmosphere()
        title = "No wind"

    sim = plant(params, atm)
    state0 = make_initial_state()

    controller = LQRHeadingController(TARGET_LANDING_POINT, plant_obj=sim)
    control = make_control_function(controller)

    print(f"Running case: {title} ...")
    times, states = sim.run_simulation(state0, T_FINAL, DT, control)

    pos_xy = states[:, 0:2]
    to_target = TARGET_LANDING_POINT - pos_xy
    distance_to_target = np.linalg.norm(to_target, axis=1)

    desired_heading = np.arctan2(to_target[:, 1], to_target[:, 0])
    actual_heading = states[:, 5]

    # Plot heading error directly so the desired heading is represented by y = 0.
    heading_error = wrap_to_pi(actual_heading - desired_heading)

    # Reconstruct the control history from the controller law.
    # For this LQR controller, differential brake dA = deltaR - deltaL is the
    # most meaningful single control-history signal to plot.
    control_history = np.zeros_like(times)
    for i, state in enumerate(states):
        deltaL, deltaR, _ = controller.computeControl(state)
        control_history[i] = deltaR - deltaL

    return {
        "title": title,
        "times": times,
        "heading_error": heading_error,
        "control_history": control_history,
    }


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def align_yaxis_zeros(ax_left, ax_right):
    """Expand both y-axes symmetrically so their zeros are vertically aligned."""
    def symmetric_limits(ax):
        lo, hi = ax.get_ylim()
        mag = max(abs(lo), abs(hi))
        ax.set_ylim(-mag, mag)

    symmetric_limits(ax_left)
    symmetric_limits(ax_right)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    no_wind = run_case(use_wind=False)
    with_wind = run_case(use_wind=True)

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

    if SAVE_FIGURE:
        fig.savefig(FIGURE_NAME, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {FIGURE_NAME}")

    plt.show()


if __name__ == "__main__":
    main()