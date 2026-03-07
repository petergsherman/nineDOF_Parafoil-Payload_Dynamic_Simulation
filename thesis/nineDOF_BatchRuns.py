import numpy as np
import matplotlib.pyplot as plt

from nineDOF_Plant import plant
from nineDOF_Control import LQRHeadingController, make_control_function
from nineDOF_Parameters import systemParameters
from nineDOF_Atmosphere import TurbulenceMode, dynamicAtmosphere

try:
    from scipy.stats import gaussian_kde
    HAS_KDE = True
except ImportError:
    HAS_KDE = False


def random_start_xy(rng, limit=1500.0, min_radius=1000.0):
    """
    Random starting position within ±limit, but at least min_radius from origin.
    """
    while True:
        x = rng.uniform(-limit, limit)
        y = rng.uniform(-limit, limit)

        if np.hypot(x, y) >= min_radius:
            return x, y


def run_batch():
    rng = np.random.default_rng()

    num_runs = 10
    target = np.array([0.0, 0.0])

    t_final = 1000.0
    dt = 0.01
    success_radius = 200.0  # successful if final landing point is within 200 m of target

    params = systemParameters()

    all_trajectories = []
    all_landings = []
    successful_landings = []

    for i in range(num_runs):
        atm = dynamicAtmosphere(
            turbulence_mode=TurbulenceMode.DRYDEN,
            turbulence_intensity="moderate"
        )

        sim = plant(params, atm)

        state0 = np.zeros(18)

        x0, y0 = random_start_xy(rng)

        state0[0] = x0
        state0[1] = y0
        state0[2] = -1000.0  # start at 1000 m altitude

        state0[3:6] = [0.0, 0.1, 0.0]
        state0[6:9] = [0.0, 0.1, 0.0]
        state0[9:12] = [10.0, 0.0, -0.5]

        controller = LQRHeadingController(
            targetLandingLocation=target,
            plant_obj=sim,
        )
        control = make_control_function(controller)

        print(f"Run {i+1:02d}: start = ({x0:.1f}, {y0:.1f})")

        times, states = sim.run_simulation(state0, t_final, dt, control)

        all_trajectories.append(states[:, 0:2])

        landing_xy = states[-1, 0:2]
        all_landings.append(landing_xy)

        landing_error = np.linalg.norm(landing_xy - target)
        if landing_error <= success_radius:
            successful_landings.append(landing_xy)

    all_landings = np.array(all_landings)
    successful_landings = (
        np.array(successful_landings) if len(successful_landings) > 0 else np.empty((0, 2))
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12), constrained_layout=True)

    # =========================================================
    # Top Plot: Top-down trajectories
    # =========================================================
    for traj in all_trajectories:
        ax1.plot(traj[:, 0], traj[:, 1], linewidth=1.0)

    ax1.scatter(all_landings[:, 0], all_landings[:, 1], s=25, alpha=0.6, label="All landing points")

    if len(successful_landings) > 0:
        ax1.scatter(
            successful_landings[:, 0],
            successful_landings[:, 1],
            s=35,
            marker="o",
            label=f"Successful landings (≤ {success_radius:.0f} m)"
        )

    circle1 = plt.Circle((target[0], target[1]), success_radius, fill=False, linestyle="--")
    ax1.add_patch(circle1)
    ax1.scatter(target[0], target[1], marker="x", s=120, label="Target")

    ax1.set_title("Top-Down Trajectories")
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.grid(True)
    ax1.axis("equal")
    ax1.legend()

    # =========================================================
    # Bottom Plot: Successful landing locations + soft density blob
    # =========================================================
    ax2.set_title(f"Successful Landing Locations")
    ax2.set_xlabel("X Position (m)")
    ax2.set_ylabel("Y Position (m)")
    ax2.grid(True)
    ax2.axis("equal")

    circle2 = plt.Circle((target[0], target[1]), success_radius, fill=False, linestyle="--")
    ax2.add_patch(circle2)
    ax2.scatter(target[0], target[1], marker="x", s=120, label="Target")

    if len(successful_landings) > 0:
        # Scatter successful points
        ax2.scatter(
            successful_landings[:, 0],
            successful_landings[:, 1],
            s=45,
            alpha=0.9,
            label="Successful landing points"
        )

        # Soft amorphous density highlighting if enough points and scipy available
        if HAS_KDE and len(successful_landings) >= 3:
            x = successful_landings[:, 0]
            y = successful_landings[:, 1]

            kde = gaussian_kde(np.vstack([x, y]))

            pad = 40.0
            xmin = min(-success_radius, np.min(x) - pad)
            xmax = max(success_radius, np.max(x) + pad)
            ymin = min(-success_radius, np.min(y) - pad)
            ymax = max(success_radius, np.max(y) + pad)

            xx, yy = np.meshgrid(
                np.linspace(xmin, xmax, 200),
                np.linspace(ymin, ymax, 200)
            )
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

            # Filled contours create the soft "blob" look
            # Mask very low density so the background stays white
            threshold = np.max(zz) * 0.05
            zz_masked = np.ma.masked_where(zz < threshold, zz)

            ax2.contourf(
                xx,
                yy,
                zz_masked,
                levels=6,
                alpha=0.35
            )

            ax2.contour(
                xx,
                yy,
                zz_masked,
                levels=4,
                linewidths=0.8,
                alpha=0.5
            )

        ax2.legend()
    else:
        ax2.text(
            0.5,
            0.5,
            f"No runs landed within {success_radius:.0f} m of the target",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12
        )

    print(f"\nSuccessful landings within {success_radius:.0f} m: {len(successful_landings)} / {num_runs}")

    plt.show()


if __name__ == "__main__":
    run_batch()