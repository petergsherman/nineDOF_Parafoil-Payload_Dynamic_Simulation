# nineDOF_BatchRuns.py
import os
import runpy
import numpy as np
import matplotlib.pyplot as plt

def main(n_runs: int = 10):
    # Path to your existing "main" script
    here = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.join(here, "nineDOF_Main.py")

    all_runs = []

    for i in range(n_runs):
        # Make each run distinct/reproducible
        np.random.seed(i)

        # Execute nineDOF_Main.py as a script, but monkeypatch plotting to no-op
        init_globals = {
            # If nineDOF_Main.py uses visualizeData.<...> already imported,
            # we can override the symbol after execution starts by providing a stub.
            # However, easiest is to provide a stub "visualizeData" up-front.
            "visualizeData": type(
                "visualizeDataStub",
                (),
                {
                    "plot_trajectory": staticmethod(lambda *args, **kwargs: None),
                    "plot_atmosphere": staticmethod(lambda *args, **kwargs: None),
                },
            )()
        }

        g = runpy.run_path(main_path, init_globals=init_globals)

        # Pull results out of the executed script's globals
        if "states" not in g or g["states"] is None:
            raise RuntimeError(f"Run {i} did not produce 'states' in nineDOF_Main.py globals.")
        states = g["states"]

        all_runs.append(states)

        print(f"[Batch] Completed run {i+1}/{n_runs}  (N={states.shape[0]} steps)")

    # --- Plot: top-down view (x vs y), different color each run, no legend ---
    fig, ax = plt.subplots()
    for states in all_runs:
        ax.plot(states[:, 0], states[:, 1], linewidth=1.5)  # x vs y

    ax.set_title("Top-Down Trajectories (10 runs)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    plt.show()


if __name__ == "__main__":
    main(n_runs=10)
