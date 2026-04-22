# plot_eval_comparison.py

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def moving_average(y, window=1):
    """Simple moving average. window=1 means no smoothing."""
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    y_valid = np.convolve(y, kernel, mode="valid")

    # Pad front so x and y stay aligned in length
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, y_valid])


def load_eval_file(npz_path):
    """
    Load one Stable-Baselines3 evaluations.npz file.

    Returns:
        timesteps: 1D array
        mean_rewards: 1D array
        std_rewards: 1D array
        mean_lengths: 1D array
    """
    data = np.load(npz_path)

    timesteps = data["timesteps"]              # shape: (n_evals,)
    results = data["results"]                  # shape: (n_evals, n_eval_episodes)
    ep_lengths = data["ep_lengths"]            # shape: (n_evals, n_eval_episodes)

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)
    mean_lengths = ep_lengths.mean(axis=1)

    return timesteps, mean_rewards, std_rewards, mean_lengths


def find_eval_files(runs_dir="runs"):
    """
    Finds all evaluation files under:
        runs/<run_name>/eval/evaluations.npz
    """
    runs_path = Path(runs_dir)
    return sorted(runs_path.glob("*/eval/evaluations.npz"))


def plot_eval_comparison(
    runs_dir="runs",
    smoothing_window=1,
    show_std=True,
    min_points=1,
):
    eval_files = find_eval_files(runs_dir)

    if not eval_files:
        raise FileNotFoundError(
            f"No evaluations.npz files found under {Path(runs_dir).resolve()}"
        )

    plotted = 0


    plt.figure(figsize=(10, 20))

    for npz_file in eval_files:
        run_name = npz_file.parent.parent.name

        try:
            timesteps, mean_rewards, std_rewards, mean_lengths = load_eval_file(npz_file)

            CLIP_VALUE = 2000  # adjust
            mean_rewards = np.clip(mean_rewards, -CLIP_VALUE, CLIP_VALUE)
            std_rewards = np.clip(std_rewards, -CLIP_VALUE, CLIP_VALUE)

            if len(timesteps) < min_points:
                print(f"Skipping {run_name}: only {len(timesteps)} eval points.")
                continue

            y_plot = moving_average(mean_rewards, smoothing_window)

            plt.plot(timesteps, y_plot, label=run_name)

            if show_std:
                lower = mean_rewards - std_rewards
                upper = mean_rewards + std_rewards
                plt.fill_between(timesteps, lower, upper, alpha=0.15)

            plotted += 1

            print(f"Loaded {run_name}")
            print(f"  File: {npz_file}")
            print(f"  Eval points: {len(timesteps)}")
            print(f"  Final mean reward: {mean_rewards[-1]:.3f}")
            print(f"  Best mean reward:  {np.max(mean_rewards):.3f}")
            print(f"  Final mean ep len: {mean_lengths[-1]:.3f}")
            print()

        except Exception as e:
            print(f"Skipping {run_name}: {e}")

    if plotted == 0:
        raise RuntimeError("No valid evaluation files could be plotted.")

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Evaluation Reward")
    plt.title("Evaluation Progress Comparison Across Runs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_eval_comparison(
        runs_dir="runs",
        smoothing_window=1,   # set to 2, 3, etc. for smoothing
        show_std=True,
        min_points=1,
    )