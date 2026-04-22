import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

# ============================================================
# USER INPUT
# Rows = policy/model sizes (6)
# Columns = training lengths (5)
# ============================================================

model_sizes = ['micro', 'tiny', 'small', 'medium', 'large', 'xlarge']
training_lengths = ['50k', '100K', '1M', '5M', '10M']

# ------------------------------------------------------------
# SUCCESS DATA (0–1 scale)
# rows = model sizes
# cols = training lengths
# ------------------------------------------------------------
success_data = [
    [0.55,	1,	0.93,	0.98,	1],
    [0.34,	0.81,	0.78,	1,	1],
    [1,	1,	1,	1,	1],
    [0.98,	0.99,	1,	0.97,	0.97],
    [0.99,	0.98,	1,	0.96,	0.97],
    [0.98,	0.97,	0.94,	0.96,	0.94],
]

# ------------------------------------------------------------
# MEAN DATA
# rows = model sizes
# cols = training lengths
# ------------------------------------------------------------
mean_data = [
    [48.8,	23.5,	34.6,	22.5,	22.9],
    [69.3,	34.1,	33.2,	30.3,	29.8],
    [20.9,	20.4,	20.2,	17.9,	19.6],
    [21.8,	20.8,	20.2,	21.7,	24.4],
    [21,	21.5,	20.6,	23.2,	25.5],
    [19.6,	22.5,	23.4,	24.3,	22.3],
]

def plot_success_and_mean(model_sizes, training_lengths, success_data, mean_data):
    success = np.array(success_data, dtype=float)
    mean = np.array(mean_data, dtype=float)

    # expected shape = (n_model_sizes, n_training_lengths) = (6, 5)
    expected_shape = (len(model_sizes), len(training_lengths))

    if success.shape != expected_shape:
        raise ValueError(f"success_data shape {success.shape} does not match expected {expected_shape}")

    if mean.shape != expected_shape:
        raise ValueError(f"mean_data shape {mean.shape} does not match expected {expected_shape}")

    success_percent = success * 100

    fig, axes = plt.subplots(2, 1, figsize=(9, 10))
    fig.suptitle("Dynamic Training | Static Evaluation", fontsize=16, fontweight='bold')

    # TOP: success %
    im1 = axes[0].imshow(
        success_percent,
        aspect='auto',
        cmap='RdYlGn',
        norm=PowerNorm(gamma=2.0, vmin=0, vmax=100)
    )

    axes[0].set_title("Success Percentage")
    axes[0].set_xlabel("Training Length")
    axes[0].set_ylabel("Model Size")
    axes[0].set_xticks(np.arange(len(training_lengths)))
    axes[0].set_xticklabels(training_lengths)
    axes[0].set_yticks(np.arange(len(model_sizes)))
    axes[0].set_yticklabels(model_sizes)

    for i in range(success.shape[0]):
        for j in range(success.shape[1]):
            axes[0].text(j, i, f"{success_percent[i, j]:.0f}%", ha='center', va='center')

    cbar1 = fig.colorbar(im1, ax=axes[0], pad=0.02)
    cbar1.set_label("Success %")

    # BOTTOM: mean
    im2 = axes[1].imshow(
        mean,
        aspect='auto',
        cmap='RdYlGn_r',
        norm=PowerNorm(gamma=0.75, vmin=np.min(mean), vmax=np.max(mean))
    )

    axes[1].set_title("Mean Landing Distance")
    axes[1].set_xlabel("Training Length")
    axes[1].set_ylabel("Model Size")
    axes[1].set_xticks(np.arange(len(training_lengths)))
    axes[1].set_xticklabels(training_lengths)
    axes[1].set_yticks(np.arange(len(model_sizes)))
    axes[1].set_yticklabels(model_sizes)

    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            axes[1].text(j, i, f"{mean[i, j]:.1f}", ha='center', va='center')

    cbar2 = fig.colorbar(im2, ax=axes[1], pad=0.02)
    cbar2.set_label("Mean Distance")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

plot_success_and_mean(model_sizes, training_lengths, success_data, mean_data)