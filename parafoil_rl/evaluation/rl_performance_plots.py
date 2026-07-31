"""Plot success percentage and CEP50 for the four PPO train/evaluation cases.

Generated from RL STATS (1).xlsx.
Each figure contains:
    top: success percentage vs. training length
    bottom: empirical CEP50 (median radial landing error) vs. training length

Run:
    python rl_performance_plots.py

The figures are saved in a folder named "rl_performance_plots" next to this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRAINING_STEPS = np.array([50000, 100000, 1000000, 5000000, 10000000], dtype=float)
TRAINING_LABELS = ["50k", "100k", "1M", "5M", "10M"]
MODELS = ['micro', 'tiny', 'small', 'medium', 'large', 'xlarge']

DATA = {
    "static_train_static_eval": {
        "title": "Static-Trained | Static-Evaluated",
        "success_percent": {
            "micro": [
                33.0,
                100.0,
                84.0,
                96.0,
                96.0
            ],
            "tiny": [
                73.0,
                100.0,
                100.0,
                100.0,
                99.0
            ],
            "small": [
                67.0,
                100.0,
                99.0,
                99.0,
                98.0
            ],
            "medium": [
                70.0,
                100.0,
                99.0,
                100.0,
                100.0
            ],
            "large": [
                100.0,
                100.0,
                96.0,
                92.0,
                99.0
            ],
            "xlarge": [
                90.0,
                100.0,
                99.0,
                91.0,
                93.0
            ]
        },
        "cep50_m": {
            "micro": [
                63.7,
                25.2,
                22.4,
                21.4,
                20.6
            ],
            "tiny": [
                39.3,
                20.6,
                19.5,
                17.5,
                18.0
            ],
            "small": [
                42.2,
                20.7,
                18.4,
                17.5,
                18.2
            ],
            "medium": [
                34.3,
                19.4,
                17.7,
                16.6,
                17.9
            ],
            "large": [
                25.7,
                18.1,
                21.4,
                24.8,
                20.4
            ],
            "xlarge": [
                29.8,
                20.9,
                19.7,
                22.0,
                21.5
            ]
        }
    },
    "dynamic_train_dynamic_eval": {
        "title": "Dynamic-Trained | Dynamic-Evaluated",
        "success_percent": {
            "micro": [
                28.0,
                97.0,
                66.0,
                85.0,
                92.0
            ],
            "tiny": [
                36.0,
                98.0,
                83.0,
                89.0,
                89.0
            ],
            "small": [
                71.0,
                99.0,
                87.0,
                86.0,
                87.0
            ],
            "medium": [
                80.0,
                100.0,
                90.0,
                81.0,
                85.0
            ],
            "large": [
                86.0,
                99.0,
                80.0,
                90.0,
                78.0
            ],
            "xlarge": [
                85.0,
                100.0,
                73.0,
                79.0,
                92.0
            ]
        },
        "cep50_m": {
            "micro": [
                62.5,
                25.3,
                28.1,
                20.6,
                15.9
            ],
            "tiny": [
                60.1,
                23.4,
                19.0,
                18.9,
                18.5
            ],
            "small": [
                37.3,
                21.9,
                19.1,
                17.4,
                19.1
            ],
            "medium": [
                36.2,
                21.4,
                19.5,
                17.2,
                18.5
            ],
            "large": [
                35.0,
                21.0,
                23.4,
                18.6,
                24.3
            ],
            "xlarge": [
                32.6,
                20.7,
                23.6,
                21.3,
                23.4
            ]
        }
    },
    "static_train_dynamic_eval": {
        "title": "Static-Trained | Dynamic-Evaluated",
        "success_percent": {
            "micro": [
                38.0,
                95.0,
                64.0,
                61.0,
                91.0
            ],
            "tiny": [
                77.0,
                87.0,
                86.0,
                79.0,
                96.0
            ],
            "small": [
                100.0,
                99.0,
                99.0,
                96.0,
                100.0
            ],
            "medium": [
                88.0,
                95.0,
                93.0,
                82.0,
                80.0
            ],
            "large": [
                98.0,
                89.0,
                93.0,
                70.0,
                62.0
            ],
            "xlarge": [
                98.0,
                94.0,
                76.0,
                72.0,
                79.0
            ]
        },
        "cep50_m": {
            "micro": [
                54.8,
                29.7,
                30.0,
                26.4,
                22.4
            ],
            "tiny": [
                35.9,
                29.2,
                29.8,
                31.6,
                29.9
            ],
            "small": [
                22.2,
                21.5,
                19.3,
                20.8,
                23.4
            ],
            "medium": [
                20.2,
                17.7,
                19.3,
                22.1,
                19.5
            ],
            "large": [
                15.6,
                15.5,
                17.1,
                26.5,
                30.2
            ],
            "xlarge": [
                19.1,
                16.9,
                21.2,
                23.8,
                23.2
            ]
        }
    },
    "dynamic_train_static_eval": {
        "title": "Dynamic-Trained | Static-Evaluated",
        "success_percent": {
            "micro": [
                55.0,
                100.0,
                93.0,
                98.0,
                100.0
            ],
            "tiny": [
                34.0,
                81.0,
                78.0,
                100.0,
                100.0
            ],
            "small": [
                100.0,
                100.0,
                100.0,
                100.0,
                100.0
            ],
            "medium": [
                98.0,
                99.0,
                100.0,
                97.0,
                97.0
            ],
            "large": [
                99.0,
                98.0,
                100.0,
                96.0,
                97.0
            ],
            "xlarge": [
                98.0,
                97.0,
                94.0,
                96.0,
                94.0
            ]
        },
        "cep50_m": {
            "micro": [
                49.4,
                24.4,
                22.0,
                21.2,
                23.4
            ],
            "tiny": [
                75.0,
                33.7,
                31.6,
                32.2,
                29.8
            ],
            "small": [
                20.8,
                20.8,
                20.6,
                17.8,
                19.6
            ],
            "medium": [
                19.3,
                19.2,
                21.0,
                19.2,
                22.1
            ],
            "large": [
                19.4,
                21.4,
                20.5,
                23.0,
                23.0
            ],
            "xlarge": [
                18.4,
                23.7,
                21.5,
                21.0,
                19.9
            ]
        }
    }
}

MARKERS = {
    "micro": "o",
    "tiny": "s",
    "small": "^",
    "medium": "D",
    "large": "v",
    "xlarge": "P",
}

LINESTYLES = {
    "micro": "-",
    "tiny": "--",
    "small": "-.",
    "medium": ":",
    "large": "-",
    "xlarge": "--",
}


def plot_case(case_key: str, output_dir: Path) -> Path:
    """Create and save one two-subplot figure for a train/evaluation case."""
    case = DATA[case_key]

    fig, (ax_success, ax_cep) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 9),
        sharex=True,
    )

    plotted_lines = []
    for model in MODELS:
        line, = ax_success.plot(
            TRAINING_STEPS,
            case["success_percent"][model],
            marker=MARKERS[model],
            linestyle=LINESTYLES[model],
            linewidth=2.0,
            markersize=6,
            label=model.capitalize(),
        )
        plotted_lines.append(line)

        ax_cep.plot(
            TRAINING_STEPS,
            case["cep50_m"][model],
            marker=MARKERS[model],
            linestyle=LINESTYLES[model],
            linewidth=2.0,
            markersize=6,
        )

    fig.suptitle(case["title"], fontsize=14, fontweight="bold", y=0.985)
    ax_success.set_ylabel("Success Percentage (%)")
    ax_success.set_ylim(0, 105)
    ax_success.grid(True, which="both", alpha=0.3)

    ax_cep.set_ylabel("CEP50 (m)")
    ax_cep.set_xlabel("Training Length (Environment Steps)")
    ax_cep.set_ylim(bottom=0)
    ax_cep.grid(True, which="both", alpha=0.3)

    ax_cep.set_xscale("log")
    ax_cep.set_xticks(TRAINING_STEPS)
    ax_cep.set_xticklabels(TRAINING_LABELS)
    ax_cep.minorticks_off()

    fig.legend(
        handles=plotted_lines,
        labels=[model.capitalize() for model in MODELS],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.945),
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    plt.show()

    output_path = output_dir / f"{case_key}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "rl_performance_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for case_key in DATA:
        path = plot_case(case_key, output_dir)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
