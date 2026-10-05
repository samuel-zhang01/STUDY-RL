"""Generate deterministic lecture figures: training curves and held-out ablation."""
from __future__ import annotations

from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.plotting import use_rl_style, RL_PALETTE
from .train import train


def make_figures(seed=0):
    use_rl_style()
    report = train(seed)
    directory = Path(__file__).resolve().parents[1]/"figs"
    directory.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for key, label, color in [
        ("baseline_training_loss", "Raw ridge", "blue"),
        ("enhanced_training_loss", "Selected randomization", "teal"),
    ]:
        axes[0].plot(report[key], label=label, color=RL_PALETTE[color])
    axes[0].set(xlabel="Gradient step", ylabel="Training objective", title="Different training objectives")
    axes[0].legend()
    keys = ["baseline_test_mse", "enhanced_test_mse", "stable_features_ablation_test_mse"]
    axes[1].bar(np.arange(3), [report[k] for k in keys],
                color=[RL_PALETTE[c] for c in ("blue", "teal", "gold")])
    axes[1].set(xticks=np.arange(3), xticklabels=["Raw", "Enhanced", "No proxy"],
                ylabel="Patient-weighted MSE", title="Sealed environment 2")
    fig.tight_layout()
    fig.savefig(directory/"causal_training.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    return directory/"causal_training.pdf"


if __name__ == "__main__":
    print(make_figures())
