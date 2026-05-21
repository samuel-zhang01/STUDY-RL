"""Generate the figures used in week-4 notes.pdf.

Run from the repo root as:
    PYTHONPATH=. python week-04-td-sarsa-q/project/make_figures.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from common.plotting import use_rl_style, RL_PALETTE, shaded  # noqa: E402


def _load_pkg():
    """Load the hyphenated `week-04-td-sarsa-q/project` as a clean package."""
    pkg_init = HERE / "__init__.py"
    pkg_spec = importlib.util.spec_from_file_location(
        "_w04_project", pkg_init, submodule_search_locations=[str(HERE)])
    pkg = importlib.util.module_from_spec(pkg_spec)
    sys.modules["_w04_project"] = pkg
    pkg_spec.loader.exec_module(pkg)
    for sub in ("agents", "env_utils", "train"):
        spec = importlib.util.spec_from_file_location(
            f"_w04_project.{sub}", HERE / f"{sub}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"_w04_project.{sub}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["_w04_project.train"]


def smooth(x, w=20):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        if x.size < w:
            return x
        c = np.cumsum(np.insert(x, 0, 0))
        return (c[w:] - c[:-w]) / w
    return np.stack([smooth(row, w) for row in x])


def main(out_dir: str | None = None, n_seeds: int = 3, n_episodes: int = 400,
         taxi_episodes: int = 1000) -> None:
    use_rl_style()
    out = Path(out_dir) if out_dir else (HERE.parent / "figs")
    out.mkdir(parents=True, exist_ok=True)

    train_mod = _load_pkg()
    res = train_mod.train(n_seeds=n_seeds, n_episodes=n_episodes,
                          taxi_episodes=taxi_episodes, do_taxi=True)

    # --- Figure 1: Cliff-walking learning curves -------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    colors = [RL_PALETTE["blue"], RL_PALETTE["orange"],
              RL_PALETTE["teal"], RL_PALETTE["gold"]]
    for (name, color) in zip(["SARSA", "Q-learning", "Expected SARSA", "Double Q"], colors):
        ys = res.returns[name]
        ys_s = smooth(ys, 20)
        x = np.arange(ys_s.shape[1])
        shaded(ax, x, ys_s, label=name, color=color)
    ax.set_xlabel("Episode (smoothed window=20)")
    ax.set_ylabel("Undiscounted return")
    ax.set_title("CliffWalking: SARSA vs Q-learning vs Expected SARSA vs Double Q")
    ax.set_ylim(bottom=-200)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_cliffwalking_curves.pdf")
    plt.close(fig)

    # --- Figure 2: cliff-jumping behaviour gap ---------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    names = ["SARSA", "Expected SARSA", "Q-learning", "Double Q"]
    means = [res.returns[n][:, -100:].mean() for n in names]
    stds = [res.returns[n][:, -100:].mean(axis=1).std() for n in names]
    ax.bar(names, means, yerr=stds, capsize=4,
           color=[RL_PALETTE["blue"], RL_PALETTE["teal"],
                  RL_PALETTE["orange"], RL_PALETTE["gold"]])
    ax.axhline(-13, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.text(0.02, -12, "optimal safe path", va="bottom",
            fontsize=8, color="k", alpha=0.7)
    ax.set_ylabel("Mean undiscounted return (last 100 ep, eps=0.1)")
    ax.set_title("Cliff-jumping gap: on-policy stays safe, off-policy keeps trying the cliff")
    fig.tight_layout()
    fig.savefig(out / "fig_cliff_gap.pdf")
    plt.close(fig)

    # --- Figure 3: Taxi Q-learning ---------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ys = res.returns["Taxi Q-learning"]
    ys_s = smooth(ys, 30)
    x = np.arange(ys_s.shape[1])
    shaded(ax, x, ys_s, label="Q-learning", color=RL_PALETTE["orange"])
    ax.set_xlabel("Episode (smoothed window=30)")
    ax.set_ylabel("Undiscounted return")
    ax.set_title("Taxi: tabular Q-learning")
    ax.axhline(8, color=RL_PALETTE["gray"], lw=0.7, ls="--", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_taxi_curve.pdf")
    plt.close(fig)

    # --- Figure 4: TD vs n-step variance schematic -----------------------
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ns = np.arange(1, 21)
    sigma_r = 1.0
    var_td = sigma_r ** 2
    var_n = sigma_r ** 2 * ns
    ax.plot(ns, [var_td] * len(ns), label="TD(0): low var, biased",
            color=RL_PALETTE["blue"])
    ax.plot(ns, var_n, label="n-step / MC: higher var, less bias",
            color=RL_PALETTE["orange"])
    ax.set_xlabel(r"$n$  (return horizon)")
    ax.set_ylabel("Variance of estimator")
    ax.set_title("Bias-variance trade-off in TD(n) family")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_bias_variance.pdf")
    plt.close(fig)

    print(f"Wrote figures to {out}")
    print("Final mean returns:", res.final_mean)


if __name__ == "__main__":
    main()
