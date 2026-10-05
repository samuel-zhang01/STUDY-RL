"""Generate figures for week-6 notes.pdf.

Runs a short Rainbow ablation on LunarLander (~5 min total on CPU) and emits
4 figures.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

from common.plotting import use_rl_style, RL_PALETTE  # noqa: E402
import _pkg_loader  # noqa: F401
from w06_project import train as train_mod, solutions  # noqa: E402
import torch  # noqa: E402


def smooth(x, w=5):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1 and x.size >= w:
        c = np.cumsum(np.insert(x, 0, 0))
        return (c[w:] - c[:-w]) / w
    return x


def main(out_dir: str | None = None, n_steps: int = 8000, seeds: int = 2) -> None:
    use_rl_style()
    out = Path(out_dir) if out_dir else (HERE.parent / "figs")
    out.mkdir(parents=True, exist_ok=True)

    # ----- Figure 1: ablation curves on LunarLander -----
    configs = {
        "Rainbow-lite (full)": dict(),
        "no Double": dict(double=False),
        "no Dueling": dict(dueling=False),
        "no PER": dict(per=False),
        "no Noisy": dict(noisy=False),
    }
    results = {name: [] for name in configs}
    for seed in range(seeds):
        for name, kw in configs.items():
            print(f"  seed {seed} | {name}")
            log = train_mod.train(seed=seed, n_steps=n_steps,
                                  eps_decay_steps=int(0.7 * n_steps),
                                  warmup_steps=500, buffer_size=10_000,
                                  log_every=500, **kw)
            results[name].append(log.episode_returns)
    # align lengths
    min_lens = {name: min(len(rs) for rs in results[name]) for name in configs}
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = [RL_PALETTE["blue"], RL_PALETTE["orange"], RL_PALETTE["teal"],
              RL_PALETTE["gold"], RL_PALETTE["red"]]
    for (name, color) in zip(configs.keys(), colors):
        L = min_lens[name]
        arr = np.array([rs[:L] for rs in results[name]])
        m = arr.mean(0); s = arr.std(0)
        ms = smooth(m, 5); ss = smooth(s, 5)
        x = np.arange(len(ms))
        ax.plot(x, ms, label=name, color=color)
        ax.fill_between(x, ms - ss, ms + ss, alpha=0.15, color=color)
    ax.axhline(0, color=RL_PALETTE["gray"], ls="--", lw=0.7)
    ax.set_xlabel("Episode (smoothed window=5)")
    ax.set_ylabel("Return")
    ax.set_title(f"Rainbow-lite ablation on LunarLander-v3 ({n_steps:,} steps, {seeds} seeds)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_rainbow_ablation.pdf")
    plt.close(fig)

    # ----- Figure 2: NoisyNet sigma decay schematic -----
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    n_resets = 200
    sig = 1.0 * (0.999 ** np.arange(n_resets))
    ax.plot(np.arange(n_resets), sig, color=RL_PALETTE["teal"],
            label=r"$\sigma$ (illustrative schedule, not measured)")
    ax.axhline(0.1, color=RL_PALETTE["red"], ls="--", lw=0.7,
               label="illustrative reference (not enforced by agent)")
    ax.set_xlabel("training reset (every batch)")
    ax.set_ylabel(r"$\sigma$ (parameter noise scale)")
    ax.set_title("NoisyNets replace the $\\varepsilon$-schedule with learned noise")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_noisynet_sigma.pdf")
    plt.close(fig)

    # ----- Figure 3: PER weight curve -----
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    td = np.linspace(0, 5, 100)
    for alpha, color in [(0.4, RL_PALETTE["blue"]),
                          (0.6, RL_PALETTE["orange"]),
                          (0.8, RL_PALETTE["red"])]:
        p = td ** alpha
        ax.plot(td, p / p.max(), label=fr"$\alpha={alpha}$", color=color)
    ax.set_xlabel(r"$|\delta|$  (absolute TD error)")
    ax.set_ylabel("Relative sampling probability")
    ax.set_title("PER priority: how aggressive is the bias?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_per_priority.pdf")
    plt.close(fig)

    # ----- Figure 4: categorical projection visualisation -----
    support = torch.linspace(-3, 3, 51)
    # peak around z=1
    z = support.numpy()
    p0 = torch.exp(-((support - 1.0) ** 2) / 0.5)
    p0 = (p0 / p0.sum()).unsqueeze(0)
    proj_done = solutions.exercise_6_categorical_project(
        p0, support, torch.tensor([0.0]), torch.tensor([1.0]), 0.99
    ).squeeze(0).numpy()
    proj_alive = solutions.exercise_6_categorical_project(
        p0, support, torch.tensor([0.0]), torch.tensor([0.0]), 0.99
    ).squeeze(0).numpy()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(z, p0.squeeze(0).numpy(), width=0.10, alpha=0.4,
           color=RL_PALETTE["gray"], label="prior")
    ax.bar(z, proj_alive, width=0.10, alpha=0.7,
           color=RL_PALETTE["blue"], label="after $\\gamma$ contraction (alive)")
    ax.bar(z, proj_done, width=0.10, alpha=0.6,
           color=RL_PALETTE["orange"], label="terminal: mass to $r=0$ bin")
    ax.set_xlabel("return atom $z_i$")
    ax.set_ylabel("probability")
    ax.set_title("C51 categorical projection $\\Phi \\mathcal{T} Z$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_c51_projection.pdf")
    plt.close(fig)

    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
