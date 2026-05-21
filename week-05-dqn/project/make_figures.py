"""Generate figures for week-5 notes.pdf.

Runs a *short* DQN training pass (~30 s on CPU) and creates 3 figures.
"""
from __future__ import annotations
import sys
from pathlib import Path
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

from common.plotting import use_rl_style, RL_PALETTE, shaded  # noqa: E402

import _pkg_loader  # noqa: F401
from w05_project import agent as agent_mod, train as train_mod, solutions  # noqa: E402


def smooth(x, w=10):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1 and x.size >= w:
        c = np.cumsum(np.insert(x, 0, 0))
        return (c[w:] - c[:-w]) / w
    return x


def main(out_dir: str | None = None, n_steps: int = 30_000,
         n_seeds: int = 2) -> None:
    use_rl_style()
    out = Path(out_dir) if out_dir else (HERE.parent / "figs")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Training DQN on LunarLander-v3 for {n_steps} steps x {n_seeds} seeds...")
    all_returns = []
    all_losses = []
    for seed in range(n_seeds):
        log = train_mod.train(seed=seed, n_steps=n_steps,
                              eps_decay_steps=int(0.7 * n_steps),
                              warmup_steps=1000, buffer_size=50_000,
                              log_every=500)
        all_returns.append(log.episode_returns)
        all_losses.append(log.losses)
        print(f"  seed {seed}: episodes={len(log.episode_returns)} "
              f"mean_last_10={log.meta['mean_last_10']:.1f} "
              f"wall={log.meta['wall_clock_s']:.1f}s")

    # Figure 1: episode-return learning curves
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for i, rs in enumerate(all_returns):
        rs_s = smooth(rs, 10)
        x = np.arange(len(rs_s))
        ax.plot(x, rs_s, label=f"seed {i}",
                color=[RL_PALETTE["blue"], RL_PALETTE["orange"]][i])
    ax.axhline(0, color=RL_PALETTE["gray"], ls="--", lw=0.7)
    ax.set_xlabel("Episode (smoothed window=10)")
    ax.set_ylabel("Episode return")
    ax.set_title(f"DQN on LunarLander-v3 ({n_steps:,} env steps)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_dqn_lunar_curve.pdf")
    plt.close(fig)

    # Figure 2: training loss
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for i, ls in enumerate(all_losses):
        ls_s = smooth(ls, 5)
        ax.plot(np.arange(len(ls_s)), ls_s,
                label=f"seed {i}",
                color=[RL_PALETTE["blue"], RL_PALETTE["orange"]][i])
    ax.set_xlabel("log step (every 500 env steps)")
    ax.set_ylabel("Huber loss (smoothed)")
    ax.set_title("DQN Huber loss over training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_dqn_loss.pdf")
    plt.close(fig)

    # Figure 3: Baird divergence demonstration
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for alpha, color in [(0.01, RL_PALETTE["teal"]),
                          (0.05, RL_PALETTE["orange"]),
                          (0.10, RL_PALETTE["red"])]:
        norms = solutions.exercise_7_baird_divergence(n_iters=300, alpha=alpha)
        ax.plot(norms, label=fr"$\alpha={alpha}$", color=color)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|w\|_2$  (log)")
    ax.set_title("Deadly triad: off-policy semi-gradient TD diverges (Baird counterexample)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_baird_divergence.pdf")
    plt.close(fig)

    # Figure 4: action-value visualisation across an evaluation episode
    # (uses the last trained agent, with eps=0)
    import gymnasium as gym
    cfg = agent_mod.DQNConfig(obs_dim=8, n_actions=4,
                              eps_decay_steps=int(0.7 * n_steps),
                              warmup_steps=1000, buffer_size=50_000)
    ag = agent_mod.DQNAgent(cfg, seed=0)
    # quick re-train so we have a non-trivial Q-net
    env = gym.make("LunarLander-v3")
    s, _ = env.reset(seed=0)
    for step in range(min(15_000, n_steps)):
        a = ag.act(np.asarray(s, dtype=np.float32))
        s2, r, term, trunc, _ = env.step(a)
        ag.push(s, a, r, s2, term or trunc)
        ag.learn()
        s = s2 if not (term or trunc) else env.reset(seed=step + 1)[0]
    env_eval = gym.make("LunarLander-v3")
    _, qs, acts, rews = solutions.exercise_9_action_value_trace(ag, env_eval, max_steps=300, seed=0)
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.0), sharex=True)
    for ai, color in zip(range(qs.shape[1]),
                          [RL_PALETTE["blue"], RL_PALETTE["teal"],
                           RL_PALETTE["orange"], RL_PALETTE["gold"]]):
        axes[0].plot(qs[:, ai], color=color,
                     label=["noop", "left", "main", "right"][ai])
    axes[0].set_ylabel(r"$Q(s, a)$")
    axes[0].legend(loc="lower right", ncols=4)
    axes[0].set_title("Action-value trace during one LunarLander episode")
    axes[1].plot(rews, color=RL_PALETTE["gray"], label="reward")
    axes[1].plot(acts, color=RL_PALETTE["red"], label="chosen action",
                 alpha=0.6, lw=0.8)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("reward / action id")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_dqn_qtrace.pdf")
    plt.close(fig)

    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
