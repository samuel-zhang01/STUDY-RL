"""Figures for Week 16 notes.pdf."""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.plotting import use_rl_style, RL_PALETTE
from common.seed import seed_everything

from .train import train


def main(seed: int = 0, out_dir: str | None = None) -> None:
    use_rl_style(); seed_everything(seed)
    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "..", "figs")
    os.makedirs(out_dir, exist_ok=True)

    out = train(seed=seed)
    curve = out["eval_curve"]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(curve[:, 0], curve[:, 1], color=RL_PALETTE["teal"], marker="o")
    ax.set_xlabel("training iteration"); ax.set_ylabel("eval return")
    ax.set_title("Dreamer-lite on CartPole-v1: imagination-trained actor")
    ax.set_ylim(0, 200)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_dreamer_curve.pdf")); plt.close(fig)

    # Imagined rollout vs real rollout (single trajectory comparison).
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed + 7)
    real_traj = [obs.astype(np.float32)]
    rssm = out["rssm"]; agent = out["agent"]
    device = next(rssm.parameters()).device
    state = rssm.initial(1, device)
    with torch.no_grad():
        a_oh = torch.zeros(1, agent.n_actions, device=device)
        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
        state, *_ = rssm.observe_step(state, a_oh, obs_t)
        # Imagine 20 steps in the world model.
        imagined = []
        actions = []
        s = state
        import torch.nn.functional as F
        for _ in range(20):
            feat = torch.cat([s.h, s.s], dim=-1)
            a = agent.actor(feat).logits.argmax(dim=-1)
            actions.append(int(a.item()))
            a_oh = F.one_hot(a, agent.n_actions).float()
            s, r, c = rssm.imagine_step(s, a_oh)
            # Decode imagined obs.
            recon = rssm.decoder(torch.cat([s.h, s.s], dim=-1))
            imagined.append(recon.detach().cpu().numpy().squeeze())
    # Execute the same action sequence from the same initial observation.
    # Different imagined/real policies would confound dynamics-error comparison.
    real = [obs]
    for a in actions:
        obs, r, term, trunc, _ = env.step(a)
        real.append(obs.astype(np.float32))
        if term or trunc: break
    env.close()

    imagined = np.asarray(imagined); real = np.asarray(real[1:1 + len(imagined)])
    if len(real):
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
        for i, name in enumerate(["cart pos", "pole angle"]):
            idx = 0 if i == 0 else 2
            axes[i].plot(real[:, idx], label="real", color=RL_PALETTE["blue"])
            axes[i].plot(imagined[:len(real), idx], label="imagined", color=RL_PALETTE["orange"],
                          ls="--")
            axes[i].set_title(name); axes[i].set_xlabel("step")
        axes[0].legend()
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_imagined_vs_real.pdf"))
        plt.close(fig)


if __name__ == "__main__":
    main()
