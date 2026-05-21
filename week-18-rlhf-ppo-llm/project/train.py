"""Convenience entry point for the week-18 mini-project.

Target: with default config, PPO drives the mean reward-model score on
the toy preference dataset upward from the SFT baseline in well under
60 s on CPU.  This is a deliberately tiny task — its purpose is to
make the full RLHF pipeline observable in one terminal.
"""
from __future__ import annotations
from .ppo import train as _train


def train(seed: int = 0, **kwargs):
    return _train(seed=seed, **kwargs)


if __name__ == "__main__":
    history, value_head, ref = train()
    print(f"Reward start={history['reward'][0]:.3f}  end={history['reward'][-1]:.3f}")
    print(f"KL start={history['kl'][0]:.4f}  end={history['kl'][-1]:.4f}")
