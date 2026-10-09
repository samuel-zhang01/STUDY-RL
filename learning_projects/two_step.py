"""The same two-step delivery problem, before adding a grid or a neural network.

Run ``python -m learning_projects.two_step`` from the repository root.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DeliveryValues:
    stop: float
    continue_: float
    policy_value: float
    continue_advantage: float
    stop_advantage: float


def delivery_values(gamma: float = 0.9, continue_probability: float = 0.25,
                    finish_reward: float = 4.0) -> DeliveryValues:
    """Exact values: stop pays 2 now; continue pays 0 then finish_reward."""
    if not 0 <= gamma <= 1 or not 0 <= continue_probability <= 1:
        raise ValueError("gamma and continue_probability must be in [0, 1]")
    if not math.isfinite(finish_reward):
        raise ValueError("finish_reward must be finite")
    continued = gamma * finish_reward
    value = (1 - continue_probability) * 2 + continue_probability * continued
    return DeliveryValues(2.0, continued, value, continued - value, 2 - value)


def mc_td_updates(gamma: float = 0.9, alpha: float = 0.5,
                  old_value: float = 1.0, next_value: float = 3.0,
                  finish_reward: float = 4.0) -> dict[str, float]:
    """Compare targets for the continued trajectory [0, finish_reward]."""
    if not 0 <= gamma <= 1 or not 0 <= alpha <= 1:
        raise ValueError("gamma and alpha must be in [0, 1]")
    if not all(math.isfinite(x) for x in (old_value, next_value, finish_reward)):
        raise ValueError("values and rewards must be finite")
    mc_target = gamma * finish_reward
    td_target = gamma * next_value
    return {"mc_target": mc_target, "td_target": td_target,
            "mc_new": old_value + alpha * (mc_target - old_value),
            "td_new": old_value + alpha * (td_target - old_value)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--continue-probability", type=float, default=0.25)
    parser.add_argument("--stage", choices=("values", "targets", "policy", "all"),
                        default="values", help="add one concept at a time; default shows returns and values")
    args = parser.parse_args()
    try:
        values = delivery_values(args.gamma, args.continue_probability)
    except ValueError as exc:
        parser.error(str(exc))
    print("Delivery choice: stop → reward 2; continue → reward 0 → finish reward 4")
    print(f"gamma={args.gamma:g}; policy P(continue)={args.continue_probability:g}")
    print(f"Q(stop)={values.stop:.3f}; Q(continue)={values.continue_:.3f}")
    print(f"V(policy)={values.policy_value:.3f}; this is a mean, not a third outcome.")
    if args.stage in ("policy", "all"):
        print(f"A(continue)={values.continue_advantage:.3f}; A(stop)={values.stop_advantage:.3f}")
    if args.stage in ("targets", "all"):
        for name, value in mc_td_updates(args.gamma).items():
            print(f"{name}={value:.3f}")
    print("Predict the switch point before trying --gamma 0.4 and --gamma 0.6.")


if __name__ == "__main__":
    main()
