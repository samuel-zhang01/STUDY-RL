"""A tiny courier game with measured Q-learning, not a prewritten learning curve.

Run ``python -m learning_projects.courier --play`` or use ``--episodes 1200``.
Coordinates are (column, row), with row 0 at the top. Goal and hazards terminate;
the training step budget only truncates collection and does not mask bootstraps.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import random
from typing import Callable

from . import reference

State = tuple[int, int]
ACTIONS = ("up", "right", "down", "left")
DELTAS = {"up": (0, -1), "right": (1, 0), "down": (0, 1), "left": (-1, 0)}
ARROWS = {"up": "↑", "right": "→", "down": "↓", "left": "←"}


@dataclass(frozen=True)
class Transition:
    state: State
    reward: float
    cost: int
    terminated: bool


@dataclass(frozen=True)
class Episode:
    path: tuple[State, ...]
    rewards: tuple[float, ...]
    cost: int
    terminated: bool
    truncated: bool

    @property
    def reward(self) -> float:
        return sum(self.rewards)


@dataclass(frozen=True)
class GridCourier:
    width: int = 4
    height: int = 4
    start: State = (0, 0)
    goal: State = (3, 3)
    hazards: tuple[State, ...] = ((1, 2), (2, 2))
    step_reward: float = -0.1
    goal_reward: float = 4.0
    hazard_reward: float = -3.0

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("grid dimensions must be positive")
        cells = (self.start, self.goal, *self.hazards)
        if any(not (0 <= x < self.width and 0 <= y < self.height) for x, y in cells):
            raise ValueError("start, goal, and hazards must lie inside the grid")
        if self.start == self.goal or self.start in self.hazards or self.goal in self.hazards:
            raise ValueError("start must be nonterminal; goal and hazards must be distinct")
        if len(set(self.hazards)) != len(self.hazards):
            raise ValueError("hazard cells must be unique")
        if not all(math.isfinite(r) for r in (self.step_reward, self.goal_reward, self.hazard_reward)):
            raise ValueError("rewards must be finite")

    def terminal(self, state: State) -> bool:
        return state == self.goal or state in self.hazards

    def step(self, state: State, action: str) -> Transition:
        if action not in DELTAS:
            raise ValueError(f"unknown action {action!r}")
        if not (0 <= state[0] < self.width and 0 <= state[1] < self.height):
            raise ValueError("state lies outside the grid")
        if self.terminal(state):
            raise ValueError("an episode must reset after entering a terminal cell")
        dx, dy = DELTAS[action]
        nxt = (max(0, min(self.width - 1, state[0] + dx)),
               max(0, min(self.height - 1, state[1] + dy)))
        if nxt == self.goal:
            return Transition(nxt, self.goal_reward, 0, True)
        if nxt in self.hazards:
            return Transition(nxt, self.hazard_reward, 1, True)
        return Transition(nxt, self.step_reward, 0, False)

    def allowed_actions(self, state: State, shielded: bool = False) -> tuple[str, ...]:
        """A toy shield: exact one-step model forbids entering a hazard."""
        if self.terminal(state):
            return ()
        if shielded:
            return tuple(a for a in ACTIONS if self.step(state, a).cost == 0)
        return ACTIONS

    def states(self) -> list[State]:
        return [(x, y) for y in range(self.height) for x in range(self.width)]


def choose_action(values: dict[str, float], actions: tuple[str, ...],
                  epsilon: float, rng: random.Random) -> str:
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    if not actions:
        raise ValueError("no allowed action; define a safe fallback for this environment")
    if rng.random() < epsilon:
        return rng.choice(actions)
    best = max(values[a] for a in actions)
    return rng.choice([a for a in actions if math.isclose(values[a], best, abs_tol=1e-12)])


def train(env: GridCourier, episodes: int = 1200, seed: int = 7,
          alpha: float = 0.3, gamma: float = 0.9, epsilon: float = 0.15,
          max_steps: int = 80, shielded: bool = False,
          target: Callable = reference.q_target) -> tuple[dict[State, dict[str, float]], list[Episode]]:
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    if not 0 < alpha <= 1 or not 0 <= gamma <= 1 or not 0 <= epsilon <= 1:
        raise ValueError("alpha is in (0, 1]; gamma and epsilon are in [0, 1]")
    rng = random.Random(seed)
    q = {s: {a: 0.0 for a in ACTIONS} for s in env.states()}
    history = []
    for _ in range(episodes):
        state = env.start
        path, rewards, cost = [state], [], 0
        ended = False
        for _ in range(max_steps):
            action = choose_action(q[state], env.allowed_actions(state, shielded), epsilon, rng)
            transition = env.step(state, action)
            next_actions = env.allowed_actions(transition.state, shielded)
            future = [q[transition.state][a] for a in next_actions]
            # Empty future values are valid at a TRUE terminal, including a hazard.
            y = target(transition.reward, gamma, future, transition.terminated)
            if not math.isfinite(y):
                raise ValueError("q_target returned a nonfinite value")
            q[state][action] += alpha * (y - q[state][action])
            rewards.append(transition.reward)
            cost += transition.cost
            state = transition.state
            path.append(state)
            if transition.terminated:
                ended = True
                break
        history.append(Episode(tuple(path), tuple(rewards), cost, ended, not ended))
    return q, history


def evaluate(env: GridCourier, q: dict[State, dict[str, float]], seed: int = 99,
             episodes: int = 20, max_steps: int = 80,
             shielded: bool = False) -> list[Episode]:
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    rng = random.Random(seed)
    history = []
    for _ in range(episodes):
        state, path, rewards, cost, ended = env.start, [env.start], [], 0, False
        for _ in range(max_steps):
            action = choose_action(q[state], env.allowed_actions(state, shielded), 0, rng)
            transition = env.step(state, action)
            state = transition.state
            path.append(state)
            rewards.append(transition.reward)
            cost += transition.cost
            if transition.terminated:
                ended = True
                break
        history.append(Episode(tuple(path), tuple(rewards), cost, ended, not ended))
    return history


def render(env: GridCourier, position: State | None = None,
           q: dict[State, dict[str, float]] | None = None, shielded: bool = False) -> str:
    lines = []
    for y in range(env.height):
        row = []
        for x in range(env.width):
            s = (x, y)
            if s == position:
                cell = "@"
            elif s == env.goal:
                cell = "G"
            elif s in env.hazards:
                cell = "!"
            elif q is not None:
                actions = env.allowed_actions(s, shielded)
                cell = ARROWS[max(actions, key=lambda a: q[s][a])]
            else:
                cell = "S" if s == env.start else "."
            row.append(cell)
        lines.append(" ".join(row))
    return "\n".join(lines)


def play(env: GridCourier, shielded: bool = False) -> None:
    controls = {"w": "up", "d": "right", "s": "down", "a": "left"}
    state, reward, cost = env.start, 0.0, 0
    print("You are @. Deliver to G; ! is a terminal hazard. Move w/a/s/d; q quits.")
    print("Coordinates are (column,row), with the top-left cell (0,0).")
    print(render(env, state))
    while True:
        try:
            key = input("Move: ").strip().lower()
        except EOFError:
            return
        if key == "q":
            return
        if key not in controls:
            print("Use w/a/s/d or q.")
            continue
        action = controls[key]
        if action not in env.allowed_actions(state, shielded):
            print("Shield blocked a known hazard entry. Choose another move.")
            continue
        transition = env.step(state, action)
        state, reward, cost = transition.state, reward + transition.reward, cost + transition.cost
        print(render(env, state))
        print(f"step reward={transition.reward:g}; total reward={reward:g}; hazard cost={cost}")
        if transition.terminated:
            print("Delivered!" if state == env.goal else "Entered a hazard: episode ended.")
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--hazard-reward", type=float, default=-3.0)
    parser.add_argument("--shield", action="store_true")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--learner", action="store_true", help="use your exercises.q_target instead of the reference")
    args = parser.parse_args()
    if not math.isfinite(args.hazard_reward):
        parser.error("hazard reward must be finite")
    env = GridCourier(hazard_reward=args.hazard_reward)
    if args.play:
        play(env, args.shield)
        return
    from . import exercises
    target = exercises.q_target if args.learner else reference.q_target
    try:
        q, history = train(env, args.episodes, args.seed, epsilon=args.epsilon,
                           shielded=args.shield, target=target)
    except (NotImplementedError, ValueError) as exc:
        parser.error(f"{exc}. Attempt exercises.py, then run python -m learning_projects.check_exercises.")
    evaluated = evaluate(env, q, args.seed + 1000, shielded=args.shield)
    print("Your implementation" if args.learner else "Reference demonstration; use --learner after exercise 3.")
    print(render(env, q=q, shielded=args.shield))
    print(f"Training: {len(history)} episodes; exploration epsilon={args.epsilon:g}; seed={args.seed}")
    print(f"Training hazard entries: {sum(e.cost for e in history)} (including exploration)")
    print(f"Evaluation: greedy policy, separate tie-breaking seed={args.seed + 1000}; deterministic map")
    print(f"Mean reward={sum(e.reward for e in evaluated) / len(evaluated):.3f}; "
          f"deliveries={sum(e.path[-1] == env.goal for e in evaluated)}/{len(evaluated)}; "
          f"hazards={sum(e.cost for e in evaluated)}; "
          f"collector cutoffs={sum(e.truncated for e in evaluated)}")
    print("First greedy path:", " → ".join(map(str, evaluated[0].path)))
    print("Try another training seed. Changing a seed here changes exploration, not the deterministic map.")


if __name__ == "__main__":
    main()
