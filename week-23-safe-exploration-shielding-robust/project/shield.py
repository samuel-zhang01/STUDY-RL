"""A minimal LTL-derived shield.

For the safety formula
    phi = G (not danger)    (Always: never enter a danger cell)

the canonical deterministic safety automaton has TWO states:
    q0 = "no danger has been seen yet" (accepting)
    q_bad = "danger has been seen" (rejecting, absorbing)

The shield wraps the environment + agent:
  * the agent proposes an action a.
  * the shield checks, against the environment's one-step model
    `env.adjacent_dangerous(s, a)`, whether `a` would transition the DFA into
    `q_bad`.
  * if so, the shield replaces a with the safest alternative from the same
    state (breaking ties by going through the agent's preference list, then
    by checking an explicit fallback list). No safe action raises an error.

This is the *pre-shield* variant from Alshiekh et al. (AAAI 2018, arXiv:1708.08611):
the shield filters the action set before sampling.

Soundness requires a safe initial state, deterministic dynamics, a complete
one-step model, and a nonempty safe action set at every visited state. A finite
training run checks observed violations; it is not itself a proof.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List


class NoSafeActionError(RuntimeError):
    """The shield cannot authorize any action under its supplied model."""


@dataclass
class DFAShield:
    """Pre-shield: takes the current state and a list of action preferences
    (most preferred first) and returns the most-preferred SAFE action.

    `is_unsafe(s, a)` is a callable closing over the env's one-step model.
    `fallback_priority` is the order of actions to try when *all* preferred
    actions are unsafe. In our gridworld this is rarely needed, since the
    initial state is safe. Other grid layouts can have no safe successor.
    """

    is_unsafe: Callable[[int, int], bool]
    n_actions: int = 4
    fallback_priority: Iterable[int] = (0, 1, 2, 3)
    n_blocked: int = field(default=0, init=False)

    def filter(self, s_idx: int, action_preferences: List[int]) -> int:
        """Return a safe action, or raise NoSafeActionError without executing."""
        for a in action_preferences:
            if not self.is_unsafe(s_idx, a):
                if action_preferences and a != action_preferences[0]:
                    self.n_blocked += 1
                return a
        # All preferences unsafe — try fallback.
        for a in self.fallback_priority:
            if not self.is_unsafe(s_idx, a):
                self.n_blocked += 1
                return a
        # Fully boxed in (should not happen in our grid): emit a deterministic
        # fallback, accepting that a violation will occur.
        raise NoSafeActionError(f"No safe action available at state {s_idx}")


def build_no_lava_visit_shield(env) -> DFAShield:
    """Convenience constructor: build the canonical 'G ¬danger' shield from
    the env's one-step-look-ahead model."""
    return DFAShield(is_unsafe=env.adjacent_dangerous, n_actions=env.n_actions)
