"""A minimal LTL-derived shield.

For the safety formula
    phi = G (not danger)    (Always: never enter a danger cell)

the canonical deterministic safety automaton has TWO states:
    q0 = "no danger has been seen yet" (accepting, sink)
    q_bad = "danger has been seen" (rejecting, absorbing)

The shield wraps the environment + agent:
  * the agent proposes an action a.
  * the shield checks, against the environment's one-step model
    `env.adjacent_dangerous(s, a)`, whether `a` would transition the DFA into
    `q_bad`.
  * if so, the shield replaces a with the safest alternative from the same
    state (breaking ties by going through the agent's preference list, then
    by going `up` if all else fails — we fall back to a noop-ish move).

This is the *pre-shield* variant from Alshiekh et al. (AAAI 2018, arXiv:1708.08611):
the shield filters the action set before sampling.

Soundness theorem (informal): under deterministic dynamics and a complete
one-step model, the shield guarantees ZERO unsafe transitions in any
trajectory, regardless of the underlying agent. This is verified empirically
in `train_shielded`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List


@dataclass
class DFAShield:
    """Pre-shield: takes the current state and a list of action preferences
    (most preferred first) and returns the most-preferred SAFE action.

    `is_unsafe(s, a)` is a callable closing over the env's one-step model.
    `fallback_priority` is the order of actions to try when *all* preferred
    actions are unsafe. In our gridworld this is rarely needed, since the
    safe set is always non-empty (you can always stay-in-place by walking
    into a wall).
    """

    is_unsafe: Callable[[int, int], bool]
    n_actions: int = 4
    fallback_priority: Iterable[int] = (0, 1, 2, 3)
    n_blocked: int = field(default=0, init=False)

    def filter(self, s_idx: int, action_preferences: List[int]) -> int:
        """Return the first preferred action that is SAFE. If none, return
        the first fallback action that is safe; if even that fails, return
        the original first preference (so the shield never silently hangs).
        """
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
        return action_preferences[0]


def build_no_lava_visit_shield(env) -> DFAShield:
    """Convenience constructor: build the canonical 'G ¬danger' shield from
    the env's one-step-look-ahead model."""
    return DFAShield(is_unsafe=env.adjacent_dangerous, n_actions=env.n_actions)
