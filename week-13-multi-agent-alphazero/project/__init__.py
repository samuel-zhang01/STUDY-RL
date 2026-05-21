"""Mini-project for week 13 — Multi-Agent RL and AlphaZero.

Two complementary deliverables:

1. AlphaZero on Tic-Tac-Toe (3x3): full MCTS+PUCT, policy/value network,
   symmetry-augmented self-play. Converges to near-optimal play in ~1 min
   on CPU; expected winrate vs.\\ random opponent >= 0.85.

2. MAPPO on (i) a 2-agent matrix coordination game (converges to perfect
   coordination in ~200 episodes) and (ii) a 3-agent simple_spread cousin
   (illustrates the harder cooperative-coverage problem; converges noisily).

Target metrics:
- AlphaZero winrate vs random >= 0.85 after 12 iterations.
- MAPPO coordination-game return >= 9.5 of 10.0 after 200 episodes.

Runtime: ~90 seconds total on CPU.
"""

from . import (
    tictactoe, alphazero, simple_spread, coord_game, mappo, train, solutions,
)  # noqa: F401
