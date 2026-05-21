"""Mini-project for week 05 - DQN on LunarLander.

Targets (single consumer GPU, < 5 min for full converge; <60 s for the
smoke target on CPU):

  Smoke    : 5000 steps, mean episode return over last 10 episodes > -250
             (random baseline on LunarLander-v3 is around -180 to -300;
             this is just "training is non-trivially happening").
  Full     : 200k env steps target mean return > 150 (project goal).
"""
