"""Check mathematical invariants, boundaries, and real command-line workflows."""
from __future__ import annotations

import math
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

from learning_projects import check_exercises, courier, exercises, reference, study_bandit, two_step

ROOT = Path(__file__).resolve().parents[2]


class TwoStepTests(unittest.TestCase):
    def test_exact_values_and_zero_mean_advantage(self):
        v = two_step.delivery_values()
        self.assertAlmostEqual(v.continue_, 3.6)
        self.assertAlmostEqual(v.policy_value, 2.4)
        self.assertAlmostEqual(0.25 * v.continue_advantage + 0.75 * v.stop_advantage, 0)
        for gamma, p in ((0, 0), (0.5, 0.5), (1, 1)):
            v = two_step.delivery_values(gamma, p)
            self.assertAlmostEqual(v.policy_value, (1 - p) * 2 + p * gamma * 4)

    def test_bootstrap_changes_target_not_actual_reward(self):
        updates = two_step.mc_td_updates()
        self.assertAlmostEqual(updates["mc_new"], 2.3)
        self.assertAlmostEqual(updates["td_new"], 1.85)
        self.assertAlmostEqual(two_step.mc_td_updates(next_value=4)["td_target"], updates["mc_target"])
        with self.assertRaises(ValueError):
            two_step.delivery_values(float("nan"))


class ExerciseTests(unittest.TestCase):
    def test_reference_cases_pass_without_calling_learner(self):
        for name, cases in check_exercises.TASKS.values():
            self.assertEqual(check_exercises.check(getattr(reference, name), cases)[0], "PASS")

    def test_checker_identifies_pending_wrong_and_nonfinite_results(self):
        def pending(*args):
            raise NotImplementedError("not done")
        self.assertEqual(check_exercises.check(pending, [(1, ())])[0], "PENDING")
        self.assertEqual(check_exercises.check(lambda: float("nan"), [(1, ())])[0], "FAIL")
        self.assertEqual(check_exercises.check(lambda: "1", [(1, ())])[0], "FAIL")
        self.assertEqual(check_exercises.check(lambda: 2, [(1, ())])[0], "FAIL")

    def test_targeted_pending_checker_returns_one_without_using_reference(self):
        with patch.object(exercises, "mean_update", side_effect=NotImplementedError("Try exercise 1")), \
                patch.object(reference, "mean_update", side_effect=AssertionError("No fallback")), \
                patch.object(sys, "argv", ["check_exercises", "--exercise", "1"]), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(check_exercises.main(), 1)
        self.assertIn("PENDING", output.getvalue())
        self.assertNotIn("Exercise 2", output.getvalue())

    def test_learner_cli_refuses_pending_target_without_fallback(self):
        with patch.object(exercises, "q_target", side_effect=NotImplementedError("Try exercise 3")), \
                patch.object(reference, "q_target", side_effect=AssertionError("No fallback")), \
                patch.object(sys, "argv", ["courier", "--learner", "--episodes", "1"]), \
                redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as exited:
                courier.main()
        self.assertEqual(exited.exception.code, 2)
        self.assertIn("Try exercise 3", output.getvalue())

    def test_returns_match_backward_recursion_and_terminal_masks(self):
        rewards, gamma = [2, -1, 4, 0.5], 0.7
        backward = 0
        for r in reversed(rewards):
            backward = r + gamma * backward
        self.assertAlmostEqual(reference.discounted_return(rewards, gamma), backward)
        self.assertEqual(reference.q_target(5, 0.9, [100], True), 5)
        self.assertEqual(reference.q_target(-3, 0.9, [], True), -3)
        self.assertEqual(reference.q_target(1, 0.9, [10], False), 10)
        with self.assertRaises(ValueError):
            reference.q_target(1, 0.9, [], False)

    def test_reference_rejects_invalid_probabilities_and_costs(self):
        for probs in ([0], [float("nan")], [1.2]):
            with self.assertRaises(ValueError):
                reference.ips([1], probs, [1])
        with self.assertRaises(ValueError):
            reference.ips([1, 2], [1], [1])
        self.assertEqual(reference.shield(["risky"], {"risky": 1}, 0), [])
        with self.assertRaises(ValueError):
            reference.shield(["missing"], {}, 0)


class CourierTests(unittest.TestCase):
    def setUp(self):
        self.env = courier.GridCourier()

    def test_map_boundaries_hazard_and_goal_termination(self):
        edge = self.env.step((0, 0), "up")
        self.assertEqual(edge.state, (0, 0))
        self.assertEqual(edge.reward, -0.1)
        self.assertFalse(edge.terminated)
        hazard = self.env.step((1, 1), "down")
        self.assertEqual((hazard.state, hazard.reward, hazard.cost, hazard.terminated), ((1, 2), -3, 1, True))
        goal = self.env.step((3, 2), "down")
        self.assertEqual((goal.state, goal.reward, goal.cost, goal.terminated), ((3, 3), 4, 0, True))
        for terminal in (self.env.goal, *self.env.hazards):
            with self.assertRaises(ValueError):
                self.env.step(terminal, "up")
        with self.assertRaises(ValueError):
            self.env.step((0, 0), "diagonal")
        with self.assertRaises(ValueError):
            courier.GridCourier(width=0)
        with self.assertRaises(ValueError):
            courier.GridCourier(hazard_reward=float("inf"))

    def test_collector_cutoff_keeps_nonterminal_bootstrap(self):
        masks = []
        def record_target(reward, gamma, future, terminated):
            masks.append(terminated)
            return reference.q_target(reward, gamma, future, terminated)
        _, history = courier.train(self.env, episodes=8, max_steps=1, target=record_target)
        self.assertEqual(masks, [False] * 8)
        self.assertTrue(all(e.truncated and not e.terminated for e in history))

    def test_reproducibility_and_known_solvable_map_across_training_seeds(self):
        q1, h1 = courier.train(self.env, episodes=20, seed=3)
        q2, h2 = courier.train(self.env, episodes=20, seed=3)
        self.assertEqual((q1, h1), (q2, h2))
        for seed in (1, 7, 19):
            q, _ = courier.train(self.env, episodes=600, seed=seed)
            results = courier.evaluate(self.env, q, seed=seed + 1000)
            self.assertTrue(all(e.path[-1] == self.env.goal and e.cost == 0 for e in results))
            self.assertTrue(all(len(e.rewards) == 6 for e in results))
            self.assertTrue(all(math.isclose(e.reward, 3.5) for e in results))

    def test_shield_prevents_hazards_even_with_bad_reward_design(self):
        env = courier.GridCourier(hazard_reward=5)
        self.assertNotIn("down", env.allowed_actions((1, 1), shielded=True))
        for state in env.states():
            for action in env.allowed_actions(state, shielded=True):
                self.assertEqual(env.step(state, action).cost, 0)
        q, history = courier.train(env, episodes=600, shielded=True)
        self.assertTrue(all(e.cost == 0 for e in history))
        self.assertTrue(all(e.path[-1] == env.goal for e in courier.evaluate(env, q, shielded=True)))
        unshielded, _ = courier.train(env, episodes=600)
        self.assertTrue(all(e.cost == 1 for e in courier.evaluate(env, unshielded)))

    def test_invalid_training_parameters_and_bad_learner_value(self):
        for kwargs in ({"episodes": 0}, {"max_steps": 0}, {"alpha": 0}, {"epsilon": 1.1}, {"gamma": float("nan")}):
            with self.assertRaises(ValueError):
                courier.train(self.env, **kwargs)
        with self.assertRaises(ValueError):
            courier.train(self.env, target=lambda *args: float("nan"))
        with self.assertRaises(ValueError):
            courier.choose_action({}, (), 0, random.Random(1))


class BanditTests(unittest.TestCase):
    def test_propensities_include_exploitation_and_ties(self):
        self.assertEqual(study_bandit.probabilities([0, 0, 0], 0.2), (1 / 3,) * 3)
        p = study_bandit.probabilities([1, 0, 0], 0.3)
        self.assertAlmostEqual(p[0], 0.8)
        self.assertAlmostEqual(sum(p), 1)
        self.assertTrue(all(x > 0 for x in p))
        self.assertEqual(study_bandit.probabilities([1, 0, 0], 0), (1, 0, 0))

    def test_logged_propensity_is_actual_collection_probability(self):
        rows, means = study_bandit.collect(rounds=100, seed=7)
        self.assertEqual((rows, means), study_bandit.collect(rounds=100, seed=7))
        self.assertTrue(all(r.probabilities[r.action] > 0 for r in rows))
        estimates = study_bandit.evaluate_logs(rows)
        self.assertAlmostEqual(estimates.oracle, 0.8)
        self.assertTrue(0 <= estimates.snips <= 1)
        self.assertTrue(0 < estimates.effective_sample_size <= len(rows) + 1e-8)

    def test_fixed_population_ips_identity_and_hand_estimates(self):
        # Four observed fresh-context rows: weights [2/3, 2/3, 2/3, 2].
        target = {c: (0.5, 0.5, 0) for c in study_bandit.CONTEXTS}
        rows = [study_bandit.LogRow("fresh", a, r, (0.75, 0.25, 0))
                for a, r in ((0, 1), (0, 0), (0, 1), (1, 1))]
        estimates = study_bandit.evaluate_logs(rows, target)
        self.assertAlmostEqual(estimates.ips, 5 / 6)
        self.assertAlmostEqual(estimates.snips, 5 / 6)
        self.assertAlmostEqual(estimates.effective_sample_size, 3)
        for context in study_bandit.CONTEXTS:
            b = (0.6, 0.3, 0.1)
            expected = sum(bi * pi / bi * mu for bi, pi, mu in
                           zip(b, study_bandit.TARGET[context], study_bandit.MEANS[context]))
            self.assertAlmostEqual(expected, max(study_bandit.MEANS[context]))

    def test_zero_support_is_rejected_for_unobserved_target_action(self):
        rows = [study_bandit.LogRow("fresh", 1, 1, (0, 1, 0))]
        with self.assertRaisesRegex(ValueError, "missing support"):
            study_bandit.evaluate_logs(rows)
        # Full population support, but unlucky sample has no target overlap.
        rows = [study_bandit.LogRow("fresh", 1, 1, (1 / 3,) * 3)]
        with self.assertRaisesRegex(ValueError, "zero target overlap"):
            study_bandit.evaluate_logs(rows)
        with self.assertRaises(ValueError):
            study_bandit.collect(rounds=0)
        with self.assertRaises(ValueError):
            study_bandit.draw_action((0.2, 0.2), random.Random(1))

    def test_fresh_evaluation_reproducible_but_not_a_ranking_claim(self):
        for seed in (1, 7, 19):
            _, means = study_bandit.collect(rounds=1500, seed=seed)
            reward = study_bandit.evaluate_greedy(means, seed=seed + 1000)
            self.assertEqual(reward, study_bandit.evaluate_greedy(means, seed=seed + 1000))
            self.assertTrue(0 <= reward <= 1)
            self.assertEqual(max(range(3), key=lambda a: means["fresh"][a]), 0)
            self.assertEqual(max(range(3), key=lambda a: means["tired"][a]), 1)


class CommandLineTests(unittest.TestCase):
    def run_cli(self, module, *args, input_text=None):
        return subprocess.run([sys.executable, "-m", f"learning_projects.{module}", *args],
                              cwd=ROOT, text=True, input=input_text, capture_output=True, timeout=10)

    def test_reference_and_learner_checker_exit_codes_are_distinct(self):
        reference_run = self.run_cli("check_exercises", "--reference")
        self.assertEqual(reference_run.returncode, 0, reference_run.stderr)
        self.assertIn("Reference answers", reference_run.stdout)
        # Learner work may later be completed locally; exercise checker unit tests
        # verify pending behavior directly without depending on pristine TODOs.
        pending = check_exercises.check(lambda *args: (_ for _ in ()).throw(NotImplementedError("TODO")), [(0, ())])
        self.assertEqual(pending[0], "PENDING")

    def test_demo_commands_and_invalid_arguments(self):
        for module, args in (("two_step", ()), ("courier", ("--episodes", "50")),
                             ("study_bandit", ("--rounds", "100"))):
            result = self.run_cli(module, *args)
            self.assertEqual(result.returncode, 0, result.stderr)
        for module, args in (("two_step", ("--gamma", "2")),
                             ("courier", ("--episodes", "0")),
                             ("study_bandit", ("--rounds", "0"))):
            result = self.run_cli(module, *args)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("Traceback", result.stderr)

    def test_actual_play_loop_delivery_quit_and_eof(self):
        result = self.run_cli("courier", "--play", input_text="d\nd\nd\ns\ns\ns\n")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Delivered!", result.stdout)
        self.assertIn("hazard cost=0", result.stdout)
        for controls in ("q\n", ""):
            self.assertEqual(self.run_cli("courier", "--play", input_text=controls).returncode, 0)


if __name__ == "__main__":
    unittest.main()
