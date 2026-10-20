import test from 'node:test';
import assert from 'node:assert/strict';
import {banditRun, GRID, gridTransition, gridRun, gridBackup, ppoPoint, importanceEstimate, makeBanditLog} from '../src/plot-model.js';

const near = (a, b, tolerance = 1e-10) => assert.ok(Math.abs(a - b) < tolerance, `${a} ≠ ${b}`);

test('bandit regret uses expected opportunity cost, and sample means use actual arm counts', () => {
  const run = banditRun({seed: 3, epsilon: 1, steps: 40, means: [0, 1]});
  const last = run.at(-1);
  near(last.regret, last.counts[0]);
  near(last.totalReward, last.counts[1]);
  assert.deepEqual(last.estimates, [0, 1]);
  assert.equal(last.counts.reduce((a, b) => a + b), 40);
  assert.deepEqual(run, banditRun({seed: 3, epsilon: 1, steps: 40, means: [0, 1]}));
});

test('greedy and epsilon zero are identical and exploration at one still sometimes picks the best arm', () => {
  const run = banditRun({epsilon: 1, seed: 7, steps: 300});
  assert.ok(run.at(-1).counts.every(n => n > 50));
  assert.ok(run.every((row, i) => !i || row.explored));
  const greedy = banditRun({epsilon: 0, seed: 7});
  assert.ok(greedy.slice(1).every(row => !row.explored));
});

test('grid pays terminal reward on entry exactly once and boundary/wall collisions stay in place', () => {
  assert.deepEqual(gridTransition(3, 0), {next: 4, reward: 1, terminal: true});
  assert.deepEqual(gridTransition(4, 0), {next: 4, reward: 0, terminal: true});
  assert.deepEqual(gridTransition(0, 3), {next: 0, reward: -.04, terminal: false});
  assert.deepEqual(gridTransition(1, 1), {next: 1, reward: -.04, terminal: false});
});

test('synchronous Bellman sweeps propagate one transition per sweep', () => {
  const [initial, first, second] = gridRun({gamma: .9, sweeps: 2});
  assert.equal(initial[3], 0);
  near(first[3], 1);
  near(first[2], -.04);
  near(second[2], .86);
  near(second[3], 1);
  assert.equal(second[GRID.goal], 0);
  assert.equal(second[GRID.trap], 0);
  const converged = gridRun({gamma: .9, sweeps: 40}).at(-1);
  gridBackup(converged, .9).forEach((v, i) => near(v, converged[i]));
});

test('gamma zero ignores every future value, including a tempting distant goal', () => {
  const run = gridRun({gamma: 0, sweeps: 4});
  assert.deepEqual(run[1], run[4]);
  near(run[4][0], -.04);
  near(run[4][3], 1);
});

test('PPO clips only the improvement direction for either advantage sign', () => {
  near(ppoPoint(1.5, 2, .2).value, 2.4);
  near(ppoPoint(.5, 2, .2).value, 1);
  near(ppoPoint(.5, -2, .2).value, -1.6);
  near(ppoPoint(1.5, -2, .2).value, -3);
  near(ppoPoint(1.5, 2, .2).slope, 0);
  near(ppoPoint(1.5, -2, .2).slope, -2);
  assert.equal(ppoPoint(1.2, 2, .2).slope, null);
  assert.equal(ppoPoint(.8, -2, .2).slope, null);
  near(ppoPoint(0, 0, 0).value, 0);
});

test('importance estimators and ESS match a hand calculation', () => {
  const log = [{action: 0, reward: 1}, {action: 0, reward: 0}, {action: 0, reward: 0}, {action: 1, reward: 1}];
  const x = importanceEstimate(log, {behavior: .25, target: .75, cap: 1});
  assert.deepEqual(x.weights, [1 / 3, 1 / 3, 1 / 3, 3]);
  near(x.ips, 5 / 6); near(x.snips, 5 / 6); near(x.clipped, 1 / 3);
  near(x.ess, 12 / 7);
  const same = importanceEstimate(log, {behavior: .25, target: .25});
  near(same.ess, 4); near(same.ips, .5); near(same.snips, .5);
});

test('population support failure is not hidden by clipping; finite missing draws are distinct', () => {
  const log = [{action: 0, reward: 1}, {action: 0, reward: 0}];
  const bad = importanceEstimate(log, {behavior: 0, target: .5});
  assert.equal(bad.supported, false); assert.equal(bad.ips, null); assert.equal(bad.clipped, null);
  const finite = importanceEstimate(log, {behavior: .01, target: 1});
  assert.equal(finite.supported, true); assert.equal(finite.ips, 0); assert.equal(finite.snips, null);
  assert.equal(finite.ess, 0); assert.deepEqual(finite.missingObserved, [1]);
  const valid = importanceEstimate(log, {behavior: 0, target: 0});
  near(valid.ips, .5); near(valid.ess, 2);
});

test('logging fixtures are reproducible and behavior endpoints never select impossible actions', () => {
  assert.deepEqual(makeBanditLog({seed: 8}), makeBanditLog({seed: 8}));
  assert.ok(makeBanditLog({behavior: 0}).every(row => row.action === 0));
  assert.ok(makeBanditLog({behavior: 1}).every(row => row.action === 1));
});
