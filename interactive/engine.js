/* Small, inspectable teaching models. No network, packages, or hidden training. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.RLLab = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  function bounded(name, value, min, max) {
    if (!Number.isFinite(value) || value < min || value > max) throw Error(name + ' must be finite and between ' + min + ' and ' + max);
  }
  const probability = (name, value) => bounded(name, value, 0, 1);
  function rng(seed) {
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw Error('Seed must be a nonnegative 32-bit integer');
    let x = seed >>> 0;
    return () => { x += 0x6D2B79F5; let t = x; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  const clip = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const sigmoid = x => 1 / (1 + Math.exp(-x));
  const argmax = xs => xs.indexOf(Math.max(...xs));
  function bandit(seed = 7) {
    return {random: rng(seed), means: [.25, .55, .75], counts: [0, 0, 0], values: [0, 0, 0], rewards: [], regrets: [], actions: []};
  }
  function pull(b, action) {
    if (!Number.isInteger(action) || action < 0 || action > 2) throw Error('Route must be 0, 1, or 2');
    const old = b.values[action];
    const reward = +(b.random() < b.means[action]);
    b.counts[action]++; b.values[action] += (reward - b.values[action]) / b.counts[action];
    b.actions.push(action); b.rewards.push(reward);
    b.regrets.push((b.regrets.at(-1) || 0) + Math.max(...b.means) - b.means[action]);
    return {action, reward, old, count: b.counts[action], estimate: b.values[action]};
  }
  function banditStep(b, epsilon) {
    probability('epsilon', epsilon);
    const unplayed = b.counts.indexOf(0);
    const action = unplayed >= 0 ? unplayed : b.random() < epsilon ? Math.floor(b.random() * 3) : argmax(b.values);
    return pull(b, action);
  }
  function delivery(gamma, probability = .25) {
    bounded('gamma', gamma, 0, 1); bounded('continue probability', probability, 0, 1);
    return {stop: 2, finish: 4, continue: gamma * 4, value: (1 - probability) * 2 + probability * gamma * 4};
  }
  function targets(gamma, nextValue, alpha, current = 1, terminal = false) {
    probability('gamma', gamma); probability('alpha', alpha);
    if (!Number.isFinite(nextValue) || !Number.isFinite(current)) throw Error('Value estimates must be finite');
    const mc = terminal ? 2 : gamma * 4;
    const td = terminal ? 2 : gamma * nextValue;
    return {mc, td, mcUpdate: current + alpha * (mc - current), tdUpdate: current + alpha * (td - current)};
  }
  const ACTIONS = [[-1, 0], [0, 1], [1, 0], [0, -1]];
  const ARROWS = ['↑', '→', '↓', '←'];
  const hazardous = s => s === 9 || s === 10;
  function transition(state, action, shield = false) {
    if (!Number.isInteger(state) || state < 0 || state > 15 || !Number.isInteger(action) || action < 0 || action > 3) throw Error('Invalid grid state/action');
    if (state === 15 || hazardous(state)) throw Error('Episode has ended: reset before moving');
    const r = clip(Math.floor(state / 4) + ACTIONS[action][0], 0, 3);
    const c = clip(state % 4 + ACTIONS[action][1], 0, 3);
    const attempted = r * 4 + c;
    const next = shield && hazardous(attempted) ? state : attempted;
    return {next, reward: next === 15 ? 4 : hazardous(next) ? -3 : -.1, cost: +hazardous(next), terminal: next === 15 || hazardous(next), blocked: next !== attempted};
  }
  function courier(seed = 7) {
    return {random: rng(seed), q: Array.from({length: 16}, () => [0, 0, 0, 0]), returns: [], costs: [], state: 0, steps: 0, total: 0, episodeCost: 0};
  }
  function courierStep(c, epsilon = .2, alpha = .3, gamma = .9, shield = false) {
    probability('epsilon', epsilon); bounded('alpha', alpha, Number.EPSILON, 1); probability('gamma', gamma);
    const action = c.random() < epsilon ? Math.floor(c.random() * 4) : argmax(c.q[c.state]);
    const from = c.state; const tr = transition(from, action, shield);
    const target = tr.reward + (tr.terminal ? 0 : gamma * Math.max(...c.q[tr.next]));
    const old = c.q[from][action]; c.q[from][action] += alpha * (target - old);
    c.state = tr.next; c.steps++; c.total += tr.reward; c.episodeCost += tr.cost;
    // A 60-step cutoff truncates interaction, not the Bellman target.
    const truncated = !tr.terminal && c.steps >= 60;
    if (tr.terminal || truncated) { c.returns.push(c.total); c.costs.push(c.episodeCost); c.state = 0; c.steps = 0; c.total = 0; c.episodeCost = 0; }
    return {...tr, from, action, old, target, updated: c.q[from][action], truncated};
  }
  function greedyPath(q, shield = false) {
    let state = 0; const path = [state];
    for (let i = 0; i < 60; i++) { const tr = transition(state, argmax(q[state]), shield); path.push(tr.next); state = tr.next; if (tr.terminal) return {path, success: state === 15, terminal: true}; }
    return {path, success: false, terminal: false};
  }
  function ppo(p, old = .25, epsilon = .2) {
    probability('new probability', p); bounded('old probability', old, Number.EPSILON, 1-Number.EPSILON); probability('clip epsilon', epsilon);
    const terms = [[p / old, 1.2, old], [(1 - p) / (1 - old), -.4, 1 - old]];
    return terms.reduce((sum, [ratio, advantage, weight]) => sum + weight * Math.min(ratio * advantage, clip(ratio, 1 - epsilon, 1 + epsilon) * advantage), 0);
  }
  function gae(deltas, gamma, lambda) {
    probability('gamma', gamma); probability('lambda', lambda);
    if (!Array.isArray(deltas) || !deltas.every(Number.isFinite)) throw Error('Residuals must be finite');
    return deltas.reduce((sum, delta, t) => sum + (gamma * lambda) ** t * delta, 0);
  }
  function logs(seed, n, behavior) {
    probability('behavior probability', behavior);
    if (!Number.isInteger(n) || n < 0 || n > 100000) throw Error('Log length must be a whole number from 0 to 100000');
    const random = rng(seed); const data = [];
    for (let i = 0; i < n; i++) { const action = +(random() < behavior); const propensity = action ? behavior : 1 - behavior; data.push({action, propensity, reward: +(random() < (action ? .75 : .55))}); }
    return data;
  }
  function ope(data, behavior, target) {
    probability('behavior probability', behavior); probability('target probability', target);
    if (!Array.isArray(data) || data.some(d => ![0,1].includes(d.action) || !Number.isFinite(d.reward) || !Number.isFinite(d.propensity) || d.propensity <= 0 || d.propensity > 1)) throw Error('Observed rows need action 0/1, finite reward and positive propensity');
    const supported = !(behavior === 0 && target > 0 || behavior === 1 && target < 1);
    const weights = data.map(d => (d.action ? target : 1 - target) / d.propensity);
    const sum = weights.reduce((a, b) => a + b, 0);
    const numerator = weights.reduce((a, w, i) => a + w * data[i].reward, 0);
    const squares = weights.reduce((a, w) => a + w * w, 0);
    return {supported, weights, ips: supported && data.length ? numerator / data.length : null, snips: supported && sum ? numerator / sum : null, ess: squares ? sum * sum / squares : 0, truth: target * .75 + (1 - target) * .55, missingSample: (target > 0 && !data.some(d => d.action === 1)) || (target < 1 && !data.some(d => d.action === 0))};
  }
  function safety(p, budget, lambda, shield) {
    probability('shortcut probability', p); bounded('cost budget', budget, 0, Number.MAX_VALUE); bounded('cost penalty', lambda, 0, Number.MAX_VALUE);
    const effective = shield ? 0 : p;
    return {reward: 3 + 2 * effective, cost: 2 * effective, feasible: 2 * effective <= budget + 1e-12, maximumP: Math.min(1, budget / 2), penalizedShortcut: 5 - 2 * lambda, effective};
  }
  function modelBias(horizon, gamma, error) { if(!Number.isInteger(horizon) || horizon<0 || horizon>100000) throw Error('Horizon must be a whole number from 0 to 100000'); probability('gamma', gamma); if(!Number.isFinite(error)) throw Error('Model error must be finite'); let bias = 0; for (let t = 0; t < horizon; t++) bias += gamma ** t * error; return bias; }
  function preference(p, reference = .5, beta = .5) {
    bounded('beta', beta, Number.EPSILON, Number.MAX_VALUE);
    if (!Number.isFinite(p) || !Number.isFinite(reference)) throw Error('Preference probabilities must be finite');
    if (p <= 0 || p >= 1 || reference <= 0 || reference >= 1) throw Error('Preference probabilities must be inside (0,1)');
    const margin = beta * (Math.log(p / (1 - p)) - Math.log(reference / (1 - reference)));
    return Math.max(0, -margin) + Math.log1p(Math.exp(-Math.abs(margin)));
  }
  function causal(mix) {
    probability('common easy-job share', mix);
    // Shortcut has better success within both difficulty strata, but receives harder jobs.
    const shortcut = {easy: [9, 10], hard: [27, 90]};
    const scenic = {easy: [72, 90], hard: [2, 10]};
    return {shortcutRaw: .36, scenicRaw: .74, shortcutAdjusted: mix * .9 + (1 - mix) * .3, scenicAdjusted: mix * .8 + (1 - mix) * .2, shortcut, scenic};
  }
  return {rng, clip, sigmoid, argmax, bandit, pull, banditStep, delivery, targets, ACTIONS, ARROWS, hazardous, transition, courier, courierStep, greedyPath, ppo, gae, logs, ope, safety, modelBias, preference, causal};
});
