/**
 * Small, explicit teaching models. None of these fixtures is a benchmark result.
 * Functions are pure so the values in the charts can be independently checked.
 */
export function seededRandom(seed = 7) {
  let state = Number(seed) >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let t = state;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

export function banditRun({epsilon = .1, seed = 7, steps = 300, means = [.25, .5, .75]} = {}) {
  const decisions = seededRandom(seed), rewards = seededRandom(seed + 1009);
  const counts = means.map(() => 0), estimates = means.map(() => 0), successes = means.map(() => 0);
  const history = [{step: 0, regret: 0, reward: 0, counts: [...counts], estimates: [...estimates]}];
  let regret = 0, totalReward = 0;
  for (let t = 1; t <= steps; t++) {
    // A fixed number of random draws each round provides common randomness
    // when comparing two policies with the same seed.
    const explore = decisions(), choice = decisions(), tie = decisions();
    const potentialRewards = means.map(p => Number(rewards() < p));
    const bestEstimate = Math.max(...estimates), greedy = estimates.flatMap((v, a) => v === bestEstimate ? [a] : []);
    const action = explore < epsilon ? Math.floor(choice * means.length) : greedy[Math.floor(tie * greedy.length)];
    const reward = potentialRewards[action];
    counts[action]++; successes[action] += reward;
    estimates[action] += (reward - estimates[action]) / counts[action];
    totalReward += reward;
    regret += Math.max(...means) - means[action];
    history.push({step: t, action, reward, totalReward, regret, explored: explore < epsilon, counts: [...counts], estimates: [...estimates], successes: [...successes]});
  }
  return history;
}

export const GRID = Object.freeze({
  width: 5, height: 4, goal: 4, trap: 19, walls: [6, 8],
  actions: [{name:'East', arrow:'→', dx:1, dy:0}, {name:'South', arrow:'↓', dx:0, dy:1}, {name:'West', arrow:'←', dx:-1, dy:0}, {name:'North', arrow:'↑', dx:0, dy:-1}]
});
export const isTerminal = s => s === GRID.goal || s === GRID.trap;
export function gridTransition(state, action, cost = -.04) {
  if (GRID.walls.includes(state) || isTerminal(state)) return {next: state, reward: 0, terminal: true};
  const a = GRID.actions[action], x = state % GRID.width, y = Math.floor(state / GRID.width);
  const nx = x + a.dx, ny = y + a.dy, candidate = ny * GRID.width + nx;
  const next = nx < 0 || nx >= GRID.width || ny < 0 || ny >= GRID.height || GRID.walls.includes(candidate) ? state : candidate;
  return {next, reward: next === GRID.goal ? 1 : next === GRID.trap ? -1 : cost, terminal: isTerminal(next)};
}
export function gridBackup(values, gamma = .9, cost = -.04) {
  return values.map((_, state) => {
    if (GRID.walls.includes(state) || isTerminal(state)) return 0;
    return Math.max(...GRID.actions.map((_, action) => {
      const tr = gridTransition(state, action, cost);
      return tr.reward + gamma * (tr.terminal ? 0 : values[tr.next]);
    }));
  });
}
export function gridRun({gamma = .9, cost = -.04, sweeps = 40} = {}) {
  const history = [Array(GRID.width * GRID.height).fill(0)];
  for (let i = 0; i < sweeps; i++) history.push(gridBackup(history.at(-1), gamma, cost));
  return history;
}
export function gridActions(values, state, gamma = .9, cost = -.04) {
  return GRID.actions.map((action, i) => {
    const tr = gridTransition(state, i, cost);
    return {...action, ...tr, future: tr.terminal ? 0 : values[tr.next], q: tr.reward + gamma * (tr.terminal ? 0 : values[tr.next])};
  });
}

export function ppoPoint(ratio, advantage, epsilon) {
  const clippedRatio = Math.max(1 - epsilon, Math.min(1 + epsilon, ratio));
  const raw = ratio * advantage, capped = clippedRatio * advantage;
  const value = Math.min(raw, capped);
  const threshold = advantage > 0 ? 1 + epsilon : 1 - epsilon;
  const kink = advantage !== 0 && Math.abs(ratio - threshold) < 1e-10;
  const plateau = advantage > 0 ? ratio > threshold : advantage < 0 ? ratio < threshold : true;
  return {ratio, raw, capped, clippedRatio, value, plateau, kink, slope: kink ? null : plateau ? 0 : advantage};
}

export function makeBanditLog({behavior = .2, seed = 7, size = 200, means = [.25, .85]} = {}) {
  const random = seededRandom(seed);
  return Array.from({length: size}, (_, i) => {
    const actionDraw = random(), rewardDraw = random();
    const action = Number(actionDraw < behavior);
    return {i: i + 1, action, reward: Number(rewardDraw < means[action])};
  });
}
export function importanceEstimate(log, {behavior = .2, target = .8, cap = 5, means = [.25, .85]} = {}) {
  const b = [1 - behavior, behavior], p = [1 - target, target];
  const supported = p.every((probability, a) => probability === 0 || b[a] > 0);
  const weights = log.map(row => b[row.action] > 0 ? p[row.action] / b[row.action] : 0);
  const sum = weights.reduce((a, w) => a + w, 0), squares = weights.reduce((a, w) => a + w * w, 0);
  const weightedReward = weights.reduce((a, w, i) => a + w * log[i].reward, 0);
  const clippedReward = weights.reduce((a, w, i) => a + Math.min(w, cap) * log[i].reward, 0);
  const summary = [0, 1].map(action => {
    const rows = log.filter(row => row.action === action), successes = rows.reduce((s, row) => s + row.reward, 0);
    return {action, count: rows.length, successes, mean: rows.length ? successes / rows.length : null, weight: b[action] ? p[action] / b[action] : null, behavior: b[action], target: p[action]};
  });
  return {
    supported, weights, sum, weightedReward, summary,
    ips: supported && log.length ? weightedReward / log.length : null,
    snips: supported && sum > 0 ? weightedReward / sum : null,
    clipped: supported && log.length ? clippedReward / log.length : null,
    ess: squares ? sum * sum / squares : 0,
    truth: p.reduce((s, prob, a) => s + prob * means[a], 0),
    observed: log.length ? log.reduce((s, row) => s + row.reward, 0) / log.length : null,
    missingObserved: summary.filter(row => row.target > 0 && row.count === 0).map(row => row.action)
  };
}
