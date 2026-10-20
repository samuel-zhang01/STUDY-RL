/* Deterministic teaching calculations, shared by the browser and Node checks. */
(function (root) {
  "use strict";
  const finite = (value, name) => {
    if (!Number.isFinite(value)) throw new Error(`${name} must be finite`);
    return value;
  };
  const unit = (value, name) => {
    finite(value, name);
    if (value < 0 || value > 1) throw new Error(`${name} must be between 0 and 1`);
    return value;
  };
  function delivery(gamma, probability, finalReward) {
    unit(gamma, "Discount"); unit(probability, "Probability"); finite(finalReward, "Reward");
    const stop = 2, go = gamma * finalReward;
    const value = (1 - probability) * stop + probability * go;
    return {stop, go, value, advantageStop: stop - value, advantageGo: go - value,
      gradient: (go - stop) * probability * (1 - probability)};
  }
  function td(reward, gamma, nextValue, oldValue, alpha, terminated) {
    [reward, nextValue, oldValue].forEach(v => finite(v, "Value"));
    unit(gamma, "Discount"); unit(alpha, "Learning rate");
    const bootstrap = terminated ? 0 : gamma * nextValue;
    const target = reward + bootstrap;
    return {bootstrap, target, error: target - oldValue, updated: oldValue + alpha * (target - oldValue)};
  }
  function ppo(ratio, advantage, epsilon) {
    finite(ratio, "Ratio"); finite(advantage, "Advantage"); unit(epsilon, "Clip radius");
    if (ratio <= 0) throw new Error("Ratio must be positive");
    const clippedRatio = Math.min(1 + epsilon, Math.max(1 - epsilon, ratio));
    const raw = ratio * advantage, candidate = clippedRatio * advantage;
    const surrogate = Math.min(raw, candidate);
    const plateau = advantage > 0 ? ratio > 1 + epsilon : advantage < 0 && ratio < 1 - epsilon;
    const kink = advantage !== 0 && Math.abs(ratio - (advantage > 0 ? 1 + epsilon : 1 - epsilon)) < 1e-10;
    return {clippedRatio, raw, candidate, surrogate, loss: -surrogate, plateau, kink,
      logGradient: kink ? null : (plateau || advantage === 0 ? 0 : -ratio * advantage)};
  }
  function support(loggingB, targetB, sample) {
    unit(loggingB, "Logging probability"); unit(targetB, "Target probability");
    if (!["mixed", "only-a"].includes(sample)) throw new Error("Unknown log");
    const actions = sample === "mixed" ? [0, 0, 0, 1] : [0, 0, 0, 0];
    // Report structural support separately from consistency of a fixed sample.
    if ((loggingB === 0 && targetB > 0) || (loggingB === 1 && targetB < 1))
      return {status: "unsupported", rows: [], ips: null, snips: null, ess: null};
    if (actions.some(a => (a ? loggingB : 1 - loggingB) === 0))
      return {status: "impossible", rows: [], ips: null, snips: null, ess: null};
    const rows = actions.map(a => {
      const reward = a ? 4 : 1, b = a ? loggingB : 1 - loggingB, pi = a ? targetB : 1 - targetB;
      return {action: a ? "B" : "A", reward, b, pi, weight: pi / b, contribution: pi / b * reward};
    });
    const totalWeight = rows.reduce((s, r) => s + r.weight, 0);
    const weightedReward = rows.reduce((s, r) => s + r.contribution, 0);
    const squaredWeight = rows.reduce((s, r) => s + r.weight ** 2, 0);
    const missing = (targetB > 0 && !actions.includes(1)) || (targetB < 1 && !actions.includes(0));
    return {status: missing ? "missing" : "ok", rows, ips: weightedReward / rows.length,
      snips: totalWeight > 0 ? weightedReward / totalWeight : null,
      ess: squaredWeight > 0 ? totalWeight ** 2 / squaredWeight : null};
  }
  const api = {delivery, td, ppo, support};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.RL = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
