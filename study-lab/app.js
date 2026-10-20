"use strict";
const $ = id => document.getElementById(id);
const value = id => Number($(id).value);
const fmt = x => x === null ? "undefined" : Number(x.toFixed(4)).toString();
const formula = text => `<p class="formula">${text}</p>`;
function syncOutputs() {
  document.querySelectorAll("input[type=range]").forEach(input => {
    $(`${input.id}-value`).value = ["0.05"].includes(input.step) ? Number(input.value).toFixed(2) : input.value;
  });
}
function updateReturn() {
  const g = value("discount"), p = value("probability"), r = value("final-reward");
  const d = RL.delivery(g, p, r);
  $("final-reward-label").textContent = `reward ${r}`;
  const decision = Math.abs(d.go - d.stop) < 1e-9 ? "Both actions tie." : `${d.go > d.stop ? "Continue" : "Stop"} has the larger return.`;
  $("return-result").innerHTML = `<p class="label">Expected return under this policy</p><p class="big">V(s₀) = ${fmt(d.value)}</p>`
    + formula(`Q(stop) = 2 · Q(continue) = 0 + ${fmt(g)} × ${r} = ${fmt(d.go)}`)
    + formula(`V = ${fmt(1-p)} × 2 + ${fmt(p)} × ${fmt(d.go)} = ${fmt(d.value)}`)
    + `<p>${decision} This is a comparison of action values; V averages them using your policy.</p>`
    + formula(`A(continue) = ${fmt(d.go)} − ${fmt(d.value)} = ${fmt(d.advantageGo)}<br>A(stop) = 2 − ${fmt(d.value)} = ${fmt(d.advantageStop)}`)
    + `<p>Policy-gradient derivative: ${fmt(d.gradient)}. ${d.gradient > 1e-9 ? "Increase the continue logit." : d.gradient < -1e-9 ? "Decrease the continue logit." : "Zero in this case (or endpoint limit)."}</p>`;
}
function updateTD() {
  const r = value("td-reward"), g = value("td-discount"), next = value("next-value");
  const end = $("ending").value, terminal = end === "terminated";
  const d = RL.td(r,g,next,1,0.5,terminal);
  const reason = terminal ? "The task ended: set future value to zero, regardless of the next-value slider."
    : end === "truncated" ? "Collection ended, but the task has a future. Bootstrap from the final real observation; stop the trace at a reset."
    : "Both continue: use the next-state estimate.";
  $("td-result").innerHTML = `<p class="label">${terminal ? "No bootstrap" : "Bootstrap retained"}</p><p class="big">Target = ${fmt(d.target)}</p>`
    + formula(`Target = ${r} + ${terminal ? "0" : `${fmt(g)} × ${fmt(next)}`} = ${fmt(d.target)}`)
    + formula(`TD error = target − old value = ${fmt(d.target)} − 1 = ${fmt(d.error)}`)
    + formula(`Updated value = 1 + 0.5 × (${fmt(d.error)}) = ${fmt(d.updated)}`)
    + `<p>${reason}</p><p>The target and the updated estimate are different quantities.</p>`;
}
function updatePPO() {
  const ratio = value("ratio"), a = value("advantage"), e = value("epsilon");
  const d = RL.ppo(ratio,a,e);
  const state = a === 0 ? "Zero advantage: no actor signal from this sample." : d.kink ? "At the kink: the classical derivative is not unique. Autodiff uses a subgradient convention."
    : d.plateau ? "Plateau: this sample's actor-loss derivative is zero."
    : `Active branch: gradient descent ${a > 0 ? "increases" : "decreases"} this action's log probability.`;
  $("ppo-result").innerHTML = `<p class="label">Surrogate to maximise</p><p class="big">${fmt(d.surrogate)}</p>`
    + formula(`Raw: ${fmt(ratio)} × (${a}) = ${fmt(d.raw)}<br>Clipped candidate: ${fmt(d.clippedRatio)} × (${a}) = ${fmt(d.candidate)}`)
    + formula(`min(${fmt(d.raw)}, ${fmt(d.candidate)}) = ${fmt(d.surrogate)}<br>Loss to minimise = ${fmt(d.loss)}`)
    + `<p>${state}</p><p>Loss derivative with respect to new log probability:
       ${d.logGradient === null ? "not unique at the kink" : fmt(d.logGradient)}.</p>`;
  drawPPO(ratio,a,e);
}
function drawPPO(ratio, a, e) {
  const svg = $("ppo-chart");
  const x = r => 60 + r / 2 * 590, y = v => 125 - v * 23;
  const line = (x1,y1,x2,y2,style) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${style}/>`;
  let parts = `<title id="ppo-chart-title">PPO surrogate, advantage ${a}</title><desc id="ppo-chart-desc">Ratio ${fmt(ratio)}, surrogate ${fmt(RL.ppo(ratio,a,e).surrogate)}. ${a > 0 ? `Flat above ${fmt(1+e)}.` : a < 0 ? `Flat below ${fmt(1-e)}.` : "Zero for all ratios."} Solid: clipped surrogate. Dashed: raw product.</desc>`;
  for (const v of [-4,-2,0,2,4]) {
    parts += line(60,y(v),650,y(v),'stroke="#d3dcd6"')+`<text x="45" y="${y(v)+5}" text-anchor="end">${v}</text>`;
  }
  for (const r of [0,0.5,1,1.5,2]) parts += `<text x="${x(r)}" y="244" text-anchor="middle">${r}</text>`;
  for (const r of [1-e,1+e]) parts += line(x(r),25,x(r),222,'stroke="#ad632c" stroke-dasharray="3 4"');
  parts += line(x(0),y(0),x(2),y(2*a),'stroke="#687772" stroke-width="2" stroke-dasharray="7 5"');
  const kink = a > 0 ? 1+e : 1-e;
  const points = [0.001,kink,2].map(r => `${x(r)},${y(RL.ppo(r,a,e).surrogate)}`).join(" ");
  parts += `<polyline points="${points}" fill="none" stroke="#12665b" stroke-width="4"/>`;
  parts += `<circle cx="${x(ratio)}" cy="${y(RL.ppo(ratio,a,e).surrogate)}" r="6" fill="#a64a18" stroke="white" stroke-width="2"/>`;
  parts += `<text x="60" y="17">surrogate ↑</text><text x="355" y="272" text-anchor="middle">new / old probability ratio →</text>`;
  svg.innerHTML = parts;
}
function updateSupport() {
  const d = RL.support(value("logging-b"),value("target-b"),$("sample").value);
  const descriptions = {
    unsupported: "No support: the target assigns probability to an action the logger never takes. A value for that target is not identified by these logs.",
    impossible: "Impossible log: at least one recorded action has zero probability under the chosen logger. Choose a compatible log or logging probability.",
    missing: "Finite-sample coverage gap: an action used by the target is absent from this log. The arithmetic below is not evidence of a zero reward for that action.",
    ok: "Both target support and this log's action coverage are present. Four rows still provide very limited evidence."
  };
  $("support-result").innerHTML = `<p class="label">Check the evidence first</p><p>${descriptions[d.status]}</p>`
    + (d.rows.length ? `<p class="big">IPS = ${fmt(d.ips)}</p>`+formula(`IPS = sum(weight × reward) / 4<br>SNIPS = ${fmt(d.snips)}<br>Weight-based effective sample size = ${fmt(d.ess)}`)
      + `<p>${d.snips === null ? "SNIPS is undefined: the total weight is zero." : "SNIPS normalises the weights; it does not add observations."} A weight diagnostic is not a confidence interval.</p>`
      : `<p class="big">No valid estimate</p><p>Adding a small denominator would hide the problem.</p>`);
  $("support-rows").innerHTML = d.rows.length ? d.rows.map(r=>`<tr><th scope="row">${r.action}</th><td>${r.reward}</td><td>${fmt(r.b)}</td><td>${fmt(r.pi)}</td><td>${fmt(r.weight)}</td><td>${fmt(r.contribution)}</td></tr>`).join("")
    : `<tr><td colspan="6">Resolve the support or log-consistency problem above.</td></tr>`;
}
const updates = {"return-controls":updateReturn,"td-controls":updateTD,"ppo-controls":updatePPO,"support-controls":updateSupport};
for (const [id, update] of Object.entries(updates)) {
  $(id).addEventListener("input", () => {syncOutputs();update();});
  $(id).addEventListener("reset", () => setTimeout(() => {
    syncOutputs();update();
    const check = $(id).closest("section").querySelector(".check");
    check.reset(); check.querySelector(".feedback").textContent = "";
  },0));
  // Controls are an experiment, not a submission workflow.
  $(id).addEventListener("submit", e=>e.preventDefault());
}
const checks = {
  return:{answer:2, explanation:"Both actions return 2 because 0.5×4=2. Every mixture therefore has value 2; 0.8 is the action probability, not an extra discount."},
  td:{answer:6.5, explanation:"Keep the future: 2+0.9×5=6.5. Target 2 would be correct for true termination, not this external collection cutoff."},
  ppo:{answer:-1.6, explanation:"Raw is −1; the clipped candidate is 0.8×(−2)=−1.6. The minimum is −1.6. The training loss would be +1.6."},
  support:{answer:4, explanation:"The B row has weight 4 and weighted reward 16; A rows have weight zero. IPS divides 16 by all four logged rows, giving 4."}
};
document.querySelectorAll(".check").forEach(form => {
  form.addEventListener("submit", event => {
    event.preventDefault();
    const field = form.elements.answer, feedback = form.querySelector(".feedback");
    const expected = checks[form.dataset.check];
    if (!field.value.trim() || !Number.isFinite(Number(field.value))) {
      feedback.textContent = "Enter a finite number before checking."; return;
    }
    const correct = Math.abs(Number(field.value)-expected.answer) <= 0.005;
    feedback.dataset.correct = String(correct);
    feedback.textContent = `${correct ? "Correct." : "Recheck the calculation."} ${expected.explanation}`;
  });
});
syncOutputs();
Object.values(updates).forEach(update => update());
