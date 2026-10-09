'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const E = require('../engine.js');
const close = (a,b,tolerance=1e-10) => assert.ok(Math.abs(a-b)<tolerance, `${a} != ${b}`);

test('seeded bandit estimates equal observed sample means; regret is oracle gap accounting',()=>{
  const a=E.bandit(99),b=E.bandit(99);
  for(let i=0;i<300;i++){E.banditStep(a,.2);E.banditStep(b,.2);}
  assert.deepEqual(a.actions,b.actions);assert.deepEqual(a.rewards,b.rewards);
  assert.equal(a.counts.reduce((x,y)=>x+y,0),300);
  a.counts.forEach((count,action)=>close(a.values[action],a.rewards.filter((_,i)=>a.actions[i]===action).reduce((x,y)=>x+y,0)/count));
  close(a.regrets.at(-1),a.actions.reduce((r,action)=>r+.75-a.means[action],0));
  assert.ok(a.regrets.every((r,i)=>!i||r>=a.regrets[i-1]-1e-12));
  assert.throws(()=>E.pull(a,3));
});
test('Bellman policy value and MC/TD targets match hand calculations at endpoints',()=>{
  close(E.delivery(.9,.25).value,2.4);close(E.delivery(.5).continue,2);
  close(E.delivery(0,1).value,0);close(E.delivery(1,0).value,2);
  const t=E.targets(.9,3,.5);close(t.mcUpdate,2.3);close(t.tdUpdate,1.85);
  assert.deepEqual(E.targets(.9,999,.5,1,true),{mc:2,td:2,mcUpdate:1.5,tdUpdate:1.5});
  close(E.targets(1,3,0).tdUpdate,1);close(E.targets(1,3,1).tdUpdate,3);
});
test('grid boundaries, terminal transitions and shielding preserve episode semantics',()=>{
  assert.deepEqual(E.transition(0,0),{next:0,reward:-.1,cost:0,terminal:false,blocked:false});
  const hazard=E.transition(8,1);assert.equal(hazard.next,9);assert.equal(hazard.reward,-3);assert.equal(hazard.terminal,true);assert.equal(hazard.cost,1);
  assert.deepEqual(E.transition(8,1,true),{next:8,reward:-.1,cost:0,terminal:false,blocked:true});
  assert.equal(E.transition(14,1).reward,4);assert.equal(E.transition(14,1).terminal,true);
  assert.throws(()=>E.transition(15,0));assert.throws(()=>E.transition(9,1));assert.throws(()=>E.transition(-1,0));
});
test('terminal Q target never bootstraps; truncation does bootstrap',()=>{
  const c=E.courier(3);c.state=14;c.q[14][1]=2;c.q[15]=[99,99,99,99];
  const end=E.courierStep(c,0,.5,.9);close(end.target,4);close(end.updated,3);assert.equal(c.state,0);assert.equal(c.returns.length,1);
  const d=E.courier(3);d.steps=59;d.q[0]=[1,0,0,0];
  const cut=E.courierStep(d,0,.5,.9);assert.equal(cut.truncated,true);assert.equal(cut.terminal,false);close(cut.target,.8);
});
test('Q-learning reaches a safe six-move route across held-out seeds; shielding avoids all hazards',()=>{
  for(const seed of [1,7,23,81,2026]){
    const c=E.courier(seed);while(c.returns.length<400)E.courierStep(c,.2,.3,.9);
    const route=E.greedyPath(c.q);assert.equal(route.success,true,`seed ${seed}`);assert.equal(route.path.length,7);assert.ok(route.path.every(s=>!E.hazardous(s)));
    assert.ok(c.q[15].every(x=>x===0));assert.ok(c.q[9].every(x=>x===0));
  }
  const shielded=E.courier(7);while(shielded.returns.length<100)E.courierStep(shielded,.5,.3,.9,true);assert.ok(shielded.costs.every(x=>x===0));
});
test('PPO positive and negative advantage clipping obeys min, not absolute clipping',()=>{
  close(E.ppo(.25),0);close(E.ppo(.3),.08);close(E.ppo(.4),.12);close(E.ppo(.8),.12);
  assert.ok(E.ppo(.05)<0);
});
test('GAE weights reduce to one-step at lambda0 and discounted residuals at lambda1',()=>{
  close(E.gae([1,-.5,2],.9,0),1);
  close(E.gae([1,-.5,2],.9,1),2.17);
  close(E.gae([], .9,.95),0);
});
test('OPE hand weights, zero support, empty evidence and finite-sample warnings',()=>{
  const data=[{action:1,propensity:.5,reward:1},{action:0,propensity:.5,reward:0}];
  const r=E.ope(data,.5,.75);close(r.ips,.75);close(r.snips,.75);close(r.ess,1.6);
  const none=E.ope(E.logs(7,200,0),0,1);assert.equal(none.supported,false);assert.equal(none.ips,null);assert.equal(none.snips,null);
  assert.equal(E.ope([], .5,1).ips,null);
  assert.equal(E.ope([{action:0,propensity:.9,reward:1}],.1,1).missingSample,true);
  const same=E.logs(7,200,.5);assert.deepEqual(same,E.logs(7,200,.5));
});
test('OPE is accurate on average across seeds, not guaranteed for a single log',()=>{
  let ips=0,snips=0;for(let seed=0;seed<100;seed++){const r=E.ope(E.logs(seed,500,.3),.3,.8);ips+=r.ips;snips+=r.snips;}
  close(ips/100,.71,.025);close(snips/100,.71,.025);
});
test('expected safety, perfect shield and model-error accumulation match independent sums',()=>{
  const s=E.safety(.25,.5,1,false);close(s.cost,.5);assert.equal(s.feasible,true);close(s.maximumP,.25);close(s.penalizedShortcut,3);
  assert.equal(E.safety(.3,.5,0,false).feasible,false);close(E.safety(1,0,0,true).cost,0);
  close(E.modelBias(20,1,.2),4);close(E.modelBias(0,.9,1),0);close(E.modelBias(10,.9,.2),.2*(1-.9**10)/.1);
});
test('DPO is stable, prefers chosen answer, and standardization reverses the raw ranking',()=>{
  close(E.preference(.5),Math.log(2));assert.ok(E.preference(.9)<E.preference(.5));assert.ok(E.preference(.01)>E.preference(.5));assert.throws(()=>E.preference(0));
  const c=E.causal(.5);assert.ok(c.shortcutRaw<c.scenicRaw);close(c.shortcutAdjusted,.6);close(c.scenicAdjusted,.5);
  for(const mix of [0,.5,1])close(E.causal(mix).shortcutAdjusted-E.causal(mix).scenicAdjusted,.1);
});
test('invalid probabilities, counts and nonfinite values fail instead of corrupting graphs',()=>{
  for(const bad of [NaN,Infinity,-.1,1.1]){
    assert.throws(()=>E.delivery(bad));assert.throws(()=>E.banditStep(E.bandit(),bad));assert.throws(()=>E.logs(7,20,bad));assert.throws(()=>E.causal(bad));
  }
  assert.throws(()=>E.rng(-1));assert.throws(()=>E.logs(7,Infinity,.5));assert.throws(()=>E.modelBias(-1,.9,.2));
  assert.throws(()=>E.ope([{action:1,propensity:0,reward:1}],.5,1));assert.throws(()=>E.ppo(.3,0));assert.throws(()=>E.preference(NaN));assert.throws(()=>E.gae([NaN],.9,.5));
});
