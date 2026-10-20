const test = require("node:test");
const assert = require("node:assert/strict");
const {delivery, td, ppo, support} = require("./model.js");
const near = (a,b) => assert.ok(Math.abs(a-b)<1e-8, `${a} != ${b}`);

test("delivery: policy average, switch point and sigmoid gradient", () => {
  const d = delivery(.9,.25,4);
  near(d.value,2.4); near(d.go,3.6); near(d.gradient,.3);
  near(.25*d.advantageGo+.75*d.advantageStop,0);
  near(delivery(.5,.8,4).value,2);
  assert.ok(delivery(.4,.25,4).go<2);
  const z=Math.log(.25/.75), h=1e-5, sigmoid=x=>1/(1+Math.exp(-x));
  near((delivery(.9,sigmoid(z+h),4).value-delivery(.9,sigmoid(z-h),4).value)/(2*h),d.gradient);
  near(delivery(1,1,8).value,8); near(delivery(0,0,8).value,2);
});
test("TD: terminal masks bootstrap, external cutoff preserves it", () => {
  near(td(0,.9,3,1,.5,false).updated,1.85);
  near(td(2,.9,5,1,.5,false).target,6.5);
  near(td(2,.9,999,1,.5,true).target,2);
  near(td(-1,0,5,1,0,false).updated,1);
});
test("PPO: all four sign/boundary combinations and zero advantage", () => {
  [[1.5,2,2.4],[1.5,-2,-3],[.5,2,1],[.5,-2,-1.6]].forEach(([r,a,expected])=>near(ppo(r,a,.2).surrogate,expected));
  near(ppo(1.5,0,.2).logGradient,0);
  assert.equal(ppo(1.2,2,.2).logGradient,null);
  assert.equal(ppo(.8,-2,.2).logGradient,null);
  // Being outside the *opposite* threshold is not a kink.
  near(ppo(.8,2,.2).logGradient,-1.6);
  near(ppo(1.2,-2,.2).logGradient,2.4);
});
test("PPO: loss derivative agrees with perturbing log probability", () => {
  for(const a of [-2,2]) for(const r of [.3,.7,.9,1,1.1,1.3,1.7]) {
    const h=1e-6, d=ppo(r,a,.2);
    const numeric=(ppo(r*Math.exp(h),a,.2).loss-ppo(r*Math.exp(-h),a,.2).loss)/(2*h);
    near(d.logGradient,numeric);
  }
});
test("IPS: actual denominator, weight concentration, SNIPS distinction", () => {
  const d=support(.25,1,"mixed"); near(d.ips,4); near(d.snips,4); near(d.ess,1);
  const changed=support(.5,1,"mixed"); near(changed.ips,2); near(changed.snips,4);
  const identity=support(.25,.25,"mixed"); near(identity.ess,4); near(identity.ips,1.75);
});
test("support: distinguish structural absence, impossible log and missing sample", () => {
  assert.equal(support(0,1,"only-a").status,"unsupported");
  assert.equal(support(1,0,"mixed").status,"unsupported");
  assert.equal(support(0,0,"mixed").status,"impossible");
  assert.equal(support(1,1,"mixed").status,"impossible");
  const missing=support(.25,1,"only-a");
  assert.equal(missing.status,"missing"); near(missing.ips,0);
  assert.equal(missing.snips,null); assert.equal(missing.ess,null);
  assert.equal(support(0,0,"only-a").status,"ok");
  near(support(0,0,"only-a").ips,1);
});
test("models reject invalid numbers instead of displaying NaN", () => {
  assert.throws(()=>delivery(NaN,.5,4));
  assert.throws(()=>delivery(.9,2,4));
  assert.throws(()=>ppo(0,2,.2));
  assert.throws(()=>support(.2,.5,"unknown"));
});
