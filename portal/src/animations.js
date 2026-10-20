const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>Number(x.toFixed(4));
export const animations={
 timeline:{title:'Follow one episode',week:'01',description:'Step through the delivery task. Watch when each reward arrives and which power of the discount applies.'},
 bellman:{title:'Watch values travel backward',week:'02',description:'One synchronous Bellman sweep at a time. Each new value uses the previous sweep’s values.'},
 gae:{title:'Build an advantage backward',week:'08',description:'Turn three fixed TD residuals into GAE. See exactly where the trace parameter changes the contribution.'},
 groups:{title:'Compare the group baselines',week:'20',description:'Change the reward of one completion. Compare inclusive centring, leave-one-out and group standardisation.'}
};
export function mountAnimation(host,type){
 let step=0,playing=null,gamma=.9,lambda=.8;const data=animations[type];
 host.innerHTML=`<div class="animation-card"><p class="eyebrow">Step through the mechanism · Week ${data.week}</p><h2>${data.title}</h2><p>${data.description}</p><div class="animation-stage" aria-label="Values at the current step"></div><div class="animation-explanation" aria-live="polite" aria-atomic="true"></div><div class="control-row"><button class="button" data-step>Next step →</button><button class="button secondary" data-play>Play slowly</button><button class="button secondary" data-reset>Reset</button><span data-counter class="tag"></span></div><div class="control-row" data-settings></div><p class="storage-note">Invented teaching fixture. Change one setting, predict the result, then step again.</p></div>`;
 const stage=host.querySelector('.animation-stage'),message=host.querySelector('.animation-explanation'),counter=host.querySelector('[data-counter]'),next=host.querySelector('[data-step]'),play=host.querySelector('[data-play]');
 const settings=host.querySelector('[data-settings]');
 if(type==='timeline'||type==='bellman')settings.innerHTML='<label>Discount γ <output data-gamma-value>0.90</output><input aria-label="Animation discount" data-gamma type="range" min="0" max="1" step="0.05" value="0.9"></label>';
 if(type==='gae')settings.innerHTML='<label>Trace λ <output data-lambda-value>0.80</output><input aria-label="GAE trace parameter" data-lambda type="range" min="0" max="1" step="0.05" value="0.8"></label>';
 if(type==='groups'){
  host.querySelector('[data-step]').hidden=true;play.hidden=true;counter.hidden=true;
  settings.innerHTML=[0,1,2,3].map(i=>`<label>Completion ${i+1}<select data-reward="${i}" aria-label="Reward for completion ${i+1}"><option value="0" ${i?'selected':''}>0 · incorrect</option><option value="1" ${i?'':'selected'}>1 · correct</option></select></label>`).join('');
 }
 function stop(){clearInterval(playing);playing=null;play.textContent='Play slowly';}
 function draw(){
  let values=[],note='',active=step;
  if(type==='timeline'){
   values=[['s₀','choose'],['s₁','finish'],['terminal','future value 0']];
   note=[`Start at s₀. Predict the return if you continue. The first reward has discount exponent zero.`,`Continue: receive 0 now and reach s₁. Accumulated discounted return = 0. One future reward remains.`,`Finish: receive 4 one step later. Return = 0 + ${fmt(gamma)} × 4 = ${fmt(4*gamma)}. The terminal state contributes no extra value.`][step];
  }else if(type==='bellman'){
   const v=step===0?[0,0,0]:step===1?[0,0,4]:step===2?[0,gamma*4,4]:[gamma*gamma*4,gamma*4,4];
   values=v.map((x,i)=>['s'+i,`V = ${fmt(x)}`]);active=step===0?-1:3-step;
   note=[`All estimates start at 0. The chain has one action per state: rewards 0, 0, then 4 on termination.`,`Sweep 1: s₂ sees the terminal reward 4. s₀ and s₁ still use old next-state values of 0.`,`Sweep 2: V(s₁) = 0 + ${fmt(gamma)} × 4 = ${fmt(gamma*4)}. V(s₀) still sees the previous V(s₁)=0.`,`Sweep 3: V(s₀) = 0 + ${fmt(gamma)} × ${fmt(gamma*4)} = ${fmt(gamma*gamma*4)}. These values now satisfy every backup.`][step];
  }else if(type==='gae'){
   const b=.9*lambda,a2=-1,a1=2+b*a2,a0=1+b*a1;
   values=[['t = 0',`Â = ${step>=3?fmt(a0):'?'}`],['t = 1',`Â = ${step>=2?fmt(a1):'?'}`],['t = 2',`Â = ${step>=1?'-1':'?'}`]];active=step===0?-1:3-step;
   note=[`TD residuals are (1, 2, −1). Discount γ=0.9, trace λ=${fmt(lambda)}, so γλ=${fmt(b)}. This trajectory terminates; the continuation advantage is 0.`,`Last step: Â₂ = δ₂ + γλ × 0 = −1. Start at the end of the trajectory.`,`Step back: Â₁ = 2 + ${fmt(b)} × (−1) = ${fmt(a1)}. The next residual reduces this estimate.`,`Step back again: Â₀ = 1 + ${fmt(b)} × ${fmt(a1)} = ${fmt(a0)}. Equivalently 1 + ${fmt(b)} × 2 − ${fmt(b*b)}. Stop this recurrence at a reset.`][step];
  }else{
   const r=[...host.querySelectorAll('[data-reward]')].map(n=>Number(n.value)),mean=r.reduce((a,b)=>a+b)/4,centered=r.map(x=>x-mean),std=Math.sqrt(centered.reduce((a,b)=>a+b*b,0)/4),rloo=centered.map(x=>4/3*x),grpo=centered.map(x=>std?x/std:0);
   values=r.map((x,i)=>[`y${i+1}`,`reward ${x}`]);active=-1;
   note=`Mean reward = ${fmt(mean)}. Population standard deviation = ${fmt(std)}.<br>Inclusive centred: (${centered.map(fmt).join(', ')}).<br>Leave-one-out: (${rloo.map(fmt).join(', ')}).<br>Group-standardised: (${grpo.map(fmt).join(', ')}).<br><strong>${std?'Signs agree, scales differ. Random reward-dependent standardisation is not just a fixed learning-rate change.':'All rewards tie: every centred coefficient is zero. A KL penalty can still update the policy, but centring cannot invent reward information.'}</strong>`;
  }
  stage.innerHTML=values.map((v,i)=>`<div class="anim-node ${i===active?'active':''}"><strong>${esc(v[0])}</strong><small>${esc(v[1])}</small></div>${i<values.length-1?'<span class="anim-arrow" aria-hidden="true">→</span>':''}`).join('');message.innerHTML=note;
  const max=type==='timeline'?2:3;counter.textContent=`Step ${step} / ${max}`;next.disabled=step===max;
  if(step===max)stop();
 }
 next.addEventListener('click',()=>{stop();step++;draw();});
 play.addEventListener('click',()=>{if(playing){stop();return;}if(step===(type==='timeline'?2:3))step=0;draw();play.textContent='Pause';playing=setInterval(()=>{step++;draw();},2200);});
 host.querySelector('[data-reset]').addEventListener('click',()=>{stop();step=0;gamma=.9;lambda=.8;host.querySelector('[data-gamma]')?.setAttribute('value','.9');if(host.querySelector('[data-gamma]'))host.querySelector('[data-gamma]').value='.9';if(host.querySelector('[data-lambda]'))host.querySelector('[data-lambda]').value='.8';host.querySelectorAll('[data-reward]').forEach((n,i)=>n.value=i?'0':'1');if(host.querySelector('[data-gamma-value]'))host.querySelector('[data-gamma-value]').value='0.90';if(host.querySelector('[data-lambda-value]'))host.querySelector('[data-lambda-value]').value='0.80';draw();});
 settings.addEventListener('input',()=>{stop();step=0;if(host.querySelector('[data-gamma]')){gamma=Number(host.querySelector('[data-gamma]').value);host.querySelector('[data-gamma-value]').value=gamma.toFixed(2);}if(host.querySelector('[data-lambda]')){lambda=Number(host.querySelector('[data-lambda]').value);host.querySelector('[data-lambda-value]').value=lambda.toFixed(2);}draw();});
 draw();return stop;
}
