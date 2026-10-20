import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {parseHTML} from 'linkedom';
import {QUESTIONS,QUESTION_BY_ID,questionsForWeek} from '../src/review-content.js';
import {normalizeQuizRecord,recordAnswer,markUncertain,reviewQueue,reviewStats} from '../src/review-model.js';
import {mountDiagnostics,mountReview} from '../src/review.js';

const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const NOW='2026-10-20T12:00:00.000Z',T=Date.parse(NOW),DAY=86400000;
const answer=id=>QUESTION_BY_ID[id].choices[QUESTION_BY_ID[id].correctIndex];
const norm=value=>value.toLowerCase().replace(/[’']/g,'').replace(/[^\p{L}\p{N}]+/gu,' ').trim();

test('all 26 weeks have three distinct, complete diagnostics with live repair and code handoffs',()=>{
  assert.equal(QUESTIONS.length,78);assert.equal(new Set(QUESTIONS.map(q=>q.id)).size,78);
  for(let week=1;week<=26;week++)assert.deepEqual(questionsForWeek(week).map(q=>q.kind),['concept','calculate','transfer']);
  for(const q of QUESTIONS){
    assert.match(q.id,/^w(?:0[1-9]|1\d|2[0-6])-[a-z0-9-]+$/);
    assert.ok(q.prompt.length>35,`${q.id}: substantive prompt`);
    assert.equal(new Set(q.choices).size,q.choices.length);
    assert.ok(q.choices.length>=3&&q.choices.length<=4);assert.equal(q.feedback.length,q.choices.length);
    for(const feedback of q.feedback)assert.ok(feedback.length>40,`${q.id}: explanatory feedback`);
    assert.ok(Number.isInteger(q.correctIndex)&&q.correctIndex>=0&&q.correctIndex<q.choices.length);
    assert.ok(q.handoff.route.startsWith(`#/lesson/${q.week}?view=workbook`));
    assert.match(q.repair.route,/^#\/lesson\/(0[1-9]|1\d|2[0-6])\?view=guide&section=.+$/);
    assert.ok(fs.existsSync(path.join(ROOT,q.notebook)),`${q.id}: actual notebook`);
    assert.ok(fs.existsSync(path.join(ROOT,q.handoff.path)),`${q.id}: actual helper`);
    assert.equal(q.handoff.source,`https://github.com/samuel-zhang01/STUDY-RL/blob/main/${q.handoff.path}${q.handoff.sourceLine?'#L'+q.handoff.sourceLine:''}`);
    if(q.handoff.sourceLine){
      const lines=fs.readFileSync(path.join(ROOT,q.handoff.path),'utf8').split('\n');
      assert.match(lines[q.handoff.sourceLine-1],new RegExp(`(?:def|class)\\s+${q.handoff.sourceSymbol}\\b`),`${q.id}: exact helper line`);
    }
    if(Number.isInteger(q.handoff.cell)){
      const notebook=JSON.parse(fs.readFileSync(path.join(ROOT,q.notebook),'utf8'));
      assert.ok(notebook.cells[q.handoff.cell-1].source.join('').includes(q.handoff.cellNeedle),`${q.id}: meaningful one-based cell target`);
      assert.equal(q.handoff.route,`#/lesson/${q.week}?view=workbook&section=cell-${q.handoff.cell}`);
    }
    const guide=fs.readFileSync(path.join(ROOT,path.dirname(q.notebook),'teaching_guide.md'),'utf8');
    assert.ok(norm(guide).includes(norm(q.sourceSection)),`${q.id}: section exists: ${q.sourceSection}`);
    const slug=s=>s.toLowerCase().replace(/[^\p{L}\p{N}_\- ]/gu,'').replace(/ /g,'-');
    const expected=new URLSearchParams(q.repair.route.split('?')[1]).get('section');
    assert.ok(guide.split('\n').filter(line=>line.startsWith('## ')).some(line=>slug(line.slice(3))===expected),`${q.id}: exact guide anchor`);
    assert.ok(q.handoff.task.length>45,`${q.id}: specific coding handoff`);
  }
  assert.deepEqual(new Set(QUESTIONS.map(q=>q.correctIndex)),new Set([0,1,2]));
});

test('numerical answer keys agree with independently calculated examples from every week',()=>{
  const close=(value,expected)=>assert.ok(Math.abs(value-expected)<1e-10,`${value} != ${expected}`);
  const key=(id,start)=>assert.ok(answer(id).startsWith(start),`${id}: ${answer(id)}`);
  // One or more independent finite calculations for every week's numeric item.
  close((2*.5+1)/3,2/3);key('w01-arm-count','2/3');
  const vb=2/(1-.5),va=1+.5*vb;assert.deepEqual([va,vb],[3,4]);key('w02-continuing-loop','(3, 4)');
  close([1,2,3].reduce((s,r,i)=>s+r*.5**i,0),2.75);key('w03-first-return','2.75');
  close(2+.5*(5-2),3.5);key('w04-terminal-target','3.5');
  close(1-.1*((1*2-3)*2),1.2);key('w05-linear-gradient','1.2');
  close([.25,.75].reduce((s,p,i)=>s+p/(2*p)*[2,6][i],0),4);key('w06-per-correction','4');
  close(.2*(1-.2),.16);key('w07-bernoulli-score','0.16');
  close(1+.9*.5*2,1.9);key('w08-gae-residuals','1.9');
  close(-Math.min(1.5*-2,Math.min(1.2,Math.max(.8,1.5))*-2),3);key('w09-ppo-loss-sign','3');
  close(1+.9*(1-1)*(Math.min(4,6)-.2*-2),1);key('w10-soft-terminal','1');
  const weights=[2/3,2],sum=weights.reduce((a,b)=>a+b);close(sum/2,4/3);close(sum/sum,1);key('w11-ips-range','(4/3, 1)');
  close(Math.abs(1/2-1/8),.375);key('w12-selection-denominator','0.375');
  const scores=[[.6,3,-.4],[.4,0,0]].map(([p,n,q])=>-q+p*Math.sqrt(16)/(1+n));assert.deepEqual(scores,[1,1.6]);key('w13-puct-choice','B: 1.6');
  const awr=[.3,.7].map(b=>b*Math.exp(3*.4));close(awr[0]/(awr[0]+awr[1]),.3);key('w14-awr-frequency','(0.3,0.7)');
  close(10-(-1),11);key('w15-remaining-return','11');
  const g1=2+.9*0*(.5*10+.5*10);close(1+.9*(.5*3+.5*g1),3.25);key('w16-lambda-terminal','3.25');
  const target=[1,-1],p=[0,0],next=p.map((x,i)=>x-.25*2*(x-target[i]));close(next.reduce((s,x,i)=>s+(x-target[i])**2,0),.5);key('w17-rnd-update','0.5');
  const shaped=[.2,-.1,.3].map((x,i)=>-.1*x+(i===2?1:0));shaped.forEach((x,i)=>close(x,[-.02,.01,.97][i]));key('w18-rlhf-shaped','(−0.02, +0.01, 0.97)');
  close(.1*((-3+3.5)-(-5+3.5)),.2);key('w19-dpo-beta','0.2');
  const rewards=[1,0,0,0],rloo=rewards.map((r,i)=>r-rewards.filter((_,j)=>j!==i).reduce((a,b)=>a+b)/3);assert.deepEqual(rloo,[1,-1/3,-1/3,-1/3]);key('w20-rloo-values','(1, −1/3');
  close(1/(1+3)*4,1);key('w21-goodhart-shrinkage','1');
  close(Math.max(0,.5+.2*(1.5-1)),.6);key('w22-multiplier-sign','0.6');
  const boundary=-1/-2;assert.ok(-2*boundary>=-1);close(boundary,.5);key('w23-barrier-direction','0.5');
  const w=[1.5,2,.5,1/3],r=[1,0,1,0];close(w.reduce((s,x,i)=>s+x*r[i],0)/w.reduce((a,b)=>a+b),6/13);key('w24-capstone-snips','6/13');
  const pi=[.75,.25],mu=[.5,.5],qm=[.6,.2],a=[0,1,0],rw=[1,0,0];const dm=pi.reduce((s,x,i)=>s+x*qm[i],0);close(a.reduce((s,j,i)=>s+dm+pi[j]/mu[j]*(rw[i]-qm[j]),0)/3,11/30);key('w25-dr-contributions','11/30');
  close(.5*(3-2)+.5*(1-0),1);key('w26-standardisation','1');
});

test('imports derive correctness from the bank and reject malformed or unrelated evidence',()=>{
  const id='w09-ppo-loss-sign',q=QUESTION_BY_ID[id];
  const good=recordAnswer(id,q.correctIndex,'confident',null,T);
  assert.equal(normalizeQuizRecord(id,{...good,correct:false}).correct,true);
  assert.equal(normalizeQuizRecord(id,{...good,answer:(q.correctIndex+1)%3,correct:true}).correct,false);
  for(const value of [null,[],{}, {...good,attempts:0},{...good,lastReviewed:'not-a-date'}])assert.equal(normalizeQuizRecord(id,value),null);
  assert.equal(normalizeQuizRecord('__proto__',good),null);
  assert.equal(normalizeQuizRecord('constructor',good),null);
  assert.equal(normalizeQuizRecord('toString',good),null);
  assert.equal(normalizeQuizRecord('unknown',good),null);
  assert.equal(normalizeQuizRecord(id,{...good,nextReview:'2099-01-01T00:00:00Z'}).nextReview,good.lastReviewed);
  assert.equal(normalizeQuizRecord(id,{...good,nextReview:'2020-01-01T00:00:00Z'}).nextReview,good.lastReviewed);
  assert.equal(normalizeQuizRecord(id,{...good,attempts:1e10}).attempts,100000);
  assert.throws(()=>recordAnswer(id,99,'confident',null,T));
  assert.throws(()=>recordAnswer(id,q.correctIndex,'probably',null,T));
  assert.throws(()=>recordAnswer(id,q.correctIndex,'confident',null,'bad'));
});

test('missed, unsure and confident evidence have distinct schedules; early retries do not double spacing',()=>{
  const q=QUESTION_BY_ID['w01-arm-count'];
  const wrong=recordAnswer(q.id,(q.correctIndex+1)%3,'confident',null,T);
  assert.equal(Date.parse(wrong.nextReview)-T,600000);
  const unsure=recordAnswer(q.id,q.correctIndex,'unsure',wrong,T);
  assert.equal(Date.parse(unsure.nextReview)-T,DAY);assert.equal(unsure.attempts,2);
  let confident=recordAnswer(q.id,q.correctIndex,'confident',unsure,T);
  assert.equal(Date.parse(confident.nextReview)-T,3*DAY);
  const early=recordAnswer(q.id,q.correctIndex,'confident',confident,T+60000);
  assert.equal(Date.parse(early.nextReview)-Date.parse(early.lastReviewed),3*DAY);
  for(const days of [6,12,24,30,30]){
    const due=Date.parse(confident.nextReview);confident=recordAnswer(q.id,q.correctIndex,'confident',confident,due);
    assert.equal(Date.parse(confident.nextReview)-due,days*DAY);
  }
  const failed=recordAnswer(q.id,(q.correctIndex+1)%3,'unsure',confident,Date.parse(confident.nextReview));
  const recovered=recordAnswer(q.id,q.correctIndex,'confident',failed,Date.parse(failed.nextReview));
  assert.equal(Date.parse(recovered.nextReview)-Date.parse(recovered.lastReviewed),3*DAY);
});

test('marking uncertainty creates no attempt and does not alter a previous answer',()=>{
  const q=QUESTION_BY_ID['w26-prediction-causation'];
  const bookmark=markUncertain(q.id,null,T);
  assert.deepEqual(bookmark,{answer:null,correct:false,confidence:'unsure',attempts:0,lastReviewed:null,nextReview:NOW});
  const first=recordAnswer(q.id,q.correctIndex,'confident',bookmark,T);
  assert.equal(first.attempts,1);
  const marked=markUncertain(q.id,first,T+1000);
  assert.equal(marked.answer,first.answer);assert.equal(marked.correct,true);assert.equal(marked.attempts,1);assert.equal(marked.confidence,'unsure');
  const stats=reviewStats({[q.id]:bookmark},{now:T});
  assert.deepEqual(stats,{total:78,attempted:0,correct:0,missed:0,unsure:1,due:1,priority:1});
});

test('mixed review interleaves weeks and distinguishes future, due, missed and unsure items',()=>{
  const a=questionsForWeek(1),b=questionsForWeek(2);const quiz={};
  for(const q of a)quiz[q.id]=markUncertain(q.id,null,T);
  quiz[b[0].id]=recordAnswer(b[0].id,(b[0].correctIndex+1)%3,'confident',null,T);
  quiz[b[1].id]=recordAnswer(b[1].id,b[1].correctIndex,'confident',null,T);
  const priority=reviewQueue(quiz,{now:T});
  assert.deepEqual(priority.slice(0,3).map(row=>row.question.week),['02','01','01']);
  assert.equal(reviewQueue(quiz,{now:T,filter:'due'}).length,3);
  assert.equal(reviewQueue(quiz,{now:T+600000,filter:'due'}).length,4);
  assert.equal(reviewQueue(quiz,{now:T,filter:'missed'}).length,1);
  assert.equal(reviewQueue(quiz,{now:T,filter:'unsure'}).length,3);
  assert.equal(reviewQueue(quiz,{now:T,filter:'all',week:2}).length,3);
  assert.equal(reviewQueue(quiz,{now:T,filter:'priority',week:2}).length,1);
  assert.equal(reviewQueue({}, {now:T}).length,0);
  assert.equal(reviewQueue({}, {now:T,filter:'all'}).length,78);
  assert.throws(()=>reviewQueue({}, {now:T,filter:'other'}));
});

function dom(callback){
  const {document,window}=parseHTML('<html><body><div id="host"></div></body></html>');
  const old=globalThis.document;globalThis.document=document;
  try{return callback({document,window,host:document.getElementById('host')});}finally{if(old===undefined)delete globalThis.document;else globalThis.document=old;}
}

test('diagnostic interface requires an answer and confidence, preserves milestones and hides answers on retry',()=>dom(({host,window})=>{
  const state={quiz:{},weeks:{'01':{explain:true}}};const original=JSON.stringify(state);const changes=[];
  const dispose=mountDiagnostics(host,{week:1,state,onChange:(...args)=>changes.push(args)});
  assert.equal(host.querySelectorAll('[data-review-question]').length,3);
  assert.equal(host.querySelectorAll('.review-choices legend').length,3);
  const q=questionsForWeek(1)[0],element=host.querySelector(`[data-review-question="${q.id}"]`),form=element.querySelector('form');
  let reports=0;form.reportValidity=()=>{reports++;};
  form.dispatchEvent(new window.Event('submit',{bubbles:true,cancelable:true}));assert.equal(reports,1);assert.equal(changes.length,0);
  // linkedom does not model :checked property state; mirror the native checked
  // attribute for this DOM-level lifecycle test. Browser QA covers real input.
  form.querySelector(`.review-choices input[value="${q.correctIndex}"]`).setAttribute('checked','');
  form.querySelector('.review-confidence input[value="unsure"]').setAttribute('checked','');
  form.dispatchEvent(new window.Event('submit',{bubbles:true,cancelable:true}));
  assert.equal(changes.length,1);assert.equal(changes[0][1].correct,true);assert.equal(changes[0][1].confidence,'unsure');
  assert.match(element.querySelector('[data-review-feedback]').textContent,/Correct — now explain why/);
  assert.equal(JSON.stringify(state),original);
  form.dispatchEvent(new window.Event('submit',{bubbles:true,cancelable:true}));assert.equal(changes.length,1,'duplicate submit cannot manufacture attempts');
  element.querySelector('[data-review-unsure]').dispatchEvent(new window.Event('click',{bubbles:true}));
  assert.ok(element.querySelector('[data-review-retry]'),'mark unsure retains the retry control after feedback');
  element.querySelector('[data-review-retry]').dispatchEvent(new window.Event('click',{bubbles:true}));
  const retried=host.querySelector(`[data-review-question="${q.id}"]`);
  assert.equal(retried.querySelector('[data-review-feedback]').textContent,'');assert.equal(retried.querySelectorAll('input[checked]').length,0);
  dispose();const before=changes.length;
  retried.querySelector('[data-review-unsure]').dispatchEvent(new window.Event('click',{bubbles:true}));
  // linkedom currently ignores AbortSignal listener teardown. Native browser
  // listener disposal is additionally protected by the implementation guard.
  assert.equal(changes.length,before);
}));

test('revision desk starts with an honest empty queue and can expose mixed new questions',()=>dom(({host,window})=>{
  const dispose=mountReview(host,{state:{quiz:{}},onChange:()=>{},catalog:{weeks:[{number:1,title:'Bandits'},{number:2,title:'Bellman'}]}});
  assert.match(host.textContent,/No questions in this queue/);assert.equal(host.querySelectorAll('[data-review-question]').length,0);
  host.querySelector('[data-review-all]').dispatchEvent(new window.Event('click',{bubbles:true}));
  assert.equal(host.querySelectorAll('[data-review-question]').length,6);
  assert.deepEqual([...host.querySelectorAll('[data-review-question]')].slice(0,3).map(el=>el.dataset.reviewQuestion.slice(0,3)),['w01','w02','w03']);
  assert.match(host.querySelector('[data-review-queue-note]').textContent,/1–6 of 78/);
  dispose();
}));

test('mounted questions use fresh tab state before submitting and refreshed queues see remote changes',()=>dom(({host,window})=>{
  const initial={quiz:{}},q=questionsForWeek(1)[0];let current=initial;
  const dispose=mountDiagnostics(host,{week:1,state:initial,getState:()=>current,onChange:(id,record)=>{current={quiz:{...current.quiz,[id]:record}};}});
  // Another tab answered twice after this component mounted.
  const first=recordAnswer(q.id,q.correctIndex,'confident',null,T);
  current={quiz:{[q.id]:recordAnswer(q.id,q.correctIndex,'confident',first,T+1000)}};
  const form=host.querySelector(`[data-review-form="${q.id}"]`);
  form.querySelector(`.review-choices input[value="${q.correctIndex}"]`).setAttribute('checked','');
  form.querySelector('.review-confidence input[value="confident"]').setAttribute('checked','');
  form.dispatchEvent(new window.Event('submit',{bubbles:true,cancelable:true}));
  assert.equal(current.quiz[q.id].attempts,3);dispose();
  current={quiz:{}};
  const stop=mountReview(host,{state:current,getState:()=>current,onChange:()=>{}});
  assert.equal(host.querySelectorAll('[data-review-question]').length,0);
  current={quiz:{[q.id]:markUncertain(q.id)}};
  host.querySelector('[data-review-refresh]').dispatchEvent(new window.Event('click',{bubbles:true}));
  assert.equal(host.querySelectorAll('[data-review-question]').length,1);stop();
}));
