import {QUESTIONS,QUESTION_BY_ID,questionsForWeek} from './review-content.js';
import {normalizeQuizRecord,recordAnswer,markUncertain,reviewQueue,reviewStats,reviewStatus} from './review-model.js';

const escape = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const kinds = {concept:'Spot the misconception',calculate:'Calculate or test an assumption',transfer:'Transfer the idea'};
let mounts = 0;
const dateLabel = value => new Date(value).toLocaleString(undefined,{day:'numeric',month:'short',hour:'2-digit',minute:'2-digit'});
const scheduling = 'Missed answers return in 10 minutes; correct but unsure answers in 1 day; confident correct answers in 3 days, then 6, 12, 24 and at most 30 days after successful due reviews. You can practise earlier. These are simple practice intervals, not a mastery estimate.';

function card(question,prefix,record) {
  const id = `${prefix}-${question.id}`, status = reviewStatus(record);
  const previous = record ? `${status.attempted?`Latest answer: ${record.correct?'correct':'needs another look'} · ${record.attempts} ${record.attempts===1?'attempt':'attempts'}`:'Not answered yet'} · ${record.confidence==='unsure'?'marked unsure':'confident'} · ${status.due?'review due now':`next review ${dateLabel(record.nextReview)}`}` : 'No attempt recorded yet.';
  return `<section class="review-question" data-review-question="${question.id}" aria-labelledby="${id}-title">
    <p class="eyebrow">Week ${question.week} · ${kinds[question.kind]}</p>
    <h3 id="${id}-title">${escape(question.prompt)}</h3>
    <p class="review-previous" data-review-previous>${escape(previous)}</p>
    <form data-review-form="${question.id}">
      <fieldset class="review-choices"><legend>Choose your answer</legend>${question.choices.map((choice,index)=>`<label class="review-choice"><input type="radio" name="answer-${id}" value="${index}" required><span>${escape(choice)}</span></label>`).join('')}</fieldset>
      <fieldset class="review-confidence"><legend>How confident are you?</legend><label><input type="radio" name="confidence-${id}" value="unsure" required> Unsure / working it out</label><label><input type="radio" name="confidence-${id}" value="confident" required> Confident in my reasoning</label></fieldset>
      <div class="review-actions"><button class="button small" type="submit">Check answer</button><button class="button secondary small" type="button" data-review-unsure="${question.id}">Save for review without answering</button></div>
    </form>
    <div class="review-feedback" data-review-feedback role="status" aria-live="polite" aria-atomic="true"></div>
  </section>`;
}

function connect(host,{state,onChange,getState},prefix,onRecord=()=>{}) {
  const controller = new AbortController(), quiz = {...(state?.quiz||{})};
  let disposed = false;
  function refreshRecords(){
    const fresh=getState?.();
    if(fresh&&typeof fresh.quiz==='object'){
      for(const id of Object.keys(quiz))delete quiz[id];
      Object.assign(quiz,fresh.quiz);
    }
  }
  const on = (type,callback) => host.addEventListener(type,callback,{signal:controller.signal});
  function save(id,record) {
    quiz[id]=record;
    onChange?.(id,record);
    if (!disposed) onRecord(quiz);
  }
  function showPrior(element,record) {
    const status=reviewStatus(record);
    element.querySelector('[data-review-previous]').textContent=`${status.attempted?`Latest answer: ${record.correct?'correct':'needs another look'} · ${record.attempts} ${record.attempts===1?'attempt':'attempts'}`:'Not answered yet'} · ${record.confidence==='unsure'?'marked unsure':'confident'} · ${status.due?'review due now':`next review ${dateLabel(record.nextReview)}`}`;
  }
  on('submit',event=>{
    if(disposed)return;
    const form=event.target.closest('[data-review-form]');
    if (!form || !host.contains(form)) return;
    event.preventDefault();
    // Also guard synthetic submissions: native required fields alone do not
    // protect programmatic submit events.
    const answer=form.querySelector('.review-choices input:checked');
    const confidence=form.querySelector('.review-confidence input:checked');
    if (!answer || !confidence) { form.reportValidity(); return; }
    const id=form.dataset.reviewForm, question=QUESTION_BY_ID[id], element=form.closest('[data-review-question]');
    if (form.dataset.submitted) return;
    refreshRecords();
    const record=recordAnswer(id,Number(answer.value),confidence.value,quiz[id]);
    form.dataset.submitted='true';
    for(const fieldset of form.querySelectorAll('fieldset'))fieldset.disabled=true;
    form.querySelector('[type="submit"]').disabled=true;
    const result=element.querySelector('[data-review-feedback]');
    result.className=`review-feedback ${record.correct?'review-correct':'review-repair'}`;
    result.innerHTML=`<p><strong>${record.correct?'Correct — now explain why.':'Not quite — inspect the reasoning.'}</strong> ${escape(question.feedback[record.answer])}</p>${record.correct?'':`<p><strong>Answer:</strong> ${escape(question.choices[question.correctIndex])} ${escape(question.feedback[question.correctIndex])}</p>`}<div class="review-repair-links"><a href="${question.repair.route}">${escape(question.repair.label)} →</a><a href="${question.handoff.route}">${escape(question.handoff.label)} →</a><a href="${question.handoff.source}" target="_blank" rel="noopener noreferrer">Inspect the matching code on GitHub ↗</a></div><p class="review-code-task"><strong>Take it into Jupyter:</strong> ${escape(question.handoff.task)}</p><p class="review-schedule">Next review: ${escape(dateLabel(record.nextReview))}. ${record.correct&&record.confidence==='unsure'?'Correct and unsure are recorded separately. ':''}Your manual lesson milestones are unchanged.</p><button type="button" class="button secondary small" data-review-retry="${id}">Try again with answers hidden</button>`;
    showPrior(element,record);save(id,record);
  });
  on('click',event=>{
    if(disposed)return;
    const retry=event.target.closest('[data-review-retry]');
    if(retry&&host.contains(retry)) {
      const id=retry.dataset.reviewRetry,element=retry.closest('[data-review-question]');
      refreshRecords();
      element.outerHTML=card(QUESTION_BY_ID[id],prefix,normalizeQuizRecord(id,quiz[id]));
      host.querySelector(`[data-review-question="${id}"] .review-choices input`)?.focus();
      return;
    }
    const unsure=event.target.closest('[data-review-unsure]');
    if(unsure&&host.contains(unsure)) {
      const id=unsure.dataset.reviewUnsure,element=unsure.closest('[data-review-question]');
      refreshRecords();
      const record=markUncertain(id,quiz[id]);
      showPrior(element,record);
      const feedback=element.querySelector('[data-review-feedback]');
      if(element.querySelector('form').dataset.submitted){
        const schedule=feedback.querySelector('.review-schedule');
        if(schedule)schedule.textContent='Review due now because you marked this idea unsure. Your manual lesson milestones are unchanged.';
        let message=feedback.querySelector('[data-review-unsure-note]');
        if(!message){message=document.createElement('p');message.dataset.reviewUnsureNote='true';feedback.append(message);}
        message.textContent='Marked unsure and added to your review queue now. The recorded answer and attempt count are unchanged.';
      }else{
        feedback.className='review-feedback';
        feedback.innerHTML='<p><strong>Saved to your review queue as unsure.</strong> This did not submit an answer or change your manual milestones. You can answer when ready.</p>';
      }
      save(id,record);
    }
  });
  return {quiz,refreshRecords,dispose(){disposed=true;controller.abort();}};
}

export function mountDiagnostics(host,{week,state,onChange,getState}) {
  const questions=questionsForWeek(week);
  if(!questions.length)throw new RangeError('No diagnostics for that week.');
  const prefix=`diagnostics-${++mounts}`;
  host.innerHTML=`<section class="review-diagnostics" aria-labelledby="${prefix}-heading"><div class="review-heading"><div><p class="eyebrow">Retrieve before rereading</p><h2 id="${prefix}-heading">Can you use the idea?</h2><p>Three short checks: a misconception, a calculation or assumption, and a new situation. Choose your answer and confidence before revealing feedback.</p></div><a class="text-link" href="#/review">Your revision queue →</a></div><p class="review-separation">Answers and confidence are saved separately from your self-reported lesson milestones. A correct click is a prompt to explain your reasoning, not proof of mastery.</p>${questions.map(q=>card(q,prefix,normalizeQuizRecord(q.id,state?.quiz?.[q.id]))).join('')}<details class="review-method"><summary>How the practice schedule works</summary><p>${scheduling}</p><p>Progress stays on this device unless you export and import a backup. This question bank and your workbook attempts are complementary evidence.</p></details></section>`;
  return connect(host,{state,onChange,getState},prefix).dispose;
}

export function mountReview(host,{state,onChange,catalog,getState}) {
  const prefix=`review-${++mounts}`;
  const weeks=catalog?.weeks||Array.from({length:26},(_,index)=>({number:index+1,title:`Week ${String(index+1).padStart(2,'0')}`}));
  host.innerHTML=`<section class="review-page" aria-labelledby="${prefix}-heading"><p class="eyebrow">A little retrieval, across the course</p><h1 id="${prefix}-heading">Your revision queue</h1><p class="review-intro">Practise missed or uncertain ideas across weeks. Check your reasoning, then open the exact explanation or related notebook exercise.</p><div class="review-stats" data-review-stats aria-live="polite"></div><p class="review-separation">Counts describe your latest answer to each question. They are separate from manual milestones and are not a course grade. Nothing here requires an account or cookies.</p><div class="review-filters"><label for="${prefix}-filter">Choose a queue<select id="${prefix}-filter" data-review-filter><option value="priority">Needs attention: missed, unsure or due</option><option value="due">Scheduled reviews due now</option><option value="missed">Latest answer was missed</option><option value="unsure">Marked unsure</option><option value="all">All 78 questions, including new ones</option></select></label><label for="${prefix}-week">Weeks to mix<select id="${prefix}-week" data-review-week><option value="">All weeks</option>${weeks.map(w=>`<option value="${String(w.number).padStart(2,'0')}">Week ${String(w.number).padStart(2,'0')} · ${escape(w.title)}</option>`).join('')}</select></label><button type="button" class="button secondary small" data-review-refresh>Refresh queue</button></div><p class="review-queue-note" data-review-queue-note></p><div data-review-cards></div><div class="review-pagination" data-review-pagination></div><details class="review-method"><summary>What changes your schedule?</summary><p>${scheduling}</p><p>Answers stay visible until you try again, refresh the queue or leave this page. Revisiting a question hides the previous answer so you can retrieve it again. Marking unsure without answering records no quiz attempt.</p><p>Use the progress page to export a backup or move your notes and answers to another device.</p></details><a class="text-link" href="#/progress">Progress, notes and backup →</a></section>`;
  const cards=host.querySelector('[data-review-cards]'),filter=host.querySelector('[data-review-filter]'),week=host.querySelector('[data-review-week]');
  const controller=new AbortController();let page=0,queue=[];
  function stats(quiz) {
    const data=reviewStats(quiz);
    host.querySelector('[data-review-stats]').innerHTML=[['answered',data.attempted],['latest correct',data.correct],['missed',data.missed],['unsure',data.unsure],['due now',data.due]].map(([label,value])=>`<div><strong>${value}</strong><span>${label}</span></div>`).join('');
  }
  const connected=connect(host,{state,onChange,getState},prefix,stats);
  function draw({refresh=true,focus=false}={}) {
    if(refresh){connected.refreshRecords();queue=reviewQueue(connected.quiz,{filter:filter.value,week:week.value});page=0;}
    stats(connected.quiz);
    const slice=queue.slice(page*6,page*6+6);
    host.querySelector('[data-review-queue-note]').textContent=queue.length?`Showing ${page*6+1}–${Math.min(queue.length,page*6+6)} of ${queue.length} questions. Different weeks are interleaved. Refresh after answering to update this selection.`:'No questions in this queue right now.';
    cards.innerHTML=slice.length?slice.map(row=>card(row.question,prefix,normalizeQuizRecord(row.question.id,connected.quiz[row.question.id]))).join(''):`<div class="review-empty"><h2>A clear desk.</h2><p>${reviewStats(connected.quiz).attempted?'Nothing matches this filter. You can practise early, choose another week or return when a scheduled review is due.':'Start with a week’s three questions, or choose all questions to build a mixed practice session. Mark anything uncertain for later.'}</p><button type="button" class="button small" data-review-all>Browse all questions</button><a href="#/lesson/01?view=guide">Begin with week 01 →</a></div>`;
    host.querySelector('[data-review-pagination]').innerHTML=queue.length>6?`<button type="button" class="button secondary small" data-review-page="previous" ${page===0?'disabled':''}>← Previous six</button><span>Set ${page+1} of ${Math.ceil(queue.length/6)}</span><button type="button" class="button secondary small" data-review-page="next" ${(page+1)*6>=queue.length?'disabled':''}>Next six →</button>`:'';
    if(focus){const title=cards.querySelector('h2,h3');if(title){title.tabIndex=-1;title.focus();}}
  }
  host.addEventListener('change',event=>{if(event.target===filter||event.target===week)draw();},{signal:controller.signal});
  host.addEventListener('click',event=>{
    if(event.target.closest('[data-review-refresh]'))draw();
    if(event.target.closest('[data-review-all]')){for(const option of filter.options)option.selected=option.value==='all';draw({focus:true});}
    const button=event.target.closest('[data-review-page]');
    if(button&&!button.disabled){page+=button.dataset.reviewPage==='next'?1:-1;draw({refresh:false,focus:true});}
  },{signal:controller.signal});
  draw();
  return ()=>{controller.abort();connected.dispose();};
}
