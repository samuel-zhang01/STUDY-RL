import {QUESTIONS, QUESTION_BY_ID} from './review-content.js';

const DAY = 86400000;
const MINUTE = 60000;
const MAX_INTERVAL = 30 * DAY;
const MAX_ATTEMPTS = 100000;
const date = value => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}T/.test(value) && Number.isFinite(Date.parse(value)) ? new Date(value).toISOString() : null;
const clock = value => {
  const milliseconds = value instanceof Date ? value.getTime() : typeof value === 'number' ? value : Date.parse(value);
  if (!Number.isFinite(milliseconds)) throw new TypeError('A valid review time is required.');
  return milliseconds;
};

/** Import boundary: content, never a backup's `correct`, determines correctness. */
export function normalizeQuizRecord(id, value) {
  const question = QUESTION_BY_ID[id];
  if (!question || !value || typeof value !== 'object' || Array.isArray(value)) return null;
  const answer = Number.isInteger(value.answer) && value.answer >= 0 && value.answer < question.choices.length ? value.answer : null;
  const confidence = value.confidence === 'confident' ? 'confident' : 'unsure';
  const lastReviewed = date(value.lastReviewed);
  let nextReview = date(value.nextReview);
  const attempts = Number.isInteger(value.attempts) ? Math.max(0,Math.min(MAX_ATTEMPTS,value.attempts)) : 0;
  // An unanswered uncertainty bookmark is a useful record but not an attempt.
  if (answer === null) {
    if (confidence !== 'unsure' || !nextReview) return null;
    return {answer:null, correct:false, confidence:'unsure', attempts:0, lastReviewed:null, nextReview};
  }
  if (!lastReviewed || attempts < 1) return null;
  const correct = answer === question.correctIndex;
  // Missing, reversed or implausibly distant imported dates cannot silently hide
  // an item. The scheduler itself never exceeds these transparent intervals.
  const maximumGap = !correct ? 10*MINUTE : confidence === 'unsure' ? DAY : MAX_INTERVAL;
  if (!nextReview || Date.parse(nextReview) < Date.parse(lastReviewed) || Date.parse(nextReview)-Date.parse(lastReviewed) > maximumGap) nextReview = lastReviewed;
  return {answer, correct, confidence, attempts, lastReviewed, nextReview};
}

export function recordAnswer(id, answer, confidence, previous = null, now = Date.now()) {
  const question = QUESTION_BY_ID[id];
  if (!question) throw new RangeError('Unknown review question.');
  if (!Number.isInteger(answer) || answer < 0 || answer >= question.choices.length) throw new RangeError('Choose one of this question’s answers.');
  if (!['unsure','confident'].includes(confidence)) throw new TypeError('Choose a confidence level.');
  const time = clock(now), prior = normalizeQuizRecord(id, previous);
  const correct = answer === question.correctIndex;
  let interval = correct ? confidence === 'unsure' ? DAY : 3*DAY : 10*MINUTE;
  if (correct && confidence === 'confident' && prior?.correct && prior.confidence === 'confident') {
    const previousGap = Date.parse(prior.nextReview) - Date.parse(prior.lastReviewed);
    // Immediate retries do not lengthen the interval. Retrieve again on/after
    // the due time before earning a longer spacing interval.
    interval = time >= Date.parse(prior.nextReview)
      ? Math.min(MAX_INTERVAL,Math.max(3*DAY,previousGap*2))
      : Math.max(3*DAY,Math.min(MAX_INTERVAL,previousGap));
  }
  return {answer, correct, confidence, attempts:Math.min(MAX_ATTEMPTS,(prior?.attempts||0)+1),
    lastReviewed:new Date(time).toISOString(), nextReview:new Date(time+interval).toISOString()};
}

/** Marking uncertainty is a self-report; it must not create a quiz attempt. */
export function markUncertain(id, previous = null, now = Date.now()) {
  if (!QUESTION_BY_ID[id]) throw new RangeError('Unknown review question.');
  const time = clock(now), prior = normalizeQuizRecord(id,previous);
  return prior
    ? {...prior, confidence:'unsure', nextReview:new Date(Math.min(time,Date.parse(prior.nextReview))).toISOString()}
    : {answer:null,correct:false,confidence:'unsure',attempts:0,lastReviewed:null,nextReview:new Date(time).toISOString()};
}

export function reviewStatus(record, now = Date.now()) {
  if (!record) return {attempted:false, missed:false, unsure:false, due:false, reason:'New question'};
  const attempted = record.answer !== null && record.attempts > 0;
  const missed = attempted && !record.correct;
  const unsure = record.confidence === 'unsure';
  const due = Date.parse(record.nextReview) <= clock(now);
  return {attempted,missed,unsure,due,reason:missed?'Last answer missed':unsure?'Marked unsure':due?'Review due':'Scheduled for later'};
}

export function reviewStats(quiz = {}, {now = Date.now()} = {}) {
  const result = {total:QUESTIONS.length,attempted:0,correct:0,missed:0,unsure:0,due:0,priority:0};
  for (const question of QUESTIONS) {
    const record = normalizeQuizRecord(question.id,quiz?.[question.id]);
    if (!record) continue;
    const status = reviewStatus(record,now);
    if (status.attempted) result.attempted++;
    if (status.attempted && record.correct) result.correct++;
    for (const key of ['missed','unsure','due']) if(status[key]) result[key]++;
    if (status.missed || status.unsure || status.due) result.priority++;
  }
  return result;
}

/** Stable interleaving mixes weeks; it does not imply a learning-science optimum. */
export function reviewQueue(quiz = {}, {now = Date.now(),filter = 'priority',week = null} = {}) {
  const time = clock(now), requested = week ? String(Number(week)).padStart(2,'0') : null;
  if (!['priority','due','missed','unsure','all'].includes(filter)) throw new RangeError('Unknown review filter.');
  const candidates = QUESTIONS.flatMap(question => {
    if (requested && question.week !== requested) return [];
    const record = normalizeQuizRecord(question.id,quiz?.[question.id]);
    const status = reviewStatus(record,time);
    const included = filter === 'all' || (filter === 'priority' ? status.missed||status.unsure||status.due : status[filter]);
    return included ? [{question,record,...status}] : [];
  });
  candidates.sort((a,b) => Number(b.missed)-Number(a.missed) || Number(b.unsure)-Number(a.unsure)
    || (Date.parse(a.record?.nextReview)||0)-(Date.parse(b.record?.nextReview)||0)
    || a.question.id.localeCompare(b.question.id));
  const buckets = new Map();
  for (const row of candidates) {
    if (!buckets.has(row.question.week)) buckets.set(row.question.week,[]);
    buckets.get(row.question.week).push(row);
  }
  const result=[];
  while (result.length < candidates.length) for (const bucket of buckets.values()) if(bucket.length)result.push(bucket.shift());
  return result;
}
