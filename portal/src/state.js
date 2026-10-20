import {normalizeQuizRecord} from './review-model.js';
export const KEY='study-rl-companion-v1';
export const empty=()=>({schema:1,weeks:{},drafts:{},last:null,focus:false,quiz:{},bookmarks:{},conflicts:{},reading:{size:100,width:'comfortable'},updatedAt:0,backupAt:0,reminderAt:0});
const object=value=>value&&typeof value==='object'&&!Array.isArray(value);
const time=value=>Number.isFinite(value)&&value>=0?Math.min(value,Date.now()+86400000):0;
export function validReadingRoute(route){
 if(typeof route!=='string'||route.length>800||!route.startsWith('#/'))return false;
 const [path,query='']=route.slice(2).split('?'),params=new URLSearchParams(query);
 if(!/^(?:lesson\/(?:0[1-9]|1\d|2[0-6])|read\/[a-z0-9_-]+|atlas|routes)$/.test(path))return false;
 for(const [key,value] of params){if(!['view','section','resume'].includes(key)||value.length>400)return false;}
 return !path.startsWith('lesson/')||['guide','lecture','workbook','reading','project'].includes(params.get('view'));
}
export function validateState(value){
 if(!object(value)||value.schema!==1||!object(value.weeks))throw new Error('This is not a STUDY–RL progress export.');
 const result=empty();
 for(const [key,row] of Object.entries(value.weeks)){
  if(!/^(?:0[1-9]|1\d|2[0-6])$/.test(key)||!object(row))continue;
  result.weeks[key]={};for(const name of ['explain','calculate','transfer','evaluate'])result.weeks[key][name]=row[name]===true;
  if(typeof row.note==='string')result.weeks[key].note=row.note.slice(0,8000);
 }
 for(const [key,code] of Object.entries(object(value.drafts)?value.drafts:{}))if(['value','td','advantage','ppo','ips'].includes(key)&&typeof code==='string')result.drafts[key]=code.slice(0,20000);
 if(object(value.last)&&validReadingRoute(value.last.route))result.last={route:value.last.route,y:Number.isFinite(value.last.y)?Math.max(0,Math.min(200000,value.last.y)):0};
 for(const [id,row] of Object.entries(object(value.quiz)?value.quiz:{})){
  const record=normalizeQuizRecord(id,row);if(record)result.quiz[id]=record;
 }
 for(const [id,row] of Object.entries(object(value.bookmarks)?value.bookmarks:{})){
  if(!/^[a-zA-Z0-9-]{1,80}$/.test(id)||['constructor','prototype'].includes(id)||!object(row)||!validReadingRoute(row.route))continue;
  result.bookmarks[id]={route:row.route,title:String(row.title||'Saved section').slice(0,160),createdAt:time(row.createdAt)};
 }
 for(const [id,row] of Object.entries(object(value.conflicts)?value.conflicts:{})){
  if(!/^[a-zA-Z0-9-]{1,80}$/.test(id)||['constructor','prototype'].includes(id)||!object(row)||!/^(?:0[1-9]|1\d|2[0-6])$/.test(row.week)||typeof row.note!=='string')continue;
  result.conflicts[id]={week:row.week,note:row.note.slice(0,8000),createdAt:time(row.createdAt)};
 }
 result.focus=value.focus===true;
 if(object(value.reading)){
  if([90,100,110,120,130].includes(value.reading.size))result.reading.size=value.reading.size;
  if(['comfortable','wide'].includes(value.reading.width))result.reading.width=value.reading.width;
 }
 for(const key of ['updatedAt','backupAt','reminderAt'])result[key]=time(value[key]);
 return result;
}
export function mergeBackup(current,incoming){
 const result=validateState(current),other=validateState(incoming);
 for(const [week,row] of Object.entries(other.weeks)){
  const before=result.weeks[week]||{};
  for(const key of ['explain','calculate','transfer','evaluate'])row[key]=row[key]||before[key]||false;
  if(before.note&&row.note!==undefined&&before.note!==row.note){
   const id=globalThis.crypto?.randomUUID?.()||'backup-'+Date.now()+'-'+Math.random().toString(36).slice(2);
   result.conflicts[id]={week,note:before.note,createdAt:Date.now()};
  }
  result.weeks[week]={...before,...row};
 }
 Object.assign(result.drafts,other.drafts);Object.assign(result.bookmarks,other.bookmarks);Object.assign(result.conflicts,other.conflicts);
 for(const [id,row] of Object.entries(other.quiz)){
  if(!result.quiz[id]||Date.parse(row.lastReviewed||row.nextReview)>Date.parse(result.quiz[id].lastReviewed||result.quiz[id].nextReview))result.quiz[id]=row;
 }
 if(object(incoming.reading))result.reading=other.reading;
 if(!result.last)result.last=other.last;
 result.backupAt=Math.max(result.backupAt,other.backupAt);
 return validateState(result);
}
export function milestoneCount(state){return Object.values(state.weeks).reduce((sum,w)=>sum+['explain','calculate','transfer','evaluate'].filter(k=>w[k]).length,0);}
export function searchDocuments(documents,query){
 const terms=query.toLowerCase().trim().split(/\s+/).filter(Boolean);
 if(!terms.length)return [];
 return documents.map(d=>{
  const title=d.title.toLowerCase(),body=d.text.toLowerCase();let score=0;
  for(const term of terms){if(!title.includes(term)&&!body.includes(term))return null;score+=title.includes(term)?12:1;}
  if(d.kind==='guide')score+=8;if(d.kind==='lecture')score+=4;if(d.kind==='code')score-=2;
  const at=body.indexOf(terms[0]);return {...d,score,snippet:d.text.slice(Math.max(0,at-75),Math.max(0,at-75)+200)};
 }).filter(Boolean).sort((a,b)=>b.score-a.score).slice(0,24);
}
