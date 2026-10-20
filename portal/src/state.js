export const KEY='study-rl-companion-v1';
export const empty=()=>({schema:1,weeks:{},drafts:{},last:null,focus:false});
export function validateState(value){
 if(!value||value.schema!==1||!value.weeks||typeof value.weeks!=='object'||Array.isArray(value.weeks))throw new Error('This is not a STUDY–RL progress export.');
 const result=empty();
 for(const [key,row] of Object.entries(value.weeks||{})){
  if(!/^\d{2}$/.test(key)||Number(key)<1||Number(key)>26||!row||typeof row!=='object')continue;
  result.weeks[key]={};for(const name of ['explain','calculate','transfer','evaluate'])result.weeks[key][name]=row[name]===true;
  if(typeof row.note==='string')result.weeks[key].note=row.note.slice(0,8000);
 }
 for(const [key,code] of Object.entries(value.drafts||{}))if(['value','td','advantage','ppo','ips'].includes(key)&&typeof code==='string')result.drafts[key]=code.slice(0,20000);
 if(value.last&&/^#\/lesson\/(0[1-9]|1\d|2[0-6])\?view=(guide|lecture|workbook|reading|project)$/.test(value.last.route||''))result.last={route:value.last.route,y:Number.isFinite(value.last.y)?Math.max(0,Math.min(200000,value.last.y)):0};
 result.focus=value.focus===true;return result;
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
