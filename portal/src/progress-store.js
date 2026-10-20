import {KEY,empty,validateState} from './state.js';
export const FIELD_PREFIX='study-rl-field-v1:';
export const NOTE_PREFIX='study-rl-note-v1:';
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const clone=value=>structuredClone(value);
const uid=()=>globalThis.crypto?.randomUUID?.()||Date.now().toString(36)+'-'+Math.random().toString(36).slice(2);
function flatten(state){
 const fields={};
 for(const group of ['weeks','drafts','quiz','bookmarks','conflicts'])for(const [id,row] of Object.entries(state[group])){
  if(group==='weeks')for(const [name,value] of Object.entries(row))fields[`weeks/${id}/${name}`]=value;
  else fields[`${group}/${id}`]=row;
 }
 for(const key of ['last','focus','reading','backupAt','reminderAt'])fields[key]=state[key];
 return fields;
}
function put(state,path,value){
 const p=path.split('/');
 if(p.length===1&&['last','focus','reading','backupAt','reminderAt'].includes(p[0]))state[p[0]]=value;
 else if(p.length===3&&p[0]==='weeks'&&/^(0[1-9]|1\d|2[0-6])$/.test(p[1])&&['explain','calculate','transfer','evaluate','note'].includes(p[2])){
  state.weeks[p[1]]??={};if(value===null)delete state.weeks[p[1]][p[2]];else state.weeks[p[1]][p[2]]=value;
 }else if(p.length===2&&['drafts','quiz','bookmarks','conflicts'].includes(p[0])&&/^[a-zA-Z0-9-]{1,100}$/.test(p[1])&&!['constructor','prototype'].includes(p[1])){
  if(value===null)delete state[p[0]][p[1]];else state[p[0]][p[1]]=value;
 }
}
function snapshot(storage){
 const raw=storage.getItem(KEY);let state=raw?validateState(JSON.parse(raw)):empty();
 const notes=new Map(),tombstones=new Set();
 for(let i=0;i<storage.length;i++){
  const key=storage.key(i);if(!key?.startsWith(FIELD_PREFIX)&&!key?.startsWith(NOTE_PREFIX))continue;
  let row;try{row=JSON.parse(storage.getItem(key));}catch{continue;}
  if(!row||typeof row!=='object'||!Number.isFinite(row.at))continue;
  state.updatedAt=Math.max(state.updatedAt,row.at);
  if(key.startsWith(FIELD_PREFIX)){
   const path=key.slice(FIELD_PREFIX.length);put(state,path,row.value);
   if(path.startsWith('conflicts/')&&row.value===null)tombstones.add(path.slice(10));
  }else if(/^(0[1-9]|1\d|2[0-6])$/.test(row.week)&&typeof row.value==='string'&&Array.isArray(row.parents))notes.set(key.slice(NOTE_PREFIX.length),row);
 }
 const parents=new Set([...notes.values()].flatMap(row=>row.parents)),tips={};
 for(const [id,row] of notes)if(!parents.has(id))(tips[row.week]??=[]).push({id,...row});
 for(const [week,rows] of Object.entries(tips)){
  rows.sort((a,b)=>b.at-a.at||b.id.localeCompare(a.id));put(state,`weeks/${week}/note`,rows[0].value);
  for(const row of rows.slice(1)){
   const id='note-'+row.id;
   if(row.value!==rows[0].value&&!tombstones.has(id))state.conflicts[id]={week,note:row.value,createdAt:row.at};
  }
 }
 return {state:validateState(state),notes,tips};
}
export function createProgressStore(storage){
 let snap,baseline,ok=true,error=null;
 try{snap=snapshot(storage);baseline=clone(snap.state);}catch(e){ok=false;error=e;snap={state:empty(),notes:new Map(),tips:{}};baseline=empty();}
 function sync(current,protectedPaths=[]){
  try{
   const fresh=snapshot(storage),before=flatten(baseline),local=flatten(validateState(current)),remote=flatten(fresh.state),next=clone(fresh.state),nextBaseline=clone(fresh.state);
   for(const path of new Set([...Object.keys(before),...Object.keys(local)])){
    if(!same(before[path],local[path])||protectedPaths.includes(path)){
     put(next,path,local[path]??null);put(nextBaseline,path,before[path]??null);
     // Keep the parent version of an in-progress note edit, even after a storage event.
     const match=path.match(/^weeks\/(\d+)\/note$/);if(match)fresh.tips[match[1]]=snap.tips[match[1]];
    }
   }
   snap=fresh;baseline=validateState(nextBaseline);ok=true;error=null;return validateState(next);
  }catch(e){ok=false;error=e;return current;}
 }
 function save(current){
  const desired=validateState(current),before=flatten(baseline),after=flatten(desired);
  try{
   const fresh=snapshot(storage);let at=Math.max(Date.now(),fresh.state.updatedAt+1);
   for(const path of new Set([...Object.keys(before),...Object.keys(after)])){
    if(same(before[path],after[path]))continue;
    const match=path.match(/^weeks\/(\d+)\/note$/);
    if(match&&after[path]!==undefined){
     const week=match[1],id=uid(),parent=snap.tips[week]?.[0]?.id,parents=parent?[parent]:[];
     storage.setItem(NOTE_PREFIX+id,JSON.stringify({week,value:after[path],parents,at:at++}));
     // Compact only ancestors of this edit; concurrent branches remain recoverable.
     const remove=[...parents],seen=new Set();while(remove.length){const ancestor=remove.pop();if(seen.has(ancestor))continue;seen.add(ancestor);const row=fresh.notes.get(ancestor);if(row){remove.push(...row.parents);storage.removeItem(NOTE_PREFIX+ancestor);}}
    }else storage.setItem(FIELD_PREFIX+path,JSON.stringify({value:after[path]??null,at:at++}));
   }
   snap=snapshot(storage);baseline=clone(snap.state);ok=true;error=null;return snap.state;
  }catch(e){ok=false;error=e;return desired;}
 }
 return {get state(){return clone(snap.state);},get ok(){return ok;},get error(){return error;},save,sync};
}
