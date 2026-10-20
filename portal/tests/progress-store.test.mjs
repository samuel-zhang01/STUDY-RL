import test from 'node:test';
import assert from 'node:assert/strict';
import {empty,KEY,mergeBackup} from '../src/state.js';
import {createProgressStore,FIELD_PREFIX,NOTE_PREFIX} from '../src/progress-store.js';
class MemoryStorage{data=new Map();get length(){return this.data.size;}key(i){return [...this.data.keys()][i];}getItem(k){return this.data.get(k)??null;}setItem(k,v){this.data.set(k,v);}removeItem(k){this.data.delete(k);}}
const row=(s,n='01')=>s.weeks[n]??=( {explain:false,calculate:false,transfer:false,evaluate:false} );
test('legacy data survives and stale tabs update distinct fields without clobbering',()=>{
 const storage=new MemoryStorage(),old=empty();row(old).note='Earlier note';old.drafts.td='legacy';storage.setItem(KEY,JSON.stringify(old));
 const a=createProgressStore(storage),b=createProgressStore(storage),as=a.state,bs=b.state;
 row(as).explain=true;a.save(as);row(bs,'02').calculate=true;b.save(bs);
 const result=createProgressStore(storage).state;assert.equal(result.weeks['01'].note,'Earlier note');assert.equal(result.drafts.td,'legacy');assert.equal(result.weeks['01'].explain,true);assert.equal(result.weeks['02'].calculate,true);
 const c=createProgressStore(storage);let cs=c.state;row(cs).explain=false;cs=c.save(cs);row(as,'04').evaluate=true;a.save(as);assert.equal(createProgressStore(storage).state.weeks['01'].explain,false,'unchanged stale field does not undo an explicit remote uncheck');
});
test('concurrent notes are recoverable; deliberate edits compact ancestors',()=>{
 const storage=new MemoryStorage(),a=createProgressStore(storage),b=createProgressStore(storage),as=a.state,bs=b.state;
 row(as).note='First tab version';a.save(as);row(bs).note='Second tab version';b.save(bs);
 let result=createProgressStore(storage).state;
 assert.equal(result.weeks['01'].note,'Second tab version');assert.deepEqual(Object.values(result.conflicts).map(v=>v.note),['First tab version']);
 const c=createProgressStore(storage);result=c.state;row(result).note='Second tab revised';result=c.save(result);
 assert.equal(result.weeks['01'].note,'Second tab revised');assert.equal(Object.values(result.conflicts)[0].note,'First tab version');
 const id=Object.keys(result.conflicts)[0];delete result.conflicts[id];c.save(result);assert.deepEqual(createProgressStore(storage).state.conflicts,{});
 assert.equal([...storage.data.keys()].filter(k=>k.startsWith(NOTE_PREFIX)).length,2);
});
test('storage events protect an active draft and still merge unrelated milestones',()=>{
 const storage=new MemoryStorage(),a=createProgressStore(storage),b=createProgressStore(storage);let as=a.state,bs=b.state;
 row(as).note='Draft A';row(bs).note='Remote B';row(bs,'03').transfer=true;b.save(bs);
 as=a.sync(as,['weeks/01/note']);assert.equal(as.weeks['01'].note,'Draft A');assert.equal(as.weeks['03'].transfer,true);
 as=a.save(as);assert.equal(as.weeks['01'].note,'Draft A');assert.equal(Object.values(as.conflicts)[0].note,'Remote B');
});
test('explicit unchecks and bookmark removal survive reload and stale writers',()=>{
 const storage=new MemoryStorage(),a=createProgressStore(storage);let s=a.state;row(s).explain=true;s.bookmarks.one={route:'#/lesson/01?view=guide',title:'Start',createdAt:1};s=a.save(s);
 row(s).explain=false;delete s.bookmarks.one;a.save(s);const restored=createProgressStore(storage).state;assert.equal(restored.weeks['01'].explain,false);assert.deepEqual(restored.bookmarks,{});
});
test('blocked storage retains exportable edits and reports failure',()=>{
 const storage=new MemoryStorage(),store=createProgressStore(storage),s=store.state;storage.setItem=()=>{throw new Error('Quota');};row(s).note='Keep me';const result=store.save(s);assert.equal(store.ok,false);assert.equal(result.weeks['01'].note,'Keep me');
});
test('import preserves both different notes and ignores unsafe journal paths',()=>{
 const a=empty(),b=empty();row(a).note='Current';row(b).note='Imported';const merged=mergeBackup(a,b);assert.equal(merged.weeks['01'].note,'Imported');assert.equal(Object.values(merged.conflicts)[0].note,'Current');
 const storage=new MemoryStorage();storage.setItem(FIELD_PREFIX+'weeks/__proto__/polluted',JSON.stringify({value:true,at:1}));assert.equal(createProgressStore(storage).state.polluted,undefined);assert.equal({}.polluted,undefined);
});
test('interleaved note writes retain both branches even when both writers read before either commit',()=>{
 const storage=new MemoryStorage(),a=createProgressStore(storage),b=createProgressStore(storage),as=a.state,bs=b.state;row(as).note='Racing A';row(bs).note='Racing B';
 const original=storage.setItem.bind(storage);let interleave=true;storage.setItem=(key,value)=>{if(interleave&&key.startsWith(NOTE_PREFIX)){interleave=false;b.save(bs);}original(key,value);};
 a.save(as);const result=createProgressStore(storage).state;assert.deepEqual(new Set([result.weeks['01'].note,...Object.values(result.conflicts).map(r=>r.note)]),new Set(['Racing A','Racing B']));
});
test('partial quota failure retains all pending fields and can recover on a later save',()=>{
 const storage=new MemoryStorage(),store=createProgressStore(storage);let state=store.state;row(state).explain=true;row(state).note='A pending note';
 const original=storage.setItem.bind(storage);let count=0;storage.setItem=(key,value)=>{if(++count>1)throw new Error('Quota');original(key,value);};state=store.save(state);assert.equal(store.ok,false);assert.equal(state.weeks['01'].note,'A pending note');
 storage.setItem=original;state=store.save(state);assert.equal(store.ok,true);assert.equal(state.weeks['01'].explain,true);assert.equal(createProgressStore(storage).state.weeks['01'].note,'A pending note');
});
