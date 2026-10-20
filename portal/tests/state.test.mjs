import test from 'node:test';
import assert from 'node:assert/strict';
import {empty,validateState,milestoneCount,searchDocuments} from '../src/state.js';
test('backup round trip preserves independent milestones, notes and code',()=>{
 const state=empty();state.weeks['04']={explain:true,calculate:true,transfer:false,note:'Cutoff retains the bootstrap.'};state.drafts.td='def td_target():\n    pass';
 const restored=validateState(JSON.parse(JSON.stringify(state)));
 assert.equal(milestoneCount(restored),2);assert.equal(restored.weeks['04'].note,state.weeks['04'].note);assert.equal(restored.drafts.td,state.drafts.td);
});
test('imports reject unrelated files and bound untrusted fields',()=>{
 for(const value of [null,{},[],{schema:2,weeks:{}},{schema:1,weeks:null}])assert.throws(()=>validateState(value));
 const imported=validateState({schema:1,weeks:{'99':{explain:true},'01':{explain:'yes',note:'x'.repeat(9000)}},drafts:{value:'x'.repeat(21000),unknown:'x'},last:{route:'#/lesson/99?view=guide',y:-10}});
 assert.equal(milestoneCount(imported),0);assert.equal(imported.weeks['01'].note.length,8000);assert.equal(imported.drafts.value.length,20000);assert.equal(imported.drafts.unknown,undefined);assert.equal(imported.last,null);
});
test('search requires all terms and prioritises teaching over code',()=>{
 const docs=[{title:'PPO',text:'Negative advantage clipping',kind:'code'},{title:'PPO',text:'Negative advantage clipping',kind:'guide'},{title:'PPO',text:'Only positive cases',kind:'lecture'}];
 const results=searchDocuments(docs,'negative clipping');
 assert.equal(results.length,2);assert.equal(results[0].kind,'guide');assert.deepEqual(searchDocuments(docs,'   '),[]);
});
