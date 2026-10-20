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
test('new backups retain reader settings, saved sections and all preserved note versions',()=>{
 const state=empty();state.reading={size:130,width:'wide'};state.bookmarks.one={route:'#/lesson/09?view=lecture&section=display-equation-1',title:'PPO equation',createdAt:1};
 for(let n=0;n<1001;n++)state.conflicts['version-'+n]={week:'01',note:'Preserved '+n,createdAt:n};
 const restored=validateState(JSON.parse(JSON.stringify(state)));assert.deepEqual(restored.reading,state.reading);assert.deepEqual(restored.bookmarks,state.bookmarks);assert.equal(Object.keys(restored.conflicts).length,1001);
});
test('malformed quiz records and unsafe bookmark routes cannot enter imported state',()=>{
 const state=validateState({schema:1,weeks:{},quiz:JSON.parse('{"__proto__":{},"constructor":{},"toString":{}}'),bookmarks:{bad:{route:'javascript:alert(1)'},external:{route:'https://example.com'},wrongWeek:{route:'#/lesson/99?view=guide'},badView:{route:'#/lesson/01?view=unknown'}}});
 assert.deepEqual(state.quiz,{});assert.deepEqual(state.bookmarks,{});assert.equal({}.polluted,undefined);
});
