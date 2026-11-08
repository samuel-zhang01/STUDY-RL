import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {parseHTML} from 'linkedom';
const root=path.resolve(import.meta.dirname,'../dist');
const read=name=>JSON.parse(fs.readFileSync(path.join(root,name)));
const catalog=read('catalog.json');
const docs=catalog.documents.map(d=>read(`content/${d.id}.json`));
const routes=new Map(docs.map(d=>[d.route,d]));

test('all 26 weeks have complete reading, workbook and implementation paths',()=>{
 assert.equal(catalog.weeks.length,26);
 for(const w of catalog.weeks){
  for(const view of ['guide','lecture','workbook','reading']){
   const d=docs.find(d=>d.id===w.views[view]);assert.ok(d,`${w.number} ${view}`);
   assert.ok(d.text.length>500,`${d.path} has substantive content`);
  }
  assert.ok(w.project.length>0);
  for(const asset of [w.pdf,w.notebook])assert.ok(fs.existsSync(path.join(root,'assets',asset)));
 }
 assert.equal(new Set(docs.map(d=>d.id)).size,docs.length);
});
test('all equations, local reading links, figures and downloads resolve',()=>{
 assert.deepEqual(read('validation.json').mathErrors,[]);
 assert.deepEqual(read('validation.json').unresolved,[]);
 for(const d of docs){
  const dom=parseHTML('<body>'+d.html+'</body>').document;
  for(const a of dom.querySelectorAll('a[href]')){
   const href=a.getAttribute('href');
   if(href.startsWith('assets/')){assert.ok(fs.existsSync(path.join(root,href)),`${d.path} → ${href}`);continue;}
   if(!href.startsWith('#/'))continue;
   const [base,query]=href.split('?'),params=new URLSearchParams(query),section=params.get('section');params.delete('section');
   const route=base+(params.size?'?'+params:'');
   if(base.startsWith('#/read/')||base.startsWith('#/lesson/')){
    const target=routes.get(route);assert.ok(target,`${d.path} → ${route}`);
    if(section)assert.ok(parseHTML(target.html).document.getElementById(section),`${d.path} → ${href}`);
   }
  }
  for(const img of dom.querySelectorAll('img[src]')){const source=img.getAttribute('src');if(source.startsWith('data:image/png;base64,')){assert.ok(img.getAttribute('alt').length>20);continue;}assert.ok(fs.existsSync(path.join(root,source)),`${d.path}: missing image`);}
  assert.equal(dom.querySelectorAll('.math-fallback').length,0);
 }
});
test('conversion preserves theorem targets, algorithm walkthroughs and lecture figures',()=>{
 const eight=docs.find(d=>d.path==='week-08-a2c-gae-trpo/notes.tex');
 assert.match(eight.html,/id="thm:surrbnd"/);
 assert.match(eight.text,/Conjugate-gradient solve/);
 assert.match(eight.text,/Synchronous A2C/);
 assert.match(eight.text,/does not inherit/);
 const eleven=docs.find(d=>d.path==='week-11-contextual-bandits-ope/notes.tex');
 assert.ok(eleven.html.includes('katex'));
 assert.ok(docs.filter(d=>d.kind==='lecture').reduce((n,d)=>n+(d.html.match(/<img /g)||[]).length,0)>=26);
 const reading=docs.find(d=>d.path==='week-21-reward-hacking-oversight/extended_reading.tex');
 assert.match(reading.text,/invented teaching family/);
 assert.doesNotMatch(reading.text,/held-out human evaluator/);
});
