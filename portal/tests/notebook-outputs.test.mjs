import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {safeOutput,readOutputManifest,outputStatus} from '../scripts/notebook-outputs.mjs';
import {sha256} from '../scripts/fingerprints.mjs';
test('notebook outputs preserve tables and literal text while removing executable content',()=>{
 const html=safeOutput({data:{'text/html':'<script>alert(1)</script><iframe src="https://example.com"></iframe><table onclick="alert(1)"><tr><th>mean</th><td>2.5</td></tr></table><img src=x onerror=alert(2)>'}});
 assert.match(html,/<table>/);assert.match(html,/2\.5/);assert.doesNotMatch(html,/script|iframe|onclick|onerror|<img/);
 assert.match(safeOutput({text:['<script>','literal</script>']}),/&lt;script&gt;/);
 assert.equal(safeOutput({data:{'application/javascript':'alert(1)','image/svg+xml':'<svg onload="alert(1)"/>'}}),'');
 assert.throws(()=>safeOutput({data:{'image/png':'bm90IGEgcG5n'}}),/Invalid/);
});
test('execution manifest rejects stale source bytes and unsafe path traversal',()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'rl-output-test-'));
 try{
  fs.mkdirSync(path.join(root,'portal/generated/notebook-outputs'),{recursive:true});fs.mkdirSync(path.join(root,'week-01-example'));
  const name='week-01-example/workbook.ipynb',source='{"cells":[]}';fs.writeFileSync(path.join(root,name),source);
  fs.mkdirSync(path.join(root,'common'));fs.writeFileSync(path.join(root,'common/statistics.py'),'def mean(x): return sum(x)/len(x)\n');
  const helper=fs.readFileSync(path.join(root,'common/statistics.py'));
  const manifest={schema:1,notebooks:{[name]:{source_sha256:sha256(source),source_files_sha256:{'common/statistics.py':sha256(helper)},status:'executed',execution_mode:'bounded-reference',overrides:[{cell_index:3,description:'Four evaluation steps'}],cells:[]}}};
  const file=path.join(root,'portal/generated/notebook-outputs/manifest.json');fs.writeFileSync(file,JSON.stringify(manifest));
  assert.equal(readOutputManifest(root).notebooks[name].status,'executed');assert.match(outputStatus(manifest.notebooks[name]),/bounded teaching run/);assert.match(outputStatus(manifest.notebooks[name]),/Four evaluation steps/);
  fs.writeFileSync(path.join(root,'common/statistics.py'),helper+'# helper changed\n');assert.throws(()=>readOutputManifest(root),/Stale execution dependency/);fs.writeFileSync(path.join(root,'common/statistics.py'),helper);
  fs.writeFileSync(path.join(root,name),source+' ');assert.throws(()=>readOutputManifest(root),/Stale notebook/);
  manifest.notebooks={'../workbook.ipynb':manifest.notebooks[name]};fs.writeFileSync(file,JSON.stringify(manifest));assert.throws(()=>readOutputManifest(root),/Invalid execution source/);
 }finally{fs.rmSync(root,{recursive:true,force:true});}
});
