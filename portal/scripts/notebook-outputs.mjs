import fs from 'node:fs';
import path from 'node:path';
import sanitizeHtml from 'sanitize-html';
import {sha256} from './fingerprints.mjs';
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const join=value=>Array.isArray(value)?value.join(''):String(value??'');
export function safeOutput(output,label='Executed notebook figure'){
 const data=output.data||{};
 if(data['image/png']){
  const png=join(data['image/png']).replace(/\s/g,'');
  if(png.length>12000000||!/^[A-Za-z0-9+/]+={0,2}$/.test(png)||Buffer.from(png,'base64').subarray(0,8).toString('hex')!=='89504e470d0a1a0a')throw new Error('Invalid or oversized notebook PNG');
  return `<figure class="notebook-result"><img src="data:image/png;base64,${png}" alt="${escape(label)}" loading="lazy"><figcaption>${escape(label)}. Recorded reference output.</figcaption></figure>`;
 }
 if(data['text/html']){
  const safe=sanitizeHtml(join(data['text/html']).slice(0,250000),{allowedTags:['table','caption','thead','tbody','tr','th','td','p','strong','em','code','pre','br','div','span'],allowedAttributes:{th:['scope','colspan','rowspan'],td:['colspan','rowspan']},allowedSchemes:[],disallowedTagsMode:'discard'});
  return `<div class="notebook-output-table" tabindex="0" role="region" aria-label="${escape(label)} · result table">${safe}</div>`;
 }
 const text=join(data['text/plain']??output.text);
 return text?`<pre class="notebook-output-text"><code>${escape(text.slice(0,18000))}${text.length>18000?'\n… output abbreviated':''}</code></pre>`:'';
}
export function readOutputManifest(root,{allowStale=false}={}){
 const filename=path.join(root,'portal/generated/notebook-outputs/manifest.json');
 if(!fs.existsSync(filename))return {schema:1,notebooks:{}};
 const manifest=JSON.parse(fs.readFileSync(filename,'utf8'));
 if(manifest.schema!==1||!manifest.notebooks||Array.isArray(manifest.notebooks))throw new Error('Unsupported notebook execution manifest');
 for(const [name,entry] of Object.entries(manifest.notebooks)){
  if(!/^(?:week-\d\d-[a-z0-9-]+|llm-lab)\/workbook\.ipynb$/.test(name))throw new Error('Invalid execution source path');
  if(fs.realpathSync(path.join(root,name))!==path.join(root,name))throw new Error('Execution notebook sources must not contain symbolic links: '+name);
  if(sha256(fs.readFileSync(path.join(root,name)))!==entry.source_sha256){if(allowStale){delete manifest.notebooks[name];continue;}throw new Error('Stale notebook execution: '+name);}
  if(!entry.source_files_sha256||Array.isArray(entry.source_files_sha256)||typeof entry.source_files_sha256!=='object')throw new Error('Missing execution dependency hashes: '+name);
  let staleDependency=false;
  for(const [dependency,hash] of Object.entries(entry.source_files_sha256)){
   if(!/^(?:common\/[a-zA-Z0-9_-]+\.py|week-\d\d-[a-z0-9-]+\/(?:project\/(?:[a-zA-Z0-9_-]+\/)*[a-zA-Z0-9_-]+\.py|(?:_pkg_loader|conftest)\.py)|llm-lab\/llm_lab\/(?:[a-zA-Z0-9_-]+\/)*[a-zA-Z0-9_-]+\.py)$/.test(dependency)||!/^[a-f0-9]{64}$/.test(hash))throw new Error('Invalid execution dependency path or hash: '+name);
   const filename=path.join(root,dependency);
   if(fs.realpathSync(filename)!==filename)throw new Error('Execution dependencies must not contain symbolic links: '+dependency);
   if(sha256(fs.readFileSync(filename))!==hash){if(allowStale){staleDependency=true;break;}throw new Error('Stale execution dependency: '+dependency+' ('+name+')');}
  }
  if(staleDependency){delete manifest.notebooks[name];continue;}
  if(!['executed','failed'].includes(entry.status))throw new Error('Unknown execution status: '+name);
  if(!Array.isArray(entry.cells))throw new Error('Missing output cell list: '+name);
  const cells=new Set();
  for(const cell of entry.cells){if(!Number.isInteger(cell.index)||cell.index<1||cells.has(cell.index)||!Array.isArray(cell.outputs))throw new Error('Invalid output cell index: '+name);cells.add(cell.index);}
 }
 return manifest;
}
export function outputStatus(entry){
 if(!entry)return '<aside class="execution-status execution-missing"><strong>Source workbook.</strong> A current reference execution is not attached to this release. Download the bundle to run it.</aside>';
 const bounded=entry.execution_mode==='bounded-reference';
 return `<aside class="execution-status ${entry.status==='failed'?'execution-missing':''}"><strong>${entry.status==='executed'?'Reference execution recorded':'Reference execution incomplete'}${bounded?' · bounded teaching run':''}.</strong> ${entry.executed_at?escape(entry.executed_at):''}${bounded?' Training budgets or evaluation settings were reduced for reproducible inspection.':''}${entry.overrides?.length?`<details><summary>Execution settings (${entry.overrides.length} adjustments)</summary><ul>${entry.overrides.map(x=>`<li>Cell ${x.cell_index}: ${escape(x.description)}</li>`).join('')}</ul></details>`:''}<p>These outputs illustrate the supplied reference path. Check the changed-input exercise independently; successful reference execution does not establish your implementation or a paper benchmark.</p></aside>`;
}
