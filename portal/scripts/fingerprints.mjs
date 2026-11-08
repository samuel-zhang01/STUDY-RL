import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';

export const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
export const sha256=data=>crypto.createHash('sha256').update(data).digest('hex');
const skipped=new Set(['.git','.env','.codex','.agents','node_modules','generated','dist','__pycache__','.pytest_cache','.ipynb_checkpoints','runs','logs','checkpoints','.venv','results']);
const extensions=new Set(['.md','.tex','.sty','.bib','.py','.ipynb','.json','.jsonl','.csv','.tsv','.html','.css','.js','.mjs','.cjs','.svg','.pdf','.txt']);
function files(directory){
 if(!fs.existsSync(directory))return [];
 return fs.readdirSync(directory,{withFileTypes:true}).sort((a,b)=>a.name.localeCompare(b.name)).flatMap(entry=>{
  if(skipped.has(entry.name)||entry.name.startsWith('.env')||entry.name.includes('.executed.'))return [];
  const name=path.join(directory,entry.name);
  if(entry.isSymbolicLink())throw new Error('Publication inputs must not contain symbolic links: '+path.relative(ROOT,name));
  return entry.isDirectory()?files(name):extensions.has(path.extname(name))?[name]:[];
 });
}
export function fingerprint(names){
 const records=[...new Set(names)].map(name=>[path.relative(ROOT,name).split(path.sep).join('/'),sha256(fs.readFileSync(name))]).sort((a,b)=>a[0].localeCompare(b[0]));
 return {sha256:sha256(JSON.stringify(records)),files:records.length};
}
export function contentFingerprint(){
 const directories=['common','docs','research','syllabus','llm-lab/llm_lab','llm-lab/lessons','llm-lab/data','llm-lab/tests','learning_projects','interactive','study-lab',...fs.readdirSync(ROOT).filter(n=>/^week-\d\d-/.test(n))];
 const roots=['README.md','CONTRIBUTING.md','SECURITY.md','LICENSE','docs/ATTRIBUTION.md','requirements.txt','requirements-foundations.txt','requirements-extras.txt','conftest.py','portal/README.md','scripts/new_week.py','scripts/execute_reference_notebooks.py','llm-lab/README.md','llm-lab/NVIDIA.md','llm-lab/VALIDATION.md','llm-lab/requirements.txt','llm-lab/workbook.ipynb','llm-lab/results/base_evaluation.json','llm-lab/results/sft_evaluation.json','llm-lab/results/sft_training.json','llm-lab/results/dpo_training.json'].map(n=>path.join(ROOT,n)).filter(n=>fs.existsSync(n));
 return fingerprint([...roots,...directories.flatMap(n=>files(path.join(ROOT,n)))]);
}
export function interfaceFingerprint(){return fingerprint([...files(path.join(ROOT,'portal/src')),...files(path.join(ROOT,'portal/scripts')),...['package.json','package-lock.json','README.md'].map(n=>path.join(ROOT,'portal',n))]);}
export function artifactFingerprint(directory=path.join(ROOT,'portal/dist')){
 function walk(base){return fs.readdirSync(base,{withFileTypes:true}).flatMap(e=>e.isDirectory()?walk(path.join(base,e.name)):e.name==='release.json'?[]:[path.join(base,e.name)]);}
 return fingerprint(walk(directory));
}
export function outputFingerprint(){const p=path.join(ROOT,'portal/generated/notebook-outputs/manifest.json');return fs.existsSync(p)?sha256(fs.readFileSync(p)):null;}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 if(process.argv.includes('--verify')){
  const release=JSON.parse(fs.readFileSync(path.join(ROOT,'portal/dist/release.json')));
  if(release.preview||release.preparationStable===false)throw new Error('Preview builds cannot pass publication verification. Prepare stable content and build without --preview.');
  if(release.notebookExecutions?.expected!==27||release.notebookExecutions?.executed!==27||release.notebookExecutions?.total!==27)throw new Error('Publication requires successful current execution evidence for all 26 weekly notebooks and the optional lab.');
  for(const [key,current] of Object.entries({content:contentFingerprint(),interface:interfaceFingerprint(),artifact:artifactFingerprint(),notebookOutputs:outputFingerprint()})){
   if(JSON.stringify(release.fingerprints?.[key])!==JSON.stringify(current))throw new Error('Stale publication: '+key+' fingerprint differs. Prepare content, build and verify again.');
  }
  process.stdout.write('Publication fingerprints verified: content, interface, executed outputs and complete static artifact.\n');
 }else process.stdout.write(JSON.stringify(contentFingerprint()));
}
