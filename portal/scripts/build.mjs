import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {marked} from 'marked';
import katex from 'katex';
import sanitizeHtml from 'sanitize-html';
import {parseHTML} from 'linkedom';
import {contentFingerprint,interfaceFingerprint,artifactFingerprint,outputFingerprint} from './fingerprints.mjs';
import {readOutputManifest,safeOutput,outputStatus} from './notebook-outputs.mjs';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const OUT=path.join(ROOT,'dist');
const snapshot=JSON.parse(fs.readFileSync(path.join(ROOT,'generated/documents.json'),'utf8'));
const preview=process.argv.includes('--preview');
const buildInputs={content:contentFingerprint(),interface:interfaceFingerprint(),notebookOutputs:outputFingerprint()};
if(!preview&&(snapshot.source.preparationStable===false||JSON.stringify(snapshot.source.contentFingerprint)!==JSON.stringify(buildInputs.content)))throw new Error('Canonical content changed since preparation. Run make portal-content before building.');
const execution=readOutputManifest(path.join(ROOT,'..'),{allowStale:preview});
const docs=snapshot.documents;
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const slug=s=>s.toLowerCase().replace(/[^\p{L}\p{N}_\- ]/gu,'').replace(/ /g,'-');
const idFor=s=>s.replace(/^docs\//,'').replace(/\.[^.]+$/,'').replace(/[\/_]/g,'-').toLowerCase();
const special={'docs/LEARNING_MAP.md':'#/routes','docs/CONCEPT_ATLAS.md':'#/atlas','docs/PROGRESS.md':'#/progress','study-lab/README.md':'#/labs','docs/statistics/README.md':'#/statistics'};
const route=d=>{
 const match=d.path.match(/^week-(\d+)-[^/]+\/(teaching_guide\.md|notes\.tex|workbook\.ipynb|extended_reading\.tex)$/);
 if(match)return `#/lesson/${match[1]}?view=${({'teaching_guide.md':'guide','notes.tex':'lecture','workbook.ipynb':'workbook','extended_reading.tex':'reading'})[match[2]]}`;
 return special[d.path]||`#/read/${idFor(d.path)}`;
};
const byPath=new Map(docs.map(d=>[d.path,{...d,id:idFor(d.path),route:route(d)}]));
const assets=new Set(snapshot.assets);
function visibleText(element){const copy=element.cloneNode(true);for(const hidden of copy.querySelectorAll('annotation,.katex-html,.section-actions'))hidden.remove();return copy.textContent.replace(/\s+/g,' ').trim();}
const hasAnchor=(d,id)=>d.body?.includes(`id="${id}"`)||d.body?.includes(`\\label{${id}}`);
const unresolved=[], mathErrors=[];
function sectionRoute(base,anchor){return base+(base.includes('?')?'&':'?')+'section='+encodeURIComponent(decodeURIComponent(anchor));}
function resolve(raw,doc,image=false){
 if(!raw)return raw;
 if(/^(https?:|mailto:|tel:|data:)/.test(raw))return raw;
 if(raw.startsWith('/assets/'))return raw.slice(1);
 const [pathname,fragment]=raw.split('#');
 if(!pathname){
  const anchor=decodeURIComponent(fragment||'');
  const lecture=byPath.get(path.posix.join(path.posix.dirname(doc.path),'notes.tex'));
  return sectionRoute(!hasAnchor(doc,anchor)&&lecture&&hasAnchor(lecture,anchor)?lecture.route:doc.route,fragment||'');
 }
 const decoded=decodeURIComponent(pathname.split('?')[0]);
 const target=path.posix.normalize(path.posix.join(path.posix.dirname(doc.path),decoded));
 if(target==='interactive/index.html')return '#/labs?experiment=courier-'+(fragment||'bandit');
 if(target==='study-lab/index.html'||target==='study-lab/README.md')return '#/labs'+(fragment?'?experiment='+fragment.replace('-lab',''):'');
 const targetDoc=byPath.get(target);
 if(targetDoc)return fragment?sectionRoute(targetDoc.route,fragment):targetDoc.route;
 if(assets.has(target))return image?'assets/'+target:target.endsWith('.pdf')?'#/asset/'+encodeURIComponent(target):'assets/'+target;
 // Directory references open an inspectable source listing in the portal.
 if([...byPath.keys()].some(p=>p.startsWith(target.replace(/\/$/,'')+'/')))return '#/library?folder='+encodeURIComponent(target);
 if(target==='research')return '#/library?category=research';
 if(target==='scripts'||target==='scripts/')return '#/library?category=maintenance';
 // Maintenance-only commands may refer to files outside the published course snapshot.
 unresolved.push({from:doc.path,target,raw});
 return '#/library?query='+encodeURIComponent(path.posix.basename(target));
}
const macros={'\\States':'\\mathcal{S}','\\Actions':'\\mathcal{A}','\\Rewards':'\\mathcal{R}','\\Tr':'\\mathcal{T}','\\Policy':'\\pi','\\Optpol':'\\pi^{\\star}','\\Disc':'\\gamma','\\Hor':'H','\\Ret':'G','\\Vfn':'V','\\Qfn':'Q','\\Afn':'A','\\Adv':'A','\\E':'\\mathbb{E}','\\Prb':'\\mathbb{P}','\\Var':'\\operatorname{Var}','\\Cov':'\\operatorname{Cov}','\\KL':'\\mathrm{KL}','\\TV':'\\mathrm{TV}','\\Real':'\\mathbb{R}','\\Nat':'\\mathbb{N}','\\indep':'\\perp\\!\\!\\!\\perp','\\argmax':'\\operatorname*{arg\\,max}','\\argmin':'\\operatorname*{arg\\,min}','\\given':'\\,\\vert\\,','\\defeq':'\\triangleq'};
function math(tex,display,doc){
 tex=tex.replace(/\\bar\\mathrm\{([^}]+)\}/g,'\\overline{\\mathrm{$1}}').replace(/\\mbox\b/g,'\\text').replace(/_\\(min|max)\b/g,'_{\\$1}');
 let labels=[];tex=tex.replace(/\\label\{([^}]+)\}/g,(_,id)=>{labels.push(id);return ''});
 tex=tex.replace(/\\begin\{equation\*?\}|\\end\{equation\*?\}/g,'').replace(/\\begin\{align\*?\}/g,'\\begin{aligned}').replace(/\\end\{align\*?\}/g,'\\end{aligned}').replace(/\\nonumber\b|\\notag\b/g,'');
 try{return labels.map(id=>`<span id="${escape(id)}" class="equation-anchor"></span>`).join('')+katex.renderToString(tex,{displayMode:display,throwOnError:true,strict:'ignore',trust:false,macros,output:'htmlAndMathml'});}
 catch(error){mathErrors.push({doc:doc.path,tex,error:error.message});return `<code class="math-fallback">${escape(tex)}</code>`;}
}
function markdown(s){
 const protectedText=s.replace(/```[\s\S]*?```|`[^`]*`|\$\$([\s\S]+?)\$\$|(?<!\\)\$([^$\n]+?)(?<!\\)\$/g,(whole,block,inline)=>{
  if(block===undefined&&inline===undefined)return whole;
  const tex=escape(block??inline).replace(/_/g,'&#95;').replace(/\*/g,'&#42;').replace(/\\/g,'&#92;');
  return `<span class="math ${block!==undefined?'display':'inline'}">${block!==undefined?'&#92;[':'&#92;('}${tex}${block!==undefined?'&#92;]':'&#92;)'}</span>`;
 });
 return marked.parse(protectedText);
}
function cleanGuide(s){
 return s.replace(/^\*\*Course navigation:\*\*.*$/gm,'').replace(/^\[Routes\].*$/gm,'').replace(/^\[Lecture PDF\].*$/gm,'').replace(/^\*\*(?:You are here|Recall first|Visual anchor|Jump within this guide|Try it):\*\*.*$/gm,'').replace(/\n---\n\n\*\*Ready to move on\?[\s\S]*$/,'');
}
function notebookOutputLabel(doc,cell){
 const preceding=doc.cells.slice(0,cell.index-1).filter(c=>c.type==='markdown').at(-1)?.source||'';
 const heading=[...preceding.matchAll(/^#{1,6}\s+(.+)$/gm)].at(-1)?.[1];
 const purpose=preceding.split(/\n\s*\n/).find(p=>!/^\s*#/.test(p)&&p.trim().length>20)?.replace(/\[([^\]]+)\]\([^)]+\)/g,'$1').replace(/[*`$]/g,'').replace(/\s+/g,' ').trim();
 const concise=purpose?.replace(/^(?:Goal|Purpose|Predict)\.\s*/i,'').split(/(?<=[.!?])\s+/)[0]?.slice(0,180);
 return `${(heading||doc.title).replace(/^Statistical investigation\s*[—–-]\s*/,'')} · cell ${cell.index}${concise?'. '+concise:'. Read the matching explanation and interpretation in this workbook.'}`;
}
function render(doc){
 let input=doc.body||'';let html='';
 if(doc.format==='code')html=`<pre class="code-file"><code>${escape(input)}</code></pre>`;
 else if(doc.format==='notebook')html=outputStatus(execution.notebooks[doc.path])+'<div class="workbook-notice">Read every exercise and code cell here. Download the weekly bundle to run this notebook with its project files in Jupyter. Supplied reference cells demonstrate an implementation; attempt the exercises before comparing.</div>'+doc.cells.map(c=>`<section class="notebook-cell" id="cell-${c.index}" data-cell="${c.index}"><p class="cell-label">Cell ${c.index} · ${c.type==='code'?'Python':'Explanation'}</p>${c.type==='code'?`<pre><code class="language-python">${escape(c.source)}</code></pre>`:markdown(c.source)}${(execution.notebooks[doc.path]?.status==='executed'?execution.notebooks[doc.path]?.cells.find(x=>x.index===c.index)?.outputs||[]:[]).map(output=>safeOutput(output,notebookOutputLabel(doc,c))).join('')}</section>`).join('');
 else if(doc.format==='html')html=input;
 else {if(doc.kind==='guide')input=cleanGuide(input);html=markdown(input);}
 html=sanitizeHtml(html,{allowedTags:sanitizeHtml.defaults.allowedTags.concat(['img','figure','figcaption','details','summary','span','embed','input','section']),allowedAttributes:{'*':['class','id','title','data-cell','data-reference','data-reference-type','aria-hidden','aria-label','role','tabindex'],a:['href','title'],img:['src','alt','width','height','loading'],embed:['src'],input:['type','checked','disabled']},allowedSchemes:['http','https','mailto'],allowedSchemesByTag:{img:['http','https','data']},allowProtocolRelative:false});
 const {document}=parseHTML('<html><body><article>'+html+'</article></body></html>');const article=document.querySelector('article');
 // Pre-existing headings are retained so cross-document references keep working.
 const counts=new Map();
 for(const h of article.querySelectorAll('h1,h2,h3,h4,h5,h6')){
  if(!h.id){const key=slug(h.textContent),count=counts.get(key)||0;h.id=key+(count?'-'+count:'');counts.set(key,count+1);}
 }
 const h1=article.querySelector('h1');if(h1 && doc.format!=='html')h1.remove();
 for(const node of article.querySelectorAll('a[href]')){
  const url=node.getAttribute('href');node.setAttribute('href',resolve(url,doc));
  if(/^https?:/.test(url)){node.setAttribute('target','_blank');node.setAttribute('rel','noopener noreferrer');}
 }
 for(const node of article.querySelectorAll('img[src],embed[src]')){
  node.setAttribute('src',resolve(node.getAttribute('src'),doc,true));
  if(node.tagName==='EMBED'){const image=document.createElement('img');image.src=node.getAttribute('src');image.alt=node.closest('figure')?.querySelector('figcaption')?.textContent||'Lecture figure';node.replaceWith(image);}
  else {node.setAttribute('loading','lazy');if(!node.hasAttribute('alt'))node.setAttribute('alt','Course illustration');}
 }
 for(const pre of article.querySelectorAll('pre')){
  const code=pre.querySelector('code');if(code?.classList.contains('language-mermaid')){pre.className='mermaid';pre.textContent=code.textContent;}
 }
 for(const node of article.querySelectorAll('span.math')){
  let tex=node.textContent;const display=node.classList.contains('display');tex=tex.slice(2,-2);node.innerHTML=math(tex,display,doc);
 }
 // Markdown inside details is retained; process math text nodes outside code.
 function walk(node){
  if(node.nodeType===1&&['PRE','CODE','SCRIPT','STYLE'].includes(node.tagName))return;
  if(node.nodeType===1&&(node.classList.contains('katex')||node.classList.contains('math')))return;
  if(node.nodeType===3){
   const value=node.textContent;const pattern=/\$\$([\s\S]+?)\$\$|(?<!\\)\$([^$\n]+?)(?<!\\)\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)/g;
   if(!pattern.test(value))return;pattern.lastIndex=0;let end=0,out='';
   for(const m of value.matchAll(pattern)){out+=escape(value.slice(end,m.index))+math(m[1]||m[2]||m[3]||m[4],m[1]!==undefined||m[3]!==undefined,doc);end=m.index+m[0].length;}
   out+=escape(value.slice(end));const fragment=document.createElement('span');fragment.innerHTML=out;node.replaceWith(fragment);return;
  }
  for(const child of [...node.childNodes])walk(child);
 }
 walk(article);
 function linkCitations(node){
  if(node.nodeType===1&&(['A','PRE','CODE'].includes(node.tagName)||node.classList.contains('math')||node.classList.contains('katex')))return;
  if(node.nodeType===3&&/arXiv:\d{4}\.\d{4,5}/.test(node.textContent)){
   const span=document.createElement('span');span.innerHTML=escape(node.textContent).replace(/arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)/g,'<a href="https://arxiv.org/abs/$1" target="_blank" rel="noopener noreferrer">arXiv:$1</a>');node.replaceWith(span);return;
  }
  for(const child of [...node.childNodes])linkCitations(child);
 }
 linkCitations(article);
 for(const table of article.querySelectorAll('table')){const div=document.createElement('div');div.className='table-scroll';div.setAttribute('tabindex','0');div.setAttribute('role','region');div.setAttribute('aria-label','Scrollable lesson table');table.replaceWith(div);div.appendChild(table);}
 // Pandoc maps paragraph headings to h6 even without intervening levels.
 // Keep the source hierarchy and IDs while removing skipped HTML levels.
 const outline=[];for(const heading of [...article.querySelectorAll('h2,h3,h4,h5,h6')]){
  const sourceLevel=Number(heading.tagName[1]);while(outline.length&&outline.at(-1).sourceLevel>=sourceLevel)outline.pop();
  const level=outline.length?outline.at(-1).level+1:2;outline.push({sourceLevel,level});
  if(level!==sourceLevel){const replacement=document.createElement('h'+level);for(const attribute of heading.attributes)replacement.setAttribute(attribute.name,attribute.value);replacement.innerHTML=heading.innerHTML;heading.replaceWith(replacement);}
 }
 const text=visibleText(article);
 const toc=[...article.querySelectorAll('h2,h3')].map(h=>({id:h.id,title:visibleText(h),level:Number(h.tagName[1])}));
 return {html:article.innerHTML,text,toc,words:text.split(/\s+/).length};
}
fs.rmSync(OUT,{recursive:true,force:true});fs.mkdirSync(path.join(OUT,'content'),{recursive:true});
fs.cpSync(path.join(ROOT,'generated/assets'),path.join(OUT,'assets'),{recursive:true});fs.cpSync(path.join(ROOT,'src'),OUT,{recursive:true});
const index=[];
for(const doc of byPath.values()){
 const result=render(doc);const base={id:doc.id,path:doc.path,route:doc.route,title:doc.title,kind:doc.kind,format:doc.format,words:result.words,pdf:doc.pdf,download:doc.download};
 fs.writeFileSync(path.join(OUT,'content',doc.id+'.json'),JSON.stringify({...base,...result}));index.push({...base,text:result.text});
}
const titles=[...byPath.get('docs/STUDY_GUIDE.md').body.matchAll(/^\| (\d\d) \| \[([^\]]+)\]\(([^)]+)\) \| (.*?) \|/gm)].map(m=>({number:Number(m[1]),title:m[2],path:path.posix.normalize(path.posix.join('docs',m[3])),question:m[4]}));
const trackNames=['Foundations','Deep value learning','Policy optimisation','Decisions from logged data','Multi-agent learning','Offline RL','World models','Exploration & adaptation','Language-model RL','Safety & constraints','Applied audit','Causal biology'];
const trackWeeks=[[1,2,3,4],[5,6],[7,8,9,10],[11,12],[13],[14,15],[16],[17],[18,19,20,21],[22,23,24],[25],[26]];
const weeks=titles.map(w=>{const doc=byPath.get(w.path),prefix=w.path.split('/')[0];const recall=doc.body.match(/^\*\*Recall first:\*\*(.*)$/m)?.[1]||'';return {...w,track:trackWeeks.findIndex(t=>t.includes(w.number)),prerequisites:[...recall.matchAll(/\[Week (\d+)\]/g)].map(m=>Number(m[1])),views:Object.fromEntries(['guide','lecture','workbook','reading'].map(kind=>[kind,index.find(d=>d.path.startsWith(prefix+'/')&&d.kind===kind)?.id])),project:index.filter(d=>d.path.startsWith(prefix+'/project/')).map(d=>({id:d.id,name:d.path.split('/').pop(),route:d.route})),pdf:prefix+'/notes.pdf',notebook:prefix+'/workbook.ipynb'};});
fs.writeFileSync(path.join(OUT,'catalog.json'),JSON.stringify({weeks,tracks:trackNames.map((title,i)=>({title,weeks:trackWeeks[i]})),documents:index.map(({text,...x})=>x),assets:snapshot.assets}));
fs.writeFileSync(path.join(OUT,'search.json'),JSON.stringify(index.map(({id,title,route,path,kind,text})=>({id,title,route,path,kind,text}))));
fs.writeFileSync(path.join(OUT,'validation.json'),JSON.stringify({documents:index.length,lessons:weeks.length,mathErrors,unresolved},null,2));

// Local copies keep reading, diagrams and plots independent of CDNs.
function copyDir(from,to,filter=()=>true){fs.cpSync(from,to,{recursive:true,filter});}
for(const name of ['mermaid','katex'])copyDir(path.join(ROOT,'node_modules',name,'dist'),path.join(OUT,'vendor',name),(p)=>!p.endsWith('.map'));
for(const name of ['mermaid','katex'])fs.copyFileSync(path.join(ROOT,'node_modules',name,'LICENSE'),path.join(OUT,'vendor',name,'LICENSE'));
fs.writeFileSync(path.join(OUT,'vendor','packages.json'),JSON.stringify(['mermaid','katex','echarts'].map(name=>{const {version,license,homepage}=JSON.parse(fs.readFileSync(path.join(ROOT,'node_modules',name,'package.json')));return {name,version,license,homepage};}),null,2));
const chartDir=path.join(OUT,'vendor/echarts');fs.mkdirSync(chartDir,{recursive:true});
for(const [source,name] of [['dist/echarts.min.js','echarts.min.js'],['LICENSE','LICENSE'],['NOTICE','NOTICE']])fs.copyFileSync(path.join(ROOT,'node_modules/echarts',source),path.join(chartDir,name));
// Keep the original courier lab self-contained while its links return to the parent lesson reader.
const courier=path.join(OUT,'assets/interactive');
let courierHTML=fs.readFileSync(path.join(courier,'index.html'),'utf8').replace('</head>','<link rel="stylesheet" href="portal-bridge.css"><script defer src="portal-bridge.js"></script></head>');
fs.writeFileSync(path.join(courier,'index.html'),courierHTML);
fs.writeFileSync(path.join(courier,'portal-bridge.css'),':root{color-scheme:light;--bg:#f7f7f2;--fg:#172c30;--panel:#fff;--muted:#425b61;--line:#b9c9c9;--a:#007466;--b:#b14a11;--c:#6752b5;--soft:#e3f1ed;--danger:#a02d35}header,footer,body>.skip,#navigation,#previous,#next{display:none}body{background:transparent;color:#173e34;color-scheme:light}main{max-width:none;padding:0 6px}');
fs.writeFileSync(path.join(courier,'portal-bridge.js'),`const bridge=()=>{for(const a of document.querySelectorAll('a[href]')){const match=a.getAttribute('href').match(/week-(\\d+)-[^/]+\\/teaching_guide\\.md/);if(match){a.href='../../#/lesson/'+match[1]+'?view=guide';a.target='_parent';}}parent.postMessage({type:'study-lab-height',height:Math.ceil(document.body.getBoundingClientRect().height)},location.origin);};new ResizeObserver(bridge).observe(document.body);new MutationObserver(bridge).observe(document.getElementById('links'),{childList:true});window.addEventListener('load',bridge);`);
let lab=fs.readFileSync(path.join(OUT,'assets/study-lab/index.html'),'utf8');
lab=lab.replace('</head>',`<style>body{background:transparent}header,.intro,nav,footer,.lesson,.skip{display:none}main{padding:0;max-width:none}section{padding:16px 0;border:0}section:not([data-selected]){display:none}</style></head>`);
lab=lab.replace('</body>',`<script>const selected=new URLSearchParams(location.search).get('experiment')||'return';document.getElementById(selected+'-lab')?.setAttribute('data-selected','');const send=()=>parent.postMessage({type:'study-lab-height',height:Math.ceil(document.body.getBoundingClientRect().height)},location.origin);new ResizeObserver(send).observe(document.body);window.addEventListener('load',send);</script></body>`);
fs.writeFileSync(path.join(OUT,'assets/study-lab/index.html'),lab);
console.log(`Built ${index.length} readable documents; ${weeks.length} weeks; ${mathErrors.length} math errors; ${unresolved.length} unresolved source references.`);
if(mathErrors.length||unresolved.length)process.exitCode=1;
else {
 const finishedInputs={content:contentFingerprint(),interface:interfaceFingerprint(),notebookOutputs:outputFingerprint()};
 if(!preview&&JSON.stringify(buildInputs)!==JSON.stringify(finishedInputs))throw new Error('Publication inputs changed during the build. Finish edits and notebook execution, then prepare and build again.');
 fs.writeFileSync(path.join(OUT,'release.json'),JSON.stringify({...snapshot.source,preview,builtAt:new Date().toISOString(),notebookExecutions:{expected:docs.filter(d=>d.kind==='workbook').length,total:Object.keys(execution.notebooks).length,executed:Object.values(execution.notebooks).filter(e=>e.status==='executed').length,bounded:Object.values(execution.notebooks).filter(e=>e.execution_mode==='bounded-reference').length},fingerprints:{...finishedInputs,artifact:artifactFingerprint()}},null,2));
}
