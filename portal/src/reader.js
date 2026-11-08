import {REPO,github} from './notebooks.js';
export const sectionRoute=(route,id)=>{const [base,query='']=route.split('?'),params=new URLSearchParams(query);params.delete('resume');params.set('section',id);return base+'?'+params.toString();};
export const issueURL=(doc,route)=>{const params=new URLSearchParams({title:`Teaching correction: ${doc.path}`,body:`Source: ${github(doc.path)}\nReading: https://study-rl.samuelzhang.co.uk/${route}\n\nWhat seems incorrect or unclear?\n\nSuggested correction and evidence:\n`});return `${REPO}/issues/new?${params}`;};
export function readingToolbar(){return '<details class="reader-settings" open><summary>Reading settings</summary><div class="reading-toolbar" aria-label="Reading preferences"><label>Text size <select data-reading-size><option value="90">90%</option><option value="100">100%</option><option value="110">110%</option><option value="120">120%</option><option value="130">130%</option></select></label><label>Line width <select data-reading-width><option value="comfortable">Comfortable</option><option value="wide">Wide</option></select></label><span>Settings stay in this browser.</span></div></details>';}
export function applyReading(state){document.documentElement.style.setProperty('--reading-scale',state.reading.size/100);document.body.classList.toggle('reading-wide',state.reading.width==='wide');document.querySelectorAll('[data-reading-size]').forEach(n=>n.value=state.reading.size);document.querySelectorAll('[data-reading-width]').forEach(n=>n.value=state.reading.width);}
export function mountReader(doc,{getState,onChange,onBookmark,toast}){
 const article=document.getElementById('article');if(!article)return()=>{};
 const settings=document.querySelector('.reader-settings');if(settings)settings.open=!matchMedia('(max-width:650px)').matches;
 applyReading(getState());const controls=document.querySelector('.reading-toolbar'),route=doc.route;
 const change=e=>{const reading={...getState().reading};if(e.target.matches('[data-reading-size]'))reading.size=Number(e.target.value);else if(e.target.matches('[data-reading-width]'))reading.width=e.target.value;else return;onChange(reading);applyReading(getState());};controls?.addEventListener('change',change);
 const link=async id=>{const value=new URL(sectionRoute(route,id),location.href).href;try{await navigator.clipboard.writeText(value);toast('Section link copied.');}catch{toast('Copy the link from the address bar.');location.hash=sectionRoute(route,id);}};
 for(const heading of article.querySelectorAll('h2[id],h3[id]')){
  const label=heading.cloneNode(true);for(const hidden of label.querySelectorAll('annotation,.katex-html,.section-actions'))hidden.remove();
  const title=label.textContent.replace(/\s+/g,' ').trim(),actions=document.createElement('span');actions.className='section-actions';
  const copy=document.createElement('button');copy.type='button';copy.textContent='Link';copy.setAttribute('aria-label','Copy link to '+title);copy.addEventListener('click',()=>link(heading.id));
  const bookmark=document.createElement('button');bookmark.type='button';bookmark.textContent='Save';bookmark.setAttribute('aria-label','Bookmark '+title);bookmark.addEventListener('click',()=>onBookmark(sectionRoute(route,heading.id),title));
  actions.append(copy,bookmark);heading.append(actions);
 }
 let count=0;for(const equation of article.querySelectorAll('.katex-display')){
  count++;equation.dataset.equationNumber=count;let anchor=equation.previousElementSibling;
  if(!anchor?.classList.contains('equation-anchor')){anchor=document.createElement('span');anchor.id='display-equation-'+count;anchor.className='equation-anchor';equation.before(anchor);}
  const copy=document.createElement('button');copy.type='button';copy.className='equation-link';copy.textContent='Equation link';copy.setAttribute('aria-label','Copy link to equation '+count);copy.addEventListener('click',()=>link(anchor.id));equation.after(copy);
 }
 for(const p of article.querySelectorAll('p')){const start=p.querySelector(':scope > strong:first-child')?.textContent?.toLowerCase()||'';for(const kind of ['assumption','worked example','common mistake','check yourself'])if(start.startsWith(kind))p.classList.add('teaching-callout');}
 const scrollFocus=()=>{for(const equation of article.querySelectorAll('.katex-display,.math.inline')){if(equation.scrollWidth>equation.clientWidth+1){equation.tabIndex=0;equation.setAttribute('role','region');equation.setAttribute('aria-label',(equation.dataset.equationNumber?'Equation '+equation.dataset.equationNumber:'Mathematical expression')+'; scroll horizontally to read');}else{equation.removeAttribute('tabindex');equation.removeAttribute('role');equation.removeAttribute('aria-label');}}};
 const observer=new ResizeObserver(scrollFocus);observer.observe(article);scrollFocus();
 return()=>{observer.disconnect();controls?.removeEventListener('change',change);};
}
export function mountFigureZoom(article){
 if(!article)return()=>{};let modal=null,opener=null;
 function open(target,title,button){
  opener=button;modal=document.createElement('dialog');modal.className='figure-dialog';modal.setAttribute('aria-label','Enlarged diagram');
  const bar=document.createElement('div');bar.className='figure-dialog-bar';const heading=document.createElement('h2');heading.textContent=title;
  const close=document.createElement('button');close.type='button';close.className='button secondary small';close.textContent='Close diagram';close.addEventListener('click',()=>modal.close());
  const label=document.createElement('label');label.textContent='Zoom ';const zoom=document.createElement('input');zoom.type='range';zoom.min='100';zoom.max='300';zoom.step='25';zoom.value='100';label.append(zoom);
  const value=document.createElement('output');value.textContent='100%';label.append(value);bar.append(heading,label,close);
  const canvas=document.createElement('div');canvas.className='figure-canvas';canvas.tabIndex=0;canvas.setAttribute('aria-label','Diagram; scroll to explore at higher zoom');
  const img=document.createElement('img');img.alt=title;
  if(target.tagName.toLowerCase()==='svg'){
   const svg=target.cloneNode(true);svg.setAttribute('xmlns','http://www.w3.org/2000/svg');svg.style.maxWidth='none';svg.removeAttribute('height');svg.setAttribute('width','1400');img.src=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)],{type:'image/svg+xml'}));
  }else img.src=target.src;
  img.style.width='100%';img.style.maxWidth='none';canvas.append(img);zoom.addEventListener('input',()=>{img.style.width=zoom.value+'%';value.textContent=zoom.value+'%';});
  modal.append(bar,canvas);modal.addEventListener('close',()=>{if(img.src.startsWith('blob:'))URL.revokeObjectURL(img.src);modal.remove();modal=null;opener?.focus();},{once:true});document.body.append(modal);modal.showModal();close.focus();
 }
 for(const target of article.querySelectorAll('img,.mermaid svg')){
  if(target.dataset.zoomReady)continue;target.dataset.zoomReady='true';const wrapper=target.closest('.mermaid')||target;
  const title=target.closest('figure')?.querySelector('figcaption')?.textContent||target.getAttribute('alt')||target.querySelector('title')?.textContent||'Lesson diagram';
  const button=document.createElement('button');button.type='button';button.className='figure-zoom button secondary small';button.textContent='Enlarge diagram ↗';button.setAttribute('aria-label','Enlarge '+title);button.addEventListener('click',()=>open(target,title,button));wrapper.after(button);
 }
 return()=>{if(modal)modal.close();};
}
