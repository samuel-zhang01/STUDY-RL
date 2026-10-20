/* Use a dedicated Browser View session.
 * node tests/browser.cjs /path/to/playwright-core http://127.0.0.1:9223 [/path/to/@axe-core/playwright]
 * Serve dist on 8767 first. Browser and axe are QA tools, not app dependencies.
 */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.argv[2]||'playwright');
const base=process.env.PORTAL_TEST_URL||'http://127.0.0.1:8767/';
const screenshots=process.env.PORTAL_QA_DIR||'/tmp/rl-round2-portal-qa';
(async()=>{
 fs.mkdirSync(screenshots,{recursive:true});
 const b=await chromium.connectOverCDP(process.argv[3]||'http://127.0.0.1:9223');
 const context=await b.newContext();const p=await context.newPage();
 const errors=[],failed=[],checks=[],accessibility=[];
 p.on('pageerror',e=>errors.push(e.message));
 p.on('response',r=>{if(r.status()>=400)failed.push([r.status(),r.url()]);});
 const goto=async route=>{await p.goto(base+'#/'+route);await p.waitForFunction(()=>document.querySelector('#main h1')&&!document.querySelector('.loading'));assert.equal(await p.locator('.error-panel').count(),0,route);};
 const check=async(name,fn)=>{await fn();checks.push(name);console.log('PASS',name);};
 try{
  await p.setViewportSize({width:1440,height:1000});await p.emulateMedia({reducedMotion:'reduce'});
  await p.goto(base);await p.evaluate(()=>localStorage.removeItem('study-rl-companion-v1'));await goto('home');
  await check('home, keyboard skip and complete course navigation',async()=>{
   assert.equal(await p.locator('[data-week]').count(),26);
   await p.locator('.skip-link').focus();await p.keyboard.press('Enter');assert.equal(await p.locator(':focus').getAttribute('id'),'main');
   await p.screenshot({path:path.join(screenshots,'home-desktop.png')});
   await goto('curriculum');assert.equal(await p.locator('.lesson-row').count(),26);
  });
  await check('all 26 teaching companions render their diagrams and full lessons',async()=>{
   for(let n=1;n<=26;n++){
    const week=String(n).padStart(2,'0');await goto(`lesson/${week}?view=guide`);
    await p.waitForSelector('.mermaid svg',{timeout:20000});
    assert.ok((await p.locator('.article').innerText()).length>3000);
    await goto(`lesson/${week}?view=lecture`);
    assert.ok(await p.locator('.article .katex').count()>0,week);
    assert.equal(await p.locator('.math-fallback').count(),0);
   }
  });
  await check('reading tabs, source reader, workbook cells and PDF download',async()=>{
   await goto('lesson/08?view=reading');assert.ok(await p.locator('.article a[href^="https"]').count()>0);
   await goto('lesson/08?view=workbook');assert.ok(await p.locator('.notebook-cell').count()>4);
   await goto('lesson/08?view=project');await p.locator('.source-grid a').first().click();await p.waitForSelector('.code-file');
   await goto('lesson/08?view=lecture');await p.locator('.lesson-meta a').filter({hasText:'Lecture PDF'}).click();await p.waitForSelector('object');
   const url=await p.locator('object').getAttribute('data');assert.equal((await p.request.get(base+url)).status(),200);
  });
  await check('search and repeated section navigation keep valid routes',async()=>{
   await p.locator('#search-button').click();await p.locator('#search-input').fill('termination');
   await p.waitForSelector('.search-result');await p.locator('.search-result').first().click();await p.waitForSelector('.article');
   await goto('read/glossary');
   for(let i=1;i<=2;i++){await p.locator('.reader-aside .toc a').nth(i).click();await p.waitForFunction(()=>document.querySelector('.article'));assert.ok(p.url().includes('?section='));assert.ok(!p.url().includes('glossary&'));}
   await goto('lesson/20?view=reading');
   await p.locator('.article a[href*="thm%3Arloo-unbiased"]').click();
   await p.waitForSelector('[id="thm:rloo-unbiased"]');assert.ok(p.url().includes('view=lecture'));
  });
  await check('four embedded experiments respond and fit their content',async()=>{
   for(const [id,control,value,result,expected] of [
    ['return','#discount','0','#return-result','Stop has the larger return'],
    ['td','#ending','terminated','#td-result','Target = 0'],
    ['ppo','#advantage','-2','#ppo-result','Active branch'],
    ['support','#logging-b','0','#support-result','No valid estimate']]){
    await goto('labs?experiment='+id);const f=p.frameLocator('.lab-iframe');
    await f.locator(control).waitFor();
    if(control==='#ending'||control==='#advantage')await f.locator(control).selectOption(value);else await f.locator(control).fill(value);
    assert.ok((await f.locator(result).innerText()).includes(expected),id);
    const frame=await p.locator('iframe').elementHandle();const body=await frame.contentFrame();
    const content=await body.locator('body').boundingBox(),box=await p.locator('iframe').boundingBox();
    assert.ok(box.height>=content.height-2&&box.height<content.height+80,`${id}: iframe ${box.height}, content ${content.height}`);
   }
  });
  await check('animations explain Bellman sweeps, GAE and tied groups',async()=>{
   await goto('labs?experiment=timeline');await p.locator('[data-step]').click();await p.locator('[data-step]').click();assert.match(await p.locator('.animation-explanation').innerText(),/3.6/);
   await goto('labs?experiment=bellman');for(let i=0;i<3;i++)await p.locator('[data-step]').click();assert.match(await p.locator('.animation-stage').innerText(),/3.24/);await p.locator('[data-reset]').click();assert.match(await p.locator('[data-counter]').innerText(),/Step 0/i);
   await goto('labs?experiment=gae');for(let i=0;i<3;i++)await p.locator('[data-step]').click();assert.match(await p.locator('.animation-stage').innerText(),/1.9216/);
   await goto('labs?experiment=groups');await p.locator('[data-reward="0"]').selectOption('0');assert.match(await p.locator('.animation-explanation').innerText(),/All rewards tie/);
  });
  await check('a newcomer can find instructions, GitHub and every downloadable notebook bundle',async()=>{
   await goto('start');assert.match(await p.locator('#main').innerText(),/first 30 minutes/i);
   assert.ok(await p.locator('#main a[href="https://github.com/samuel-zhang01/STUDY-RL"]').count()>0);
   await p.locator('#main a[href="#/downloads"]').click();await p.waitForSelector('#download-week');
   assert.equal(await p.locator('#download-week option').count(),26);
   for(let n=1;n<=26;n++){
    await p.locator('#download-week').selectOption(String(n));await p.waitForFunction(n=>document.querySelector('.download-picker h2')?.textContent.startsWith(String(n).padStart(2,'0')+'.'),n);
    const notebook=await p.locator('.download-actions a[download][href$=".ipynb"]').getAttribute('href');
    const response=await p.request.get(base+notebook);assert.equal(response.status(),200);const json=await response.json();assert.equal(json.nbformat,4);assert.ok(json.cells.length>0);
    const bundle=await p.locator('.download-actions a[download][href$=".zip"]').getAttribute('href');assert.ok(bundle.endsWith('week-'+String(n).padStart(2,'0')+'.zip'));
    const zipped=await p.request.get(base+bundle);assert.equal(zipped.status(),200);assert.equal((await zipped.body()).subarray(0,2).toString(),'PK');
    assert.ok((await p.locator('.download-actions a[href*="/blob/main/"]').getAttribute('href')).endsWith(notebook.replace(/^assets\//,'')));
   }
   const received=p.waitForEvent('download');await p.locator('.download-actions a[download][href$=".zip"]').click();assert.equal((await received).suggestedFilename(),'week-26.zip');
   await goto('lesson/09?view=workbook');assert.match(await p.locator('.notebook-bridge').innerText(),/edit and run them in Jupyter/);
   await goto('practice');await p.waitForSelector('#download-week');assert.equal(await p.locator('#python-editor,#run-python').count(),0);
   assert.equal(await p.evaluate(()=>performance.getEntriesByType('resource').some(r=>/pyodide|python-worker/.test(r.name))),false);
   await p.screenshot({path:path.join(screenshots,'downloads-desktop.png'),fullPage:true});
  });
  await check('four SVG teaching instruments change models, expose exact data and link back to notebooks',async()=>{
   for(const id of ['bandits','gridworld','ppo-geometry','importance-weights']){
    await goto('labs?experiment='+id);await p.waitForSelector('[data-chart="0"] svg');assert.equal(await p.locator('.plot-chart svg').count(),2);
    assert.ok(await p.locator('.plot-notebook a[download][href$=".ipynb"]').count()===1);
    const first=p.locator('.plot-control input[type="range"]').first();await first.focus();const before=await first.inputValue();await p.keyboard.press('ArrowRight');assert.notEqual(await first.inputValue(),before);
    await p.locator('[data-next]').click();await p.locator('[data-play]').click();await p.waitForFunction(()=>document.querySelector('[data-play]').textContent==='Pause');await p.locator('[data-play]').click();
    await p.locator('[data-reset]').click();await p.locator('.plot-data summary').click();assert.ok(await p.locator('.plot-data table tbody tr').count()>1);
    const received=p.waitForEvent('download');await p.locator('[data-csv]').click();assert.equal((await received).suggestedFilename(),'study-rl-'+id+'.csv');
    if(id==='ppo-geometry'){await p.locator('[data-input="advantage"]').fill('-1');await p.locator('[data-input="advantage"]').dispatchEvent('input');assert.match(await p.locator('[data-calculation]').innerText(),/sloped branch/);}
    if(id==='importance-weights'){await p.locator('[data-input="behavior"]').fill('0');await p.locator('[data-input="behavior"]').dispatchEvent('input');assert.match(await p.locator('[data-calculation]').innerText(),/support failure/i);}
   }
   await goto('lesson/02?view=guide');await p.locator('.inline-experiment summary').click();await p.waitForSelector('[data-chart="0"] svg');await p.locator('[data-next]').click();assert.match(await p.locator('[data-step-label]').innerText(),/Sweep 1/);
  });
  await check('ten courier investigations stay inside the portal and expose actionable experiments',async()=>{
   for(const id of ['bandit','bellman','targets','courier','policy','ope','safety','model','preferences','causal']){
    await goto('labs?experiment=courier-'+id);const frame=p.frameLocator('.lab-iframe');await frame.locator('#title').waitFor();assert.ok((await frame.locator('#title').innerText()).length>5);
    assert.ok((await frame.locator('#question').innerText()).length>10);assert.ok(await frame.locator('#controls input,#controls select,#actions button').count()>0);
    assert.ok(await frame.locator('#links a[href*="#/lesson/"]').count()>0);
   }
  });
  await check('milestones, notes, backup export, merge and invalid import',async()=>{
   await goto('lesson/04?view=guide');await p.locator('[data-milestone="calculate"]').check();await p.locator('#week-note').fill('I kept the bootstrap at a cutoff.');await goto('progress');
   const exported=p.waitForEvent('download');await p.locator('#export-progress').click();const download=await exported;const backup=JSON.parse(fs.readFileSync(await download.path(),'utf8'));assert.equal(backup.weeks['04'].calculate,true);assert.match(backup.weeks['04'].note,/cutoff/);
   await p.locator('#progress-file').setInputFiles({name:'progress.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({schema:1,weeks:{'04':{transfer:true,note:'Changed gamma independently.'}},drafts:{}}))});
   await p.waitForFunction(()=>document.getElementById('toast').textContent==='Progress imported.');
   assert.equal(await p.locator('[data-number="04"][data-milestone="calculate"]').isChecked(),true);
   assert.equal(await p.locator('[data-number="04"][data-milestone="transfer"]').isChecked(),true);
   await p.locator('#progress-file').setInputFiles({name:'invalid.json',mimeType:'application/json',buffer:Buffer.from('{}')});
   await p.waitForFunction(()=>document.getElementById('toast').textContent.includes('not a STUDY'));
   await p.reload();await p.waitForSelector('.milestone-card');assert.equal(await p.locator('[data-number="04"][data-milestone="transfer"]').isChecked(),true);
  });
  await check('320px, 390px and tablet layouts contain the reader and primary tools',async()=>{
   for(const width of [320,390,768]){
    await p.setViewportSize({width,height:844});
    for(const route of ['home','lesson/09?view=guide','lesson/08?view=lecture','lesson/01?view=workbook','labs?experiment=groups','labs?experiment=gridworld','labs?experiment=importance-weights','start','downloads','progress','library']){
     await goto(route);assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true,`${width}px ${route}`);
    }
   }
   await p.setViewportSize({width:390,height:844});await goto('lesson/09?view=guide');await p.waitForSelector('.mermaid svg');
   assert.equal(await p.locator('.mobile-toc').isVisible(),true);
   await p.locator('#menu-button').click();assert.equal(await p.locator('#close-menu').isVisible(),true);await p.keyboard.press('Escape');assert.equal(await p.locator('#menu-button').getAttribute('aria-expanded'),'false');
   await p.screenshot({path:path.join(screenshots,'lesson-mobile.png')});
  });
  if(process.argv[4])await check('automated accessibility for desktop, mobile, lessons, downloads and teaching instruments',async()=>{
   const AxeBuilder=require(process.argv[4]).default;
   for(const [width,route,scheme='light'] of [[1440,'home'],[1440,'lesson/08?view=lecture'],[1440,'labs?experiment=return'],[1440,'labs?experiment=groups'],[1440,'downloads'],[1440,'start'],[1440,'labs?experiment=bandits'],[1440,'labs?experiment=ppo-geometry'],[390,'home'],[390,'lesson/09?view=guide'],[390,'progress'],[320,'labs?experiment=gridworld'],[320,'labs?experiment=importance-weights'],[320,'downloads'],...['bandit','bellman','targets','courier','policy','ope','safety','model','preferences','causal'].map(id=>[320,'labs?experiment=courier-'+id]),[320,'labs?experiment=courier-causal','dark']]){
    await p.setViewportSize({width,height:1000});await p.emulateMedia({colorScheme:scheme,reducedMotion:'reduce'});await goto(route);
    if(route.includes('view=guide'))await p.waitForSelector('.mermaid svg');
    if(route.includes('experiment=return'))await p.frameLocator('iframe').locator('#return-result').waitFor();
    if(route.includes('experiment=courier-'))await p.frameLocator('iframe').locator('#title').waitFor();
    if(['bandits','gridworld','ppo-geometry','importance-weights'].some(id=>route.endsWith('experiment='+id)))await p.waitForSelector('[data-chart="0"] svg');
    const result=await new AxeBuilder({page:p}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
    accessibility.push({width,route,scheme,violations:result.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>({target:n.target,summary:n.failureSummary}))}))});
   }
   fs.writeFileSync(path.join(screenshots,'accessibility.json'),JSON.stringify(accessibility,null,2));
   assert.equal(accessibility.reduce((n,x)=>n+x.violations.length,0),0,JSON.stringify(accessibility.filter(x=>x.violations.length)));
  });
  assert.deepEqual(errors,[]);assert.deepEqual(failed,[]);
  const result={checks,passed:checks.length,pageErrors:errors,failedRequests:failed,accessibility,screenshots};
  fs.writeFileSync(path.join(screenshots,'results.json'),JSON.stringify(result,null,2));console.log(JSON.stringify({passed:checks.length,pageErrors:errors,failedRequests:failed,screenshots}));
 }finally{await context.close();await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
