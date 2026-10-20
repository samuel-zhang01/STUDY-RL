// Run against a dedicated Browser View session; no production user data is used.
const assert=require('node:assert/strict'),fs=require('node:fs');
const {chromium}=require(process.argv[2]);
const base=process.env.PORTAL_TEST_URL||'http://127.0.0.1:8767/',out=process.env.PORTAL_QA_DIR||'/tmp/rl-next-qa';
(async()=>{
 fs.mkdirSync(out,{recursive:true});const browser=await chromium.connectOverCDP(process.argv[3]||'http://127.0.0.1:9223');const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce',permissions:['clipboard-read','clipboard-write']});
 const page=await context.newPage(),errors=[],failed=[],checks=[],axe=[];context.on('page',p=>{p.on('pageerror',e=>errors.push(e.message));p.on('response',r=>{if(r.status()>=400)failed.push(r.url());});});page.on('pageerror',e=>errors.push(e.message));
 const goto=async(p,route)=>{await p.goto(base+route);await p.waitForSelector('#main h1');assert.equal(await p.locator('.error-panel').count(),0);};
 const check=async(name,fn)=>{await fn();checks.push(name);console.log('PASS',name);};
 const backup=async p=>{await goto(p,'#/progress');const event=p.waitForEvent('download');await p.locator('#export-progress').click();return JSON.parse(fs.readFileSync(await (await event).path(),'utf8'));};
 try{
  await check('legacy milestones, notes and drafts migrate without data loss',async()=>{
   await goto(page,'#/home');await page.evaluate(()=>{localStorage.clear();localStorage.setItem('study-rl-companion-v1',JSON.stringify({schema:1,weeks:{'04':{explain:true,note:'Legacy cutoff note'}},drafts:{td:'legacy Python draft'},last:{route:'#/lesson/04?view=guide',y:300},focus:false}));});await page.reload();await goto(page,'#/lesson/04?view=guide');assert.equal(await page.locator('#week-note').inputValue(),'Legacy cutoff note');assert.equal(await page.locator('[data-milestone="explain"]').isChecked(),true);const saved=await backup(page);assert.equal(saved.drafts.td,'legacy Python draft');
  });
  await check('reader settings, section bookmarks, GitHub correction links and diagram zoom work',async()=>{
   await goto(page,'#/lesson/09?view=guide');await page.waitForSelector('.mermaid svg');await page.locator('[data-reading-size]').selectOption('120');await page.locator('[data-reading-width]').selectOption('wide');
   await page.locator('.section-actions button').filter({hasText:'Save'}).first().click();await page.locator('.section-actions button').filter({hasText:'Link'}).first().click();const copied=await page.evaluate(()=>navigator.clipboard.readText());assert.match(copied,/lesson\/09\?view=guide&section=/);
   const report=await page.locator('.reader-aside a').filter({hasText:'Report a teaching error'}).getAttribute('href');assert.ok(report.startsWith('https://github.com/samuel-zhang01/STUDY-RL/issues/new?'));assert.ok(decodeURIComponent(report).includes('teaching_guide.md'));assert.ok(!report.includes('Legacy'));
   const zoom=page.locator('.figure-zoom').first();await zoom.click();await page.waitForSelector('.figure-dialog[open]');await page.locator('.figure-dialog input').fill('200');assert.equal(await page.locator('.figure-dialog output').textContent(),'200%');await page.screenshot({path:out+'/diagram-enlarged.png'});await page.keyboard.press('Escape');await page.waitForSelector('.figure-dialog',{state:'detached'});assert.equal(await page.locator('.figure-dialog').count(),0);assert.equal(await zoom.evaluate(e=>e===document.activeElement),true);
   await goto(page,'#/progress');assert.equal(await page.locator('.saved-item').count(),1);await page.locator('.saved-item a').click();await page.waitForSelector('#article');assert.equal(await page.locator('[data-reading-size]').inputValue(),'120');assert.ok(page.url().includes('section='));
   await goto(page,'#/lesson/09?view=lecture');await page.locator('.equation-link').first().click();const eq=await page.evaluate(()=>navigator.clipboard.readText());await page.goto(eq);await page.waitForSelector('#article');const id=new URLSearchParams(page.url().split('?')[1]).get('section');assert.equal(await page.locator(`[id="${id}"]`).count(),1);
  });
  await check('diagnostics explain wrong answers, track confidence, retry and join cumulative review',async()=>{
   await goto(page,'#/lesson/01?view=guide');const q=page.locator('[data-review-question]').first();await q.locator('.review-choices input').nth(0).check();await q.locator('.review-confidence input[value="unsure"]').check();await q.locator('[type="submit"]').click();assert.ok((await q.locator('[data-review-feedback]').innerText()).length>250);assert.equal(await q.locator('[data-review-feedback] a[href*="view=workbook"]').count(),1);
   await q.locator('[data-review-retry]').click();assert.equal(await page.locator('[data-review-question]').first().locator('[data-review-feedback]').textContent(),'');await page.locator('[data-review-question]').nth(1).locator('[data-review-unsure]').click();
   await goto(page,'#/review');assert.ok(await page.locator('[data-review-question]').count()>=2);await page.locator('[data-review-filter]').selectOption('all');await page.locator('[data-review-week]').selectOption('26');assert.equal(await page.locator('[data-review-question]').count(),3);await page.reload();await page.waitForSelector('[data-review-stats]');const saved=await backup(page);assert.equal(Object.keys(saved.quiz).length,2);assert.equal(Object.values(saved.quiz).filter(r=>r.attempts===0).length,1);assert.equal(saved.weeks['01']?.explain||false,false);
  });
  await check('two tabs preserve concurrent note versions and merge independent milestones',async()=>{
   const other=await context.newPage();await goto(page,'#/lesson/04?view=guide');await goto(other,'#/lesson/04?view=guide');
   await page.locator('#week-note').fill('First tab reasoning');await other.locator('#week-note').fill('Second tab reasoning');
   await page.evaluate(()=>window.dispatchEvent(new Event('pagehide')));await other.evaluate(()=>window.dispatchEvent(new Event('pagehide')));
   await page.locator('[data-milestone="calculate"]').check();await other.locator('[data-milestone="transfer"]').check();
   await other.locator('#focus-button').click();await page.waitForFunction(()=>document.body.classList.contains('focused'));assert.equal(await page.locator('#focus-button').getAttribute('aria-pressed'),'true');await page.locator('#focus-button').click();await other.waitForFunction(()=>!document.body.classList.contains('focused'));
   const saved=await backup(page);assert.equal(saved.weeks['04'].calculate,true);assert.equal(saved.weeks['04'].transfer,true);const versions=[saved.weeks['04'].note,...Object.values(saved.conflicts).map(v=>v.note)];assert.ok(versions.includes('First tab reasoning'));assert.ok(versions.includes('Second tab reasoning'));assert.equal(await page.locator('.note-conflicts').count(),1);
   await page.locator('.note-conflicts summary').first().click();await page.locator('[data-restore-note]').first().click();const restored=await backup(page);assert.ok([restored.weeks['04'].note,...Object.values(restored.conflicts).map(v=>v.note)].includes('Second tab reasoning'));await other.close();
  });
  await check('mobile lifecycle flush and backup imports retain notes, settings, quiz and saved sections',async()=>{
   await goto(page,'#/lesson/03?view=guide');await page.locator('#week-note').fill('A last-second mobile note');await page.evaluate(()=>window.dispatchEvent(new Event('pagehide')));await page.reload();await page.waitForSelector('#week-note');assert.equal(await page.locator('#week-note').inputValue(),'A last-second mobile note');
   const saved=await backup(page);await page.evaluate(()=>localStorage.clear());await page.reload();await page.waitForSelector('#progress-file',{state:'attached'});await page.locator('#progress-file').setInputFiles({name:'backup.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(saved))});await page.waitForFunction(()=>document.getElementById('toast').textContent==='Progress imported.');const restored=await backup(page);assert.equal(restored.weeks['03'].note,'A last-second mobile note');assert.equal(restored.reading.size,120);assert.equal(Object.keys(restored.bookmarks).length,1);assert.equal(Object.keys(restored.quiz).length,2);
  });
  await check('blocked storage shows a warning while export still keeps new notes',async()=>{
   const blocked=await browser.newContext();await blocked.addInitScript(()=>{Storage.prototype.setItem=function(){throw new DOMException('blocked','SecurityError');};});const p=await blocked.newPage();await goto(p,'#/lesson/02?view=guide');await p.locator('#week-note').fill('Unsaved but exportable');await p.evaluate(()=>window.dispatchEvent(new Event('pagehide')));assert.match(await p.locator('#save-indicator').innerText(),/Not saved/);const saved=await backup(p);assert.equal(saved.weeks['02'].note,'Unsaved but exportable');await blocked.close();
  });
  await check('reading and review controls fit narrow screens and pass accessibility checks',async()=>{
   const Axe=process.argv[4]?require(process.argv[4]).default:null;
   for(const width of [320,390,1440])for(const route of ['#/lesson/01?view=guide','#/review','#/progress']){
    await page.setViewportSize({width,height:1000});await goto(page,route);if(route.includes('guide'))await page.waitForSelector('.mermaid svg');assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true,`${width} ${route}`);
    if(Axe){const result=await new Axe({page}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();axe.push({width,route,violations:result.violations});}
    if(width===390)await page.screenshot({path:out+'/'+(route.includes('guide')?'reader':route.slice(2))+'-mobile.png'});
   }
   assert.equal(axe.reduce((sum,r)=>sum+r.violations.length,0),0,JSON.stringify(axe.filter(r=>r.violations.length)));
  });
  assert.deepEqual(errors,[]);assert.deepEqual(failed,[]);fs.writeFileSync(out+'/learning-results.json',JSON.stringify({checks,errors,failed,axe},null,2));console.log(JSON.stringify({passed:checks.length,accessibility:axe.length,errors,failed}));
 }finally{await context.close();await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
