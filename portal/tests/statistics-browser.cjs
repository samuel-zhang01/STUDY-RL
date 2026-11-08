// Dedicated CDP test context; no real learner progress is touched.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.argv[2]||'playwright-core');
const base=process.env.PORTAL_TEST_URL||'http://127.0.0.1:8767/',out=process.env.PORTAL_QA_DIR||'/tmp/studyrl-statistics-qa';
(async()=>{
 fs.mkdirSync(out,{recursive:true});const browser=await chromium.connectOverCDP(process.argv[3]||'http://127.0.0.1:9233');const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce'});const page=await context.newPage(),errors=[],failed=[],checks=[],accessibility=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)failed.push([r.status(),r.url()]);});
 async function goto(route){await page.goto(base+'#/'+route);await page.waitForSelector('#main h1');await page.waitForFunction(()=>!document.querySelector('.loading'));assert.equal(await page.locator('.error-panel').count(),0,route);}
 async function check(name,fn){await fn();checks.push(name);console.log('PASS',name);}
 try{
  await check('statistics companion exposes nine original lessons and three investigations',async()=>{
   await goto('statistics');assert.equal(await page.locator('.statistics-chapters>a').count(),9);assert.equal(await page.locator('.statistics-lab-cards>a').count(),3);assert.equal(await page.evaluate(()=>document.activeElement.tagName),'H1');await page.screenshot({path:path.join(out,'statistics-desktop.png'),fullPage:true});
   await page.locator('.statistics-chapters>a').nth(2).click();await page.waitForSelector('.article');assert.ok((await page.locator('.article').innerText()).length>2000);
  });
  for(const id of ['sampling-distribution','paired-comparisons','clustered-bootstrap'])await check(id+' changes a real model, supports keys, and exports data',async()=>{
   await goto('labs?experiment='+id);await page.waitForSelector('.statistics-svg');const before=await page.locator('[data-stat-result]').innerText();const slider=page.locator('[data-stat-input]').first();await slider.focus();await page.keyboard.press('ArrowRight');assert.notEqual(await page.locator('[data-stat-result]').innerText(),before);assert.equal(await slider.evaluate(e=>document.activeElement===e),true);
   await page.locator('.plot-data summary').click();const download=page.waitForEvent('download');await page.locator('[data-stat-csv]').click();const file=await download;assert.equal(file.suggestedFilename(),'study-rl-'+id+'.csv');assert.ok(fs.readFileSync(await file.path(),'utf8').split('\n').length>5);
   await page.locator('[data-stat-reset]').click();assert.ok(await page.locator('.plot-table-wrap tbody tr').count()>=5);await page.screenshot({path:path.join(out,id+'-desktop.png'),fullPage:true});
  });
  await check('every week has an explicit statistics handoff and stable workbook links',async()=>{
   for(let n=1;n<=26;n++){await goto('lesson/'+String(n).padStart(2,'0')+'?view=guide');assert.equal(await page.locator('.statistical-thread a').count(),1);assert.equal(await page.locator('.lesson-journey a').count(),4);}
   await goto('lesson/09?view=lecture');await page.screenshot({path:path.join(out,'lesson-desktop.png'),fullPage:true});
  });
  await check('mobile lecture shows learning content in the first viewport',async()=>{await page.setViewportSize({width:390,height:844});await goto('lesson/09?view=lecture');assert.ok((await page.locator('.article').boundingBox()).y<600);assert.equal(await page.locator('.lesson-meta-panel').getAttribute('open'),null);assert.equal(await page.locator('.reader-settings').getAttribute('open'),null);await page.locator('.reader-settings>summary').click();assert.ok(await page.locator('[data-reading-size]').isVisible());});
  await check('mobile investigation selector offers all 25 instruments',async()=>{await page.setViewportSize({width:390,height:844});await goto('labs?experiment=sampling-distribution');assert.equal(await page.locator('#lab-select option').count(),25);await page.locator('#lab-select').selectOption('paired-comparisons');await page.waitForFunction(()=>document.querySelector('.statistics-instrument h2')?.textContent.includes('seed by seed'));assert.equal(await page.evaluate(()=>document.activeElement.id),'lab-select');});
  await check('new learning routes fit narrow screens at default and enlarged reading sizes',async()=>{
   for(const width of [320,390,768])for(const route of ['home','statistics','labs?experiment=sampling-distribution','labs?experiment=paired-comparisons','labs?experiment=clustered-bootstrap','lesson/09?view=lecture']){
    await page.setViewportSize({width,height:844});await goto(route);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true,width+' '+route);
    if(width===390)await page.screenshot({path:path.join(out,route.startsWith('labs')?route.split('=')[1]+'-mobile.png':route.split('?')[0].replace('/','-')+'-mobile.png'),fullPage:true});
   }
  });
  if(process.argv[4])await check('new statistics and redesigned routes meet automated WCAG AA checks',async()=>{
   const Axe=require(process.argv[4]).default;
   for(const width of [390,1440])for(const route of ['home','statistics','labs?experiment=sampling-distribution','labs?experiment=paired-comparisons','labs?experiment=clustered-bootstrap','lesson/09?view=lecture','lesson/01?view=workbook']){
    await page.setViewportSize({width,height:1000});await goto(route);const result=await new Axe({page}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();accessibility.push({width,route,violations:result.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>({target:n.target,summary:n.failureSummary}))}))});
   }
   assert.equal(accessibility.reduce((n,r)=>n+r.violations.length,0),0,'Accessibility findings saved in '+path.join(out,'results.json'));
  });
  assert.deepEqual(errors,[]);assert.deepEqual(failed,[]);
 }finally{fs.writeFileSync(path.join(out,'results.json'),JSON.stringify({checks,errors,failed,accessibility},null,2));await context.close();await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
