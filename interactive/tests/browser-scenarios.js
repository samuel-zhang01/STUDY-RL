/* Reusable with any Playwright Page. Tests observable behavior, not DOM snapshots. */
module.exports = async function scenarios(page, baseURL, options = {}) {
  page.setDefaultTimeout(10000);page.setDefaultNavigationTimeout(10000);
  await page.emulateMedia({reducedMotion:'no-preference',colorScheme:'light'});
  const errors = [], requests = [], checks = [];
  const verify = (condition, message) => { if (!condition) throw Error(message); checks.push(message); };
  page.on('pageerror', error => errors.push(error.message));
  page.on('requestfailed', request => errors.push(request.url() + ': ' + request.failure().errorText));
  page.on('request', request => requests.push(request.url()));
  const go = async id => { await page.goto(baseURL + '/interactive/#' + id); await page.locator('#title').waitFor(); };
  const slider = async (name,value) => page.getByRole('slider',{name,exact:true}).evaluate((el,v)=>{el.value=v;el.dispatchEvent(new Event('input',{bubbles:true}));},value);
  const result = () => page.locator('#experiment').innerText();
  const counts = () => page.locator('#experiment tbody tr').evaluateAll(rows=>rows.reduce((sum,row)=>sum+Number(row.cells[1].textContent),0));

  await go('bandit');
  verify(await counts()===0,'new bandit has no fabricated observations');
  await page.getByRole('button',{name:'Try route A',exact:true}).click();
  verify(await counts()===1 && /Estimate: 0.00 \+/.test(await result()),'manual choice exposes one numeric mean update');
  await page.getByRole('button',{name:'Run 100 decisions',exact:true}).click();
  verify(await counts()===101,'automatic run adds exactly100 observed choices');
  await page.getByLabel('Reveal simulator rates and regret').check();
  verify((await result()).includes('Cumulative mean-gap regret'),'oracle information is explicitly revealed');
  await page.locator('#seed').fill('');await page.getByRole('button',{name:'Reset with this seed',exact:true}).click();
  verify((await page.locator('#error').innerText()).includes('whole-number') && await counts()===101,'empty seed reports error and preserves data');
  await page.locator('#seed').fill('-1');await page.getByRole('button',{name:'Reset with this seed',exact:true}).click();
  verify((await page.locator('#error').innerText()).includes('whole-number'),'negative seed is rejected');
  await page.locator('#seed').fill('7');await page.getByRole('button',{name:'Reset with this seed',exact:true}).click();
  verify(await counts()===0 && await page.locator('#error').innerText()==='','valid reset clears evidence and validation error');
  await page.getByRole('button',{name:'Animate 20 / pause',exact:true}).click();
  await page.waitForFunction(()=>[...document.querySelectorAll('#experiment tbody tr')].some(r=>Number(r.cells[1].textContent)>0));
  await page.getByRole('button',{name:'Animate 20 / pause',exact:true}).click();const paused=await counts();
  await page.waitForTimeout(400);verify(await counts()===paused,'pause stops simulation updates');
  await page.getByRole('button',{name:'Animate 20 / pause',exact:true}).click();
  await page.getByRole('button',{name:'2 · Send value backward',exact:true}).click();
  await page.waitForTimeout(400);verify((await page.locator('#title').innerText()).startsWith('2'),'changing lessons cancels the previous animation');
  verify((await result()).includes('2.40'),'Bellman policy value matches hand calculation');
  await slider('Discount gamma',.5);verify((await result()).includes('Best action: tie'),'discount threshold yields the action tie');
  await slider('Chance of continuing',1);await slider('Discount gamma',0);verify((await result()).includes('policy value = (1 − 1.00) × 2 + 1.00 × 0.00 = 0.00'),'discount and probability endpoint arithmetic stays correct');
  await go('targets');verify((await result()).includes('2.30')&&(await result()).includes('1.85'),'MC and TD update from identical initial value');
  await page.getByLabel('Use the terminal stop transition (reward 2)').check();await slider('Estimated next-state value',8);
  verify((await result()).includes('TD new estimate = 1.50'),'terminal target ignores the next-state estimate');
  await go('courier');verify(await page.locator('.hazard').count()===2,'courier renders exactly two hazards');
  for(let i=0;i<4;i++)await page.getByRole('button',{name:'Train 100 episodes',exact:true}).click();
  verify((await result()).includes('Completed episodes: 400'),'courier completes the requested training episodes');
  await page.getByRole('button',{name:'Watch greedy route / pause',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('#announcement').textContent==='Delivery reached the goal.');
  verify(await page.locator('.goal.agent').count()===1,'learned greedy route visibly reaches the goal');
  await go('policy');await page.getByLabel('Show PPO fixed-batch surrogate').check();await slider('Chance of continuing',.8);
  verify((await result()).includes('Clipped surrogate = 0.120')&&(await result()).includes('Weighted minimum'),'PPO plateau has individual signed contributions');
  for(let i=0;i<35;i++)await page.getByRole('button',{name:'Take one exact-return gradient step',exact:true}).click();
  const p=await page.getByRole('slider',{name:'Chance of continuing',exact:true}).inputValue();
  verify(Number(p)<=.999 && (await result()).includes('p='+Number(p).toFixed(3)),'repeated gradient steps keep model and control probability consistent');
  await page.getByLabel('Week 8 extension: inspect GAE credit weights').check();await slider('GAE lambda',0);
  verify((await result()).includes('GAE advantage = 1.000'),'GAE lambda0 keeps only the first residual');
  await slider('GAE lambda',1);verify((await result()).includes('GAE advantage = 2.170'),'GAE lambda1 includes discounted later residuals');
  await go('ope');await slider('Logger chance of route C',0);
  verify((await result()).includes('Unsupported target')&&(await result()).includes('IPS = unavailable'),'unsupported logged target does not fabricate an estimate');
  await slider('New policy chance of route C',0);verify(!(await result()).includes('Unsupported target'),'supported single-action target remains evaluable');
  if(options.downloads!==false){const downloading=page.waitForEvent('download');await page.getByRole('button',{name:'Export these logs',exact:true}).click();const download=await downloading;verify(download.suggestedFilename()==='courier-route-logs.json','synthetic logs export has a deterministic filename');const exported=JSON.parse(await require('node:fs/promises').readFile(await download.path(),'utf8'));verify(exported.version===1 && exported.seed===7 && exported.behaviorProbability===0 && exported.targetProbability===0 && exported.rows.length===200 && exported.rows.every(r=>r.action===0 && r.propensity===1 && [0,1].includes(r.reward)),'export contains the displayed seeded dataset and propensities');}
  await go('safety');verify((await result()).includes('satisfied in expectation'),'safety budget is correctly labeled as expectation');
  await slider('Chance of shortcut',1);verify((await result()).includes('violated'),'excess expected cost violates budget');
  await page.getByLabel('Shield every shortcut (substitute scenic)').check();verify((await result()).includes('expected cost = 0.00'),'perfect toy shield removes shortcut cost');
  await go('model');await slider('Discount gamma',1);await slider('Planning horizon (steps)',20);verify((await result()).includes('extra return = 4.00'),'undiscounted model bias accumulates linearly');
  await go('preferences');await slider('Length bonus in proxy',2);verify((await result()).includes('Optimizer prefers repetition'),'bad length proxy favors the wrong behavior');
  await go('causal');verify((await result()).includes('shortcut 60.0%')&&(await result()).includes('scenic 50.0%'),'standardization matches the within-stratum causal example');

  for(const width of [320,390,768,1280]){
    await page.setViewportSize({width,height:900});
    for(const id of ['bandit','bellman','targets','courier','policy','ope','safety','model','preferences','causal']){
      await go(id);
      verify(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),id+' fits viewport'+width);
      verify(await page.locator('#question').innerText()!=='' && await page.locator('#exercise').innerText()!=='',id+' has prediction and transfer at'+width);
      for(const link of await page.locator('#links a, footer a').evaluateAll(xs=>xs.map(a=>a.href))){const response=await page.request.get(link);verify(response.status()===200,'course link resolves '+new URL(link).pathname);}
    }
  }
  await page.emulateMedia({reducedMotion:'reduce'});await go('bandit');await page.getByRole('button',{name:'Animate 20 / pause',exact:true}).click();
  verify(await counts()===1 && (await page.locator('#announcement').innerText()).includes('Reduced motion'),'reduced motion advances one step without animation');
  await page.waitForTimeout(400);verify(await counts()===1,'reduced motion leaves no running timer');
  await page.emulateMedia({colorScheme:'dark'});await go('policy');verify(await page.evaluate(()=>getComputedStyle(document.body).backgroundColor!=='rgb(247, 247, 242)'),'dark preference changes the lab palette');
  await page.getByRole('slider',{name:'Chance of continuing',exact:true}).focus();const before=Number(await page.getByRole('slider',{name:'Chance of continuing',exact:true}).inputValue());await page.keyboard.press('ArrowRight');
  verify(Number(await page.getByRole('slider',{name:'Chance of continuing',exact:true}).inputValue())>before,'native keyboard changes the policy control');
  verify(errors.length===0,'no browser exceptions or failed resource requests: '+errors.join('; '));
  verify(requests.every(url=>url.startsWith(baseURL)),'lab makes no requests to third-party origins');
  return {checks:checks.length,errors};
};
