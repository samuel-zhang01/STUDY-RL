import {banditRun, GRID, gridRun, gridBackup, gridActions, isTerminal, ppoPoint, makeBanditLog, importanceEstimate} from './plot-model.js';

const REPOSITORY = 'https://github.com/samuel-zhang01/STUDY-RL';
export const PLOTS = {
  bandits: {title:'The cost of learning', week:'01', folder:'week-01-mdps-bandits', description:'Compare exploration and greedy learning on the same seeded bandit. Follow reward estimates and cumulative expected opportunity cost.', question:'Can a good early guess become a bad habit?', kind:'Exploration'},
  gridworld: {title:'Watch the future travel', week:'02', folder:'week-02-bellman-dp', description:'A value-iteration heatmap. Step through synchronous backups and inspect every action behind a state’s next value.', question:'How many sweeps until a distant reward can affect this state?', kind:'Dynamic programming'},
  'ppo-geometry': {title:'Where PPO stops rewarding change', week:'09', folder:'week-09-ppo', description:'Move along the surrogate objective and its slope. Flip the advantage sign to see which side becomes flat.', question:'Does going outside the clip interval always remove the gradient?', kind:'Policy optimisation'},
  'importance-weights': {title:'When a few samples carry everything', week:'11', folder:'week-11-contextual-bandits-ope', description:'See importance weights, effective sample size and target-policy estimates change together on a reproducible log.', question:'Does having 200 logged decisions mean 200 useful observations?', kind:'Off-policy evaluation'}
};
const colors = {green:'#1c6454', amber:'#a35b23', blue:'#356a9a', violet:'#79518a', light:'#e6eddd', ink:'#193c36', muted:'#5b6e67'};
const f = (n, digits = 3) => n === null || !Number.isFinite(n) ? 'undefined' : Number(n.toFixed(digits)).toLocaleString('en-GB', {maximumFractionDigits:digits});
const escape = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const range = (id, label, value, min, max, step, help = '') => `<label class="plot-control" for="plot-${id}"><span>${label}<output data-value="${id}" for="plot-${id}">${value}</output></span><input id="plot-${id}" data-input="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${value}" ${help ? `aria-describedby="plot-help-${id}"` : ''}>${help ? `<small id="plot-help-${id}">${help}</small>` : ''}</label>`;
const stat = (label, value, detail = '') => `<div class="plot-stat"><span>${label}</span><strong>${value}</strong>${detail ? `<small>${detail}</small>` : ''}</div>`;
const table = (caption, headings, rows) => `<div class="plot-table-wrap" tabindex="0" role="region" aria-label="${escape(caption)}"><table><caption>${caption}</caption><thead><tr>${headings.map(x=>`<th scope="col">${x}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map((cell,i)=>`<${i?'td':'th scope="row"'}>${cell}</${i?'td':'th'}>`).join('')}</tr>`).join('')}</tbody></table></div>`;
const coordinate = state => `(${state % GRID.width}, ${Math.floor(state / GRID.width)})`;
let chartLibrary;
function loadCharts() {
  if (globalThis.echarts) return Promise.resolve(globalThis.echarts);
  if (!chartLibrary) chartLibrary = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = new URL('vendor/echarts/echarts.min.js', import.meta.url).href;
    script.async = true;
    const timer = setTimeout(() => {script.remove(); chartLibrary = null; reject(new Error('Chart library timeout'));}, 15000);
    script.onload = () => {clearTimeout(timer); globalThis.echarts ? resolve(globalThis.echarts) : reject(new Error('Chart library unavailable'));};
    script.onerror = () => {clearTimeout(timer); script.remove(); chartLibrary = null; reject(new Error('Chart library unavailable'));};
    document.head.append(script);
  });
  return chartLibrary;
}

/**
 * Render a self-contained teaching instrument. A synchronous return value makes
 * route cleanup safe even while the local chart library is still loading.
 */
export function mountPlot(host, id) {
  if (!PLOTS[id]) throw new Error(`Unknown teaching plot: ${id}`);
  const metadata = PLOTS[id], abort = new AbortController(), signal = abort.signal;
  let destroyed = false, charts = [], timer = null, csv = '', drawCurrent = () => {};
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  host.innerHTML = `<section class="plot-instrument" aria-label="${metadata.title}">
    <div class="plot-intro"><div><p class="eyebrow">Visual instrument · Week ${metadata.week}</p><h2>${metadata.title}</h2><p>${metadata.description}</p></div><span class="plot-kind">${metadata.kind}</span></div>
    <div class="plot-predict"><span aria-hidden="true">?</span><p><strong>Predict first.</strong> ${metadata.question} Change one setting, then explain the evidence.</p></div>
    <div class="plot-controls" data-controls></div>
    <div class="plot-actions"><button class="button" data-next>Next step →</button><button class="button secondary" data-play>Play</button><button class="button secondary" data-reset>Reset</button><span data-step-label></span></div>
    <div class="plot-stats" data-stats></div>
    <div class="plot-figure"><h3 data-chart-title></h3><div class="plot-legend" data-legend="0"></div><div class="plot-chart" data-chart="0" role="img"></div><p class="plot-chart-caption" data-caption="0"></p></div>
    <div class="plot-figure"><h3 data-chart-title-secondary></h3><div class="plot-legend" data-legend="1"></div><div class="plot-chart plot-chart-secondary" data-chart="1" role="img"></div><p class="plot-chart-caption" data-caption="1"></p></div>
    <div class="plot-calculation" data-calculation aria-live="polite" aria-atomic="true"></div>
    <details class="plot-data"><summary>Read the values as tables / download chart data</summary><div data-table></div><button class="button secondary small" data-csv>Download this view’s data (.csv)</button></details>
    <details class="plot-explain"><summary>What to try next, and what this does not show</summary><div data-explanation></div></details>
    <div class="plot-notebook"><div><p class="eyebrow">Take the idea into code</p><h3>Reproduce it in the week ${metadata.week} notebook</h3><p>The charts use small teaching fixtures. The workbook and project provide the course’s actual exercises.</p></div><div class="plot-notebook-links"><a class="button small" href="#/lesson/${metadata.week}?view=workbook">Read the workbook</a><a href="assets/${metadata.folder}/workbook.ipynb" download>Download Jupyter notebook ↓</a><a href="${REPOSITORY}/tree/main/${metadata.folder}" target="_blank" rel="noopener">Week’s files on GitHub ↗</a><a href="#/lesson/${metadata.week}?view=guide">Return to the explanation →</a></div></div>
    <p class="plot-library-note">Charts: Apache ECharts, rendered as SVG. Controls work with a keyboard and each chart has a table alternative. Displayed numbers are rounded; CSV downloads retain full computed precision.</p>
  </section>`;
  const query = selector => host.querySelector(selector);
  const set = (selector, html) => {query(selector).innerHTML = html;};
  function output(name, value) {query(`[data-value="${name}"]`).value = value;}
  function setInput(name, value) {query(`[data-input="${name}"]`).value = value; output(name, value);}
  const stop = () => {clearInterval(timer);timer = null;query('[data-play]').textContent = 'Play';query('[data-calculation]').setAttribute('aria-live','polite');};
  function playback(tick, finished, interval = 160) {
    if(timer){stop();return;}
    query('[data-play]').textContent = 'Pause';
    if(finished()) return stop();
    query('[data-calculation]').setAttribute('aria-live','off');
    timer = setInterval(()=>{tick();if(finished())stop();}, interval);
  }
  const on = (selector, event, callback) => query(selector).addEventListener(event, callback, {signal});
  function settings(callback) {
    on('[data-controls]', 'input', event => {
      if (!event.target.matches('[data-input]')) return;
      stop();callback(event.target.dataset.input, Number(event.target.value));
    });
  }
  function options(description) {
    return {animation:!reduced.matches, animationDuration:300, animationDurationUpdate:timer?100:250,
      color:[colors.green,colors.amber,colors.blue,colors.violet], textStyle:{fontFamily:'Inter, ui-sans-serif, system-ui, sans-serif',color:colors.ink},
      tooltip:{trigger:'axis',renderMode:'richText',confine:true}, grid:{top:30,right:20,bottom:48,left:40,containLabel:true},
      legend:{show:false},
      xAxis:{type:'value',nameLocation:'middle',nameGap:29,axisLabel:{fontSize:10,color:colors.muted},splitLine:{show:false},axisLine:{lineStyle:{color:'#bacab4'}}},
      yAxis:{type:'value',axisLabel:{fontSize:10,color:colors.muted},splitLine:{lineStyle:{color:'#e8eee1'}}},
      aria:{enabled:false}, ...description};
  }
  function paint(index, option, description) {
    const element = query(`[data-chart="${index}"]`);
    element.setAttribute('aria-label', description);
    if(option.yAxis?.name) option.yAxis={...option.yAxis,nameTextStyle:{align:'left',fontSize:10}};
    if(option.xAxis?.name) option.xAxis={...option.xAxis,nameTextStyle:{fontSize:10}};
    if(charts[index]) charts[index].setOption(options(option), {notMerge:true});
  }
  function legend(index, items) {
    set(`[data-legend="${index}"]`,items.map(([label,color,dashed])=>`<span><i aria-hidden="true" style="border-color:${color};${dashed?'border-top-style:dashed':''}"></i>${escape(label)}</span>`).join(''));
  }
  function line(name, data, extra = {}) {return {name,type:'line',data,showSymbol:false,lineStyle:{width:2.5},...extra};}
  const setExplanation = html => set('[data-explanation]', html);
  const setCaption = (index, text) => {query(`[data-caption="${index}"]`).textContent = text;};
  function announceStep(text) {query('[data-step-label]').textContent = text;}
  const cell = query('[data-controls]');

  if(id === 'bandits') {
    let epsilon=.1, seed=2, step=60, explored, greedy;
    const recompute = () => {explored=banditRun({epsilon,seed});greedy=banditRun({epsilon:0,seed});};
    cell.innerHTML=range('epsilon','Explore probability ε',epsilon,0,1,.05,'Exploration picks uniformly from all three arms, including the greedy arm.')+range('seed','Random seed',seed,1,30,1,'Both policies see the same potential rewards at each round.')+range('round','Round',step,0,300,1,'Scrub the history; no samples are skipped in the calculation.');
    query('[data-chart-title]').textContent='Expected opportunities forgone';
    query('[data-chart-title-secondary]').textContent='What the exploring learner currently believes';
    legend(0,[['Exploring policy',colors.green],['Greedy ε = 0',colors.amber,true]]);
    legend(1,[['Estimate Q · bars',colors.green],['True probability · diamonds',colors.amber]]);
    setExplanation(`<p><strong>Try:</strong> compare ε = 0, 0.1 and 1 at the same seed. Then compare seeds 2 and 7: the greedy learner starts more luckily in one of them. A curve from one run cannot establish which algorithm is better on average.</p><p><strong>Mechanism.</strong> Each arm pays 1 with probability 0.25, 0.50 or 0.75; otherwise it pays 0. The learner does not observe these probabilities. Q starts at 0, greedy ties are broken randomly, and after a pull Q ← Q + (reward − Q) / count.</p><p><strong>Read the plot carefully.</strong> Pseudo-regret adds μ* − μ<sub>chosen</sub> for each decision. It measures expected opportunity cost using the simulator’s known means, not the difference between two noisy realised rewards. Uniform exploration can choose the best arm too.</p><p><strong>Boundary check.</strong> At ε = 0, both curves coincide. At ε = 1, all choices are uniform; estimates improve but the policy never exploits them. These are stationary Bernoulli arms, without contexts, confidence intervals or a regret guarantee.</p><p>Source: <a href="https://mitpress.mit.edu/9780262039246/reinforcement-learning/" target="_blank" rel="noopener">Sutton and Barto, chapter 2</a>. <strong>Notebook challenge:</strong> run 100 seeds and plot mean cumulative pseudo-regret with uncertainty across seeds.</p>`);
    function draw() {
      const x=explored[step], base=greedy[step], previous=explored[Math.max(0,step-1)];
      output('epsilon',epsilon.toFixed(2));output('seed',seed);output('round',step);
      set('[data-stats]',stat('Exploring policy',f(x.regret),'cumulative pseudo-regret')+stat('Greedy baseline',f(base.regret),'same seed, ε = 0')+stat('Observed rewards',`${x.totalReward||0} / ${step}`,'exploring policy'));
      paint(0,{xAxis:{type:'value',min:0,max:Math.max(50,Math.ceil(step/50)*50),name:'Decision round',nameLocation:'middle',nameGap:27},yAxis:{type:'value',name:'Pseudo-regret',min:0,max:Math.max(5,Math.ceil(Math.max(x.regret,base.regret)/5)*5)},
        series:[line(`ε = ${epsilon.toFixed(2)}`,explored.slice(0,step+1).map(r=>[r.step,r.regret])),line('Greedy ε = 0',greedy.slice(0,step+1).map(r=>[r.step,r.regret]),{lineStyle:{type:'dashed',width:2.5}})]},
        `At round ${step}, exploring cumulative pseudo-regret is ${f(x.regret)} and greedy pseudo-regret is ${f(base.regret)}. Full values are in the table.`);
      paint(1,{grid:{top:25,bottom:35,left:30,right:15,containLabel:true},xAxis:{type:'category',data:['Arm A','Arm B','Arm C']},yAxis:{type:'value',min:0,max:1,name:'Mean reward'},series:[
        {name:'Learned estimate Q',type:'bar',data:x.estimates,barMaxWidth:35,itemStyle:{borderRadius:[4,4,0,0]}},
        {name:'True probability μ',type:'scatter',symbol:'diamond',symbolSize:14,data:[.25,.5,.75],itemStyle:{color:colors.amber}}
      ]},`Learned estimates for A, B, C: ${x.estimates.map(v=>f(v)).join(', ')}. True probabilities: 0.25, 0.50, 0.75.`);
      setCaption(0,'Lower is better. A flat segment means the policy selected a truly optimal arm; the reward can still be 0. Axes expand as the displayed history grows.');
      setCaption(1,'Bars are the learner’s sample means. Diamonds are simulator truths, hidden from the learner. Unpulled arms retain their initial estimate of zero.');
      set('[data-calculation]',step?`<p class="eyebrow">Inspect decision ${step}</p><p>${x.explored?'An exploration draw':'A greedy decision'} chose <strong>arm ${'ABC'[x.action]}</strong>, which paid ${x.reward}. Its ${x.counts[x.action]} observations now give</p><p class="plot-equation">Q = ${f(previous.estimates[x.action])} + (${x.reward} − ${f(previous.estimates[x.action])}) / ${x.counts[x.action]} = <strong>${f(x.estimates[x.action])}</strong></p><p>This decision adds 0.75 − ${[.25,.5,.75][x.action]} = <strong>${f(.75-[.25,.5,.75][x.action])}</strong> expected opportunity cost. Observed reward and regret answer different questions.</p>`:`<p><strong>No decisions yet.</strong> Each Q estimate starts at 0. All three arms tie, so the first greedy choice is random. Predict which early reward might make the learner keep choosing one arm.</p>`);
      set('[data-table]',table('Current arm estimates for the exploring policy',['Arm','True μ','Pulls','Reward sum','Estimate Q'],x.counts.map((count,a)=>['ABC'[a],f([.25,.5,.75][a]),count,x.successes?.[a]||0,f(x.estimates[a])]))+table('Current policy comparison',['Policy','Decisions','Reward sum','Pseudo-regret'],[[`ε = ${f(epsilon)}`,step,x.totalReward||0,f(x.regret)],['Greedy ε = 0',step,base.totalReward||0,f(base.regret)]]));
      csv='round,exploring_action,exploring_reward,exploring_pseudo_regret,greedy_pseudo_regret\n'+explored.slice(0,step+1).map((r,i)=>[i,r.action===undefined?'':r.action,r.reward,r.regret,greedy[i].regret].join(',')).join('\n');
      announceStep(`Round ${step} / 300`);query('[data-next]').disabled=step>=300;
    }
    recompute();drawCurrent=draw;
    const advance=()=>{step=Math.min(300,step+1);setInput('round',step);draw();};
    on('[data-next]','click',()=>{stop();advance();});
    on('[data-play]','click',()=>{if(step===300){step=0;draw();}playback(advance,()=>step>=300,100);});
    on('[data-reset]','click',()=>{stop();epsilon=.1;seed=2;step=0;setInput('epsilon',epsilon);setInput('seed',seed);setInput('round',step);recompute();draw();});
    settings((name,value)=>{if(name==='epsilon')epsilon=value;if(name==='seed')seed=value;if(name==='round')step=value;else recompute();draw();});
  }

  if(id === 'gridworld') {
    let gamma=.9,cost=-.04,step=0,selected=0,history;
    const recompute=()=>{history=gridRun({gamma,cost});};
    cell.innerHTML=range('discount','Discount γ',gamma,0,.99,.01,'A new discount restarts the sweeps at V₀ = 0.')+range('living','Reward on a non-terminal move',cost,-.2,0,.01,'Entering the goal pays +1; entering the trap pays −1 instead.')+`<label class="plot-control" for="plot-cell"><span>Inspect a state</span><select id="plot-cell" data-input="cell">${Array.from({length:20},(_,s)=>s).filter(s=>!GRID.walls.includes(s)&&!isTerminal(s)).map(s=>`<option value="${s}">${coordinate(s)}${s===0?' · start':''}</option>`).join('')}</select><small>Coordinates are (column, row), starting at 0 in the upper left.</small></label>`+range('sweep','Completed sweeps',step,0,40,1,'All cells in one sweep read the previous grid.');
    query('[data-chart-title]').textContent='Value estimates and the next greedy decision';
    query('[data-chart-title-secondary]').textContent='How much another sweep would change';
    legend(1,[['Bellman residual',colors.green]]);
    setExplanation(`<p><strong>Try:</strong> step from sweep 0 to 1, then 2. A neighbouring cell sees the +1 immediately; a more distant cell must wait for that information to arrive. Select (0, 0), predict its next backup, and check the action table.</p><p><strong>Environment.</strong> Four deterministic actions; a wall or boundary leaves the agent in place and pays the movement reward. The two marked outcomes terminate the episode. Their stored values are always zero because their reward has already been paid on entry.</p><p><strong>Mechanism.</strong> V<sub>k+1</sub>(s) = max<sub>a</sub>[r(s,a) + γ V<sub>k</sub>(s′)]. Every backup in a sweep reads V<sub>k</sub>. Arrows show all actions tied for the largest <em>next</em> backup using the displayed grid; they do not indicate an action already taken.</p><p><strong>Boundary check.</strong> Set γ = 0: only the immediate reward matters. Set movement reward to 0: some early backups tie. The residual is max<sub>s</sub>|T V<sub>k</sub>(s) − V<sub>k</sub>(s)|; it is a consistency check, not a measured episode return.</p><p>Source: <a href="https://mitpress.mit.edu/9780262039246/reinforcement-learning/" target="_blank" rel="noopener">Sutton and Barto, chapter 4</a>. <strong>Notebook challenge:</strong> add a 20% chance of slipping sideways and replace each deterministic backup with an expectation.</p>`);
    const residualAt=k=>Math.max(...gridBackup(history[k],gamma,cost).map((v,s)=>Math.abs(v-history[k][s])));
    function draw() {
      const values=history[step],next=gridBackup(values,gamma,cost),actions=gridActions(values,selected,gamma,cost),best=Math.max(...actions.map(a=>a.q));
      output('discount',gamma.toFixed(2));output('living',cost.toFixed(2));output('sweep',step);
      set('[data-stats]',stat('Completed sweeps',step,'synchronous backups')+stat(`V${step} at ${coordinate(selected)}`,f(values[selected]),'current estimate')+stat('Largest next change',f(residualAt(step),5),'Bellman residual'));
      const data=values.map((v,s)=>{
        const wall=GRID.walls.includes(s),terminal=isTerminal(s),a=gridActions(values,s,gamma,cost),maximum=Math.max(...a.map(x=>x.q)),arrows=a.filter(x=>Math.abs(x.q-maximum)<1e-10).map(x=>x.arrow).join('');
        return {value:[s%5,Math.floor(s/5),v],state:s,itemStyle:{color:wall?'#d3d9cf':terminal?s===GRID.goal?'#e0ecac':'#f2dfd5':undefined,borderColor:s===selected?'#a35b23':'#fafbf8',borderWidth:s===selected?4:2},label:{formatter:wall?'WALL':terminal?s===GRID.goal?'GOAL +1\nV = 0':'TRAP −1\nV = 0':`${f(v,2)}\n${arrows}`,fontSize:host.clientWidth<500?10:13,color:'#153f3a'}};
      });
      paint(0,{grid:{top:12,right:12,left:12,bottom:54},legend:{show:false},tooltip:{trigger:'item',renderMode:'richText',formatter:p=>{const s=p.data.state;return GRID.walls.includes(s)?'Wall':isTerminal(s)?`${s===GRID.goal?'Goal +1':'Trap −1'} on entry; future value 0`:`State ${coordinate(s)}\nV after ${step} sweeps: ${f(values[s],5)}\nNext optimal backup: ${f(next[s],5)}`;}},
        xAxis:{type:'category',data:[0,1,2,3,4],position:'top',show:false,splitArea:{show:true}},yAxis:{type:'category',data:[0,1,2,3],inverse:true,show:false},
        visualMap:{min:-1,max:1,orient:'horizontal',left:'center',bottom:2,itemWidth:15,itemHeight:150,inRange:{color:['#d8ab8d','#fafbf1','#8eb88b']},text:['+1','−1'],textStyle:{color:colors.muted,fontSize:10}},
        series:[{name:'State value',type:'heatmap',data,label:{show:true},emphasis:{disabled:true}}]},
        `A 5-column 4-row value grid after ${step} sweeps. Selected state ${coordinate(selected)} has value ${f(values[selected])}; next backup is ${f(best)}. All states and actions are available in the tables.`);
      paint(1,{xAxis:{type:'value',min:0,max:40,name:'Completed sweep',nameLocation:'middle',nameGap:28},yAxis:{type:'value',min:0,max:1,name:'Maximum change'},series:[line('Bellman residual',Array.from({length:step+1},(_,k)=>[k,residualAt(k)]))]},
        `Maximum change from one more sweep is ${f(residualAt(step),5)} at sweep ${step}.`);
      setCaption(0,'Click a non-terminal cell, or use “Inspect a state”. The amber border marks your selection. Tied arrows are all shown; wall and terminal labels override the value colours.');
      setCaption(1,'A zero residual means every state already equals its Bellman backup, within floating-point precision. The grid is deterministic and fully known.');
      set('[data-calculation]',`<p class="eyebrow">The next backup at ${coordinate(selected)}</p><p>Choose the largest action value using <strong>only the displayed V<sub>${step}</sub></strong>:</p><p class="plot-equation">V<sub>${step+1}</sub>(${coordinate(selected)}) = max(${actions.map(a=>f(a.q)).join(', ')}) = <strong>${f(best)}</strong></p><p>Current V<sub>${step}</sub> = ${f(values[selected])}. Next change = ${f(best-values[selected])}. Best action${actions.filter(a=>Math.abs(a.q-best)<1e-10).length>1?'s':''}: <strong>${actions.filter(a=>Math.abs(a.q-best)<1e-10).map(a=>a.name).join(', ')}</strong>.</p>`);
      set('[data-table]',table(`Next backup for ${coordinate(selected)} at sweep ${step}`,['Action','Next state','Reward','Future V','r + γV'],actions.map(a=>[`${a.arrow} ${a.name}`,coordinate(a.next),f(a.reward),f(a.future),`${f(a.q)}${Math.abs(a.q-best)<1e-10?' · best':''}`]))+table('All non-wall state values',['State','Current V','Next backup'],values.flatMap((v,s)=>GRID.walls.includes(s)?[]:[[`${coordinate(s)}${isTerminal(s)?' · terminal':''}`,f(v,5),f(next[s],5)]])));
      csv='column,row,current_value,next_backup,terminal\n'+values.flatMap((v,s)=>GRID.walls.includes(s)?[]:[[s%5,Math.floor(s/5),v,next[s],isTerminal(s)].join(',')]).join('\n');
      announceStep(`Sweep ${step} / 40${residualAt(step)<1e-10?' · fixed point reached':''}`);query('[data-next]').disabled=step>=40;
    }
    recompute();drawCurrent=draw;
    const advance=()=>{step=Math.min(40,step+1);setInput('sweep',step);draw();};
    on('[data-next]','click',()=>{stop();advance();});
    on('[data-play]','click',()=>{if(!timer&&(step===40||residualAt(step)<1e-10)){step=0;setInput('sweep',0);draw();}playback(advance,()=>step>=40||residualAt(step)<1e-10,650);});
    on('[data-reset]','click',()=>{stop();gamma=.9;cost=-.04;step=0;selected=0;setInput('discount',gamma);setInput('living',cost);setInput('sweep',0);query('[data-input="cell"]').value='0';recompute();draw();});
    settings((name,value)=>{if(name==='discount')gamma=value;if(name==='living')cost=value;if(name==='cell')selected=value;if(name==='sweep')step=value;if(name==='discount'||name==='living'){step=0;setInput('sweep',0);recompute();}draw();});
    host._plotCellSelect=s=>{if(!GRID.walls.includes(s)&&!isTerminal(s)){stop();selected=s;query('[data-input="cell"]').value=String(s);draw();}};
  }

  if(id === 'ppo-geometry') {
    let advantage=1,epsilon=.2,ratio=1.4;
    cell.innerHTML=range('advantage','Advantage Â',advantage,-3,3,.25,'Hold this sampled advantage fixed while changing the policy.')+range('clip','Clip width ε',epsilon,0,.5,.05,'Clipping limits the objective’s incentive in one direction.')+range('ratio','Probability ratio r',ratio,0,2.5,.05,'Old action probability = 0.4, so new probability = 0.4 × r.');
    query('[data-next]').textContent='Move ratio +0.05 →';
    query('[data-chart-title]').textContent='The objective contribution we maximise';
    query('[data-chart-title-secondary]').textContent='Its slope with respect to the probability ratio';
    legend(0,[['Raw rÂ',colors.amber,true],['PPO objective',colors.green],['Selected point',colors.violet]]);
    legend(1,[['PPO slope',colors.green],['Selected point',colors.violet]]);
    setExplanation(`<p><strong>Try:</strong> keep r = 1.4 and flip Â from +1 to −1. Then move r to 0.6 and repeat. Which change is favourable for each sample? The flat part swaps sides.</p><p><strong>Mechanism.</strong> Compare rÂ with clip(r, 1−ε, 1+ε)Â and choose the smaller. For Â &gt; 0 the objective stops increasing above 1+ε. For Â &lt; 0 it stops improving below 1−ε. An unfavourable move can remain sloped outside the interval.</p><p><strong>The derivative.</strong> This plot shows dL/dr for one fixed sample, not the full neural-network gradient. Multiplying by dr/dθ and averaging over a batch changes the parameter update. At the active corner the ordinary derivative is undefined; implementations choose a subgradient through their operations.</p><p><strong>Boundary check.</strong> Â = 0 removes this sample’s signal everywhere. ε = 0 still leaves a sloped penalty in the unfavourable direction. Ratio clipping does not constrain r itself and does not guarantee a KL bound.</p><p>Source: <a href="https://arxiv.org/abs/1707.06347" target="_blank" rel="noopener">Schulman et al., PPO, equation 7</a>. <strong>Notebook challenge:</strong> calculate the two branches for a mixed-sign batch and compare your loss and gradients with the project implementation.</p>`);
    function draw(){
      const x=ppoPoint(ratio,advantage,epsilon),curve=Array.from({length:101},(_,i)=>ppoPoint(i/40,advantage,epsilon)),probability=.4*ratio;
      output('advantage',advantage.toFixed(2));output('clip',epsilon.toFixed(2));output('ratio',ratio.toFixed(2));
      set('[data-stats]',stat('Objective L',f(x.value),'one fixed sample')+stat('Slope dL / dr',x.kink?'corner':f(x.slope),x.kink?'ordinary derivative undefined':x.plateau?'flat at this ratio':'still supplies a ratio gradient')+stat('New action probability',f(probability),'old probability 0.4'));
      const clipLines={silent:true,symbol:['none','none'],label:{show:false},lineStyle:{color:'#a8b79c',type:'dashed'},data:[{xAxis:1-epsilon},{xAxis:1+epsilon}]};
      paint(0,{xAxis:{type:'value',min:0,max:2.5,name:'Probability ratio r',nameLocation:'middle',nameGap:30},yAxis:{type:'value',name:'Objective L (maximise)'},series:[
        line('Raw rÂ',curve.map(p=>[p.ratio,p.raw]),{lineStyle:{width:2,type:'dashed',color:colors.amber}}),
        line('PPO min of branches',curve.map(p=>[p.ratio,p.value]),{lineStyle:{width:3.5,color:colors.green},markLine:clipLines}),
        {name:'Selected ratio',type:'scatter',symbolSize:12,data:[[ratio,x.value]],itemStyle:{color:colors.violet}}
      ]},`At ratio ${f(ratio)}, advantage ${f(advantage)}, and clip width ${f(epsilon)}, raw objective is ${f(x.raw)}, clipped branch is ${f(x.capped)}, and PPO chooses ${f(x.value)}.`);
      const threshold=advantage>0?1+epsilon:1-epsilon,slopeSegments=advantage===0?[[[0,0],[2.5,0]]]:advantage>0?[[[0,advantage],[threshold,advantage]],[[threshold,0],[2.5,0]]]:[[[0,0],[threshold,0]],[[threshold,advantage],[2.5,advantage]]];
      paint(1,{xAxis:{type:'value',min:0,max:2.5,name:'Ratio r',nameLocation:'middle',nameGap:28},yAxis:{type:'value',min:-3,max:3,name:'dL / dr'},legend:{show:false},series:slopeSegments.map((points,i)=>line('PPO slope',points,{lineStyle:{width:3,color:colors.green},markLine:i===0?clipLines:undefined})).concat(x.kink?[]:[{name:'Selected slope',type:'scatter',symbolSize:12,data:[[ratio,x.slope]],itemStyle:{color:colors.violet}}])},
        x.kink?`The chosen ratio is the active clipping corner, so the ordinary derivative does not exist.`:`At the chosen ratio, dL/dr is ${f(x.slope)}.`);
      setCaption(0,`Dashed vertical lines mark [${f(1-epsilon)}, ${f(1+epsilon)}]. The violet point is your selected ratio; the raw line is dashed amber.`);
      setCaption(1,'The two slope segments have one-sided limits at the active corner. No derivative is assigned at that corner. The derivative is with respect to r, not model parameters.');
      set('[data-calculation]',`<p class="eyebrow">Do the branch comparison</p><p class="plot-equation">L = min(${f(ratio)} × ${f(advantage)}, ${f(x.clippedRatio)} × ${f(advantage)})<br>= min(${f(x.raw)}, ${f(x.capped)}) = <strong>${f(x.value)}</strong></p><p>${advantage===0?'This advantage is zero: both branches are zero at every ratio.':x.kink?'You are exactly at the active corner. Check values just to the left and right before discussing a derivative.':x.plateau?`This sample is on its flat branch. ${advantage>0?'Increasing':'Decreasing'} its action probability further provides no extra surrogate improvement.`:`This sample remains on the sloped branch. ${advantage>0?'Increasing':'Decreasing'} its action probability would improve its surrogate contribution locally.`} Clipping a branch and taking the minimum are two distinct operations.</p>`);
      const probes=[0,1-epsilon,1,1+epsilon,ratio,2.5].filter((r,i,a)=>a.indexOf(r)===i).sort((a,b)=>a-b).map(r=>ppoPoint(r,advantage,epsilon));
      set('[data-table]',table('Selected ratio, boundaries, and reference points',['r','rÂ','clip(r)Â','PPO L','dL/dr'],probes.map(p=>[f(p.ratio),f(p.raw),f(p.capped),f(p.value),p.kink?'undefined at corner':f(p.slope)])));
      csv='ratio,advantage,epsilon,raw_branch,clipped_branch,ppo_objective,slope\n'+curve.map(p=>[p.ratio,advantage,epsilon,p.raw,p.capped,p.value,p.slope===null?'undefined':p.slope].join(',')).join('\n');
      announceStep(`r = ${ratio.toFixed(2)}; new probability ${probability.toFixed(2)}`);query('[data-next]').disabled=ratio>=2.5;
    }
    drawCurrent=draw;
    const advance=()=>{ratio=Math.min(2.5,Math.round((ratio+.05)*100)/100);setInput('ratio',ratio);draw();};
    on('[data-next]','click',()=>{stop();advance();});
    on('[data-play]','click',()=>{if(ratio>=2.5){ratio=0;setInput('ratio',ratio);draw();}playback(advance,()=>ratio>=2.5,160);});
    on('[data-reset]','click',()=>{stop();advantage=1;epsilon=.2;ratio=1.4;setInput('advantage',1);setInput('clip',.2);setInput('ratio',1.4);draw();});
    settings((name,value)=>{if(name==='advantage')advantage=value;if(name==='clip')epsilon=value;if(name==='ratio')ratio=value;draw();});
  }

  if(id === 'importance-weights') {
    let behavior=.2,target=.8,cap=5,seed=7,log;
    const recompute=()=>{log=makeBanditLog({behavior,seed});};
    cell.innerHTML=range('behavior','Logging probability b(B)',behavior,0,1,.01,'Changing the logging policy generates a new 200-row log at the same seed.')+range('target','Target probability π(B)',target,0,1,.01,'Changing the target reuses exactly the same logged actions and rewards.')+range('cap','Weight cap c',cap,1,20,.5,'Only the clipped IPS curve uses this cap; clipping changes the estimator.')+range('logseed','Log seed',seed,1,30,1,'Try multiple seeds before comparing estimators.');
    query('[data-next]').textContent='Move target +0.02 →';
    query('[data-chart-title]').textContent='Evaluate many target policies from one log';
    query('[data-chart-title-secondary]').textContent='How much weight each logged row receives';
    legend(0,[['Simulator truth',colors.ink,true],['IPS',colors.green],['Self-normalised',colors.blue,true],['Clipped IPS',colors.amber],['Selected target',colors.violet]]);
    legend(1,[['Action A rows',colors.amber],['Action B rows',colors.green]]);
    setExplanation(`<p><strong>Try:</strong> set π(B) = b(B). All weights become 1 and ESS equals 200. Keep π(B) = 0.8, then decrease b(B). Rare B rows now carry more of the estimate. Compare several seeds.</p><p><strong>Environment.</strong> A and B pay binary rewards with means 0.25 and 0.85. The logging policy independently picks B with probability b(B). Reward distributions are identical under logging and target policies; propensities are known. The simulator can show true target value 0.25(1−π(B)) + 0.85π(B); a real log usually cannot.</p><p><strong>Three estimates.</strong> IPS = Σwᵢrᵢ/n. Self-normalised IPS = Σwᵢrᵢ/Σwᵢ. Clipped IPS = Σmin(wᵢ,c)rᵢ/n. Normalisation and clipping can trade variance for bias. The cap is not a replacement for support.</p><p><strong>ESS is a diagnostic.</strong> (Σw)²/Σw² measures weight concentration. It is not a confidence interval, does not prove accuracy, and ignores which rows have informative rewards.</p><p><strong>Boundary check.</strong> If b(B) = 0 but π(B) &gt; 0, support fails: no amount of reweighting identifies B’s value. If b(B) is positive but the finite log happens to contain no B rows, population support holds but sample coverage is poor; these are different failures. If all observed weights are zero, self-normalisation is undefined.</p><p>Source: <a href="https://proceedings.mlr.press/v37/swaminathan15.html" target="_blank" rel="noopener">Swaminathan and Joachims, counterfactual risk minimisation</a>. <strong>Notebook challenge:</strong> repeat the experiment over 100 logs, then compare estimator bias and variance against the known simulator value.</p>`);
    function draw() {
      const x=importanceEstimate(log,{behavior,target,cap}),curve=Array.from({length:51},(_,i)=>({p:i/50,...importanceEstimate(log,{behavior,target:i/50,cap})})),largest=Math.max(...x.weights);
      output('behavior',behavior.toFixed(2));output('target',target.toFixed(2));output('cap',cap.toFixed(1));output('logseed',seed);
      set('[data-stats]',stat('True target value',f(x.truth),'known only in this simulator')+stat('IPS estimate',x.supported?f(x.ips):'unsupported',x.supported?'weighted reward / 200':'target action has zero logging probability')+stat('Weight ESS',`${f(x.ess,1)} / 200`,'concentration diagnostic'));
      paint(0,{xAxis:{type:'value',min:0,max:1,name:'Target probability π(B)',nameLocation:'middle',nameGap:28},yAxis:{type:'value',name:'Mean reward / estimate',min:0},
        series:[line('Simulator truth',curve.map(p=>[p.p,p.truth]),{lineStyle:{color:colors.ink,type:'dashed',width:2}}),line('IPS',curve.map(p=>[p.p,p.ips]),{lineStyle:{color:colors.green,width:3}}),line('Self-normalised',curve.map(p=>[p.p,p.snips]),{lineStyle:{color:colors.blue,width:2,type:'dotted'}}),line(`Clipped, c=${f(cap)}`,curve.map(p=>[p.p,p.clipped]),{lineStyle:{color:colors.amber,width:2}}),{name:'Selected target',type:'scatter',data:[[target,x.truth]],symbolSize:10,itemStyle:{color:colors.violet},markLine:{silent:true,symbol:['none','none'],label:{show:false},lineStyle:{color:colors.violet,type:'dashed'},data:[{xAxis:target}]}}]},
        `At target B probability ${f(target)}, true value is ${f(x.truth)}; IPS ${x.supported?f(x.ips):'unsupported'}, self-normalised IPS ${x.supported?f(x.snips):'unsupported'}, clipped IPS ${x.supported?f(x.clipped):'unsupported'}.`);
      paint(1,{legend:{show:false},grid:{top:30,left:45,bottom:46,right:16,containLabel:true},xAxis:{type:'category',data:log.map(r=>r.i),name:'Logged row (chronological)',nameLocation:'middle',nameGap:28,axisLabel:{interval:49,fontSize:10}},yAxis:{type:'value',min:0,name:'Importance weight',max:Math.max(1,largest*1.1)},series:[{name:'Weight',type:'bar',barCategoryGap:'0%',data:log.map((row,i)=>({value:x.weights[i],itemStyle:{color:row.action===1?colors.green:colors.amber}}))}]},
        `Largest observed weight is ${f(largest)}. A rows are amber and B rows green; the aggregate counts, rewards, and weights are in the table. ESS is ${f(x.ess,1)}.`);
      setCaption(0,'Every point reuses the same 200-row log. The vertical line selects your target. Unsupported estimates are omitted. Unnormalised IPS can exceed 1; a single log’s closeness to truth does not rank estimators.');
      setCaption(1,`A rows: ${x.summary[0].count} (amber). B rows: ${x.summary[1].count} (green). Largest weight ${f(largest)}. A weight can be zero when the target never chooses that action.`);
      set('[data-calculation]',!x.supported?`<p class="eyebrow">Support failure</p><p><strong>The target gives positive probability to an action the logger never chooses.</strong> We withhold the estimates for this target. The known simulator truth is ${f(x.truth)}, but observed rewards alone cannot identify it. Try moving the target back onto the logger’s supported action.</p>`:`<p class="eyebrow">Follow the weight into the estimate</p><p class="plot-equation">w(A) = ${f(1-target)} / ${f(1-behavior)} = ${f(x.summary[0].weight)}<br>w(B) = ${f(target)} / ${f(behavior)} = ${f(x.summary[1].weight)}</p><p>Weighted reward sum = ${f(x.weightedReward)}; total weight = ${f(x.sum)}.<br><strong>IPS = ${f(x.weightedReward)} / 200 = ${f(x.ips)}</strong>.<br>Self-normalised IPS = ${f(x.snips)}; clipped IPS (cap ${f(cap)}) = ${f(x.clipped)}.</p>${x.missingObserved.length?`<p class="plot-warning"><strong>Finite-log coverage gap:</strong> target action ${x.missingObserved.map(a=>'AB'[a]).join(', ')} has no observed rows. This can occur even with positive logging probability. ${x.sum===0?'Every observed weight is zero, so self-normalised IPS is undefined.':''}</p>`:''}<p>ESS = (Σw)² / Σw² = <strong>${f(x.ess,2)}</strong>. This is a weight diagnostic, not an accuracy guarantee.</p>`);
      set('[data-table]',table('Every logged row is represented in these action totals',['Action','Log b','Target π','Rows','Reward sum','Weight'],x.summary.map(s=>['AB'[s.action],f(s.behavior),f(s.target),s.count,s.successes,s.weight===null?'not defined (b = 0)':f(s.weight)]))+table('Selected policy estimates',['Quantity','Value'],[['Simulator truth',f(x.truth)],['Unweighted logged reward',f(x.observed)],['IPS',x.supported?f(x.ips):'unsupported'],['Self-normalised IPS',x.supported?f(x.snips):'unsupported'],['Clipped IPS',x.supported?f(x.clipped):'unsupported'],['Weight ESS',f(x.ess,2)]]));
      csv='row,action,reward,logging_probability,target_probability,weight,clipped_weight\n'+log.map((r,i)=>[r.i,'AB'[r.action],r.reward,r.action?behavior:1-behavior,r.action?target:1-target,x.weights[i],Math.min(cap,x.weights[i])].join(',')).join('\n');
      announceStep(`Same ${log.length}-row log; target B = ${target.toFixed(2)}`);query('[data-next]').disabled=target>=1;
    }
    recompute();drawCurrent=draw;
    const advance=()=>{target=Math.min(1,Math.round((target+.02)*100)/100);setInput('target',target);draw();};
    on('[data-next]','click',()=>{stop();advance();});
    on('[data-play]','click',()=>{if(target>=1){target=0;setInput('target',target);draw();}playback(advance,()=>target>=1,160);});
    on('[data-reset]','click',()=>{stop();behavior=.2;target=.8;cap=5;seed=7;setInput('behavior',behavior);setInput('target',target);setInput('cap',cap);setInput('logseed',seed);recompute();draw();});
    settings((name,value)=>{if(name==='behavior')behavior=value;if(name==='target')target=value;if(name==='cap')cap=value;if(name==='logseed')seed=value;if(name==='behavior'||name==='logseed')recompute();draw();});
  }

  drawCurrent();
  const resize = new ResizeObserver(()=>charts.forEach(chart=>chart.resize()));
  resize.observe(host);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();},{signal});
  reduced.addEventListener('change',()=>{stop();drawCurrent();},{signal});
  on('[data-csv]','click',()=>{
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');
    a.href=url;a.download=`study-rl-${id}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  loadCharts().then(library=>{
    if(destroyed)return;
    charts=[...host.querySelectorAll('[data-chart]')].map(element=>library.init(element,null,{renderer:'svg'}));
    if(id==='gridworld')charts[0].on('click',event=>{if(Number.isInteger(event.data?.state))host._plotCellSelect(event.data.state);});
    drawCurrent();
  }).catch(()=>{
    if(destroyed)return;
    host.querySelectorAll('[data-chart]').forEach(element=>{element.innerHTML='<p class="plot-chart-error">The chart library could not load. The controls, calculation, data table and download still work.</p>';});
  });
  return ()=>{destroyed=true;stop();abort.abort();resize.disconnect();charts.forEach(chart=>chart.dispose());delete host._plotCellSelect;};
}
