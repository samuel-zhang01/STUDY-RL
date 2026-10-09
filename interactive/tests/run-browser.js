'use strict';
const http = require('node:http');
const {spawn} = require('node:child_process');
const {resolve} = require('node:path');
const {chromium,firefox,webkit} = require('playwright');
const scenarios = require('./browser-scenarios.js');
const root=resolve(__dirname,'../..');
const port=Number(process.env.RL_LAB_TEST_PORT || 8877);
const baseURL='http://127.0.0.1:'+port;
const server=spawn(process.env.PY || 'python3',['-m','http.server',String(port),'--bind','127.0.0.1'],{cwd:root,stdio:['ignore','ignore','pipe']});
const stopServer=()=>{server.kill();};
process.once('SIGINT',()=>{stopServer();process.exit(130);});
process.once('SIGTERM',()=>{stopServer();process.exit(143);});
let serverError='';server.stderr.on('data',data=>{serverError+=data;});
server.on('error',error=>{serverError+=error.message;});
// A crashed browser must fail the run instead of hanging indefinitely.
const watchdog=setTimeout(()=>{console.error('Browser run exceeded five minutes');server.kill();process.exit(1);},300000);
const available=()=>new Promise(resolveReady=>{const request=http.get(baseURL+'/interactive/',response=>{response.resume();resolveReady(response.statusCode===200);});request.on('error',()=>resolveReady(false));});
(async()=>{
  let ready=false;for(let attempt=0;attempt<50;attempt++){if(server.exitCode!==null)throw Error('Test server failed: '+serverError);if(await available()){ready=true;break;}await new Promise(r=>setTimeout(r,100));}
  if(!ready)throw Error('Test server did not become available');
  const requested=(process.env.RL_LAB_BROWSERS || 'chromium,firefox,webkit').split(',');
  for(const name of requested){const type={chromium,firefox,webkit}[name];if(!type)throw Error('Unknown browser '+name);console.log(name+': starting');const browser=await type.launch({timeout:15000});try{const context=await browser.newContext({viewport:{width:1280,height:900},acceptDownloads:true});const page=await context.newPage();const report=await scenarios(page,baseURL);console.log(name+': '+report.checks+' browser assertions passed');await context.close();}finally{await browser.close();}}
})().catch(error=>{console.error(error);process.exitCode=1;}).finally(()=>{clearTimeout(watchdog);server.kill();});
