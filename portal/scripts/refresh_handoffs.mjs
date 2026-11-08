// Repair source-line anchors by their authored Python symbol after course edits.
// This is an explicit maintenance command; builds never rewrite authored sources.
import fs from 'node:fs';
import path from 'node:path';
import {ROOT} from './fingerprints.mjs';
import {QUESTIONS} from '../src/review-content.js';
const filename=path.join(ROOT,'portal/src/review-content.js'),source=fs.readFileSync(filename,'utf8');
const match=source.match(/const HANDOFF_TARGETS = (\{[\s\S]*?\n\});/);
if(!match)throw new Error('Cannot locate the authored handoff bank');
const targets=JSON.parse(match[1]);let changed=0;
for(const question of QUESTIONS){
 const target=targets[question.id];if(!target?.symbol)continue;
 const expression=new RegExp(`^\\s*(?:def|class)\\s+${target.symbol}\\b`);
 const lines=fs.readFileSync(path.join(ROOT,question.handoff.path),'utf8').split('\n');
 if(expression.test(lines[target.line-1]||''))continue;
 const matches=lines.flatMap((line,i)=>expression.test(line)?[i+1]:[]);
 if(matches.length!==1)throw new Error('Source symbol must resolve uniquely: '+question.id+' '+target.symbol);
 if(target.line!==matches[0]){target.line=matches[0];changed++;}
}
fs.writeFileSync(filename,source.replace(match[0],'const HANDOFF_TARGETS = '+JSON.stringify(targets,null,2)+';'));
process.stdout.write(`Refreshed ${changed} diagnostic source-line anchors; existing anchors verified; shifted symbols resolved uniquely.\n`);
