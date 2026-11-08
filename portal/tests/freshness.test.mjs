import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
test('tests assess an artifact matching current content, interface and execution outputs',()=>{
 const result=execFileSync(process.execPath,['scripts/fingerprints.mjs','--verify'],{cwd:new URL('..',import.meta.url),encoding:'utf8'});
 assert.match(result,/Publication fingerprints verified/);
});
