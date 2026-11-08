import test from 'node:test';
import assert from 'node:assert/strict';
import {mean,variance,studentCDF,studentCritical,tInterval,samplingExperiment,pairedExperiment,clusteredBootstrap,quantile} from '../src/statistics-model.js';
const near=(a,b,tolerance=1e-8)=>assert.ok(Math.abs(a-b)<tolerance,`${a} != ${b}`);
test('t intervals match tabulated quantiles and a hand-worked example',()=>{
 near(studentCDF(0,9),.5);near(studentCritical(1),12.706204736,1e-8);near(studentCritical(9),2.262157163,1e-8);near(studentCritical(29),2.045229642,1e-8);
 const result=tInterval([1,2,3,4,5]);near(result.mean,3);near(result.sd,Math.sqrt(2.5));near(result.se,Math.sqrt(.5));near(result.high,4.963243161,1e-8);
});
test('sampling distinguishes observation spread from uncertainty of a mean',()=>{
 const x=samplingExperiment({n:25,sigma:10,repetitions:2000});near(x.theoreticalSE,2);assert.ok(Math.abs(x.empiricalSE-2)<.12);assert.ok(x.coverage>.93&&x.coverage<.97);assert.deepEqual(samplingExperiment({n:5,repetitions:3}),samplingExperiment({n:5,repetitions:3}));
 const y=samplingExperiment({n:100,sigma:10,repetitions:2000});near(y.theoreticalSE,1);assert.ok(Math.abs(y.empiricalSE-1)<.06);
});
test('pairing subtracts common variation and retains the difference estimand',()=>{
 const paired=pairedExperiment({n:30,sharedSD:25,noiseSD:2});near(paired.paired.mean,paired.unpaired.mean);assert.ok(paired.paired.se<paired.unpaired.se/5);near(paired.paired.mean,mean(paired.rows.map(r=>r.b-r.a)));near(paired.paired.se,Math.sqrt(variance(paired.rows.map(r=>r.difference))/30));
});
test('cluster resampling preserves dependence while row resampling can hide it',()=>{
 const x=clusteredBootstrap({clusters:20,rowsPerCluster:10,rho:1,repetitions:1500});near(x.designEffect,10);assert.ok(x.groups.every(g=>g.every(r=>r.value===g[0].value)));assert.ok(x.cluster.se>2.5*x.naive.se);near(x.theoreticalSE,10/Math.sqrt(20));near(quantile([1,2,3,4],.5),2.5);
 const y=clusteredBootstrap({clusters:20,rowsPerCluster:1,rho:0,repetitions:1500});near(y.designEffect,1);assert.ok(Math.abs(y.cluster.se/y.naive.se-1)<.12);
});
