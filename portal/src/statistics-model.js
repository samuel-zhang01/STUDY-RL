// Transparent teaching simulations. Random units and interval assumptions are explicit.
import {seededRandom} from './plot-model.js';
export const mean=xs=>xs.reduce((a,b)=>a+b,0)/xs.length;
export const variance=xs=>xs.length>1?xs.reduce((s,x)=>s+(x-mean(xs))**2,0)/(xs.length-1):NaN;
export function normalGenerator(seed){const random=seededRandom(seed);return()=>Math.sqrt(-2*Math.log(Math.max(Number.EPSILON,random())))*Math.cos(2*Math.PI*random());}
// Lanczos log-gamma and the continued fraction for the regularised incomplete beta.
function logGamma(z){const p=[676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,12.507343278686905,-.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];if(z<.5)return Math.log(Math.PI)-Math.log(Math.sin(Math.PI*z))-logGamma(1-z);z--;let x=.99999999999980993;for(let i=0;i<p.length;i++)x+=p[i]/(z+i+1);const t=z+p.length-.5;return .5*Math.log(2*Math.PI)+(z+.5)*Math.log(t)-t+Math.log(x);}
function betaFraction(a,b,x){const tiny=1e-30;let c=1,d=1-(a+b)*x/(a+1);if(Math.abs(d)<tiny)d=tiny;d=1/d;let h=d;for(let m=1;m<=200;m++){const m2=2*m;let aa=m*(b-m)*x/((a+m2-1)*(a+m2));d=1+aa*d;if(Math.abs(d)<tiny)d=tiny;c=1+aa/c;if(Math.abs(c)<tiny)c=tiny;d=1/d;h*=d*c;aa=-(a+m)*(a+b+m)*x/((a+m2)*(a+m2+1));d=1+aa*d;if(Math.abs(d)<tiny)d=tiny;c=1+aa/c;if(Math.abs(c)<tiny)c=tiny;d=1/d;const delta=d*c;h*=delta;if(Math.abs(delta-1)<3e-14)break;}return h;}
function beta(x,a,b){if(x<=0)return 0;if(x>=1)return 1;const factor=Math.exp(logGamma(a+b)-logGamma(a)-logGamma(b)+a*Math.log(x)+b*Math.log1p(-x));return x<(a+1)/(a+b+2)?factor*betaFraction(a,b,x)/a:1-factor*betaFraction(b,a,1-x)/b;}
export function studentCDF(t,df){if(!(df>0))throw new Error('Positive degrees of freedom required');const tail=.5*beta(df/(df+t*t),df/2,.5);return t>=0?1-tail:tail;}
export function studentCritical(df,probability=.975){if(!(probability>.5&&probability<1&&df>0))throw new Error('Invalid t quantile');let lo=0,hi=1;while(studentCDF(hi,df)<probability)hi*=2;for(let i=0;i<65;i++){const mid=(lo+hi)/2;if(studentCDF(mid,df)<probability)lo=mid;else hi=mid;}return (lo+hi)/2;}
export function tInterval(values){const estimate=mean(values),sd=Math.sqrt(variance(values)),se=sd/Math.sqrt(values.length),critical=studentCritical(values.length-1);return {mean:estimate,sd,se,critical,low:estimate-critical*se,high:estimate+critical*se,n:values.length};}
export function samplingExperiment({n=20,sigma=10,seed=7,repetitions=400}={}){
 if(!Number.isInteger(n)||n<2||!Number.isFinite(sigma)||sigma<=0||!Number.isInteger(repetitions)||repetitions<2)throw new Error('Invalid sampling parameters');
 const random=normalGenerator(seed),truth=50,critical=studentCritical(n-1);
 const trials=Array.from({length:repetitions},(_,i)=>{const values=Array.from({length:n},()=>truth+sigma*random()),estimate=mean(values),sd=Math.sqrt(variance(values)),se=sd/Math.sqrt(n);return {trial:i+1,mean:estimate,sd,se,low:estimate-critical*se,high:estimate+critical*se,values};});
 return {truth,n,sigma,repetitions,critical,trials,theoreticalSE:sigma/Math.sqrt(n),empiricalSE:Math.sqrt(variance(trials.map(t=>t.mean))),coverage:mean(trials.map(t=>Number(t.low<=truth&&truth<=t.high)))};
}
export function pairedExperiment({n=20,sharedSD=15,noiseSD=4,effect=3,seed=7}={}){
 if(!Number.isInteger(n)||n<3||sharedSD<0||noiseSD<=0)throw new Error('Invalid paired parameters');
 const random=normalGenerator(seed),rows=Array.from({length:n},(_,i)=>{const common=sharedSD*random(),a=50+common+noiseSD*random(),b=50+effect+common+noiseSD*random();return {seed:i+1,a,b,difference:b-a};});
 const a=rows.map(r=>r.a),b=rows.map(r=>r.b),paired=tInterval(rows.map(r=>r.difference));
 const va=variance(a),vb=variance(b),se=Math.sqrt((va+vb)/n),df=(va+vb)**2/((va**2+vb**2)/(n-1)),critical=studentCritical(df);
 return {rows,effect,paired,unpaired:{mean:mean(b)-mean(a),se,df,critical,low:paired.mean-critical*se,high:paired.mean+critical*se},covariance:rows.reduce((s,r)=>s+(r.a-mean(a))*(r.b-mean(b)),0)/(n-1)};
}
export function quantile(xs,q){if(!xs.length)throw new Error('Empty quantile');const sorted=[...xs].sort((a,b)=>a-b),p=(sorted.length-1)*q,i=Math.floor(p);return sorted[i]+(sorted[Math.ceil(p)]-sorted[i])*(p-i);}
export function clusteredBootstrap({clusters=12,rowsPerCluster=8,rho=.7,seed=7,repetitions=600}={}){
 if(!Number.isInteger(clusters)||clusters<3||!Number.isInteger(rowsPerCluster)||rowsPerCluster<1||rho<0||rho>1)throw new Error('Invalid cluster parameters');
 const normal=normalGenerator(seed),random=seededRandom(seed+12345),truth=50,sigma=10;
 const groups=Array.from({length:clusters},(_,i)=>{const shared=normal();return Array.from({length:rowsPerCluster},(_,j)=>({cluster:i+1,row:j+1,value:truth+sigma*(Math.sqrt(rho)*shared+Math.sqrt(1-rho)*normal())}));});
 const rows=groups.flat(),values=rows.map(r=>r.value),clusterMeans=groups.map(g=>mean(g.map(r=>r.value))),bootRows=[],bootClusters=[];
 for(let b=0;b<repetitions;b++){
  bootRows.push(mean(Array.from({length:values.length},()=>values[Math.floor(random()*values.length)])));
  bootClusters.push(mean(Array.from({length:clusters},()=>clusterMeans[Math.floor(random()*clusters)])));
 }
 const interval=xs=>({low:quantile(xs,.025),high:quantile(xs,.975),se:Math.sqrt(variance(xs))});
 return {truth,rows,groups,clusterMeans,estimate:mean(values),naive:interval(bootRows),cluster:interval(bootClusters),bootRows,bootClusters,designEffect:1+(rowsPerCluster-1)*rho,theoreticalSE:sigma*Math.sqrt((1+(rowsPerCluster-1)*rho)/values.length),repetitions};
}
