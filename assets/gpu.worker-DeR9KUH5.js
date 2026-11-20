(function(){var e=class extends Error{reason;constructor(e,t){super(e),this.reason=t,this.name=`GpuFallbackError`}};function t(t,n){for(let r of t){let t=r.bound?Math.min(n.maxStorageBufferBindingSize,n.maxBufferSize):n.maxBufferSize;if(r.bytes>t)throw new e(`${r.label}: ${r.bytes} bytes over the device limit of ${t} bytes`,`limits`)}}async function n(t,n){t.pushErrorScope(`validation`),t.pushErrorScope(`out-of-memory`);let r,i=null,a=!1;try{r=await n()}catch(e){a=!0,i=e}let o=await t.popErrorScope().catch(()=>null),s=await t.popErrorScope().catch(()=>null);if(o)throw new e(`GPU out of memory: ${o.message}`,`out-of-memory`);if(s)throw new e(`GPU validation error: ${s.message}`,`validation`);if(a)throw i;return r}function r(e,t){let n=Math.fround,r=e.length/3,i=n(t),a=n(i*i),o=i*1.02,s=(e,t,n)=>`${e},${t},${n}`,c=/* @__PURE__ */ new Map;for(let t=0;t<r;t++){let n=s(Math.floor(e[t*3]/o),Math.floor(e[t*3+1]/o),Math.floor(e[t*3+2]/o)),r=c.get(n);r?r.push(t):c.set(n,[t])}let l=new Uint32Array(r);for(let t=0;t<r;t++){let r=e[t*3],i=e[t*3+1],u=e[t*3+2],d=Math.floor(r/o),f=Math.floor(i/o),p=Math.floor(u/o),m=0;for(let t=-1;t<=1;t++)for(let o=-1;o<=1;o++)for(let l=-1;l<=1;l++){let h=c.get(s(d+l,f+o,p+t));if(h)for(let t of h){let o=n(r-e[t*3]),s=n(i-e[t*3+1]),c=n(u-e[t*3+2]);n(n(n(o*o)+n(s*s))+n(c*c))<a&&m++}}l[t]=m}return l}let i=Math.fround,a=34028234663852886e22;var o=class{dims;vind;rootLow=/* @__PURE__ */ new Float32Array(3);rootHigh=/* @__PURE__ */ new Float32Array(3);child1;lo;hi;divfeat;divlow;divhigh;box;nodes=0;partitioned=0;constructor(e,t=!1){let n=e.length/3,r=new Float32Array(n),i=new Float32Array(n),a=new Float32Array(n);for(let o=0;o<n;o++)r[o]=e[o*3],i[o]=e[o*3+1],a[o]=t?0:e[o*3+2];this.dims=[r,i,a],this.vind=new Int32Array(n);for(let e=0;e<n;e++)this.vind[e]=e;let o=1024;if(this.child1=new Int32Array(o),this.lo=new Int32Array(o),this.hi=new Int32Array(o),this.divfeat=new Uint8Array(o),this.divlow=new Float32Array(o),this.divhigh=new Float32Array(o),this.box=new Float32Array(o*6),n===0)return;for(let e=0;e<3;e++){let t=this.dims[e],r=t[0],i=r;for(let e=1;e<n;e++){let n=t[e];n<r&&(r=n),n>i&&(i=n)}this.rootLow[e]=r,this.rootHigh[e]=i}let s=this.alloc(0,n);for(let e=0;e<3;e++)this.box[s*6+2*e]=this.rootLow[e],this.box[s*6+2*e+1]=this.rootHigh[e]}get size(){return this.vind.length}alloc(e,t){if(this.nodes===this.child1.length){let e=e=>{let t=new e.constructor(e.length*2);return t.set(e),t};this.child1=e(this.child1),this.lo=e(this.lo),this.hi=e(this.hi),this.divfeat=e(this.divfeat),this.divlow=e(this.divlow),this.divhigh=e(this.divhigh),this.box=e(this.box)}let n=this.nodes++;return this.child1[n]=-2,this.lo[n]=e,this.hi[n]=t,n}minMax(e,t,n){let r=this.vind,i=n[r[e]],a=i;for(let o=e+1;o<t;o++){let e=n[r[o]];e<i&&(i=e),e>a&&(a=e)}return[i,a]}planeSplit(e,t,n,r){let i=this.vind,a=0,o=t-1;for(;;){for(;a<=o&&n[i[e+a]]<r;)++a;for(;a<=o&&n[i[e+o]]>=r;)--o;if(a>o)break;let t=i[e+a];i[e+a]=i[e+o],i[e+o]=t,++a,--o}let s=a;for(o=t-1;;){for(;a<=o&&n[i[e+a]]<=r;)++a;for(;a<=o&&n[i[e+o]]>r;)--o;if(a>o)break;let t=i[e+a];i[e+a]=i[e+o],i[e+o]=t,++a,--o}return[s,a]}build(e){let t=this.lo[e],n=this.hi[e],r=n-t;if(r<=15){this.child1[e]=-1;return}this.partitioned+=r;let a=e*6,o=this.box,s=i(o[a+1]-o[a]),c=0;for(let e=1;e<3;e++){let t=i(o[a+2*e+1]-o[a+2*e]);t>s&&(s=t,c=e)}let[l,u]=this.minMax(t,n,this.dims[c]),d=i(i(l+u)/2);s=i(u-l);let f=c;for(let e=0;e<3;e++){if(e===f)continue;let r=i(o[a+2*e+1]-o[a+2*e]);r>s&&([l,u]=this.minMax(t,n,this.dims[e]),r=i(u-l),r>s&&(s=r,c=e,d=i(i(l+u)/2)))}let p=this.dims[c],[m,h]=this.planeSplit(t,r,p,d),g=Math.trunc(r/2),_=m>g?m:h<g?h:g,v=this.alloc(t,t+_),y=this.alloc(t+_,n),b=this.box;for(let e=0;e<6;e++)b[v*6+e]=b[a+e],b[y*6+e]=b[a+e];b[v*6+2*c+1]=d,b[y*6+2*c]=d,this.child1[e]=v,this.divfeat[e]=c,this.divlow[e]=this.minMax(t,t+_,p)[1],this.divhigh[e]=this.minMax(t+_,n,p)[0]}nearest(e,t,n){if(this.vind.length===0)return{index:-1,d2:1/0};let[r,o,s]=this.dims,c=this.vind,l=[i(e),i(t),i(n)],u=[0,0,0],d=0;for(let e=0;e<3;e++){if(l[e]<this.rootLow[e]){let t=i(l[e]-this.rootLow[e]);u[e]=i(t*t),d=i(d+u[e])}if(l[e]>this.rootHigh[e]){let t=i(l[e]-this.rootHigh[e]);u[e]=i(t*t),d=i(d+u[e])}}let f=a,p=-1,m=(e,t)=>{this.child1[e]===-2&&this.build(e);let n=this.child1[e];if(n===-1){let t=f;for(let n=this.lo[e];n<this.hi[e];n++){let e=c[n],a=i(l[0]-r[e]),u=i(l[1]-o[e]),d=i(l[2]-s[e]),m=i(i(i(a*a)+i(u*u))+i(d*d));m<t&&m<f&&(f=m,p=e)}return}let a=this.divfeat[e],d=l[a],h=this.divlow[e],g=this.divhigh[e],_=i(d-h),v=i(d-g),y,b,x;if(i(_+v)<0){y=n,b=n+1;let e=i(d-g);x=i(e*e)}else{y=n+1,b=n;let e=i(d-h);x=i(e*e)}m(y,t);let S=u[a],C=i(i(t+x)-S);u[a]=x,C<=f&&m(b,C),u[a]=S};return m(0,d),{index:p,d2:f}}knn(e,t,n,r){if(this.vind.length===0||r<=0)return/* @__PURE__ */ new Int32Array;let[o,s,c]=this.dims,l=this.vind,u=[i(e),i(t),i(n)],d=[0,0,0],f=0;for(let e=0;e<3;e++){if(u[e]<this.rootLow[e]){let t=i(u[e]-this.rootLow[e]);d[e]=i(t*t),f=i(f+d[e])}if(u[e]>this.rootHigh[e]){let t=i(u[e]-this.rootHigh[e]);d[e]=i(t*t),f=i(f+d[e])}}let p=new Float64Array(r),m=new Int32Array(r),h=0,g=a,_=(e,t)=>{if(e>=g)return;h<r&&h++;let n=h-1;for(;n>0&&p[n-1]>e;--n)p[n]=p[n-1],m[n]=m[n-1];p[n]=e,m[n]=t,h===r&&(g=p[r-1])},v=(e,t)=>{this.child1[e]===-2&&this.build(e);let n=this.child1[e];if(n===-1){let t=g;for(let n=this.lo[e];n<this.hi[e];n++){let e=l[n],r=i(u[0]-o[e]),a=i(u[1]-s[e]),d=i(u[2]-c[e]),f=i(i(i(r*r)+i(a*a))+i(d*d));f<t&&_(f,e)}return}let r=this.divfeat[e],a=u[r],f=i(a-this.divlow[e]),p=i(a-this.divhigh[e]),m,h,y;i(f+p)<0?(m=n,h=n+1,y=i(p*p)):(m=n+1,h=n,y=i(f*f)),v(m,t);let b=d[r],x=i(i(t+y)-b);d[r]=y,x<=g&&v(h,x),d[r]=b};return v(0,f),m.slice(0,h)}};let s=Math.fround,c=1.1920928955078125e-7,l=s(1/3),u=s(Math.sqrt(3)),d=s(1e-5);function f(e){let[t,n,r,,i,a,,,o]=[e[0],e[1],e[2],e[3],e[4],e[5],e[6],e[7],e[8]],d=s(s(s(s(s(s(t*i)*o)+s(s(s(2*n)*r)*a))-s(s(t*a)*a))-s(s(i*r)*r))-s(s(o*n)*n)),f=s(s(s(s(s(s(t*i)-s(n*n))+s(t*o))-s(r*r))+s(i*o))-s(a*a)),p=s(s(t+i)+o),m=()=>{let e=s(s(p*p)-4*f);e<0&&(e=0);let t=s(Math.sqrt(e));return[0,s(.5*s(p-t)),s(.5*s(p+t))]};if(Math.abs(d)<c)return m();let h=s(p*l),g=s(s(f-s(p*h))*l);g>0&&(g=0);let _=s(.5*s(d+s(h*s(s(s(2*h)*h)-f)))),v=s(s(_*_)+s(s(g*g)*g));v>0&&(v=0);let y=s(Math.sqrt(-g)),b=s(s(Math.atan2(s(Math.sqrt(-v)),_))*l),x=s(Math.cos(b)),S=s(Math.sin(b)),C=[s(h+s(s(2*y)*x)),s(h-s(y*s(x+s(u*S)))),s(h-s(y*s(x-s(u*S))))],w=(e,t)=>{let n=C[e];C[e]=C[t],C[t]=n};return C[0]>=C[1]&&w(0,1),C[1]>=C[2]&&(w(1,2),C[0]>=C[1]&&w(0,1)),C[0]<=0?m():C}function p(e){let t=t=>[e[t*3],e[t*3+1],e[t*3+2]],n=(e,t)=>[s(s(e[1]*t[2])-s(e[2]*t[1])),s(s(e[2]*t[0])-s(e[0]*t[2])),s(s(e[0]*t[1])-s(e[1]*t[0]))],r=t(0),i=t(1),a=t(2),o=[n(r,i),n(r,a),n(i,a)],c=0,l=-1;for(let e=0;e<3;e++){let t=o[e],n=s(Math.sqrt(s(s(s(t[0]*t[0])+s(t[1]*t[1]))+s(t[2]*t[2]))));n>l&&(l=n,c=e)}let u=o[c];return[s(u[0]/l),s(u[1]/l),s(u[2]/l)]}function m(e){let t=(e,t)=>Math.abs(e)<=s(Math.abs(t)*d);if(!t(e[0],e[2])||!t(e[1],e[2])){let t=s(1/s(Math.sqrt(s(s(e[0]*e[0])+s(e[1]*e[1])))));return[s(-e[1]*t),s(e[0]*t),0]}let n=s(1/s(Math.sqrt(s(s(e[1]*e[1])+s(e[2]*e[2])))));return[0,s(-e[2]*n),s(e[1]*n)]}function h(e){let t=0;for(let n=0;n<9;n++)t=Math.max(t,Math.abs(e[n]));t<=11754943508222875e-54&&(t=1);let n=Float32Array.from(e,e=>e/t),r=f(n);if(s(r[1]-r[0])>c){for(let e=0;e<3;e++)n[e*4]=s(n[e*4]-r[0]);return p(n)}if(s(r[2]-r[0])>c){for(let e=0;e<3;e++)n[e*4]=s(n[e*4]-r[2]);return m(p(n))}return[1,0,0]}function g(e,t,n){let r=t.length;if(r<3)return[NaN,NaN,NaN];let i=e[t[0]*3],a=e[t[0]*3+1],o=e[t[0]*3+2],c=/* @__PURE__ */ new Float32Array(9);for(let n=0;n<r;n++){let r=t[n]*3,l=s(e[r]-i),u=s(e[r+1]-a),d=s(e[r+2]-o);c[0]+=s(l*l),c[1]+=s(l*u),c[2]+=s(l*d),c[3]+=s(u*u),c[4]+=s(u*d),c[5]+=s(d*d),c[6]+=l,c[7]+=u,c[8]+=d}let l=s(r);for(let e=0;e<9;e++)c[e]=c[e]/l;let u=/* @__PURE__ */ new Float32Array(9);u[0]=s(c[0]-s(c[6]*c[6])),u[1]=s(c[1]-s(c[6]*c[7])),u[2]=s(c[2]-s(c[6]*c[8])),u[4]=s(c[3]-s(c[7]*c[7])),u[5]=s(c[4]-s(c[7]*c[8])),u[8]=s(c[5]-s(c[8]*c[8])),u[3]=u[1],u[6]=u[2],u[7]=u[5];let[d,f,p]=h(u),m=e[n*3],g=e[n*3+1],_=e[n*3+2],v=s(0-m),y=s(0-g),b=s(0-_);return s(s(s(v*d)+s(y*f))+s(b*p))<0&&(d=-d,f=-f,p=-p),[d,f,p]}function _(e,t){let n=e.length/3,r=new o(e),i=new Float32Array(n*3);for(let a=0;a<n;a++)i.set(g(e,r.knn(e[a*3],e[a*3+1],e[a*3+2],t),a),a*3);return i}function v(e,t,n,r){let i=Number.isNaN(e[t*3]),a=Number.isNaN(n[r*3]);if(i||a)return i&&a?0:180;let o=e[t*3]*n[r*3]+e[t*3+1]*n[r*3+1]+e[t*3+2]*n[r*3+2],s=Math.hypot(e[t*3],e[t*3+1],e[t*3+2]),c=Math.hypot(n[r*3],n[r*3+1],n[r*3+2]);return Math.acos(Math.min(1,Math.max(-1,o/(s*c))))*180/Math.PI}let y=Math.fround,b={core:{x0:-1e9,y0:-1e9,x1:1e9,y1:1e9},nodes:[]};function x(e=3e3){let t=99,n=()=>(t=t*1664525+1013904223>>>0)/4294967296,r=(e,t)=>{let r=new Float32Array(e*3);for(let i=0;i<e;i++){let e=i%4==3,a=e?r[(i-1)*3]:Math.round(n()*400)/8,o=e?r[(i-1)*3+1]:Math.round(n()*400)/8;r[i*3]=a,r[i*3+1]=o,r[i*3+2]=5*Math.sin(a/7)+n()+(a>30?t:0)}return r};return{positions:r(e,0),reference:r(Math.round(e*.8),-.75)}}function S(e,t){let n=r(e,t),i=y(t),a=Math.PI*i*i;return Float32Array.from(n,e=>e/a)}function C(e,t,n,r){let i=e.length/3,a=y(n),s=new Float32Array(i).fill(r===1?0:a);if(t.length===0)return s;let c=new o(t,r===1);for(let n=0;n<i;n++){let i=c.nearest(e[n*3],e[n*3+1],r===1?0:e[n*3+2]),o=y(Math.sqrt(i.d2));r===1?o<=a&&(s[n]=e[n*3+2]-t[i.index*3+2]):s[n]=Math.min(o,a)}return s}function w(e,t){if(e.length!==t.length)return 0;for(let n=0;n<e.length;n++)if(!Object.is(e[n],t[n]))return n;return-1}async function T(e){let{positions:t,reference:n}=x(),r=[{label:`density`,run:async()=>(await e.run(t,b,0,{op:`density`,radius:1.5})).scalarField,want:()=>S(t,1.5)},{label:`c2c 3D`,run:async()=>(await e.run(t,b,0,{op:`c2c`,maxDistance:4,mode:0,reference:n})).scalarField,want:()=>C(t,n,4,0)},{label:`c2c elevation change`,run:async()=>(await e.run(t,b,0,{op:`c2c`,maxDistance:4,mode:1,reference:n})).scalarField,want:()=>C(t,n,4,1)}];for(let e of r){let t=await e.run(),n=e.want(),r=t?w(t,n):0;if(r>=0)throw Error(`GPU self-test ${e.label}: point ${r} is ${t?.[r]}, expected ${n[r]}`)}}function E(e=3e3){let t=7,n=()=>(t=t*1664525+1013904223>>>0)/4294967296,r=new Float32Array(e*3);for(let t=0;t<e;t++){let e=n()*60,i=n()*60,a=4*Math.sin(e/9)+2*Math.cos(i/7)+(e>40?3:0)+.01*n();r.set([e,i,a],t*3)}return r}async function D(e){let t=E();if(e.normalsError)return`kernel: ${e.normalsError}`;let n=t.length/3;for(let r of[5,10,30]){let i=(await e.run(t,b,0,{op:`normals`,kSearch:r})).normals,a=_(t,r);if(!i||i.length!==a.length)return`normals k=${r}: wrong length`;let o=0,s=0;for(let e=0;e<n;e++){let t=v(i,e,a,e);t>.1&&o++,s=Math.max(s,t)}if(o>n*.001)return`normals k=${r}: ${o} of ${n} points over 0.1 deg (max ${s.toFixed(2)} deg)`}return null}function O(){return typeof navigator<`u`&&`gpu`in navigator&&!!navigator.gpu}function k(e){let t=e;return t.info?.isFallbackAdapter===!0||t.isFallbackAdapter===!0}async function A(){if(!O())return null;try{let e=await navigator.gpu.requestAdapter({powerPreference:`high-performance`});if(!e||k(e))return null;let t=e.features.has(`timestamp-query`),n=t?[`timestamp-query`]:[],r=await e.requestDevice({requiredFeatures:n,requiredLimits:{maxStorageBufferBindingSize:e.limits.maxStorageBufferBindingSize,maxBufferSize:e.limits.maxBufferSize,maxComputeWorkgroupsPerDimension:e.limits.maxComputeWorkgroupsPerDimension}}),i=e.info;return{device:r,adapterInfo:[i.vendor,i.architecture,i.description].filter(Boolean).join(` / `),timestamps:t,maxStorageBufferBindingSize:r.limits.maxStorageBufferBindingSize,maxBufferSize:r.limits.maxBufferSize}}catch{return null}}function j(e,t,n){let r=Math.max(16,Math.ceil(t.byteLength/4)*4),i=e.createBuffer({label:n,size:r,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,mappedAtCreation:!0});return new Uint8Array(i.getMappedRange()).set(new Uint8Array(t.buffer,t.byteOffset,t.byteLength)),i.unmap(),i}function M(e,t,n){return e.createBuffer({label:n,size:Math.max(16,Math.ceil(t/4)*4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST})}function N(e,t,n){let r=e.createBuffer({label:n,size:Math.max(16,Math.ceil(t.byteLength/16)*16),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,mappedAtCreation:!0});return new Uint8Array(r.getMappedRange()).set(new Uint8Array(t)),r.unmap(),r}function P(e,t=256){let n=Math.max(1,Math.ceil(e/t)),r=Math.min(n,65535);return[r,Math.ceil(n/r)]}function F(e,t,n){return e.createBindGroup({layout:t.getBindGroupLayout(0),entries:n.map((e,t)=>({binding:t,resource:{buffer:e}}))})}function I(e,t,n,r){let i=e.createBuffer({size:Math.max(16,r),usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});return t.copyBufferToBuffer(n,0,i,0,r),i}var L=class{device;capacity;querySet;names=[];resolveBuffer=null;readBuffer=null;constructor(e,t,n=128){this.device=e,this.capacity=n,this.querySet=t?e.createQuerySet({type:`timestamp`,count:n*2}):null}pass(e,t){if(!this.querySet||this.names.length>=this.capacity)return e.beginComputePass({label:t});let n=this.names.length;return this.names.push(t),e.beginComputePass({label:t,timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:n*2,endOfPassWriteIndex:n*2+1}})}resolve(e){if(!this.querySet||this.names.length===0)return;let t=this.names.length*16;this.resolveBuffer=this.device.createBuffer({size:t,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),this.readBuffer=this.device.createBuffer({size:t,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),e.resolveQuerySet(this.querySet,0,this.names.length*2,this.resolveBuffer,0),e.copyBufferToBuffer(this.resolveBuffer,0,this.readBuffer,0,t)}async read(){if(!this.readBuffer)return null;await this.readBuffer.mapAsync(GPUMapMode.READ);let e=new BigUint64Array(this.readBuffer.getMappedRange().slice(0));this.readBuffer.unmap();let t={};return this.names.forEach((n,r)=>{let i=Number(e[r*2+1]-e[r*2])/1e6;t[n]=(t[n]??0)+i}),t}destroy(){this.querySet?.destroy(),this.resolveBuffer?.destroy(),this.readBuffer?.destroy()}};let R=2**31;function z(e){let t=1/0,n=1/0,r=1/0,i=-1/0,a=-1/0,o=-1/0;for(let s=0;s<e.length;s+=3){let c=e[s],l=e[s+1],u=e[s+2];c<t&&(t=c),c>i&&(i=c),l<n&&(n=l),l>a&&(a=l),u<r&&(r=u),u>o&&(o=u)}return t<=i?{min:[t,n,r],max:[i,a,o]}:{min:[0,0,0],max:[0,0,0]}}function B(e){return 2**Math.ceil(Math.log2(Math.max(1,e)))}function V(e,t,n,r,i){let a=[Math.fround(e.min[0]),Math.fround(e.min[1]),r?0:Math.fround(e.min[2])],o=[e.max[0]-a[0],e.max[1]-a[1],r?0:e.max[2]-a[2]],s=Math.max(t,1e-6),c=[1,1,1];for(;;){let e=Math.fround(1/s);if(c=[0,1,2].map(t=>Math.floor(o[t]*e)+1),c[0]*c[1]*c[2]<R)break;s*=1.25}let l=c[0]*c[1]*c[2],u=l<=Math.max(4*n,65536)&&l+1<=i,d=u?l:Math.min(B(2*Math.max(n,512)),B(i)/2);return{gmin:a,cell:Math.fround(s),invCell:Math.fround(1/s),dims:c,hashed:!u,buckets:d,mask:u?0:d-1,flatten:r,n}}function ee(e){let t=/* @__PURE__ */ new ArrayBuffer(48),n=new Float32Array(t),r=new Uint32Array(t);return n[0]=e.gmin[0],n[1]=e.gmin[1],n[2]=e.gmin[2],n[3]=e.cell,r[4]=e.dims[0],r[5]=e.dims[1],r[6]=e.dims[2],n[7]=e.invCell,r[8]=+!!e.hashed,r[9]=e.mask,r[10]=e.n,r[11]=+!!e.flatten,t}function te(e,t){let n=Math.max(1e-12,(e.max[0]-e.min[0])*(e.max[1]-e.min[1]));return 2*Math.sqrt(n/Math.max(1,t))}function H(e,t,n,r,i,a,o){let s=Math.ceil(a/1024),c=M(e,s*4,`scan sums`),l=N(e,new Uint32Array([a,0,0,0]).buffer,`scan params`);o.push(c,l);let[u,d]=P(s,1),f=r.pass(n,`scan`);f.setPipeline(t.scanBlocks),f.setBindGroup(0,F(e,t.scanBlocks,[i,c,l])),f.dispatchWorkgroups(u,d),f.end(),s<=1||(H(e,t,n,r,c,s,o),f=r.pass(n,`scan`),f.setPipeline(t.addSums),f.setBindGroup(0,F(e,t.addSums,[i,c,l])),f.dispatchWorkgroups(u,d),f.end())}function U(e,t,n,r,i,a,o){let s=a.n,c=N(e,ee(a),`grid`),l=M(e,s*4,`point key`),u=M(e,(a.buckets+1)*4,`bucket counter`),d=M(e,(a.buckets+1)*4,`bucket starts`),f=M(e,s*16,`sorted points`),p=M(e,s*4,`sorted keys`);o.push(l,u);let[m,h]=P(s),g=r.pass(n,`grid count`);return g.setPipeline(t.countCells),g.setBindGroup(0,F(e,t.countCells,[c,i,l,u])),g.dispatchWorkgroups(m,h),g.end(),H(e,t,n,r,u,a.buckets+1,o),n.copyBufferToBuffer(u,0,d,0,(a.buckets+1)*4),g=r.pass(n,`grid scatter`),g.setPipeline(t.scatter),g.setBindGroup(0,F(e,t.scatter,[c,i,l,u,f,p])),g.dispatchWorkgroups(m,h),g.end(),{plan:a,uniform:c,sorted:f,sortedKey:p,starts:d,destroy:()=>{c.destroy(),f.destroy(),p.destroy(),d.destroy()}}}var W=`// Shared by every grid kernel. A uniform grid over tile-local f32 coordinates:
// cell = floor((p - gmin) * invCell), key = linear cell id (< 2^31, checked on
// the CPU), bucket = key (dense grid) or a hash of it (sparse grid, the table
// is a power of two). Points of a bucket are contiguous after the scatter;
// a hashed bucket can hold several cells, so lookups compare the key.

struct Grid {
  gmin: vec3<f32>,
  cell: f32,
  dims: vec3<u32>,
  invCell: f32,
  hashed: u32,
  mask: u32,
  n: u32,
  flatten: u32,
};

// lowbias32 (Chris Wellons, public domain integer hash).
fn hash32(v: u32) -> u32 {
  var x = v;
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

fn cellOfIn(g: Grid, p: vec3<f32>) -> vec3<u32> {
  let c = floor((p - g.gmin) * g.invCell);
  let hi = vec3<f32>(g.dims - vec3<u32>(1u));
  return vec3<u32>(clamp(c, vec3<f32>(0.0), hi));
}

fn keyOfIn(g: Grid, c: vec3<u32>) -> u32 {
  return c.x + g.dims.x * (c.y + g.dims.y * c.z);
}

fn bucketOfIn(g: Grid, key: u32) -> u32 {
  if (g.hashed == 1u) {
    return hash32(key) & g.mask;
  }
  return key;
}

// 2D dispatch (more than 65535 workgroups): flat invocation index.
fn flatIndex(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
  return gid.x + gid.y * nwg.x * 256u;
}

// FLANN's L2_Simple squared distance in f32: ((dx*dx + dy*dy) + dz*dz), every
// product rounded on its own. Metal fuses a*b + c into an fma otherwise (1 ulp
// off PCL on ~10% of C2C distances, measured); OR-ing the product's bits with
// a runtime zero from a uniform is an integer op no compiler can see through.
fn sqDist(d: vec3<f32>, zero: u32) -> f32 {
  let xx = bitcast<f32>(bitcast<u32>(d.x * d.x) | zero);
  let yy = bitcast<f32>(bitcast<u32>(d.y * d.y) | zero);
  let zz = bitcast<f32>(bitcast<u32>(d.z * d.z) | zero);
  return (xx + yy) + zz;
}

// Segmented grids (batched node jobs): the bucket of (cell key, segment).
fn segBucket(g: Grid, key: u32, seg: u32) -> u32 {
  return hash32(key ^ hash32(seg + 0x9e3779b9u)) & g.mask;
}
`,ne=`// Counting sort of points into grid buckets: count (atomicAdd per bucket),
// exclusive scan (scan.wgsl), scatter (atomicAdd on a copy of the offsets).
// Input points are packed xyz f32. Output: sorted vec4<u32> (xyz bits, payload) and the
// key of each sorted point. Payload = original index or, when the
// grid is flattened to 2D, the original z (elevation-change C2C needs it).

@group(0) @binding(0) var<uniform> g: Grid;
@group(0) @binding(1) var<storage, read> pts: array<f32>;
@group(0) @binding(2) var<storage, read_write> pointKey: array<u32>;
@group(0) @binding(3) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> sorted: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> sortedKey: array<u32>;

fn pointAt(i: u32) -> vec3<f32> {
  let z = select(pts[3u * i + 2u], 0.0, g.flatten == 1u);
  return vec3<f32>(pts[3u * i], pts[3u * i + 1u], z);
}

@compute @workgroup_size(256)
fn countCells(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = flatIndex(gid, nwg);
  if (i >= g.n) { return; }
  let key = keyOfIn(g, cellOfIn(g, pointAt(i)));
  pointKey[i] = key;
  atomicAdd(&counter[bucketOfIn(g, key)], 1u);
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = flatIndex(gid, nwg);
  if (i >= g.n) { return; }
  let key = pointKey[i];
  let slot = atomicAdd(&counter[bucketOfIn(g, key)], 1u);
  let p = pointAt(i);
  // Bits are moved as u32: an index bitcast to f32 is a denormal or NaN
  // that a float path could flush or canonicalise.
  let payload = select(i, bitcast<u32>(pts[3u * i + 2u]), g.flatten == 1u);
  sorted[slot] = vec4<u32>(bitcast<vec3<u32>>(p), payload);
  sortedKey[slot] = key;
}
`,G=`// Exclusive prefix sum of u32, 1024 elements per workgroup (256 threads x 4).
// Multi-level: scanBlocks writes each block's total to \`sums\`; the caller
// scans \`sums\` recursively and addSums adds the block offsets back.
// Written for this repo (Hillis-Steele inside the workgroup), no subgroups
// so it runs where subgroups are not exposed (Firefox).

struct ScanParams {
  len: u32,
  _a: u32,
  _b: u32,
  _c: u32,
};

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> sums: array<u32>;
@group(0) @binding(2) var<uniform> sp: ScanParams;

var<workgroup> tmp: array<u32, 256>;

@compute @workgroup_size(256)
fn scanBlocks(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let block = wid.x + wid.y * nwg.x;
  // Spare workgroups of a 2D dispatch (uniform per workgroup, so the early
  // return keeps the barriers in uniform control flow).
  if (block * 1024u >= sp.len) { return; }
  let base = block * 1024u + lid.x * 4u;
  var local: array<u32, 4>;
  var s = 0u;
  for (var k = 0u; k < 4u; k++) {
    let i = base + k;
    local[k] = s;
    if (i < sp.len) { s += data[i]; }
  }
  tmp[lid.x] = s;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off <<= 1u) {
    var t = 0u;
    if (lid.x >= off) { t = tmp[lid.x - off]; }
    workgroupBarrier();
    tmp[lid.x] += t;
    workgroupBarrier();
  }
  var excl = 0u;
  if (lid.x > 0u) { excl = tmp[lid.x - 1u]; }
  for (var k = 0u; k < 4u; k++) {
    let i = base + k;
    if (i < sp.len) { data[i] = local[k] + excl; }
  }
  if (lid.x == 255u) { sums[block] = tmp[255]; }
}

@compute @workgroup_size(256)
fn addSums(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let block = wid.x + wid.y * nwg.x;
  if (block * 1024u >= sp.len) { return; }
  let add = sums[block];
  let base = block * 1024u + lid.x * 4u;
  for (var k = 0u; k < 4u; k++) {
    let i = base + k;
    if (i < sp.len) { data[i] += add; }
  }
}
`,re=`// Neighbour count within radius r (PCL computeDensity semantics: strict
// d2 < r2 like FLANN's RadiusResultSet, the query point counts itself).
// One thread per sorted point, so neighbouring threads read the same cells.
// The cell size is >= r, so the 27 surrounding cells hold every neighbour.
// d2 is computed like FLANN (sqDist in common.wgsl) for bit parity.
// inclusive = 1: d2 <= r2 instead (radius outlier: PCL keeps a point whose
// (minNeighbors+1)-th neighbour has d2 <= r*r in double; r2 is then the
// largest float not above that double).

struct DensityParams {
  r2: f32,
  offset: u32,
  count: u32,
  zero: u32,
  inclusive: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> g: Grid;
@group(0) @binding(1) var<uniform> dp: DensityParams;
@group(0) @binding(2) var<storage, read> sorted: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> sortedKey: array<u32>;
@group(0) @binding(4) var<storage, read> starts: array<u32>;
@group(0) @binding(5) var<storage, read_write> counts: array<u32>;

@compute @workgroup_size(256)
fn density(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let t = flatIndex(gid, nwg);
  if (t >= dp.count) { return; }
  let s = dp.offset + t;
  let a = sorted[s];
  let p = bitcast<vec3<f32>>(a.xyz);
  let key = sortedKey[s];
  let plane = g.dims.x * g.dims.y;
  let c = vec3<i32>(vec3<u32>(key % g.dims.x, (key / g.dims.x) % g.dims.y, key / plane));
  let lo = max(c - vec3<i32>(1), vec3<i32>(0));
  let hi = min(c + vec3<i32>(1), vec3<i32>(g.dims) - vec3<i32>(1));
  var n = 0u;
  for (var z = lo.z; z <= hi.z; z++) {
    for (var y = lo.y; y <= hi.y; y++) {
      for (var x = lo.x; x <= hi.x; x++) {
        let k = keyOfIn(g, vec3<u32>(vec3<i32>(x, y, z)));
        let b = bucketOfIn(g, k);
        let end = starts[b + 1u];
        for (var j = starts[b]; j < end; j++) {
          if (sortedKey[j] != k) { continue; }
          let d = p - bitcast<vec3<f32>>(sorted[j].xyz);
          let d2 = sqDist(d, dp.zero);
          if (d2 < dp.r2 || (dp.inclusive == 1u && d2 == dp.r2)) { n++; }
        }
      }
    }
  }
  counts[a.w] = n;
}
`,ie=`// Ring search on one grid; instantiated twice (fine grid F, coarse grid C) by
// replacing $G, because WGSL functions cannot take storage buffers as
// arguments without an extension. Visits Chebyshev shells k = 0..maxShell
// around the query cell and keeps the smallest d2 (ties: smallest payload).
// best = (d2, payload z, ambiguous): \`ambiguous\` is 1 when another point lies
// at exactly the best d2 with a different z. Only elevation change reads it:
// FLANN keeps the first tied point its traversal meets, so the CPU resolves
// those queries with the emulated tree (flannKdTree.ts).
// Returns true when nothing outside the visited block can be nearer than the
// best so far or within maxD, i.e. the answer is final.

fn visit$G(q: vec3<f32>, cell: vec3<i32>, best: ptr<function, vec3<f32>>) {
  let g = g$G;
  // Skip the cell when its box is farther than the best (margin covers the
  // floor() rounding of points sitting on a cell face).
  let lo = g.gmin + vec3<f32>(cell) * g.cell;
  let hi = lo + vec3<f32>(g.cell);
  let m = g.cell * 1e-3;
  let dv = max(max(lo - q, q - hi) - vec3<f32>(m), vec3<f32>(0.0));
  if (dot(dv, dv) > (*best).x) { return; }
  let k = keyOfIn(g, vec3<u32>(cell));
  let b = bucketOfIn(g, k);
  let end = start$G[b + 1u];
  for (var j = start$G[b]; j < end; j++) {
    if (key$G[j] != k) { continue; }
    let s = sorted$G[j];
    let d = q - bitcast<vec3<f32>>(s.xyz);
    let d2 = sqDist(d, cp.zero);
    let z = bitcast<f32>(s.w);
    if (d2 < (*best).x) {
      *best = vec3<f32>(d2, z, 0.0);
    } else if (d2 == (*best).x && z != (*best).y) {
      *best = vec3<f32>(d2, min(z, (*best).y), 1.0);
    }
  }
}

fn search$G(q: vec3<f32>, best: ptr<function, vec3<f32>>, maxShell: i32, maxD: f32) -> bool {
  let g = g$G;
  let c = vec3<i32>(cellOfIn(g, q));
  let top = vec3<i32>(g.dims) - vec3<i32>(1);
  for (var k = 0; k <= maxShell; k++) {
    let lo = c - vec3<i32>(k);
    let hi = c + vec3<i32>(k);
    let clo = max(lo, vec3<i32>(0));
    let chi = min(hi, top);
    for (var z = clo.z; z <= chi.z; z++) {
      for (var y = clo.y; y <= chi.y; y++) {
        if (z == lo.z || z == hi.z || y == lo.y || y == hi.y) {
          for (var x = clo.x; x <= chi.x; x++) { visit$G(q, vec3<i32>(x, y, z), best); }
        } else {
          if (lo.x >= 0) { visit$G(q, vec3<i32>(lo.x, y, z), best); }
          if (hi.x <= top.x) { visit$G(q, vec3<i32>(hi.x, y, z), best); }
        }
      }
    }
    // Lower bound on the distance to any cell outside the visited block.
    var lb = 3.0e38;
    for (var a = 0; a < 3; a++) {
      if (lo[a] > 0) { lb = min(lb, q[a] - (g.gmin[a] + f32(lo[a]) * g.cell)); }
      if (hi[a] < top[a]) { lb = min(lb, g.gmin[a] + f32(hi[a] + 1) * g.cell - q[a]); }
    }
    if (lb >= 3.0e38) { return true; }
    lb -= g.cell * 1e-3;
    if (lb > 0.0 && (lb * lb >= (*best).x || lb > maxD)) { return true; }
  }
  return false;
}
`,ae=`// Cloud-to-cloud nearest distance (PCL cloudToCloudDistance semantics).
// mode 0: 3D nearest; mode 1 (flatten): nearest in XY, payload = its z.
// Fine grid first (a few shells); points whose nearest reference point is
// farther fall through to the coarse grid. Output per query point:
// (best d2 bits, top bit set when the nearest is ambiguous (ties with a
// different z), or 3e38 when nothing lies within maxD; payload z bits). The CPU takes
// sqrt / caps / subtracts so the f32 rounding matches the C++ exactly.

struct C2cParams {
  maxD: f32,
  flatten: u32,
  offset: u32,
  count: u32,
  fineShells: i32,
  coarseShells: i32,
  zero: u32,
  _b: u32,
};

@group(0) @binding(0) var<uniform> gF: Grid;
@group(0) @binding(1) var<uniform> gC: Grid;
@group(0) @binding(2) var<uniform> cp: C2cParams;
@group(0) @binding(3) var<storage, read> sortedF: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> keyF: array<u32>;
@group(0) @binding(5) var<storage, read> startF: array<u32>;
@group(0) @binding(6) var<storage, read> sortedC: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> keyC: array<u32>;
@group(0) @binding(8) var<storage, read> startC: array<u32>;
@group(0) @binding(9) var<storage, read> qpts: array<f32>;
@group(0) @binding(10) var<storage, read_write> outBest: array<vec2<u32>>;

@compute @workgroup_size(256)
fn c2c(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let t = flatIndex(gid, nwg);
  if (t >= cp.count) { return; }
  let i = cp.offset + t;
  let z = select(qpts[3u * i + 2u], 0.0, cp.flatten == 1u);
  let q = vec3<f32>(qpts[3u * i], qpts[3u * i + 1u], z);
  var best = vec3<f32>(3.0e38, 0.0, 0.0);
  if (!searchF(q, &best, cp.fineShells, cp.maxD)) {
    _ = searchC(q, &best, cp.coarseShells, cp.maxD);
  }
  let amb = select(0u, 0x80000000u, best.z != 0.0);
  outBest[i] = vec2<u32>(bitcast<u32>(best.x) | amb, bitcast<u32>(best.y));
}
`,oe=`// Normals (PCL NormalEstimation, k nearest neighbours). One thread per sorted
// point: k+1 nearest on the grid (Chebyshev shells), then PCL's shifted
// single-pass covariance, pcl::eigen33's closed form and the flip towards the
// origin, step for step as gpu/pclNormals.ts (the reference). Products go
// through mul() so no compiler fuses them into an fma (wasm has none).
// Canonical order: neighbours are kept sorted by (squared distance, then the
// x, y, z bit patterns), never by the grid's scatter order (atomics, not
// stable) nor by array position, and the covariance is summed in that order.
// Points that compare equal are exact duplicates, which contribute the same
// values whichever is taken. So a point's normal depends only on the SET of
// points in the tile: bit-stable run to run, and bit-identical on any two
// tiles (a node job, a batch tile) holding the same neighbourhood, in any order.
// flags[i]: 0 = done here; 1 = the k-th and (k+1)-th distances tie (FLANN's
// traversal order decides the set: with cpuTies the CPU answers with the
// emulated tree, else the set found here is used);
// 2 = not settled within maxShell shells: retried on a 4x coarser grid, then
// on ever coarser grids (retry = level), maxShell shells each, until a grid so
// coarse that maxShell shells cover it: every query is answered here, and no
// query walks an unbounded number of cells in one dispatch.

const KCAP: u32 = 104u;
const EPS: f32 = 1.1920928955078125e-7;
const FLT_MIN: f32 = 1.1754943508222875e-38;
const PREC: f32 = 1e-5;

struct NormalParams {
  k: u32,
  offset: u32,
  count: u32,
  zero: u32,
  maxShell: i32,
  inv3: f32,
  sqrt3: f32,
  // 1: flag-1 queries (k-th distance tie) are left to the CPU; 0: answered here
  // with the set the grid search found (any of the tied points).
  cpuTies: u32,
  // > 0: a coarser retry level, only for queries still flagged 2.
  retry: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> g: Grid;
@group(0) @binding(1) var<uniform> np: NormalParams;
@group(0) @binding(2) var<storage, read> sorted: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> sortedKey: array<u32>;
@group(0) @binding(4) var<storage, read> starts: array<u32>;
@group(0) @binding(5) var<storage, read_write> outN: array<f32>;
@group(0) @binding(6) var<storage, read_write> flags: array<u32>;

var<private> bd: array<f32, KCAP>;
var<private> bs: array<u32, KCAP>;
var<private> cnt: u32;
var<private> cap: u32;

fn mul(a: f32, b: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(a * b) | np.zero);
}

fn worst() -> f32 {
  if (cnt == cap) { return bd[cap - 1u]; }
  return 3.0e38;
}

// (da, sorted point ja) comes before (db, sorted point jb) in the canonical
// order: distance, then the coordinates' bit patterns (a total order).
fn less(da: f32, ja: u32, db: f32, jb: u32) -> bool {
  if (da != db) { return da < db; }
  let a = sorted[ja].xyz;
  let b = sorted[jb].xyz;
  if (a.x != b.x) { return a.x < b.x; }
  if (a.y != b.y) { return a.y < b.y; }
  return a.z < b.z;
}

fn insert(d2: f32, j: u32) {
  if (cnt == cap && !less(d2, j, bd[cap - 1u], bs[cap - 1u])) { return; }
  if (cnt < cap) { cnt++; }
  var i = cnt - 1u;
  loop {
    if (i == 0u) { break; }
    if (less(d2, j, bd[i - 1u], bs[i - 1u])) {
      bd[i] = bd[i - 1u];
      bs[i] = bs[i - 1u];
      i--;
    } else { break; }
  }
  bd[i] = d2;
  bs[i] = j;
}

fn visit(q: vec3<f32>, cell: vec3<i32>) {
  let lo = g.gmin + vec3<f32>(cell) * g.cell;
  let hi = lo + vec3<f32>(g.cell);
  let m = g.cell * 1e-3;
  let dv = max(max(lo - q, q - hi) - vec3<f32>(m), vec3<f32>(0.0));
  if (dot(dv, dv) > worst()) { return; }
  let k = keyOfIn(g, vec3<u32>(cell));
  let b = bucketOfIn(g, k);
  let end = starts[b + 1u];
  for (var j = starts[b]; j < end; j++) {
    if (sortedKey[j] != k) { continue; }
    let d = q - bitcast<vec3<f32>>(sorted[j].xyz);
    insert(sqDist(d, np.zero), j);
  }
}

// True once the k+1 nearest are final (or the whole grid was visited).
fn search(q: vec3<f32>) -> bool {
  let c = vec3<i32>(cellOfIn(g, q));
  let top = vec3<i32>(g.dims) - vec3<i32>(1);
  for (var k = 0; k <= np.maxShell; k++) {
    let lo = c - vec3<i32>(k);
    let hi = c + vec3<i32>(k);
    let clo = max(lo, vec3<i32>(0));
    let chi = min(hi, top);
    for (var z = clo.z; z <= chi.z; z++) {
      for (var y = clo.y; y <= chi.y; y++) {
        if (z == lo.z || z == hi.z || y == lo.y || y == hi.y) {
          for (var x = clo.x; x <= chi.x; x++) { visit(q, vec3<i32>(x, y, z)); }
        } else {
          if (lo.x >= 0) { visit(q, vec3<i32>(lo.x, y, z)); }
          if (hi.x <= top.x) { visit(q, vec3<i32>(hi.x, y, z)); }
        }
      }
    }
    var lb = 3.0e38;
    for (var a = 0; a < 3; a++) {
      if (lo[a] > 0) { lb = min(lb, q[a] - (g.gmin[a] + f32(lo[a]) * g.cell)); }
      if (hi[a] < top[a]) { lb = min(lb, g.gmin[a] + f32(hi[a] + 1) * g.cell - q[a]); }
    }
    if (lb >= 3.0e38) { return true; }
    lb -= g.cell * 1e-3;
    if (cnt == cap && lb > 0.0 && lb * lb >= bd[cap - 1u]) { return true; }
  }
  return false;
}

// pcl::computeRoots2 (b = c2, c = c1), ascending.
fn roots2(b: f32, c: f32) -> vec3<f32> {
  var d = mul(b, b) - mul(4.0, c);
  if (d < 0.0) { d = 0.0; }
  let sd = sqrt(d);
  return vec3<f32>(0.0, mul(0.5, b - sd), mul(0.5, b + sd));
}

// pcl::computeRoots of the scaled symmetric matrix (rows r0, r1, r2).
fn computeRoots(r0: vec3<f32>, r1: vec3<f32>, r2: vec3<f32>) -> vec3<f32> {
  let m00 = r0.x; let m01 = r0.y; let m02 = r0.z; let m11 = r1.y; let m12 = r1.z; let m22 = r2.z;
  let c0 = mul(mul(m00, m11), m22) + mul(mul(mul(2.0, m01), m02), m12) - mul(mul(m00, m12), m12)
    - mul(mul(m11, m02), m02) - mul(mul(m22, m01), m01);
  let c1 = mul(m00, m11) - mul(m01, m01) + mul(m00, m22) - mul(m02, m02) + mul(m11, m22) - mul(m12, m12);
  let c2 = m00 + m11 + m22;
  if (abs(c0) < EPS) { return roots2(c2, c1); }
  let c2over3 = mul(c2, np.inv3);
  var aOver3 = mul(c1 - mul(c2, c2over3), np.inv3);
  if (aOver3 > 0.0) { aOver3 = 0.0; }
  let halfB = mul(0.5, c0 + mul(c2over3, mul(mul(2.0, c2over3), c2over3) - c1));
  var q = mul(halfB, halfB) + mul(mul(aOver3, aOver3), aOver3);
  if (q > 0.0) { q = 0.0; }
  let rho = sqrt(-aOver3);
  let theta = mul(atan2(sqrt(-q), halfB), np.inv3);
  let ct = cos(theta);
  let st = sin(theta);
  var r = vec3<f32>(
    c2over3 + mul(mul(2.0, rho), ct),
    c2over3 - mul(rho, ct + mul(np.sqrt3, st)),
    c2over3 - mul(rho, ct - mul(np.sqrt3, st)),
  );
  if (r.x >= r.y) { r = vec3<f32>(r.y, r.x, r.z); }
  if (r.y >= r.z) {
    r = vec3<f32>(r.x, r.z, r.y);
    if (r.x >= r.y) { r = vec3<f32>(r.y, r.x, r.z); }
  }
  if (r.x <= 0.0) { return roots2(c2, c1); }
  return r;
}

fn crossNF(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(mul(a.y, b.z) - mul(a.z, b.y), mul(a.z, b.x) - mul(a.x, b.z), mul(a.x, b.y) - mul(a.y, b.x));
}

fn len3(c: vec3<f32>) -> f32 {
  return sqrt(mul(c.x, c.x) + mul(c.y, c.y) + mul(c.z, c.z));
}

// detail::getLargest3x3Eigenvector (first maximum wins, like Eigen's maxCoeff).
fn largestCross(r0: vec3<f32>, r1: vec3<f32>, r2: vec3<f32>) -> vec3<f32> {
  let c0 = crossNF(r0, r1);
  let c1 = crossNF(r0, r2);
  let c2 = crossNF(r1, r2);
  var best = c0;
  var l = len3(c0);
  let l1 = len3(c1);
  if (l1 > l) { best = c1; l = l1; }
  let l2 = len3(c2);
  if (l2 > l) { best = c2; l = l2; }
  return best / l;
}

fn unitOrthogonal(v: vec3<f32>) -> vec3<f32> {
  if (!(abs(v.x) <= mul(abs(v.z), PREC)) || !(abs(v.y) <= mul(abs(v.z), PREC))) {
    let inv = 1.0 / sqrt(mul(v.x, v.x) + mul(v.y, v.y));
    return vec3<f32>(mul(-v.y, inv), mul(v.x, inv), 0.0);
  }
  let inv = 1.0 / sqrt(mul(v.y, v.y) + mul(v.z, v.z));
  return vec3<f32>(0.0, mul(-v.z, inv), mul(v.y, inv));
}

fn eigen33Smallest(m0: vec3<f32>, m1: vec3<f32>, m2: vec3<f32>) -> vec3<f32> {
  var scale = max(max(max(abs(m0.x), abs(m0.y)), max(abs(m0.z), abs(m1.x))),
    max(max(abs(m1.y), abs(m1.z)), max(max(abs(m2.x), abs(m2.y)), abs(m2.z))));
  if (scale <= FLT_MIN) { scale = 1.0; }
  var r0 = m0 / scale;
  var r1 = m1 / scale;
  var r2 = m2 / scale;
  let ev = computeRoots(r0, r1, r2);
  if (ev.y - ev.x > EPS) {
    r0.x -= ev.x; r1.y -= ev.x; r2.z -= ev.x;
    return largestCross(r0, r1, r2);
  }
  if (ev.z - ev.x > EPS) {
    r0.x -= ev.z; r1.y -= ev.z; r2.z -= ev.z;
    return unitOrthogonal(largestCross(r0, r1, r2));
  }
  return vec3<f32>(1.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn normals(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let t = gid.x + gid.y * nwg.x * 64u;
  if (t >= np.count) { return; }
  let s = np.offset + t;
  let a = sorted[s];
  let p = bitcast<vec3<f32>>(a.xyz);
  if (np.retry != 0u && flags[a.w] != 2u) { return; }
  let o = a.w * 3u;
  let nan = bitcast<f32>(0x7fc00000u | np.zero);
  cap = np.k + 1u;
  cnt = 0u;
  var flag = 0u;
  if (!search(p)) {
    flag = 2u;
  } else if (cnt == cap && bd[np.k] == bd[np.k - 1u]) {
    flag = 1u;
  }
  flags[a.w] = flag;
  let m = min(cnt, np.k);
  if (flag == 2u || (flag == 1u && np.cpuTies == 1u) || m < 3u) {
    outN[o] = nan; outN[o + 1u] = nan; outN[o + 2u] = nan;
    return;
  }
  // computeMeanAndCovarianceMatrix, shifted by the first neighbour (the
  // nearest: the query itself or an exact duplicate of it).
  let kp = bitcast<vec3<f32>>(sorted[bs[0]].xyz);
  var acc0 = 0.0; var acc1 = 0.0; var acc2 = 0.0; var acc3 = 0.0; var acc4 = 0.0;
  var acc5 = 0.0; var acc6 = 0.0; var acc7 = 0.0; var acc8 = 0.0;
  for (var j = 0u; j < m; j++) {
    let d = bitcast<vec3<f32>>(sorted[bs[j]].xyz) - kp;
    acc0 += mul(d.x, d.x);
    acc1 += mul(d.x, d.y);
    acc2 += mul(d.x, d.z);
    acc3 += mul(d.y, d.y);
    acc4 += mul(d.y, d.z);
    acc5 += mul(d.z, d.z);
    acc6 += d.x;
    acc7 += d.y;
    acc8 += d.z;
  }
  let c = f32(m);
  acc0 /= c; acc1 /= c; acc2 /= c; acc3 /= c; acc4 /= c; acc5 /= c; acc6 /= c; acc7 /= c; acc8 /= c;
  let cxx = acc0 - mul(acc6, acc6);
  let cxy = acc1 - mul(acc6, acc7);
  let cxz = acc2 - mul(acc6, acc8);
  let cyy = acc3 - mul(acc7, acc7);
  let cyz = acc4 - mul(acc7, acc8);
  let czz = acc5 - mul(acc8, acc8);
  var n = eigen33Smallest(vec3<f32>(cxx, cxy, cxz), vec3<f32>(cxy, cyy, cyz), vec3<f32>(cxz, cyz, czz));
  let v = -p;
  let cosTheta = mul(v.x, n.x) + mul(v.y, n.y) + mul(v.z, n.z);
  if (cosTheta < 0.0) { n = -n; }
  outN[o] = n.x; outN[o + 1u] = n.y; outN[o + 2u] = n.z;
}

// k nearest squared distances (ascending; the query itself included), for
// SOR / noise: outN[i * k + j]. Distances are order-free, so they equal
// FLANN's whatever the tie order. Same passes and flags as normals (no tie flag).
@compute @workgroup_size(64)
fn knnDists(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let t = gid.x + gid.y * nwg.x * 64u;
  if (t >= np.count) { return; }
  let s = np.offset + t;
  let a = sorted[s];
  if (np.retry != 0u && flags[a.w] != 2u) { return; }
  let p = bitcast<vec3<f32>>(a.xyz);
  cap = np.k;
  cnt = 0u;
  var flag = 0u;
  if (!search(p)) { flag = 2u; }
  flags[a.w] = flag;
  let o = a.w * np.k;
  for (var j = 0u; j < np.k; j++) {
    outN[o + j] = select(3.0e38, bd[j], j < cnt);
  }
}
`,se=`// Segmented grid for batched node jobs: many small jobs share one grid, one
// dispatch and one readback. Points of every job are concatenated; a point's
// segment is its job (found from segStarts by binary search, so the host
// uploads only the points and the offsets). The bucket hashes (cell key,
// segment), and lookups compare both, so a query only ever sees points of its
// own job (jobs overlap in space: a node's halo is its neighbours' core).
// Always hashed. Output: sorted vec4<u32> (xyz bits, segment) and the cell
// key of each sorted point.

@group(0) @binding(0) var<uniform> g: Grid;
@group(0) @binding(1) var<storage, read> pts: array<f32>;
// Job j owns points [segStarts[j], segStarts[j + 1]); padded with 0xffffffff.
@group(0) @binding(2) var<storage, read> segStarts: array<u32>;
@group(0) @binding(3) var<storage, read_write> pointKey: array<u32>;
@group(0) @binding(4) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> sorted: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read_write> sortedKey: array<u32>;

fn segPoint(i: u32) -> vec3<f32> {
  return vec3<f32>(pts[3u * i], pts[3u * i + 1u], pts[3u * i + 2u]);
}

// The last j with segStarts[j] <= i (empty jobs are skipped).
fn segOf(i: u32) -> u32 {
  var lo = 0u;
  var hi = arrayLength(&segStarts) - 1u;
  loop {
    if (lo >= hi) { break; }
    let mid = (lo + hi + 1u) / 2u;
    if (segStarts[mid] <= i) { lo = mid; } else { hi = mid - 1u; }
  }
  return lo;
}

@compute @workgroup_size(256)
fn countCellsSeg(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = flatIndex(gid, nwg);
  if (i >= g.n) { return; }
  let key = keyOfIn(g, cellOfIn(g, segPoint(i)));
  pointKey[i] = key;
  atomicAdd(&counter[segBucket(g, key, segOf(i))], 1u);
}

@compute @workgroup_size(256)
fn scatterSeg(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = flatIndex(gid, nwg);
  if (i >= g.n) { return; }
  let key = pointKey[i];
  let seg = segOf(i);
  let slot = atomicAdd(&counter[segBucket(g, key, seg)], 1u);
  sorted[slot] = vec4<u32>(bitcast<vec3<u32>>(segPoint(i)), seg);
  sortedKey[slot] = key;
}
`,ce=`// Radius count for batched node jobs on the segmented grid (segGrid.wgsl):
// one thread per core point of every job (query t is core point
// t - coreStarts[j] of the job j holding it, found by binary search),
// counting points of its own segment in the 27 surrounding cells. Same test
// as density.wgsl (strict d2 < r2, or <= with inclusive for radius outlier),
// so counts equal a single-tile run on the same points.

struct SegParams {
  r2: f32,
  offset: u32,
  count: u32,
  zero: u32,
  inclusive: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> g: Grid;
@group(0) @binding(1) var<uniform> sp: SegParams;
@group(0) @binding(2) var<storage, read> sorted: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> sortedKey: array<u32>;
@group(0) @binding(4) var<storage, read> starts: array<u32>;
// Job j owns points [segStarts[j], segStarts[j + 1]) and queries
// [coreStarts[j], coreStarts[j + 1]); both padded with 0xffffffff.
@group(0) @binding(5) var<storage, read> segStarts: array<u32>;
@group(0) @binding(8) var<storage, read> coreStarts: array<u32>;
@group(0) @binding(6) var<storage, read> pts: array<f32>;
@group(0) @binding(7) var<storage, read_write> counts: array<u32>;

@compute @workgroup_size(256)
fn radiusSeg(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let t = flatIndex(gid, nwg);
  if (t >= sp.count) { return; }
  let qi = sp.offset + t;
  var ja = 0u;
  var jb = arrayLength(&coreStarts) - 1u;
  loop {
    if (ja >= jb) { break; }
    let mid = (ja + jb + 1u) / 2u;
    if (coreStarts[mid] <= qi) { ja = mid; } else { jb = mid - 1u; }
  }
  let q = vec2<u32>(segStarts[ja] + (qi - coreStarts[ja]), ja);
  let p = vec3<f32>(pts[3u * q.x], pts[3u * q.x + 1u], pts[3u * q.x + 2u]);
  let c = vec3<i32>(cellOfIn(g, p));
  let lo = max(c - vec3<i32>(1), vec3<i32>(0));
  let hi = min(c + vec3<i32>(1), vec3<i32>(g.dims) - vec3<i32>(1));
  var n = 0u;
  for (var z = lo.z; z <= hi.z; z++) {
    for (var y = lo.y; y <= hi.y; y++) {
      for (var x = lo.x; x <= hi.x; x++) {
        let k = keyOfIn(g, vec3<u32>(vec3<i32>(x, y, z)));
        let b = segBucket(g, k, q.y);
        let end = starts[b + 1u];
        for (var j = starts[b]; j < end; j++) {
          let s = sorted[j];
          if (sortedKey[j] != k || s.w != q.y) { continue; }
          let d2 = sqDist(p - bitcast<vec3<f32>>(s.xyz), sp.zero);
          if (d2 < sp.r2 || (sp.inclusive == 1u && d2 == sp.r2)) { n++; }
        }
      }
    }
  }
  counts[qi] = n;
}
`;async function K(e,t,n){let r=e.createShaderModule({label:n,code:t}),i=(await r.getCompilationInfo()).messages.filter(e=>e.type===`error`);if(i.length>0)throw Error(`WGSL ${n}: ${i.map(e=>`${e.lineNum}:${e.linePos} ${e.message}`).join(`; `)}`);return e.createComputePipelineAsync({label:n,layout:`auto`,compute:{module:r,entryPoint:n}})}async function le(e){let t=`${W}\n${ne}`,n=`${W}\n${re}`,r=e=>ie.replaceAll(`$G`,e),i=`${W}\n${ae}\n${r(`F`)}\n${r(`C`)}`,a=`${W}\n${oe}`,o=`${W}\n${se}`,s=`${W}\n${ce}`,[c,l,u]=await Promise.all([K(e,o,`countCellsSeg`),K(e,o,`scatterSeg`),K(e,s,`radiusSeg`)]),[d,f,p,m,h,g,_,v]=await Promise.all([K(e,t,`countCells`),K(e,t,`scatter`),K(e,G,`scanBlocks`),K(e,G,`addSums`),K(e,n,`density`),K(e,i,`c2c`),K(e,a,`normals`).catch(e=>e instanceof Error?e.message:String(e)),K(e,a,`knnDists`).catch(e=>e instanceof Error?e.message:String(e))]),y=typeof _!=`string`&&typeof v!=`string`;return{countCells:d,scatter:f,scanBlocks:p,addSums:m,density:h,c2c:g,countCellsSeg:c,scatterSeg:l,radiusSeg:u,normals:y?_:null,knnDists:y?v:null,normalsError:y?null:typeof _==`string`?_:String(v)}}let q=/* @__PURE__ */ new Float32Array(1),ue=new Uint32Array(q.buffer);function de(e){let t=Math.fround(e),n=t*t;return q[0]=n,q[0]>n&&--ue[0],q[0]}function fe(e,t,n,r){let i=Math.min(t,n),a=new Float64Array(t),o=new Uint8Array(t);if(i<=1)return o;let s=0,c=0;for(let r=0;r<t;r++){let t=0;for(let a=1;a<i;a++)t+=Math.sqrt(e[r*n+a]);let o=t/(i-1);a[r]=o,s+=o,c+=o*o}if(t<=1)return o;let l=s/t,u=(c-s*s/t)/(t-1),d=l+Math.fround(r)*Math.sqrt(Math.max(u,0));for(let e=0;e<t;e++)o[e]=+(a[e]<=d);return o}let pe=1<<18;function me(e){return 2**Math.ceil(Math.log2(Math.max(1,e)))}function he(e,t,n,r){let i=V(z(e),t,n,!1,r),a=Math.min(me(2*Math.max(n,512)),me(r)/2);return{...i,hashed:!0,buckets:a,mask:a-1}}async function ge(e,n,r){let{device:i,kernels:a}=e,o=n.cores.length,s=n.positions.length/3,c=Math.fround(n.radius);if(!(c>0))throw Error(`Radius must be > 0.`);let l=n.op===`radiusOutlier`,u=l?de(n.radius):Math.fround(c*c),d=e=>{let t=new Uint32Array(Math.max(4,e.length)).fill(4294967295);return t.set(e),t},f=d(n.segStarts.subarray(0,o)),p=new Uint32Array(o),m=0;for(let e=0;e<o;e++)p[e]=m,m+=n.cores[e];let h=d(p),g=he(n.positions,c*1.02,s,e.maxBuckets),_=(g.buckets+1)*4;t([{label:`batch points`,bytes:s*12,bound:!0},{label:`point key`,bytes:s*4,bound:!0},{label:`bucket counter`,bytes:_,bound:!0},{label:`sorted points`,bytes:s*16,bound:!0},{label:`batch counts`,bytes:m*4,bound:!0}],e.limits);let v=[],y=[],b=new L(i,e.timestamps);try{let e=j(i,n.positions,`batch points`),t=j(i,f,`batch segment starts`),d=j(i,h,`batch core starts`),p=M(i,m*4,`batch counts`),x=N(i,ee(g),`seg grid`),S=M(i,s*4,`point key`),C=M(i,_,`bucket counter`),w=M(i,_,`bucket starts`),T=M(i,s*16,`sorted points`),E=M(i,s*4,`sorted keys`);v.push(e,t,d,p,x,S,C,w,T,E);let D=i.createCommandEncoder(),[O,k]=P(s),A=b.pass(D,`seg grid count`);A.setPipeline(a.countCellsSeg),A.setBindGroup(0,F(i,a.countCellsSeg,[x,e,t,S,C])),A.dispatchWorkgroups(O,k),A.end(),H(i,a,D,b,C,g.buckets+1,v),D.copyBufferToBuffer(C,0,w,0,_),A=b.pass(D,`seg grid scatter`),A.setPipeline(a.scatterSeg),A.setBindGroup(0,F(i,a.scatterSeg,[x,e,t,S,C,T,E])),A.dispatchWorkgroups(O,k),A.end();for(let n=0;n<m;n+=pe){let r=Math.min(pe,m-n),o=/* @__PURE__ */ new ArrayBuffer(32);new Float32Array(o,0,1)[0]=u,new Uint32Array(o,4,4).set([n,r,0,+!!l]);let s=N(i,o,`seg radius params`);v.push(s),A=b.pass(D,`seg radius`),A.setPipeline(a.radiusSeg),A.setBindGroup(0,F(i,a.radiusSeg,[x,s,T,E,w,t,e,p,d])),A.dispatchWorkgroups(...P(r)),A.end()}let L=I(i,D,p,m*4);y.push(L),b.resolve(D);let R=performance.now(),z=R-r;i.queue.submit([D.finish()]),await L.mapAsync(GPUMapMode.READ);let B=performance.now()-R,V=new Uint32Array(L.getMappedRange().slice(0,m*4));L.unmap();let te=await b.read(),U=performance.now(),W=e=>({totalMs:performance.now()-r,gpuWallMs:B,passesMs:te,jobs:o,points:s,queries:m,prepMs:z,finishMs:e});if(l){let e=Math.trunc(n.minNeighbors??0)+1,t=new Uint8Array(m);for(let n=0;n<m;n++)t[n]=+(V[n]>=e);return{keptMask:t,timings:W(performance.now()-U)}}let ne=Math.PI*c*c,G=new Float32Array(m);for(let e=0;e<m;e++)G[e]=V[e]/ne;return{scalarField:G,timings:W(performance.now()-U)}}finally{for(let e of v)e.destroy();for(let e of y)e.destroy();b.destroy()}}function _e(e){return Math.max(16384,Math.floor((1<<18)*16/Math.max(16,e+1)))}function ve(e,t,n){let r=z(e),i=Math.max(1e-12,(r.max[0]-r.min[0])*(r.max[1]-r.min[1]));return .5*Math.sqrt((n+1)*i/(Math.PI*Math.max(1,t)))}function ye(e,t,n,r){let i=[V(e,t,n,!1,r)],a=Math.max(t*4,i[0].cell*2);for(;Math.max(...i[i.length-1].dims)>7&&i.length<=16;){let t=V(e,a,n,!1,r);i.push(t),a=Math.max(a*4,t.cell*2)}return i}async function be(e,n,r,i,a,o){let s=i===`normals`?e.kernels.normals:e.kernels.knnDists;if(!s)throw Error(`GPU kNN unavailable: ${e.kernels.normalsError}`);let c=n.length/3,{device:l,kernels:u}=e,d=ye(z(n),ve(n,c,i===`normals`?r:r-1),c,e.maxBuckets),f=(Math.max(...d.map(e=>e.buckets))+1)*4;t([{label:`tile points`,bytes:c*12,bound:!0},{label:`knn out`,bytes:c*o*4,bound:!0},{label:`knn flags`,bytes:c*4,bound:!0},{label:`sorted points`,bytes:c*16,bound:!0},{label:`point key`,bytes:c*4,bound:!0},{label:`bucket counter`,bytes:f,bound:!0},{label:`bucket starts`,bytes:f,bound:!0}],e.limits);let p=[],m=[],h=[],g=new L(l,e.timestamps),_=[d[0]];try{let e=j(l,n,`tile points`),t=M(l,c*o*4,`knn out`),f=M(l,c*4,`knn flags`);p.push(e,t,f);let v=performance.now(),y=_e(r),b=0,x=(e,n)=>[e.uniform,n,e.sorted,e.sortedKey,e.starts,t,f],S=async(t,n)=>{let o=[],d=l.createCommandEncoder(),_=U(l,u,d,g,e,t,o);h.push(_),l.queue.submit([d.finish()]);try{for(let e=0;e<c;e+=y){let t=Math.min(y,c-e),o=/* @__PURE__ */ new ArrayBuffer(48);new Uint32Array(o,0,4).set([r,e,t,0]),new Int32Array(o,16,1)[0]=6,new Float32Array(o,20,2).set([1/3,Math.sqrt(3)]),new Uint32Array(o,28,2).set([+!!a,n]);let u=N(l,o,`knn params`);p.push(u),d=l.createCommandEncoder();let f=g.pass(d,n===0?i:`${i} retry ${n}`);f.setPipeline(s),f.setBindGroup(0,F(l,s,x(_,u))),f.dispatchWorkgroups(...P(t,64)),f.end(),l.queue.submit([d.finish()])}d=l.createCommandEncoder();let e=I(l,d,f,c*4);m.push(e),l.queue.submit([d.finish()]),await e.mapAsync(GPUMapMode.READ);let t=new Uint32Array(e.getMappedRange().slice(0,c*4));return e.unmap(),t}finally{_.destroy(),h.splice(h.indexOf(_),1);for(let e of o)e.destroy()}},C=await S(d[0],0);for(let e=1;e<d.length&&C.includes(2);e++)e===2&&(b=C.reduce((e,t)=>e+ +(t===2),0)),_.push(d[e]),C=await S(d[e],e);if(C.includes(2))throw Error(`GPU kNN: queries left unsettled on the coarsest grid`);let w=l.createCommandEncoder(),T=I(l,w,t,c*o*4);m.push(T),g.resolve(w),l.queue.submit([w.finish()]),await T.mapAsync(GPUMapMode.READ);let E=performance.now()-v,D=new Float32Array(T.getMappedRange().slice(0,c*o*4));T.unmap();let O=await g.read();return{raw:D,flags:C,gpuWallMs:E,passesMs:O,usedPlans:_,unbounded:b,levels:_.length-1}}finally{for(let e of p)e.destroy();for(let e of m)e.destroy();for(let e of h)e.destroy();g.destroy()}}async function xe(e,t,n,r,i=!1){if(!(Number.isInteger(n)&&n>=3&&n<=100))throw Error(`kSearch must be an integer in 3..100.`);let a=t.length/3;if(a===0)return{normals:/* @__PURE__ */ new Float32Array,timings:{totalMs:0,gpuWallMs:0,passesMs:null,grids:[],ties:0,tieMs:0,tiePartitioned:0,unbounded:0,levels:0}};let s=await be(e,t,n,`normals`,i,3),c=s.raw,l=performance.now(),u=null,d=0;if(i)for(let e=0;e<a;e++){if(s.flags[e]!==1)continue;d++,u??=new o(t);let r=u.knn(t[e*3],t[e*3+1],t[e*3+2],n);c.set(g(t,r,e),e*3)}return{normals:c,timings:{totalMs:performance.now()-r,gpuWallMs:s.gpuWallMs,passesMs:s.passesMs,grids:s.usedPlans,ties:d,tieMs:performance.now()-l,tiePartitioned:u?.partitioned??0,unbounded:s.unbounded,levels:s.levels}}}async function Se(e,t,n){if(!(Number.isInteger(n)&&n>=1&&n<=101))throw Error(`k must be an integer in 1..101.`);if(t.length===0)return{dists:/* @__PURE__ */ new Float32Array,gpuWallMs:0,passesMs:null,grids:[]};let r=await be(e,t,n,`dists`,!1,n);return{dists:r.raw,gpuWallMs:r.gpuWallMs,passesMs:r.passesMs,grids:r.usedPlans}}let J=1<<18;var Ce=class r{ctx;kernels;limits;lost=null;tail=Promise.resolve();lostListeners=[];constructor(e,t,n){this.ctx=e,this.kernels=t,this.limits=n,e.device.lost.then(e=>{if(this.lost??=`${e.reason}: ${e.message}`,e.reason!==`destroyed`){console.warn(`[gpu] device lost, using WASM`,e.message);for(let e of this.lostListeners)e(this.lost)}}),e.device.onuncapturederror=e=>console.warn(`[gpu] uncaptured error`,e.error.message)}static async create(e){let t=await A();if(!t)return null;let n={maxStorageBufferBindingSize:Math.min(t.maxStorageBufferBindingSize,e?.maxStorageBufferBindingSize??1/0),maxBufferSize:Math.min(t.maxBufferSize,e?.maxBufferSize??1/0)};try{return new r(t,await le(t.device),n)}catch(e){return console.warn(`[gpu] kernels failed to compile, using WASM`,e),t.device.destroy(),null}}get adapterInfo(){return this.ctx.adapterInfo}get timestamps(){return this.ctx.timestamps}get normalsError(){return this.kernels.normalsError}onLost(e){this.lostListeners.push(e)}get usable(){return this.lost===null}run(e,t,n,r){let i=this.tail.then(()=>this.runNow(e,r));return this.tail=i.catch(()=>void 0),i}runBatch(e){let t=this.tail.then(()=>this.scoped(()=>ge(this.env,e,performance.now())));return this.tail=t.catch(()=>void 0),t}async scoped(t){if(this.lost)throw new e(`GPU device lost (${this.lost})`,`device-lost`);try{return await n(this.ctx.device,t)}catch(t){throw this.lost&&!(t instanceof e)?new e(`GPU device lost (${this.lost})`,`device-lost`):t}}async runNow(e,t){let n=performance.now(),r=e.length/3;return this.scoped(()=>{switch(t.op){case`density`:return this.density(e,r,t.radius,n);case`radiusOutlier`:return this.radiusOutlier(e,r,t,n);case`normals`:return this.normals(e,t.kSearch,n,t.cpuTies??!1);case`sor`:return this.sor(e,r,t,n);default:return this.c2c(e,r,t,n)}})}destroy(){this.lost??=`destroyed`,this.ctx.device.destroy()}maxBuckets(){return Math.floor(Math.min(this.limits.maxStorageBufferBindingSize,this.limits.maxBufferSize)/4)-1}async normals(e,t,n,r){return xe(this.env,e,t,n,r)}get env(){return{device:this.ctx.device,kernels:this.kernels,limits:this.limits,timestamps:this.ctx.timestamps,maxBuckets:this.maxBuckets()}}async radiusOutlier(e,t,n,r){let i=await this.density(e,t,n.radius,r,!0),a=Math.trunc(n.minNeighbors)+1,o=i.counts,s=new Uint8Array(t);for(let e=0;e<t;e++)s[e]=+(o[e]>=a);return{keptMask:s,counts:o,timings:{...i.timings,totalMs:performance.now()-r}}}async sor(e,t,n,r){let i=Math.trunc(n.meanK)+1,a=await Se(this.env,e,i);return{keptMask:fe(a.dists,t,i,n.stddevMul),timings:{totalMs:performance.now()-r,gpuWallMs:a.gpuWallMs,passesMs:a.passesMs,grids:a.grids}}}async density(e,n,r,i,a=!1){if(!(r>0))throw Error(`Radius must be > 0.`);let{device:o}=this.ctx,s=Math.fround(r),c=a?de(r):Math.fround(s*s),l=V(z(e),s*1.02,n,!1,this.maxBuckets());t([{label:`tile points`,bytes:n*12,bound:!0},{label:`density counts`,bytes:n*4,bound:!0},...we(l)],this.limits);let u=[],d=[],f=[],p=new L(o,this.ctx.timestamps);try{let t=j(o,e,`tile points`),r=M(o,n*4,`density counts`);u.push(t,r);let m=o.createCommandEncoder(),h=U(o,this.kernels,m,p,t,l,u);f.push(h);for(let e=0;e<n;e+=J){let t=Math.min(J,n-e),i=/* @__PURE__ */ new ArrayBuffer(32);new Float32Array(i,0,1)[0]=c,new Uint32Array(i,4,4).set([e,t,0,+!!a]);let s=N(o,i,`density params`);u.push(s);let l=p.pass(m,`density`);l.setPipeline(this.kernels.density),l.setBindGroup(0,F(o,this.kernels.density,[h.uniform,s,h.sorted,h.sortedKey,h.starts,r])),l.dispatchWorkgroups(...P(t)),l.end()}let g=I(o,m,r,n*4);d.push(g),p.resolve(m);let _=performance.now();o.queue.submit([m.finish()]),await g.mapAsync(GPUMapMode.READ);let v=performance.now()-_,y=new Uint32Array(g.getMappedRange().slice(0,n*4));g.unmap();let b=await p.read(),x=Math.PI*s*s,S=new Float32Array(n);for(let e=0;e<n;e++)S[e]=y[e]/x;return{scalarField:S,counts:y,timings:{totalMs:performance.now()-i,gpuWallMs:v,passesMs:b,grids:[l]}}}finally{Te(u,d,f,p)}}async c2c(e,n,r,i){let a=Math.fround(r.maxDistance);if(!(a>0))throw Error(`maxDistance must be > 0.`);let s=r.mode===1,c=r.reference,l=c.length/3,u=new Float32Array(n).fill(s?0:a);if(l===0||n===0)return{scalarField:u,timings:{totalMs:performance.now()-i,gpuWallMs:0,passesMs:null,grids:[]}};let{device:d}=this.ctx,f=z(c),p=V(f,Math.min(te(f,l),a),l,s,this.maxBuckets()),m=Math.max(p.cell*8,a/8),h=p.cell*2<a,g=h?V(f,m,l,s,this.maxBuckets()):null;t([{label:`reference points`,bytes:l*12,bound:!0},{label:`query points`,bytes:n*12,bound:!0},{label:`c2c best`,bytes:n*8,bound:!0},...we(p),...g?we(g):[]],this.limits);let _=[],v=[],y=[],b=new L(d,this.ctx.timestamps);try{let t=j(d,c,`reference points`),r=j(d,e,`query points`),l=M(d,n*8,`c2c best`);_.push(t,r,l);let f=d.createCommandEncoder(),m=U(d,this.kernels,f,b,t,p,_);y.push(m);let x=g?U(d,this.kernels,f,b,t,g,_):m;g&&y.push(x);let S=h?2:Math.ceil(a/p.cell)+1,C=g?Math.ceil(a/g.cell)+1:0;for(let e=0;e<n;e+=J){let t=Math.min(J,n-e),i=/* @__PURE__ */ new ArrayBuffer(32);new Float32Array(i,0,1)[0]=a,new Uint32Array(i,4,3).set([+!!s,e,t]),new Int32Array(i,16,2).set([S,C]);let o=N(d,i,`c2c params`);_.push(o);let c=b.pass(f,`c2c`);c.setPipeline(this.kernels.c2c),c.setBindGroup(0,F(d,this.kernels.c2c,[m.uniform,x.uniform,o,m.sorted,m.sortedKey,m.starts,x.sorted,x.sortedKey,x.starts,r,l])),c.dispatchWorkgroups(...P(t)),c.end()}let w=I(d,f,l,n*8);v.push(w),b.resolve(f);let T=performance.now();d.queue.submit([f.finish()]),await w.mapAsync(GPUMapMode.READ);let E=performance.now()-T,D=new Uint32Array(w.getMappedRange().slice(0,n*8));w.unmap();let O=await b.read(),k=new Float32Array(D.buffer),A=performance.now(),L=null,R=0;for(let t=0;t<n;t++){let n=!!(D[t*2]&2147483648);D[t*2]&=2147483647;let r=k[t*2];if(r>=3e38)continue;let i=Math.fround(Math.sqrt(r));if(s){if(i>a)continue;let r=k[t*2+1];n&&(R++,L??=new o(c,!0),r=c[L.nearest(e[t*3],e[t*3+1],0).index*3+2]),u[t]=e[t*3+2]-r}else u[t]=Math.min(i,a)}let z=performance.now()-A,B=g?[p,g]:[p];return{scalarField:u,timings:{totalMs:performance.now()-i,gpuWallMs:E,passesMs:O,grids:B,ties:R,tieMs:z,tiePartitioned:L?.partitioned??0}}}finally{Te(_,v,y,b)}}};function we(e){let t=(e.buckets+1)*4;return[{label:`sorted points`,bytes:e.n*16,bound:!0},{label:`point key`,bytes:e.n*4,bound:!0},{label:`bucket counter`,bytes:t,bound:!0},{label:`bucket starts`,bytes:t,bound:!0}]}function Te(e,t,n,r){for(let t of e)t.destroy();for(let e of t)e.destroy();for(let e of n)e.destroy();r.destroy()}let Y=null;function X(e,t=[]){self.postMessage(e,t)}async function Ee(e,t){let n=performance.now();try{if(Y=await Ce.create(t.cap),!Y){X({type:`ready`,id:e,ok:!1,error:`no WebGPU adapter in the worker`});return}await T(Y);let r=await D(Y).catch(e=>e instanceof Error?e.message:String(e));Y.onLost(e=>X({type:`lost`,message:e})),X({type:`ready`,id:e,ok:!0,adapter:Y.adapterInfo,timestamps:Y.timestamps,selfTestMs:performance.now()-n,normalsError:r})}catch(t){Y?.destroy(),Y=null,X({type:`ready`,id:e,ok:!1,error:t instanceof Error?t.message:String(t)})}}function De(e){switch(e.op){case`density`:return{op:`density`,radius:e.radius};case`normals`:return{op:`normals`,kSearch:e.kSearch,cpuTies:e.cpuTies};case`radiusOutlier`:return{op:`radiusOutlier`,radius:e.radius,minNeighbors:e.minNeighbors};case`sor`:return{op:`sor`,meanK:e.meanK,stddevMul:e.stddevMul};default:return{op:`c2c`,maxDistance:e.maxDistance,mode:e.mode,reference:e.reference}}}let Z=[],Q=/* @__PURE__ */ new Set,Oe=!1,ke=null;function $(e){Q.delete(e),X({type:`error`,id:e,reason:`cancelled`,message:`tile cancelled`})}async function Ae(){if(!Oe){Oe=!0;try{for(let e=Z.shift();e;e=Z.shift())Q.has(e.id)?$(e.id):(ke=e.id,await je(e),ke=null)}finally{Oe=!1}}}async function je(t){let{id:n,job:r}=t;if(!Y)X({type:`error`,id:n,reason:`not-ready`,message:`GPU backend not initialised`});else{X({type:`started`,id:n});try{if(r.op===`batch`){let{inner:e,radius:t,minNeighbors:i,positions:a,segStarts:o,cores:s}=r,c=await Y.runBatch({op:e,radius:t,minNeighbors:i,positions:a,segStarts:o,cores:s});if(Q.has(n)){$(n);return}let{totalMs:l,gpuWallMs:u,passesMs:d,prepMs:f,finishMs:p}=c.timings,m={totalMs:l,gpuWallMs:u,passesMs:d,prepMs:f,finishMs:p};c.keptMask?X({type:`result`,id:n,keptMask:c.keptMask,timings:m},[c.keptMask.buffer]):c.scalarField&&X({type:`result`,id:n,scalarField:c.scalarField,timings:m},[c.scalarField.buffer]);return}let e=await Y.run(r.positions,{core:{x0:0,y0:0,x1:0,y1:0},nodes:[]},0,De(r));if(Q.has(n)){$(n);return}let{totalMs:t,gpuWallMs:i,passesMs:a,ties:o,tieMs:s,unbounded:c,levels:l}=e.timings,u={totalMs:t,gpuWallMs:i,passesMs:a,ties:o,tieMs:s,unbounded:c,levels:l};if(e.keptMask){let t=e.keptMask;X({type:`result`,id:n,keptMask:t,timings:u},[t.buffer]);return}if(r.op===`normals`){let t=e.normals??new Float32Array(r.positions.length);X({type:`result`,id:n,normals:t,timings:u},[t.buffer]);return}let d=e.scalarField??new Float32Array(r.positions.length/3);X({type:`result`,id:n,scalarField:d,timings:u},[d.buffer])}catch(t){Q.delete(n),X({type:`error`,id:n,reason:t instanceof e?t.reason:`error`,message:t instanceof Error?t.message:String(t)})}}}self.onmessage=e=>{let t=e.data;t.type===`init`?Ee(t.id,t):t.type===`cancel`?(ke===t.id||Z.some(e=>e.id===t.id))&&Q.add(t.id):(Z.push(t),Ae())}})();