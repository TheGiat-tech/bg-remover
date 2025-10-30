
/* Zero-control auto-run pipeline
   - exposes window.runAuto(file)
   - auto-processes file on drop/select
   - lightweight utilities: otsuThreshold, adaptiveFeather, deSpill
*/
import { ModelWorkerClient } from './modelWorkerClient.js';

// Ensure ORT loads WASM from the local vendor folder when available
if (typeof ort !== 'undefined'){
  ort.env = ort.env || {};
  ort.env.wasm = ort.env.wasm || {};
  ort.env.wasm.wasmPaths = '/js/vendor/ort/';
}

// prefer the lightweight U2‑NetP shipped with the site (fast, client-friendly)
const modelPath = '/models/u2netp.onnx';

const els = {
  drop: document.getElementById('dropzone'),
  file: document.getElementById('fileInput'),
  displayCanvas: document.getElementById('displayCanvas'),
  controls: document.getElementById('controls'),
  resultPreview: document.getElementById('resultPreview'),
  download: document.getElementById('downloadBtn'),
  replace: document.getElementById('replaceBtn'),
  spinner: document.getElementById('spinner'),
  useFallback: document.getElementById('useFallback')
};

let session = null; // ONNX session placeholder (worker handles inference)
let mw = null; // ModelWorkerClient
let workerReady = false;
let workerEventsBound = false;
let currentFile = null;

// --- Status / Log API --------------------------------------------------
const statusEls = {
  bar: document.getElementById('statusbar'),
  text: document.getElementById('status-text'),
  progressFill: document.getElementById('progress-fill'),
  toggleLog: document.getElementById('toggle-log'),
  logPanel: document.getElementById('log-panel')
};
let logLines = [];
function timeNow(){ const d=new Date(); return d.toLocaleTimeString(); }
function appendLog(msg){ const ts = timeNow(); const line = `${ts} ${msg}`; logLines.push(line); if (logLines.length>200) logLines.shift(); if (statusEls.logPanel){ const div=document.createElement('div'); div.className='log-line'; div.textContent=line; statusEls.logPanel.appendChild(div); statusEls.logPanel.scrollTop = statusEls.logPanel.scrollHeight; }
  try{ console.log(line); }catch(_){}
}
function setProgress(pct){ pct = Math.max(0, Math.min(100, Math.round(pct))); if (statusEls.progressFill) statusEls.progressFill.style.width = pct+'%'; }
function setStatus(txt){ if (statusEls.text) statusEls.text.textContent = txt; }
function addStatus(step){ appendLog(step); setStatus(step); }

// log panel toggle
if (statusEls.toggleLog){ statusEls.toggleLog.addEventListener('click', ()=>{ const lp=statusEls.logPanel; if (!lp) return; const isHidden = lp.hasAttribute('hidden'); if (isHidden){ lp.removeAttribute('hidden'); statusEls.toggleLog.setAttribute('aria-expanded','true'); statusEls.toggleLog.textContent = 'Hide Logs'; } else { lp.setAttribute('hidden',''); statusEls.toggleLog.setAttribute('aria-expanded','false'); statusEls.toggleLog.textContent = 'Logs'; } }); }

function showSpinner(){ if (els.spinner) els.spinner.style.display = 'block'; }
function hideSpinner(){ if (els.spinner) els.spinner.style.display = 'none'; }
function showToast(msg, timeout=2000){
  let t = document.querySelector('.toast');
  if (!t){ t = document.createElement('div'); t.className='toast'; document.body.appendChild(t); }
  t.textContent = msg; t.classList.add('show'); setTimeout(()=>t.classList.remove('show'), timeout);
}

async function loadModelOnce(){
  if (session && workerReady) return session;
  if (!mw) mw = new ModelWorkerClient();
  if (!workerEventsBound){
    mw.on('log', (m)=>{ appendLog(m); });
    mw.on('load:start', ({name})=>{ addStatus('Loading model…'); setProgress(10); appendLog('⏳ loading '+name); });
    mw.on('load:done', ({name})=>{ appendLog('✅ model from IndexedDB or cache: '+name); setProgress(40); });
    mw.on('load:error', ({name, error})=>{ appendLog('❌ model load failed: '+name+' '+ (error && error.message ? error.message : error)); setStatus('Error ⚠️'); setProgress(0); });
    mw.on('infer:start', ()=>{ setStatus('Removing background…'); setProgress(90); appendLog('▶️ inference started'); });
    mw.on('infer:done', ()=>{ appendLog('✅ inference finished'); setProgress(95); });
    mw.on('infer:error', ({error})=>{ appendLog('❌ inference error: '+(error && error.message? error.message : error)); setStatus('Error ⚠️'); setProgress(0); });
    workerEventsBound = true;
  }
  addStatus('Ensuring U2‑NetP model…'); setProgress(15);
  const bytes = await ensureU2NetpModel();
  try{
    if (!session){
      if (typeof ort !== 'undefined' && ort.env && ort.env.wasm){ ort.env.wasm.wasmPaths = '/js/vendor/ort/'; }
      const sess = await ort.InferenceSession.create(bytes, { executionProviders: ['webgl','wasm'] });
      window.segModel = sess;
      session = sess;
      appendLog('🧠 session ready (u2netp)');
    }
    if (!workerReady){
      await mw.load({ name: 'u2netp.onnx', url: modelPath });
      workerReady = true;
      appendLog('🤖 worker ready (u2netp)');
    }
    setProgress(70);
    return session;
  }catch(err){
    appendLog('❌ Failed to initialise segmentation: '+(err && err.message? err.message : err));
    session = null;
    workerReady = false;
    throw err;
  }
}

// Ensure u2netp.onnx is available locally (/models/u2netp.onnx) or cached in IndexedDB
async function ensureU2NetpModel(){
  try{
    const response = await fetch(modelPath, { cache: 'force-cache' });
    if (!response.ok) throw new Error('Missing local model');
    appendLog('✅ Model found locally.');
    const arrayBuffer = await response.arrayBuffer();
    return new Uint8Array(arrayBuffer);
  }catch(err){
    // If local model missing, rethrow — we prefer not to auto-download in this local-first setup
    appendLog('❌ local model not found: '+(err && err.message? err.message : err));
    throw err;
  }
}

// --- Dynamic MediaPipe loader for optional fallback -------------------
async function ensureMediaPipeLoaded(){
  if (window.SelfieSegmentation) return;
  return new Promise((resolve, reject)=>{
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/selfie_segmentation.js';
    script.async = true;
    script.onload = ()=>resolve();
    script.onerror = (e)=>reject(e);
    document.head.appendChild(script);
  });
}

function bilinearResizeGray(src, srcW, srcH, dstW, dstH){
  const out = new Float32Array(dstW*dstH);
  for (let y=0;y<dstH;y++){ const gy = (y*(srcH-1))/(dstH-1); const y0=Math.floor(gy), y1=Math.min(y0+1, srcH-1); const wy=gy-y0;
    for (let x=0;x<dstW;x++){ const gx=(x*(srcW-1))/(dstW-1); const x0=Math.floor(gx), x1=Math.min(x0+1, srcW-1); const wx=gx-x0;
      const i00=y0*srcW+x0, i01=y0*srcW+x1, i10=y1*srcW+x0, i11=y1*srcW+x1;
      out[y*dstW+x] = (1-wy)*((1-wx)*src[i00]+wx*src[i01]) + wy*((1-wx)*src[i10]+wx*src[i11]);
  }}
  return out;
}

// Guided filter helpers (fast JS implementation for gray guidance)
function _integralImage(src, w, h){
  const S = new Float64Array((w+1)*(h+1));
  for (let y=0;y<h;y++){
    let rowSum = 0;
    for (let x=0;x<w;x++){
      rowSum += src[y*w + x];
      S[(y+1)*(w+1) + (x+1)] = S[y*(w+1) + (x+1)] + rowSum;
    }
  }
  return S;
}
function _boxSum(S, x0,y0,x1,y1, w){
  const W = w+1;
  return S[(y1+1)*W + (x1+1)] - S[(y0)*W + (x1+1)] - S[(y1+1)*W + (x0)] + S[(y0)*W + (x0)];
}
function _boxFilter(arr, w, h, r){
  const S = _integralImage(arr, w, h);
  const out = new Float32Array(w*h);
  for (let y=0;y<h;y++){
    const y0 = Math.max(0, y - r), y1 = Math.min(h-1, y + r);
    for (let x=0;x<w;x++){
      const x0 = Math.max(0, x - r), x1 = Math.min(w-1, x + r);
      const area = (y1 - y0 + 1) * (x1 - x0 + 1);
      out[y*w + x] = _boxSum(S, x0,y0,x1,y1, w) / area;
    }
  }
  return out;
}
function guidedFilterGray(rgbData, pFloat32, w, h, r=8, eps=1e-3){
  // rgbData: Uint8ClampedArray (RGBA) length w*h*4
  const N = w*h;
  const I = new Float32Array(N);
  for (let i=0, j=0;i<N;i++, j+=4){
    I[i] = (0.299*rgbData[j] + 0.587*rgbData[j+1] + 0.114*rgbData[j+2]) / 255.0;
  }
  // normalize p to 0..1
  const p = new Float32Array(N);
  let pmin=Infinity, pmax=-Infinity;
  for (let i=0;i<N;i++){ const v=pFloat32[i]; if (v<pmin) pmin=v; if (v>pmax) pmax=v; p[i]=v; }
  const prange = (pmax - pmin) || 1;
  for (let i=0;i<N;i++) p[i] = (p[i] - pmin) / prange;

  const mean_I = _boxFilter(I, w, h, r);
  const mean_p = _boxFilter(p, w, h, r);

  const Ip = new Float32Array(N);
  const II = new Float32Array(N);
  for (let i=0;i<N;i++){ Ip[i] = I[i]*p[i]; II[i] = I[i]*I[i]; }
  const mean_Ip = _boxFilter(Ip, w, h, r);
  const mean_II = _boxFilter(II, w, h, r);

  const a = new Float32Array(N);
  const b = new Float32Array(N);
  for (let i=0;i<N;i++){
    const cov_Ip = mean_Ip[i] - mean_I[i]*mean_p[i];
    const var_I = mean_II[i] - mean_I[i]*mean_I[i];
    a[i] = cov_Ip / (var_I + eps);
    b[i] = mean_p[i] - a[i]*mean_I[i];
  }
  const mean_a = _boxFilter(a, w, h, r);
  const mean_b = _boxFilter(b, w, h, r);

  const q = new Float32Array(N);
  for (let i=0;i<N;i++) q[i] = mean_a[i]*I[i] + mean_b[i];

  const out = new Float32Array(N);
  for (let i=0;i<N;i++) out[i] = q[i]*prange + pmin;
  return out;
}

function makeInputTensorFromImageBitmap(imgBitmap, target=320){
  const tmp = document.createElement('canvas'); tmp.width=target; tmp.height=target; const ctx=tmp.getContext('2d'); ctx.drawImage(imgBitmap,0,0,target,target);
  const id = ctx.getImageData(0,0,target,target).data; const data = new Float32Array(3*target*target);
  for (let i=0;i<target*target;i++){ const r=id[i*4]/255,g=id[i*4+1]/255,b=id[i*4+2]/255; data[i]=r; data[i+target*target]=g; data[i+2*target*target]=b; }
  return { data, dims: [1,3,target,target] };
}

async function runU2NetOnBitmap(imgBitmap){
  // delegate to worker
  if (!mw || !workerReady) throw new Error('model worker not initialized');
  setStatus('Removing background…'); setProgress(80);
  const res = await mw.infer(imgBitmap);
  // res contains maskBuffer (ArrayBuffer) for 320x320
  const buf = res.maskBuffer;
  const arr = new Uint8ClampedArray(buf);
  const out = new Float32Array(arr.length);
  for (let i=0;i<arr.length;i++) out[i] = arr[i];
  setProgress(100);
  return out;
}

async function selfieSegFallbackBitmap(imgBitmap){
  await ensureMediaPipeLoaded();
  return new Promise((resolve)=>{
    const ss = new SelfieSegmentation({locateFile:(f)=>`https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${f}`});
    ss.setOptions({ modelSelection: 1 });
    ss.onResults((res)=>{
      const mask = res.segmentationMask; const tmp = document.createElement('canvas'); tmp.width=mask.width; tmp.height=mask.height; const g=tmp.getContext('2d'); g.drawImage(mask,0,0);
      const d = g.getImageData(0,0,tmp.width,tmp.height).data; const out = new Float32Array(tmp.width*tmp.height); for (let i=0;i<out.length;i++) out[i]=d[i*4];
      resolve(bilinearResizeGray(out,tmp.width,tmp.height,320,320));
    });
    const c = document.createElement('canvas'); c.width=imgBitmap.width; c.height=imgBitmap.height; c.getContext('2d').drawImage(imgBitmap,0,0);
    const frame = new Image(); frame.onload = ()=> ss.send({image: frame}); frame.src = c.toDataURL('image/png');
  });
}

// utils: Otsu threshold on Uint8ClampedArray or Float32Array
function otsuThreshold(arr){
  // arr values 0-255
  const hist = new Uint32Array(256); let total=0;
  for (let i=0;i<arr.length;i++){ const v=Math.max(0,Math.min(255,Math.round(arr[i]))); hist[v]++; total++; }
  let sum=0; for (let t=0;t<256;t++) sum += t*hist[t];
  let sumB=0, wB=0, wF=0, varMax=0, threshold=0;
  for (let t=0;t<256;t++){
    wB += hist[t]; if (wB===0) continue; wF = total - wB; if (wF===0) break;
    sumB += t*hist[t]; const mB = sumB / wB; const mF = (sum - sumB) / wF; const varBetween = wB * wF * (mB - mF) * (mB - mF);
    if (varBetween > varMax){ varMax = varBetween; threshold = t; }
  }
  return threshold;
}

function percentileThreshold(arr, p=0.9){
  // compute p-th percentile (0-1)
  const copy = new Uint8Array(arr.length); for (let i=0;i<arr.length;i++) copy[i]=Math.max(0,Math.min(255,Math.round(arr[i])));
  const hist = new Uint32Array(256); for (let i=0;i<copy.length;i++) hist[copy[i]]++;
  const target = Math.round(copy.length * p); let cum=0; for (let t=0;t<256;t++){ cum += hist[t]; if (cum>=target) return t; } return 255;
}

function computeEdgeGradient(alpha, w, h){
  // Sobel-like gradient on alpha (0-255)
  const gx = new Float32Array(w*h); const gy = new Float32Array(w*h); const mag = new Float32Array(w*h);
  for (let y=1;y<h-1;y++) for (let x=1;x<w-1;x++){ const i=y*w+x;
    const a00=alpha[(y-1)*w + (x-1)], a01=alpha[(y-1)*w + x], a02=alpha[(y-1)*w + (x+1)];
    const a10=alpha[y*w + (x-1)], a11=alpha[y*w + x], a12=alpha[y*w + (x+1)];
    const a20=alpha[(y+1)*w + (x-1)], a21=alpha[(y+1)*w + x], a22=alpha[(y+1)*w + (x+1)];
    const gxv = -a00 - 2*a10 - a20 + a02 + 2*a12 + a22;
    const gyv = -a00 - 2*a01 - a02 + a20 + 2*a21 + a22;
    gx[i]=gxv; gy[i]=gyv; mag[i]=Math.hypot(gxv,gyv);
  }
  return mag;
}

function adaptiveFeather(alpha, w, h){
  const mag = computeEdgeGradient(alpha,w,h);
  // compute 90th percentile of mag
  const vals = []; for (let i=0;i<mag.length;i++) vals.push(Math.round(mag[i])); vals.sort((a,b)=>a-b);
  const p90 = vals[Math.floor(vals.length*0.9)] || 0;
  // map p90 to feather radius 0..3
  const maxMag = vals[vals.length-1] || 1;
  const v = Math.min(1, p90/(maxMag||1));
  return Math.round((1 - v) * 3); // stronger edges -> smaller feather
}

function boxBlurAlpha(alpha, w, h, r){
  if (!r) return new Uint8ClampedArray(alpha);
  const out = new Uint8ClampedArray(w*h);
  for (let y=0;y<h;y++) for (let x=0;x<w;x++){ let sum=0,cnt=0; for (let dy=-r;dy<=r;dy++) for (let dx=-r;dx<=r;dx++){ const sx=x+dx, sy=y+dy; if (sx<0||sy<0||sx>=w||sy>=h) continue; sum+=alpha[sy*w+sx]; cnt++; } out[y*w+x]=cnt?Math.round(sum/cnt):alpha[y*w+x]; }
  return out;
}

function antiAliasAlpha(alpha, w, h, maxDist=3){
  // simple distance-based anti-alias around edges
  const isFg = new Uint8Array(w*h); for (let i=0;i<w*h;i++) isFg[i]= alpha[i]>127?1:0;
  const dist = new Uint32Array(w*h); const inf=1e8; for (let i=0;i<w*h;i++) dist[i]=isFg[i]?0:inf;
  for (let y=0;y<h;y++) for (let x=0;x<w;x++){ const i=y*w+x; if (x>0) dist[i]=Math.min(dist[i], dist[i-1]+1); if (y>0) dist[i]=Math.min(dist[i], dist[i-w]+1); }
  for (let y=h-1;y>=0;y--) for (let x=w-1;x>=0;x--){ const i=y*w+x; if (x<w-1) dist[i]=Math.min(dist[i], dist[i+1]+1); if (y<h-1) dist[i]=Math.min(dist[i], dist[i+w]+1); }
  const out=new Uint8ClampedArray(w*h);
  for (let i=0;i<w*h;i++){ if (isFg[i]) out[i]=255; else if (dist[i]>maxDist) out[i]=0; else out[i]=Math.round(255*(1 - dist[i]/(maxDist+1))); }
  return out;
}

function deSpill(canvas, alpha){
  // mild de-spill: sample background around contour and desaturate pixels near contour
  const w=canvas.width, h=canvas.height; const ctx=canvas.getContext('2d'); const img=ctx.getImageData(0,0,w,h); const data=img.data;
  // find contour pixels
  const N=w*h; const contour = new Uint8Array(N);
  for (let y=1;y<h-1;y++) for (let x=1;x<w-1;x++){ const i=y*w+x; if (alpha[i]>200) continue; let hasFg=false; for (let dy=-1;dy<=1;dy++) for (let dx=-1;dx<=1;dx++){ if (dy===0&&dx===0) continue; if (alpha[(y+dy)*w + (x+dx)]>200) hasFg=true; } if (hasFg) contour[i]=1; }
  // process contour: desaturate towards complementary of local bg average
  for (let i=0;i<N;i++){
    if (!contour[i]) continue;
    const x = i % w, y = Math.floor(i / w);
    // sample small ring outside
    let sr=0,sg=0,sb=0,cnt=0; const radius=3;
    for (let dy=-radius;dy<=radius;dy++) for (let dx=-radius;dx<=radius;dx++){ const sx=x+dx, sy=y+dy; if (sx<0||sy<0||sx>=w||sy>=h) continue; const si=sy*w+sx; if (alpha[si]>220) continue; const gi=si*4; sr+=data[gi]; sg+=data[gi+1]; sb+=data[gi+2]; cnt++; }
    if (!cnt) continue; sr/=cnt; sg/=cnt; sb/=cnt;
    const pi=i*4; const fr=data[pi], fg=data[pi+1], fb=data[pi+2];
    // compute simple desaturation towards bg
    const a = alpha[i]/255; const desat = Math.min(0.6, (1-a)*0.6);
    // convert to hsv and reduce saturation
    const [hH,hS,hV] = rgbToHsv(fr,fg,fb); const newS = Math.max(0, hS * (1 - desat)); const [nr,ng,nb] = hsvToRgb(hH, newS, hV);
    data[pi]=nr; data[pi+1]=ng; data[pi+2]=nb;
  }
  ctx.putImageData(img,0,0);
}

function rgbToHsv(r,g,b){ r/=255; g/=255; b/=255; const max=Math.max(r,g,b), min=Math.min(r,g,b); const d=max-min; let h=0; if (d===0) h=0; else if (max===r) h = (60*((g-b)/d)+360)%360; else if (max===g) h = (60*((b-r)/d)+120)%360; else h=(60*((r-g)/d)+240)%360; const s = max===0?0:d/max; const v = max; return [h,s,v]; }
function hsvToRgb(h,s,v){ const c = v*s; const x = c*(1 - Math.abs((h/60)%2 -1)); const m=v-c; let r=0,g=0,b=0; if (0<=h&&h<60){ r=c; g=x; b=0; } else if (60<=h&&h<120){ r=x; g=c; b=0; } else if (120<=h&&h<180){ r=0; g=c; b=x; } else if (180<=h&&h<240){ r=0; g=x; b=c; } else if (240<=h&&h<300){ r=x; g=0; b=c; } else { r=c; g=0; b=x; } return [Math.round((r+m)*255), Math.round((g+m)*255), Math.round((b+m)*255)]; }

function compositeToCanvas(canvas, rgbaData, alpha, w, h){
  const ctx=canvas.getContext('2d'); const img = ctx.getImageData(0,0,canvas.width,canvas.height); // assume canvas already has image drawn
  // write corrected colors and alpha
  for (let i=0;i<w*h;i++){ const pi=i*4; img.data[pi]=rgbaData[pi]; img.data[pi+1]=rgbaData[pi+1]; img.data[pi+2]=rgbaData[pi+2]; img.data[pi+3]=alpha[i]; }
  ctx.putImageData(img,0,0);
}

// expose runAuto
window.runAuto = async function runAuto(file){
  if (!file) return; currentFile = file;
  // reset UI
  if (els.download) els.download.disabled = true;
  if (els.resultPreview) els.resultPreview.style.display='none';
  showSpinner();
  try{
    setStatus('Loading model…'); setProgress(10);
    const imgBitmap = await createImageBitmap(file);
    const canvas = els.displayCanvas; const ctx = canvas.getContext('2d');
    // scale to max 1600 width
    const maxW = Math.min(1600, imgBitmap.width); const scale = Math.min(maxW / imgBitmap.width, 1);
    const w = Math.round(imgBitmap.width * scale); const h = Math.round(imgBitmap.height * scale);
    canvas.width = w; canvas.height = h; ctx.clearRect(0,0,w,h); ctx.drawImage(imgBitmap,0,0,w,h);

    // load model
    let useFallback = false;
    if (els.useFallback && els.useFallback.checked) useFallback = true;
    if (!useFallback){ try { await loadModelOnce(); } catch(e){ console.warn('model load failed, will fallback'); appendLog('⚠️ model load failed: '+(e && e.message? e.message : e)); useFallback = true; } }

    // give a mid progress update
    appendLog('📁 image loaded, starting analysis'); setProgress(75);

    // inference
    let raw320;
    if (!useFallback && session && workerReady){ raw320 = await runU2NetOnBitmap(imgBitmap); }
    else { setStatus('Analyzing…'); setProgress(80); raw320 = await selfieSegFallbackBitmap(imgBitmap); }

    // normalize raw to 0-255
    const rawU8 = new Uint8ClampedArray(320*320);
    let min=Infinity,max=-Infinity; for (let i=0;i<raw320.length;i++){ const v=raw320[i]; if (v<min) min=v; if (v>max) max=v; }
    const range = (max-min) || 1; for (let i=0;i<raw320.length;i++) rawU8[i]=Math.max(0,Math.min(255,Math.round((raw320[i]-min)/range*255)));

    // resize to canvas
    let alphaResizedF = bilinearResizeGray(rawU8,320,320,w,h);
    // guided filter refinement at output size using canvas RGB as guidance
    try{
      const imgGuidance = ctx.getImageData(0,0,w,h).data;
      alphaResizedF = guidedFilterGray(imgGuidance, alphaResizedF, w, h, 8, 1e-3);
      appendLog('🧩 guided filter applied');
    }catch(e){
      appendLog('⚠️ guided filter failed: '+(e && e.message? e.message : e));
    }
    const alphaResized = new Uint8ClampedArray(w*h); for (let i=0;i<w*h;i++) alphaResized[i]=Math.max(0,Math.min(255,Math.round(alphaResizedF[i])));

    // threshold: try Otsu, fallback to 90th percentile on edge histogram
    let thresh = otsuThreshold(alphaResized);
    if (!thresh || thresh<5){ thresh = percentileThreshold(alphaResized, 0.9); }
    for (let i=0;i<w*h;i++) alphaResized[i] = alphaResized[i] > thresh ? alphaResized[i] : 0;

    // adaptive feather
    const feather = adaptiveFeather(alphaResized, w, h);
    const feathered = boxBlurAlpha(alphaResized, w, h, feather);

    // anti-alias
    const aa = antiAliasAlpha(feathered, w, h, Math.max(1, feather));

    // mild de-spill/color cleanup
    deSpill(canvas, aa);

    // composite: write alpha into canvas imageData
    const imgData = ctx.getImageData(0,0,w,h);
    for (let i=0;i<w*h;i++){ imgData.data[i*4+3] = aa[i]; }
    ctx.putImageData(imgData,0,0);

    // show preview and enable download
    if (els.resultPreview) els.resultPreview.style.display='block';
    if (els.download) els.download.disabled = false;
    setStatus('Done ✓'); setProgress(100);
    appendLog('✅ processing complete');
    showToast('Background removed');
  } catch (err){ console.error(err); appendLog('❌ processing error: '+(err && err.stack? err.stack : err && err.message? err.message : err)); setStatus('Error ⚠️'); setProgress(0); showToast('Processing error'); }
  finally{ hideSpinner(); }
};

// UI wiring: drag & drop, file input, replace, download, keyboard
function setupSimpleUI(){
  if (els.drop){ els.drop.addEventListener('dragover', e=>{ e.preventDefault(); els.drop.classList.add('hover'); }); els.drop.addEventListener('dragleave', ()=>els.drop.classList.remove('hover'));
    els.drop.addEventListener('drop', e=>{ e.preventDefault(); els.drop.classList.remove('hover'); const f = e.dataTransfer.files?.[0]; if (f) { if (els.file) els.file.value=''; window.runAuto(f); } }); }
  if (els.file){ els.file.addEventListener('change', ()=>{ const f = els.file.files?.[0]; if (f) window.runAuto(f); }); }
  if (els.replace){ els.replace.addEventListener('click', ()=>{ if (els.file) els.file.click(); }); }
  if (els.download){ els.download.addEventListener('click', ()=>{ const link=document.createElement('a'); link.download='background-removed.png'; link.href = els.displayCanvas.toDataURL('image/png'); link.click(); }); }
  window.addEventListener('keydown', (e)=>{ if (e.key==='d' || e.key==='D'){ if (!e.target || e.target.tagName==='BODY') { e.preventDefault(); if (!els.download.disabled) els.download.click(); } } if (e.key==='r' || e.key==='R'){ if (els.file) els.file.click(); } });
}

// initialize
setupSimpleUI();

