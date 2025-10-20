/* model.worker.js - runs inside a Web Worker (classic worker)
   Responsibilities:
   - load model from IndexedDB or fetch and cache
   - create ort.InferenceSession with provider preference (webgpu, webgl, wasm)
   - run inference on ImageBitmap and return mask buffer (Uint8ClampedArray.buffer)
*/

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js');

// IndexedDB helpers
function openDB(){
  return new Promise((resolve,reject)=>{
    const req = indexedDB.open('modelsDB', 1);
    req.onupgradeneeded = ()=>{ const db = req.result; if (!db.objectStoreNames.contains('files')) db.createObjectStore('files'); };
    req.onsuccess = ()=> resolve(req.result);
    req.onerror = ()=> reject(req.error);
  });
}

async function idbGet(name){
  const db = await openDB();
  return new Promise((resolve,reject)=>{
    const tx = db.transaction('files','readonly'); const st = tx.objectStore('files'); const r = st.get(name);
    r.onsuccess = ()=> resolve(r.result || null);
    r.onerror = ()=> reject(r.error);
  });
}
async function idbPut(name, blob){
  const db = await openDB();
  return new Promise((resolve,reject)=>{
    const tx = db.transaction('files','readwrite'); const st = tx.objectStore('files'); const r = st.put(blob, name);
    r.onsuccess = ()=> resolve(true);
    r.onerror = ()=> reject(r.error);
  });
}

let session = null;
let currentModelName = null;

async function createSessionFromBlob(blob){
  const ab = await blob.arrayBuffer();
  const bytes = new Uint8Array(ab);
  // choose providers: prefer webgpu if available
  const providers = [];
  try { if (self.navigator && self.navigator.gpu) providers.push('webgpu'); } catch(e){}
  providers.push('webgl','wasm');
  session = await ort.InferenceSession.create(bytes, { executionProviders: providers });
  return session;
}

async function tryLoadModel(name, url){
  // 1. try idb
  try{
    const cached = await idbGet(name);
    if (cached){
      postMessage({ type:'log', msg:'✅ Model from cache: '+name });
      const sess = await createSessionFromBlob(cached);
      currentModelName = name;
      return { ok:true };
    }
  }catch(e){ postMessage({ type:'log', msg:'⚠️ IDB read failed: '+(e && e.message) }); }
  // 2. fetch
  try{
    const resp = await fetch(url, { mode:'cors' });
    if (!resp.ok) throw new Error('fetch failed '+resp.status);
    const blob = await resp.blob();
    try{ await idbPut(name, blob); postMessage({ type:'log', msg:'⬇️ Downloaded & cached: '+name }); } catch(err){ postMessage({ type:'log', msg:'⚠️ Cache write failed: '+(err && err.message) }); }
    await createSessionFromBlob(blob);
    currentModelName = name;
    return { ok:true };
  }catch(err){ return { ok:false, error: err && err.message } }
}

// convert ImageBitmap -> tensor data (CHW float32) at target size
async function imageBitmapToTensor(bitmap, target=320){
  const off = new OffscreenCanvas(target, target);
  const ctx = off.getContext('2d');
  // draw scaled
  ctx.drawImage(bitmap, 0, 0, target, target);
  const img = ctx.getImageData(0,0,target,target).data;
  const chw = new Float32Array(3*target*target);
  for (let i=0;i<target*target;i++){ const r = img[i*4]/255, g = img[i*4+1]/255, b = img[i*4+2]/255; chw[i]=r; chw[i+target*target]=g; chw[i+2*target*target]=b; }
  return { data: chw, dims: [1,3,target,target] };
}

async function runInferenceOnBitmap(bitmap){
  if (!session) throw new Error('session-not-initialized');
  const t = await imageBitmapToTensor(bitmap, 320);
  const tensor = new ort.Tensor('float32', t.data, t.dims);
  const inputName = (session && session.inputNames && session.inputNames[0]) || 'input';
  const feeds = {}; feeds[inputName] = tensor;
  const results = await session.run(feeds);
  const first = Object.values(results)[0];
  const arr = first.data || first;
  // normalize to 0-255 and return Uint8ClampedArray buffer
  const raw = new Float32Array(arr);
  let min = Infinity, max = -Infinity;
  for (let i=0;i<raw.length;i++){ const v = raw[i]; if (v<min) min=v; if (v>max) max=v; }
  const range = (max - min) || 1;
  const out = new Uint8ClampedArray(raw.length);
  for (let i=0;i<raw.length;i++){ out[i] = Math.max(0, Math.min(255, Math.round((raw[i]-min)/range*255))); }
  return { width: 320, height: 320, maskBuffer: out.buffer };
}

self.onmessage = async (ev) => {
  const { id, type, name, url, imageBitmap } = ev.data || {};
  try{
    if (type === 'load'){
      const res = await tryLoadModel(name, url);
      if (res.ok) postMessage({ id, ok:true, type:'load:ok' });
      else postMessage({ id, ok:false, type:'load:error', error: res.error });
    } else if (type === 'infer'){
      // imageBitmap arrives as transferable; run inference
      const r = await runInferenceOnBitmap(imageBitmap);
      // transfer maskBuffer
      postMessage({ id, ok:true, type:'infer:ok', width: r.width, height: r.height, maskBuffer: r.maskBuffer }, [r.maskBuffer]);
    } else if (type === 'cancel'){
      // no-op for now
      postMessage({ id, ok:true, type:'cancel:ok' });
    }
  }catch(err){ postMessage({ id, ok:false, error: (err && err.message) || 'unknown' }); }
};

// allow main thread to receive worker logs
// nothing else

