// model.js — loads ONNX model (full U2Net or MODNet if found), caches in IndexedDB, exposes a worker-based inference API.

const MODEL_CACHE_DB = 'bgremover-models-v1';
const MODEL_STORE = 'models';

function openDb(){
  return new Promise((resolve, reject)=>{
    const req = indexedDB.open(MODEL_CACHE_DB, 1);
    req.onupgradeneeded = ()=>{
      req.result.createObjectStore(MODEL_STORE);
    };
    req.onsuccess = ()=> resolve(req.result);
    req.onerror = ()=> reject(req.error);
  });
}

async function getCachedModel(name){
  const db = await openDb();
  return new Promise((resolve,reject)=>{
    const tx = db.transaction(MODEL_STORE, 'readonly');
    const st = tx.objectStore(MODEL_STORE);
    const r = st.get(name);
    r.onsuccess = ()=> resolve(r.result);
    r.onerror = ()=> reject(r.error);
  });
}
async function cacheModel(name, bytes){
  const db = await openDb();
  return new Promise((resolve,reject)=>{
    const tx = db.transaction(MODEL_STORE, 'readwrite');
    const st = tx.objectStore(MODEL_STORE);
    const r = st.put(bytes, name);
    r.onsuccess = ()=> resolve(true);
    r.onerror = ()=> reject(r.error);
  });
}

// try multiple model urls; prefer full u2net then modnet
const MODEL_CANDIDATES = [
  // Full U2-Net ONNX (compressed/common naming) - placeholder URL (user should host or use a CORS-enabled source)
  'https://huggingface.co/chwshuang/Stable_diffusion_remove_background_model/resolve/main/u2net.onnx?download=true',
  // fallback u2netp already in use
  'https://huggingface.co/chwshuang/Stable_diffusion_remove_background_model/resolve/06d038aa68503bfc2ba4d4ce4a81ef8b768995b9/u2netp.onnx?download=true',
  // example MODNet (if available)
  'https://huggingface.co/your-org/modnet/resolve/main/modnet.onnx?download=true'
];

// load model bytes (from cache or network)
export async function loadModelBytes(preferredName='u2net-full'){
  // check cache
  try {
    const cached = await getCachedModel(preferredName);
    if (cached) return cached;
  } catch (err){ console.warn('IndexedDB read failed', err); }

  // try download candidates
  for (const url of MODEL_CANDIDATES){
    try {
      const res = await fetch(url, { mode: 'cors' });
      if (!res.ok) continue;
      const blob = await res.blob();
      const bytes = await blob.arrayBuffer();
      try { await cacheModel(preferredName, bytes); } catch(e){ console.warn('cache failed', e); }
      return bytes;
    } catch (err){ console.warn('fetch model candidate failed', url, err); }
  }
  throw new Error('no-model-available');
}

let worker = null;
let inputName = 'input';
let pendingCalls = new Map();
let callId = 1;

export async function initModelWorker(){
  if (worker) return worker;
  const bytes = await loadModelBytes();
  // spin up worker (uses importScripts to load ort)
  worker = new Worker('/src/js/modelWorker.js');
  worker.onmessage = (ev)=>{
    const { type, id, payload } = ev.data || {};
    if (type === 'init:done') {
      // ready
      const p = pendingCalls.get(id); if (p) { p.resolve(); pendingCalls.delete(id); }
    } else if (type === 'run:done'){
      const p = pendingCalls.get(id);
      if (p) { p.resolve(payload); pendingCalls.delete(id); }
    } else if (type === 'error'){
      const p = pendingCalls.get(id);
      if (p) { p.reject(new Error(payload && payload.message)); pendingCalls.delete(id); }
    }
  };
  // create init handshake
  await new Promise((resolve,reject)=>{
    const id = ''+(callId++);
    pendingCalls.set(id, { resolve, reject });
    // transfer bytes (ArrayBuffer) to worker for efficiency
    const ab = bytes instanceof ArrayBuffer ? bytes : bytes.buffer || bytes;
    worker.postMessage({ type: 'init', id, payload: { bytes: ab, name: 'u2net-full' } }, [ab]);
  });
  return worker;
}

export async function runInference(tensor, inputNameOverride){
  if (!worker) throw new Error('worker-not-initialized');
  return new Promise((resolve,reject)=>{
    const id = ''+(callId++);
    pendingCalls.set(id, { resolve, reject });
    // transfer tensor data buffer if available
    let transfer = [];
    if (tensor && tensor.data && tensor.data.buffer) transfer.push(tensor.data.buffer);
    worker.postMessage({ type: 'run', id, payload: { inputName: inputNameOverride || inputName, tensor } }, transfer);
  });
}
