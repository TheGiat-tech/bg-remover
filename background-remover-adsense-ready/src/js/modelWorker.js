// modelWorker.js
// Runs ONNX inference in a WebWorker context. Receives model bytes or URL and runs inference requests.
// Communicates with main thread via postMessage({type, id, payload})

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js');

let session = null;
let modelName = null;

async function createSessionFromBytes(bytes, backendHints = ['webgpu','webgl','wasm']){
  try {
    // try WebGPU first if available
    const providers = [];
    if (backendHints.includes('webgpu') && ort.env.wasm && ort.env.wasm.simd) {
      providers.push('webgpu');
    }
    if (backendHints.includes('webgl')) providers.push('webgl');
    if (backendHints.includes('wasm')) providers.push('wasm');
    // ort.InferenceSession.create accepts ArrayBuffer or Uint8Array
    const modelBytes = bytes instanceof ArrayBuffer ? new Uint8Array(bytes) : bytes;
    session = await ort.InferenceSession.create(modelBytes, { executionProviders: providers });
    return session;
  } catch (err) {
    console.warn('Failed to create ONNX session in worker', err);
    throw err;
  }
}

self.onmessage = async (ev) => {
  const { type, id, payload } = ev.data || {};
  try {
    if (type === 'init'){
      const { bytes, name } = payload;
      modelName = name || 'model';
      // bytes may be an ArrayBuffer transferred; create session
      await createSessionFromBytes(bytes);
      self.postMessage({ type: 'init:done', id });
    }
    if (type === 'run'){
      if (!session) throw new Error('session-not-initialized');
      const { inputName, tensor } = payload;
      // Reconstruct ort.Tensor from plain payload
      let tdata = tensor && tensor.data;
      if (!(tdata instanceof Float32Array)) tdata = new Float32Array(tdata);
      const ortTensor = new ort.Tensor(tensor.type || 'float32', tdata, tensor.dims);
      const feeds = {};
      feeds[inputName] = ortTensor;
      const results = await session.run(feeds);
      // extract first output
      const first = Object.values(results)[0];
      const arr = first.data || first;
      const outArr = (arr instanceof Float32Array) ? arr : new Float32Array(arr);
      // transfer the underlying buffer back to main thread
      self.postMessage({ type: 'run:done', id, payload: { data: outArr, dims: first.dims } }, [outArr.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', id, payload: { message: err && err.message } });
  }
};
