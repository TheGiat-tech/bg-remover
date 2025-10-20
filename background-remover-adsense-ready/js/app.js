
// ONNX + Fallback (client-only)
const MODEL_URL = 'https://huggingface.co/chwshuang/Stable_diffusion_remove_background_model/resolve/06d038aa68503bfc2ba4d4ce4a81ef8b768995b9/u2netp.onnx?download=true';
const els = {
  drop: document.getElementById('dropzone'),
  file: document.getElementById('fileInput'),
  browse: document.getElementById('browseBtn'),
  display: document.getElementById('displayCanvas'),
  controls: document.getElementById('controls'),
  canvasWrap: document.getElementById('canvasWrap'),
  modelStatus: document.getElementById('modelStatus'),
  download: document.getElementById('downloadBtn'),
  reset: document.getElementById('resetBtn'),
  bgTransparent: document.getElementById('bgTransparent'),
  bgWhite: document.getElementById('bgWhite'),
  bgColor: document.getElementById('bgColor'),
  thresh: document.getElementById('thresh'),
  feather: document.getElementById('feather')
};
let session = null;
let originalImageBitmap = null;

async function fetchModelAsBlob(url){
  const res = await fetch(url, { mode: 'cors' });
  if (!res.ok) throw new Error('Failed to fetch model: '+res.status);
  return await res.blob();
}

async function loadModel() {
  try {
    const blob = await fetchModelAsBlob(MODEL_URL);
    const arrayBuf = await blob.arrayBuffer();
    const model = new Uint8Array(arrayBuf);
    session = await ort.InferenceSession.create(model, { executionProviders: ['webgl','wasm'] });
    console.info('ONNX model loaded, session:', session);
    els.modelStatus.textContent = 'U²‑Netp loaded (CDN)';
  } catch (e) {
    console.warn('ONNX model failed, fallback (people only).', e);
    els.modelStatus.textContent = 'Model failed. Using fallback (people only)';
    session = null;
  }
}

function setupDnD() {
  const dz = els.drop;
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('hover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('hover'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('hover');
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  });
  els.browse.addEventListener('click', ()=> els.file.click());
  els.file.addEventListener('change', ()=> {
    const f = els.file.files?.[0];
    if (f) handleFile(f);
  });
  window.addEventListener('paste', (e)=>{
    const item = [...e.clipboardData.items].find(i=>i.type.startsWith('image/'));
    if (item) handleFile(item.getAsFile());
  });
}

async function handleFile(file) {
  const blobURL = URL.createObjectURL(file);
  const img = await createImageBitmap(await (await fetch(blobURL)).blob());
  URL.revokeObjectURL(blobURL);
  originalImageBitmap = img;
  els.controls.style.display = 'block';
  els.canvasWrap.style.display = 'block';
  await processImage();
}
function drawToCanvas(img) {
  const canvas = els.display;
  const ctx = canvas.getContext('2d');
  const maxW = Math.min(1600, img.width);
  const scale = Math.min(maxW / img.width, 1);
  const w = Math.round(img.width * scale);
  const h = Math.round(img.height * scale);
  canvas.width = w; canvas.height = h;
  ctx.drawImage(img, 0, 0, w, h);
  return {w, h};
}
function makeTensorFromImage(img, target=320) {
  const tmp = document.createElement('canvas');
  tmp.width = target; tmp.height = target;
  const tctx = tmp.getContext('2d');
  tctx.drawImage(img, 0, 0, target, target);
  const { data } = tctx.getImageData(0, 0, target, target);
  const chw = new Float32Array(3 * target * target);
  for (let i=0; i<target*target; i++) {
    const r = data[i*4] / 255, g = data[i*4 + 1] / 255, b = data[i*4 + 2] / 255;
    chw[i] = r; chw[i + target*target] = g; chw[i + 2*target*target] = b;
  }
  return new ort.Tensor('float32', chw, [1,3,target,target]);
}
function bilinearResizeGray(src, srcW, srcH, dstW, dstH) {
  const out = new Float32Array(dstW*dstH);
  for (let y=0; y<dstH; y++) {
    const gy = (y*(srcH-1))/(dstH-1);
    const y0 = Math.floor(gy), y1 = Math.min(y0+1, srcH-1);
    const wy = gy - y0;
    for (let x=0; x<dstW; x++) {
      const gx = (x*(srcW-1))/(dstW-1);
      const x0 = Math.floor(gx), x1 = Math.min(x0+1, srcW-1);
      const wx = gx - x0;
      const i00 = y0*srcW + x0, i01 = y0*srcW + x1, i10 = y1*srcW + x0, i11 = y1*srcW + x1;
      out[y*dstW + x] = (1-wy)*((1-wx)*src[i00] + wx*src[i01]) + wy*((1-wx)*src[i10] + wx*src[i11]);
    }
  }
  return out;
}
function refineAlpha(alpha, w, h, threshold=10, feather=2) {
  const out = new Uint8ClampedArray(w*h);
  for (let i=0; i<w*h; i++) out[i] = Math.max(0, Math.min(255, alpha[i]));
  if (threshold>0) for (let i=0; i<w*h; i++) out[i] = out[i] > threshold ? out[i] : 0;
  if (feather>0) {
    const rad = feather|0, tmp = new Uint8ClampedArray(out);
    for (let y=0; y<h; y++) for (let x=0; x<w; x++) {
      let sum=0, cnt=0;
      for (let dy=-rad; dy<=rad; dy++) {
        const yy = y+dy; if (yy<0||yy>=h) continue;
        for (let dx=-rad; dx<=rad; dx++) {
          const xx = x+dx; if (xx<0||xx>=w) continue;
          sum += tmp[yy*w+xx]; cnt++;
        }
      }
      out[y*w+x] = (sum/cnt)|0;
    }
  }
  return out;
}
async function runU2Net(img) {
  const input = makeTensorFromImage(img, 320);
  try {
    // Build feeds using the model's first input name (safer than hardcoding 'input')
    const feeds = {};
    const inputName = (session && (session.inputNames && session.inputNames[0])) || 'input';
    feeds[inputName] = input;
    const results = await session.run(feeds);
    const first = Object.values(results)[0];
    const arr = first.data || first;
    const dims = first.dims || [1,1,320,320];
    const len = (dims[2] || 320) * (dims[3] || 320);
    const out = new Float32Array(len);
    let min = Number.POSITIVE_INFINITY, max = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < len; i++) { const v = arr[i]; if (v < min) min = v; if (v > max) max = v; }
    const range = max - min + 1e-6;
    for (let i = 0; i < len; i++) { out[i] = ((arr[i] - min) / range) * 255; }
    return out;
  } catch (err) {
    console.error('runU2Net failed, falling back to selfie segmentation', err);
    // rethrow so caller can catch and switch to fallback, or return null
    throw err;
  }
}
async function selfieSegFallback(img) {
  return new Promise((resolve)=>{
    const ss = new SelfieSegmentation({locateFile: (f)=>`https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${f}`});
    ss.setOptions({ modelSelection: 1 });
    ss.onResults((res)=>{
      const mask = res.segmentationMask;
      const tmp = document.createElement('canvas');
      tmp.width = mask.width; tmp.height = mask.height;
      tmp.getContext('2d').drawImage(mask,0,0);
      const {data} = tmp.getContext('2d').getImageData(0,0,tmp.width,tmp.height);
      const gray = new Float32Array(tmp.width*tmp.height);
      for (let i=0;i<gray.length;i++){ gray[i] = data[i*4]; }
      resolve((function resize(){return (function(src,sw,sh,dw,dh){const out = new Float32Array(dw*dh);for(let y=0;y<dh;y++){const gy=(y*(sh-1))/(dh-1);const y0=Math.floor(gy),y1=Math.min(y0+1,sh-1);const wy=gy-y0;for(let x=0;x<dw;x++){const gx=(x*(sw-1))/(dw-1);const x0=Math.floor(gx),x1=Math.min(x0+1,sw-1);const wx=gx-x0;const i00=y0*sw+x0,i01=y0*sw+x1,i10=y1*sw+x0,i11=y1*sw+x1;out[y*dw+x]=(1-wy)*((1-wx)*src[i00]+wx*src[i01])+wy*((1-wx)*src[i10]+wx*src[i11]);}}return out;})(gray,tmp.width,tmp.height,320,320);})());
    });
    const c = document.createElement('canvas');
    c.width = img.width; c.height = img.height;
    c.getContext('2d').drawImage(img,0,0);
    const frame = new Image();
    frame.onload = ()=> ss.send({image: frame});
    frame.src = c.toDataURL();
  });
}
function composite(img, alpha320, dispW, dispH) {
  const a = (function(src,sw,sh,dw,dh){const out = new Float32Array(dw*dh);for(let y=0;y<dh;y++){const gy=(y*(sh-1))/(dh-1);const y0=Math.floor(gy),y1=Math.min(y0+1,sh-1);const wy=gy-y0;for(let x=0;x<dw;x++){const gx=(x*(sw-1))/(dw-1);const x0=Math.floor(gx),x1=Math.min(x0+1,sw-1);const wx=gx-x0;const i00=y0*sw+x0,i01=y0*sw+x1,i10=y1*sw+x0,i11=y1*sw+x1;out[y*dw+x]=(1-wy)*((1-wx)*src[i00]+wx*src[i01])+wy*((1-wx)*src[i10]+wx*src[i11]);}}return out;})(alpha320,320,320,dispW,dispH);
  const refined = (function(alpha,w,h,threshold=10,feather=2){const out=new Uint8ClampedArray(w*h);for(let i=0;i<w*h;i++)out[i]=Math.max(0,Math.min(255,alpha[i]));if(threshold>0)for(let i=0;i<w*h;i++)out[i]=out[i]>threshold?out[i]:0;if(feather>0){const rad=feather|0,tmp=new Uint8ClampedArray(out);for(let y=0;y<h;y++)for(let x=0;x<w;x++){let sum=0,cnt=0;for(let dy=-rad;dy<=rad;dy++){const yy=y+dy;if(yy<0||yy>=h)continue;for(let dx=-rad;dx<=rad;dx++){const xx=x+dx;if(xx<0||xx>=w)continue;sum+=tmp[yy*w+xx];cnt++;}}out[y*w+x]=(sum/cnt)|0;}}return out;})(a,dispW,dispH,parseInt(els.thresh.value,10),parseInt(els.feather.value,10));
  const ctx = els.display.getContext('2d');
  const imgData = ctx.getImageData(0,0,dispW,dispH);
  const out = imgData.data;
  for (let i=0;i<dispW*dispH;i++){ out[i*4+3] = refined[i]; }
  ctx.clearRect(0,0,dispW,dispH);
  const transparent = els.bgTransparent.classList.contains('selected');
  const white = els.bgWhite.classList.contains('selected');
  if (!transparent){
    ctx.fillStyle = white ? '#ffffff' : (els.bgColor.value || '#00000000');
    ctx.fillRect(0,0,dispW,dispH);
  }
  ctx.putImageData(imgData,0,0);
}
async function processImage() {
  if (!originalImageBitmap) return;
  const {w,h} = drawToCanvas(originalImageBitmap);
  els.modelStatus.textContent = 'Processing…';
  try {
    let alpha320;
    if (session) {
      els.modelStatus.textContent = 'Processing (U²‑Netp)…';
      alpha320 = await runU2Net(originalImageBitmap);
      console.info('Processed with U2Netp');
    } else {
      els.modelStatus.textContent = 'Processing (fallback)…';
      alpha320 = await selfieSegFallback(originalImageBitmap);
      console.info('Processed with selfie segmentation fallback');
    }
    composite(originalImageBitmap, alpha320, w, h);
    els.modelStatus.textContent = session ? 'Done (U²‑Netp)' : 'Done (fallback: people only)';
  } catch (e) {
    console.error(e);
    els.modelStatus.textContent = 'Error during segmentation';
  }
}
function initUI() {
  function selectBg(which){
    [els.bgTransparent, els.bgWhite].forEach(el=>el.classList.remove('selected'));
    if (which==='transparent') els.bgTransparent.classList.add('selected');
    if (which==='white') els.bgWhite.classList.add('selected');
    if (originalImageBitmap) processImage();
  }
  els.bgTransparent.addEventListener('click', ()=>selectBg('transparent'));
  els.bgWhite.addEventListener('click', ()=>selectBg('white'));
  els.bgColor.addEventListener('input', ()=>{ [els.bgTransparent, els.bgWhite].forEach(el=>el.classList.remove('selected')); processImage();});
  els.thresh.addEventListener('input', ()=> processImage());
  els.feather.addEventListener('input', ()=> processImage());
  els.download.addEventListener('click', ()=>{
    const link = document.createElement('a');
    link.download = 'background-removed.png';
    els.display.toBlob((blob)=>{
      const url = URL.createObjectURL(blob);
      link.href = url; document.body.appendChild(link); link.click();
      setTimeout(()=>{ URL.revokeObjectURL(url); link.remove(); }, 1000);
    }, 'image/png');
  });
  els.reset.addEventListener('click', ()=> window.location.reload());
}
(async function main(){
  // load scripts from footer
  setupDnD();
  initUI();
  await loadModel();
})();
