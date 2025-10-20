// main.js — orchestrates model, matting, brush, and UI
import { initModelWorker, runInference } from './model.js';
import { generateTrimap, guidedMatting, antiAliasAlpha, decontaminateColors, morphOpenClose, blendAlpha } from './matting.js';
import { BrushEditor } from './brush.js';

const els = {
  displayCanvas: document.getElementById('displayCanvas'),
  resultImg: document.getElementById('result'),
  resultPreview: document.getElementById('resultPreview'),
  modelStatus: document.getElementById('modelStatus'),
  downloadBtn: document.getElementById('downloadBtn'),
  preserveShadows: null,
};

let originalImageBitmap = null;
let brushEditor = null;

async function init(){
  // append a small settings panel
  const panel = document.createElement('div'); panel.style.marginTop='0.5rem';
  panel.innerHTML = `<label><input id="preserveShadows" type="checkbox" checked/> Preserve shadows</label> <label style="margin-left:1rem"><input id="upscale" type="checkbox"/> Upscale 1.5x</label>`;
  document.querySelector('.controls').appendChild(panel);
  els.preserveShadows = document.getElementById('preserveShadows');
  const upscaleChk = document.getElementById('upscale');

  // initialize model worker
  try{
    els.modelStatus.textContent = 'Loading model…';
    await initModelWorker();
    els.modelStatus.textContent = 'Model ready';
  } catch (err){ els.modelStatus.textContent = 'Model failed'; console.warn(err); }

  // setup download
  els.downloadBtn.addEventListener('click', ()=>{
    els.displayCanvas.toBlob((blob)=>{ const link=document.createElement('a'); link.download='bg-removed.png'; link.href=URL.createObjectURL(blob); link.click(); setTimeout(()=>URL.revokeObjectURL(link.href),1000); }, 'image/png');
  });

  // brush editor on top of display canvas
  brushEditor = new BrushEditor(els.displayCanvas, { w: els.displayCanvas.width, h: els.displayCanvas.height, radius: 12 });
}

async function processImage(imgBitmap){
  originalImageBitmap = imgBitmap;
  // draw to canvas with optional upscale
  const canvas = els.displayCanvas; const ctx = canvas.getContext('2d');
  const targetScale = document.getElementById('upscale')?.checked ? 1.5 : 1.0;
  const maxW = Math.min(1600, Math.round(imgBitmap.width * targetScale));
  const scale = Math.min(maxW / imgBitmap.width, 1);
  const w = Math.round(imgBitmap.width * scale);
  const h = Math.round(imgBitmap.height * scale);
  canvas.width = w; canvas.height = h; ctx.clearRect(0,0,w,h); ctx.drawImage(imgBitmap,0,0,w,h);

  // prepare tensor (reuse earlier logic) - create a 320 input if model expects fixed size
  const tmp = document.createElement('canvas'); tmp.width=320; tmp.height=320; const tctx = tmp.getContext('2d'); tctx.drawImage(imgBitmap,0,0,320,320);
  const idata = tctx.getImageData(0,0,320,320).data; const chw = new Float32Array(3*320*320);
  for (let i=0;i<320*320;i++){ const r=idata[i*4]/255,g=idata[i*4+1]/255,b=idata[i*4+2]/255; chw[i]=r; chw[i+320*320]=g; chw[i+2*320*320]=b; }
  // create ort tensor-like plain object for worker to use — the worker expects an ort.Tensor, but postMessage cannot transfer complex class; we will send plain Float32Array and reconstruct in worker
  const floatData = new Float32Array(chw.buffer ? chw.buffer : chw);
  const tensor = { type:'float32', data: floatData, dims:[1,3,320,320] };
  // run inference
  els.modelStatus.textContent = 'Running model…';
  try{
  const res = await runInference(tensor, 'input');
  // res has { data: Float32Array, dims: [...] }
  const raw = res.data; const dims = res.dims || [1,1,320,320];
    const alpha320 = new Uint8ClampedArray(320*320);
    // normalize
  let min=Infinity,max=-Infinity; for (let i=0;i<raw.length;i++){ const v = raw[i]; if (v<min) min=v; if (v>max) max=v; }
    const range = (max-min)||1;
    for (let i=0;i<320*320;i++) alpha320[i] = Math.max(0, Math.min(255, Math.round((raw[i]-min)/range*255)));

    // resize alpha to canvas size
    const alphaResized = new Uint8ClampedArray(w*h);
    // simple bilinear
    for (let y=0;y<h;y++) for (let x=0;x<w;x++){ const gx = x*(320-1)/(w-1), gy = y*(320-1)/(h-1); const x0=Math.floor(gx), x1=Math.min(x0+1,319); const y0=Math.floor(gy), y1=Math.min(y0+1,319); const wx = gx-x0, wy=gy-y0; const i00=y0*320+x0, i01=y0*320+x1, i10=y1*320+x0, i11=y1*320+x1; const v = (1-wy)*((1-wx)*alpha320[i00]+wx*alpha320[i01]) + wy*((1-wx)*alpha320[i10]+wx*alpha320[i11]); alphaResized[y*w+x] = v|0; }

    // generate trimap
    const trimap = generateTrimap(alphaResized, w, h, 4);
    // get guide (image RGBA)
    const guideData = ctx.getImageData(0,0,w,h).data;
    // guided matting
    const refined = guidedMatting(guideData, w, h, alphaResized, 4, 1e-3);
    // anti-alias
    const aa = antiAliasAlpha(refined, w, h, 3);
    // decontaminate
    const correctedRGBA = decontaminateColors(guideData, aa, w, h, 3);
    // blend with original alpha using heuristic
    const blended = blendAlpha(alphaResized, aa, w, h, 10);

    // write to canvas
    const imgData = ctx.getImageData(0,0,w,h);
    for (let i=0;i<w*h;i++){ const pi=i*4; imgData.data[pi]=correctedRGBA[pi]; imgData.data[pi+1]=correctedRGBA[pi+1]; imgData.data[pi+2]=correctedRGBA[pi+2]; imgData.data[pi+3]=blended[i]; }
    ctx.putImageData(imgData,0,0);
    // update preview
    if (els.resultImg && els.resultPreview) { els.resultImg.src = canvas.toDataURL('image/png'); els.resultPreview.style.display='block'; }
    els.modelStatus.textContent = 'Done';
  } catch (err){ console.error(err); els.modelStatus.textContent = 'Model error'; }
}

// expose functions
window.BGRemover = { init, processImage };

// auto-init
window.addEventListener('load', ()=>{ init().catch(e=>console.warn(e)); });
