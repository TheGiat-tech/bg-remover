// matting.js
// Provides trimap generation, closed-form/guided matting wrapper, anti-aliasing, feathering and color decontamination.

// Trimap: take alpha mask (0-255) and produce 0=bg,128=unknown,255=fg. band size in px
export function generateTrimap(alpha, w, h, band=4){
  const out = new Uint8ClampedArray(w*h);
  // binary threshold at 127
  const bin = new Uint8Array(w*h);
  for (let i=0;i<w*h;i++) bin[i] = alpha[i] > 127 ? 1 : 0;
  // distance transform (approx) using Manhattan distance via two-pass
  const inf = 1e8;
  const distF = new Uint32Array(w*h);
  const distB = new Uint32Array(w*h);
  for (let i=0;i<w*h;i++){ distF[i] = bin[i] ? 0 : inf; distB[i] = bin[i] ? inf : 0; }
  // forward/backward pass
  for (let y=0;y<h;y++){
    for (let x=0;x<w;x++){
      const i = y*w + x;
      if (x>0) distF[i] = Math.min(distF[i], distF[i-1] + 1);
      if (y>0) distF[i] = Math.min(distF[i], distF[i-w] + 1);
    }
  }
  for (let y=h-1;y>=0;y--){
    for (let x=w-1;x>=0;x--){
      const i = y*w + x;
      if (x<w-1) distF[i] = Math.min(distF[i], distF[i+1] + 1);
      if (y<h-1) distF[i] = Math.min(distF[i], distF[i+w] + 1);
    }
  }
  for (let y=0;y<h;y++){
    for (let x=0;x<w;x++){
      const i = y*w + x;
      if (x>0) distB[i] = Math.min(distB[i], distB[i-1] + 1);
      if (y>0) distB[i] = Math.min(distB[i], distB[i-w] + 1);
    }
  }
  for (let y=h-1;y>=0;y--){
    for (let x=w-1;x>=0;x--){
      const i = y*w + x;
      if (x<w-1) distB[i] = Math.min(distB[i], distB[i+1] + 1);
      if (y<h-1) distB[i] = Math.min(distB[i], distB[i+w] + 1);
    }
  }
  for (let i=0;i<w*h;i++){
    if (bin[i]){
      out[i] = distB[i] <= band ? 128 : 255; // if close to background, unknown else fg
    } else {
      out[i] = distF[i] <= band ? 128 : 0;
    }
  }
  return out;
}

// Guided filter from previous code (single-channel guidance)
function boxFilter(src, w, h, r) {
  const iw = w + 1;
  const integral = new Float64Array(iw * (h + 1));
  for (let y = 0; y < h; y++) {
    let rowSum = 0;
    for (let x = 0; x < w; x++) {
      rowSum += src[y * w + x];
      integral[(y + 1) * iw + (x + 1)] = integral[y * iw + (x + 1)] + rowSum;
    }
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r);
    const y1 = Math.min(h - 1, y + r);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(w - 1, x + r);
      const A = integral[y0 * iw + x0];
      const B = integral[y0 * iw + (x1 + 1)];
      const C = integral[(y1 + 1) * iw + x0];
      const D = integral[(y1 + 1) * iw + (x1 + 1)];
      const area = (y1 - y0 + 1) * (x1 - x0 + 1);
      out[y * w + x] = (D - B - C + A) / area;
    }
  }
  return out;
}

export function guidedMatting(guideRGBA, w, h, alpha, r=4, eps=1e-3){
  // guideRGBA: Uint8ClampedArray (RGBA), alpha: Uint8ClampedArray (0-255)
  const N = w*h;
  const I = new Float32Array(N);
  for (let i = 0; i < N; i++){
    const gi = i*4;
    const rc = guideRGBA[gi]/255, gc = guideRGBA[gi+1]/255, bc = guideRGBA[gi+2]/255;
    I[i] = 0.299*rc + 0.587*gc + 0.114*bc;
  }
  const P = new Float32Array(N); for (let i=0;i<N;i++) P[i] = alpha[i]/255;
  const meanI = boxFilter(I,w,h,r);
  const meanP = boxFilter(P,w,h,r);
  const Ip = new Float32Array(N); for (let i=0;i<N;i++) Ip[i] = I[i]*P[i];
  const meanIp = boxFilter(Ip,w,h,r);
  const covIp = new Float32Array(N); for (let i=0;i<N;i++) covIp[i] = meanIp[i] - meanI[i]*meanP[i];
  const II = new Float32Array(N); for (let i=0;i<N;i++) II[i] = I[i]*I[i];
  const meanII = boxFilter(II,w,h,r);
  const varI = new Float32Array(N); for (let i=0;i<N;i++) varI[i] = meanII[i] - meanI[i]*meanI[i];
  const a = new Float32Array(N); const b = new Float32Array(N);
  for (let i=0;i<N;i++) { a[i] = covIp[i]/(varI[i] + eps); b[i] = meanP[i] - a[i]*meanI[i]; }
  const meanA = boxFilter(a,w,h,r); const meanB = boxFilter(b,w,h,r);
  const q = new Uint8ClampedArray(N);
  for (let i=0;i<N;i++){ const v = meanA[i]*I[i] + meanB[i]; q[i] = Math.max(0,Math.min(255,Math.round(v*255))); }
  return q;
}

// Anti-alias edges: perform distance transform on alpha edges and blend
export function antiAliasAlpha(alpha, w, h, maxDist=3){
  const N = w*h;
  const isFg = new Uint8Array(N); for (let i=0;i<N;i++) isFg[i] = alpha[i] > 127 ? 1 : 0;
  const dist = new Uint32Array(N);
  const inf = 1e8;
  for (let i=0;i<N;i++) dist[i] = isFg[i] ? 0 : inf;
  for (let y=0;y<h;y++) for (let x=0;x<w;x++){ const i=y*w+x; if (x>0) dist[i]=Math.min(dist[i], dist[i-1]+1); if (y>0) dist[i]=Math.min(dist[i], dist[i-w]+1); }
  for (let y=h-1;y>=0;y--) for (let x=w-1;x>=0;x--){ const i=y*w+x; if (x<w-1) dist[i]=Math.min(dist[i], dist[i+1]+1); if (y<h-1) dist[i]=Math.min(dist[i], dist[i+w]+1); }
  const out = new Uint8ClampedArray(N);
  for (let i=0;i<N;i++){
    if (isFg[i]) out[i] = 255;
    else if (dist[i] > maxDist) out[i] = 0;
    else out[i] = Math.round(255 * (1 - dist[i]/(maxDist+1)));
  }
  return out;
}

// Simple color decontamination: sample background color near edge and desaturate foreground toward complementary hue
export function decontaminateColors(srcRGBA, alpha, w, h, sampleRadius=3){
  // compute contour mask
  const N = w*h; const contour = new Uint8Array(N);
  for (let y=1;y<h-1;y++) for (let x=1;x<w-1;x++){
    const i = y*w + x; if (alpha[i] > 200) continue; // only consider near-edge
    // if any neighbor is strong fg -> contour
    let hasFgNeighbor = false;
    for (let dy=-1;dy<=1;dy++) for (let dx=-1;dx<=1;dx++){ if (dx===0 && dy===0) continue; const j=(y+dy)*w + (x+dx); if (alpha[j]>200) hasFgNeighbor = true; }
    if (hasFgNeighbor) contour[i]=1;
  }
  // sample background color for each contour pixel by averaging outside region
  const out = new Uint8ClampedArray(srcRGBA.length);
  out.set(srcRGBA);
  for (let y=0;y<h;y++) for (let x=0;x<w;x++){
    const i = y*w + x;
    if (!contour[i]) continue;
    // sample ring outside (radius sampleRadius) and average
    let sr=0, sg=0, sb=0, cnt=0;
    for (let dy=-sampleRadius;dy<=sampleRadius;dy++) for (let dx=-sampleRadius;dx<=sampleRadius;dx++){
      const sx = x + dx, sy = y + dy; if (sx<0||sy<0||sx>=w||sy>=h) continue; const si = sy*w + sx; if (alpha[si] > 220) continue; // only background
      const gi = si*4; sr += srcRGBA[gi]; sg += srcRGBA[gi+1]; sb += srcRGBA[gi+2]; cnt++;
    }
    if (cnt===0) continue;
    sr = sr/cnt; sg = sg/cnt; sb = sb/cnt;
    // compute foreground pixel
    const pi = i*4; const fr = srcRGBA[pi], fg = srcRGBA[pi+1], fb = srcRGBA[pi+2];
    // convert to HSV and push hue away from background (simple desaturation toward complementary)
    const bgHue = rgbToHue(sr,sg,sb);
    const fgHue = rgbToHue(fr,fg,fb);
    const comp = (bgHue + 180) % 360; // complementary
    // compute desaturation amount based on alpha proximity
    const a = alpha[i]/255;
    const desat = Math.min(0.7, (1-a) * 0.7);
    const [nr,ng,nb] = desaturateTowards(fr,fg,fb, comp, desat);
    out[pi] = nr; out[pi+1] = ng; out[pi+2] = nb;
  }
  return out;
}

function rgbToHue(r,g,b){ // r,g,b 0-255
  r/=255; g/=255; b/=255;
  const max = Math.max(r,g,b), min=Math.min(r,g,b);
  if (max===min) return 0;
  let h;
  if (max===r) h = (60 * ((g-b)/(max-min)) + 360) % 360;
  else if (max===g) h = (60 * ((b-r)/(max-min)) + 120) % 360;
  else h = (60 * ((r-g)/(max-min)) + 240) % 360;
  return h;
}

function desaturateTowards(r,g,b, targetHue, amount){
  // convert to HSV, interpolate hue toward targetHue and reduce saturation
  r/=255; g/=255; b/=255;
  const max = Math.max(r,g,b), min=Math.min(r,g,b); const v = max;
  const delta = max - min;
  const s = max===0 ? 0 : delta/max;
  let h = rgbToHue(r*255,g*255,b*255);
  // lerp hue
  const newH = (h + (angleDiff(targetHue,h))) % 360;
  const newS = Math.max(0, s * (1 - amount));
  const [nr,ng,nb] = hsvToRgb(newH, newS, v);
  return [Math.round(nr*255), Math.round(ng*255), Math.round(nb*255)];
}
function angleDiff(a,b){ let d = a - b; while(d>180) d-=360; while(d<-180) d+=360; return d; }
function hsvToRgb(h,s,v){ // h [0,360)
  const c = v*s; const x = c * (1 - Math.abs((h/60)%2 -1)); const m = v - c;
  let r=0,g=0,b=0;
  if (0<=h && h<60){ r=c; g=x; b=0; }
  else if (60<=h && h<120){ r=x; g=c; b=0; }
  else if (120<=h && h<180){ r=0; g=c; b=x; }
  else if (180<=h && h<240){ r=0; g=x; b=c; }
  else if (240<=h && h<300){ r=x; g=0; b=c; }
  else { r=c; g=0; b=x; }
  return [r+m, g+m, b+m];
}

// morphological open/close on noisy regions (simple) — open then close
export function morphOpenClose(alpha, w, h, radius=1){
  const N = w*h; const tmp = new Uint8ClampedArray(alpha);
  // erosion
  const eroded = new Uint8ClampedArray(N);
  for (let y=0;y<h;y++) for (let x=0;x<w;x++){
    let minv=255;
    for (let dy=-radius;dy<=radius;dy++) for (let dx=-radius;dx<=radius;dx++){
      const sx=x+dx, sy=y+dy; if (sx<0||sy<0||sx>=w||sy>=h) continue; minv = Math.min(minv, tmp[sy*w + sx]);
    }
    eroded[y*w+x]=minv;
  }
  // dilation
  const dilated = new Uint8ClampedArray(N);
  for (let y=0;y<h;y++) for (let x=0;x<w;x++){
    let maxv=0;
    for (let dy=-radius;dy<=radius;dy++) for (let dx=-radius;dx<=radius;dx++){
      const sx=x+dx, sy=y+dy; if (sx<0||sy<0||sx>=w||sy>=h) continue; maxv = Math.max(maxv, eroded[sy*w + sx]);
    }
    dilated[y*w+x]=maxv;
  }
  return dilated;
}

// utility to blend alpha gracefully
export function blendAlpha(originalAlpha, refinedAlpha, w, h, edgeHeuristicThresh=10){
  // compute local variance on alpha to find noisy regions
  const N = w*h; const out = new Uint8ClampedArray(N);
  for (let y=1;y<h-1;y++) for (let x=1;x<w-1;x++){
    const i=y*w+x; let sum=0, sum2=0, cnt=0;
    for (let dy=-1;dy<=1;dy++) for (let dx=-1;dx<=1;dx++){ const j=(y+dy)*w + (x+dx); const v=originalAlpha[j]; sum+=v; sum2+=v*v; cnt++; }
    const mean = sum/cnt; const varr = sum2/cnt - mean*mean;
    if (varr*100 > edgeHeuristicThresh) out[i]=refinedAlpha[i]; else out[i]=originalAlpha[i];
  }
  // border copy
  for (let x=0;x<w;x++){ out[x]=originalAlpha[x]; out[(h-1)*w + x]=originalAlpha[(h-1)*w + x]; }
  for (let y=0;y<h;y++){ out[y*w]=originalAlpha[y*w]; out[y*w + (w-1)]=originalAlpha[y*w + (w-1)]; }
  return out;
}
