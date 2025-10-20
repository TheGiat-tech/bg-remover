// brush.js
// Provides brush-based erase/restore on an alpha mask drawn over a canvas. Includes undo/redo and zoom/pan support.

export class BrushEditor{
  constructor(canvas, options={}){
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.mask = options.mask || null; // Uint8ClampedArray of alpha values (0-255)
    this.w = options.w || canvas.width; this.h = options.h || canvas.height;
    this.radius = options.radius || 10;
    this.mode = 'erase'; // 'erase' or 'restore'
    this.isDrawing = false;
    this.last = null;
    this.undoStack = [];
    this.redoStack = [];
    this.scale = 1;
    this.offset = {x:0,y:0};
    this._bindEvents();
  }
  setMask(mask, w, h){ this.mask = mask; this.w = w; this.h = h; }
  setRadius(r){ this.radius = r; }
  setMode(m){ this.mode = m; }
  pushUndo(){ if (!this.mask) return; this.undoStack.push(new Uint8ClampedArray(this.mask)); if (this.undoStack.length>50) this.undoStack.shift(); this.redoStack=[]; }
  undo(){ if (!this.undoStack.length) return; this.redoStack.push(new Uint8ClampedArray(this.mask)); this.mask.set(this.undoStack.pop()); this._renderMask(); }
  redo(){ if (!this.redoStack.length) return; this.undoStack.push(new Uint8ClampedArray(this.mask)); this.mask.set(this.redoStack.pop()); this._renderMask(); }
  _bindEvents(){
    this.canvas.addEventListener('pointerdown', (e)=>{ this.canvas.setPointerCapture(e.pointerId); this.isDrawing=true; this.last = this._evtToPoint(e); this.pushUndo(); this._stroke(this.last); });
    this.canvas.addEventListener('pointermove', (e)=>{ if (!this.isDrawing) return; const p = this._evtToPoint(e); this._line(this.last, p); this.last = p; });
    this.canvas.addEventListener('pointerup', (e)=>{ this.isDrawing=false; this.last=null; this.canvas.releasePointerCapture(e.pointerId); });
    // wheel for zoom
    this.canvas.addEventListener('wheel', (e)=>{ if (e.ctrlKey){ e.preventDefault(); const delta = -e.deltaY * 0.001; this.scale = Math.max(0.1, Math.min(4, this.scale * (1+delta))); this._renderMask(); } });
    // panning with middle mouse
    let panning=false, startPan=null;
    this.canvas.addEventListener('mousedown', (e)=>{ if (e.button===1){ panning=true; startPan={x:e.clientX, y:e.clientY}; } });
    window.addEventListener('mousemove', (e)=>{ if (!panning) return; const dx = e.clientX - startPan.x; const dy = e.clientY - startPan.y; this.offset.x += dx; this.offset.y += dy; startPan={x:e.clientX,y:e.clientY}; this._renderMask(); });
    window.addEventListener('mouseup', ()=>{ panning=false; });
  }
  _evtToPoint(e){ const rect = this.canvas.getBoundingClientRect(); const x = (e.clientX - rect.left - this.offset.x)/this.scale; const y = (e.clientY - rect.top - this.offset.y)/this.scale; return {x,y}; }
  _stroke(p){ this._applyBrush(Math.round(p.x), Math.round(p.y)); }
  _line(a,b){ const dx = b.x - a.x, dy = b.y - a.y, steps = Math.max(Math.abs(dx), Math.abs(dy)); for (let i=0;i<=steps;i++){ const t = i/steps; const x = Math.round(a.x + dx*t); const y = Math.round(a.y + dy*t); this._applyBrush(x,y); } }
  _applyBrush(cx, cy){ if (!this.mask) return; const r = Math.max(1, Math.round(this.radius)); for (let y=cy-r;y<=cy+r;y++){ if (y<0||y>=this.h) continue; for (let x=cx-r;x<=cx+r;x++){ if (x<0||x>=this.w) continue; const d = Math.hypot(x-cx, y-cy); if (d>r) continue; const i = y*this.w + x; const t = 1 - (d/r); const delta = Math.round(255 * t);
      if (this.mode==='erase'){
        this.mask[i] = Math.max(0, this.mask[i] - delta);
      } else {
        this.mask[i] = Math.min(255, this.mask[i] + delta);
      }
    }} this._renderMask(); }
  _renderMask(){ // render alpha mask as overlay on canvas
    const ctx = this.ctx; ctx.save(); ctx.clearRect(0,0,this.canvas.width,this.canvas.height); ctx.translate(this.offset.x, this.offset.y); ctx.scale(this.scale, this.scale);
    const img = ctx.createImageData(this.w, this.h);
    for (let i=0;i<this.w*this.h;i++){ img.data[i*4+0]=0; img.data[i*4+1]=0; img.data[i*4+2]=0; img.data[i*4+3]=this.mask[i]; }
    ctx.putImageData(img, 0, 0);
    ctx.restore(); }
}
