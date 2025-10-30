// modelWorkerClient.js - client wrapper for model.worker.js
export class ModelWorkerClient {
  constructor(){
    const workerUrl = new URL('./model.worker.js', import.meta.url);
    try {
      this.w = new Worker(workerUrl, { type: 'classic' });
    } catch (err){
      this.w = new Worker(workerUrl);
    }
    this.req = 0;
    this.waiters = new Map();
    this.eventListeners = new Map();
    this.w.onmessage = (e)=>{
      const data = e.data || {};
      // if worker sent a simple log message
      if (data && data.type === 'log' && data.msg){
        this._emitEvent('log', data.msg);
      }
      const id = data.id;
      if (!id) return; const waiter = this.waiters.get(id); if (!waiter) return;
      if (data.ok) waiter.res(data); else waiter.rej(data.error || 'error');
      this.waiters.delete(id);
    };
  }
  post(type, payload){
    const id = ++this.req;
    return new Promise((res,rej)=>{
      this.waiters.set(id, { res, rej });
      try{
        // transfer ImageBitmap if present
        if (payload && payload.imageBitmap){ this.w.postMessage({ id, type, ...payload }, [payload.imageBitmap]); }
        else this.w.postMessage({ id, type, ...payload });
      }catch(err){ this.waiters.delete(id); rej(err); }
    });
  }
  // events: on/off
  on(name, cb){ if (!this.eventListeners.has(name)) this.eventListeners.set(name, new Set()); this.eventListeners.get(name).add(cb); }
  off(name, cb){ if (!this.eventListeners.has(name)) return; this.eventListeners.get(name).delete(cb); }
  _emitEvent(name, payload){ const s = this.eventListeners.get(name); if (!s) return; for (const cb of s) try{ cb(payload); }catch(_){} }
  load({name, url}){ 
    this._emitEvent('load:start', { name, url });
    return this.post('load', { name, url }).then((r)=>{ this._emitEvent('load:done', { name, url }); return r; }).catch((e)=>{ this._emitEvent('load:error', { name, url, error:e }); throw e; });
  }
  infer(imageBitmap){ 
    this._emitEvent('infer:start');
    return this.post('infer', { imageBitmap }).then((r)=>{ this._emitEvent('infer:done'); return r; }).catch((e)=>{ this._emitEvent('infer:error', { error:e }); throw e; });
  }
  cancel(){ return this.post('cancel', {}); }
  terminate(){ this.w.terminate(); }
}
