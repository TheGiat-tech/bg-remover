# Background Remover — Static Site

This folder contains a small static site for Background Remover. It is designed to run completely client-side and demonstrates a privacy-first background removal tool.

Quick start (local):

```bash
cd background-remover-adsense-ready
python3 -m http.server 5173 --bind 0.0.0.0
# open http://localhost:5173/ in your browser (or use the Codespace forwarded port)
```

Notes before publishing:
- Replace `YOUR-SITE-DOMAIN` placeholders in HTML files with your real domain.
- Replace `YOUR-ADSENSE-PUB-ID` in the HTML/ads.txt files if you plan to use Google AdSense.
- The ONNX model is loaded from a CDN in `index.html` — verify remote hosting & licensing before production.

Files of interest:
- `index.html`, `about.html`, `privacy.html`, `terms.html`, `contact.html`
- `css/style.css`, `js/app.js`, `js/banner.js`
Advanced modules (new)
- `src/js/model.js` — model loader, IndexedDB cache, and WebWorker orchestration for ONNX Runtime Web (WebGPU/WebGL/WASM fallbacks).
- `src/js/modelWorker.js` — WebWorker that creates an ONNX session and runs inference off the main thread.
- `src/js/matting.js` — trimap generation, guided matting, anti-aliasing, color decontamination, morphology and blending utilities.
- `src/js/brush.js` — brush editor with erase/restore, undo/redo, zoom/pan.
- `src/js/main.js` — orchestrates the pipeline and wires UI controls.

- Switched to a fuller matting pipeline: optional upscale → model inference (U²‑Net full if available) → trimap → guided matting → anti-alias → decontamination → blend.

Zero-control one-click flow
- The UI was simplified for a "zero-control" flow: drag & drop or click to upload, processing runs automatically, and you get a preview + Download PNG.
- Controls removed: background selector, threshold, feather sliders, brush controls (brush tool remains in `src/js/brush.js` for advanced edits but is not shown by default).
- Advanced panel: a collapsed <details> contains a single checkbox "Use MediaPipe fallback" for troubleshooting.

How to test the zero-control flow:
1. Serve locally:

```bash
cd background-remover-adsense-ready
python3 -m http.server 5173 --bind 0.0.0.0
# open http://localhost:5173/ in your browser
```
2. Drag & drop or click Upload. The UI will show a small "Processing…" spinner. When finished you'll see the checkerboard preview and the "Download PNG" button will enable.
3. Keyboard: press `D` to download, `R` to replace (open file dialog).
- Model runs in a WebWorker and model bytes are cached in IndexedDB to speed subsequent loads.
- Brush-based manual correction added (erase/restore, size, undo/redo).
- Exports remain PNG with alpha preserved; checkerboard preview added.

How to test (manual):
1. Serve the folder locally:

```bash
cd background-remover-adsense-ready
python3 -m http.server 5173 --bind 0.0.0.0
# open http://localhost:5173/ in your browser
```

2. Upload a PNG with transparency or a standard photo. The app will attempt to load the U²‑NetP ONNX model from the local `/models` folder. If loading fails it will surface an error instead of silently falling back to a different segmentation engine.
3. After processing you'll see a checkerboard preview and can use the brush tools (in the Controls) to erase/restore edges. Use Undo/Redo for corrections.
4. Toggle "Preserve shadows" in the Controls to keep soft shadows (they are treated as semi-transparent alpha).
5. Download the result — it will always be a PNG with alpha.

Notes & limitations
- The provided model URLs are placeholders and may not serve CORS-enabled responses; for reliable use host your ONNX model on a CORS-enabled host (or place it alongside the site) and update `src/js/model.js`.
- Closed-form matting is approximated via a guided filter implementation here; for production consider a WebAssembly-compiled closed-form matting solver for highest quality.
- Performance: large images (>4K) will be tiled in upcoming iterations; currently the pipeline includes optional upscale but will process within the limits of the browser.

License & attribution
- This project bundles third-party model references and CDN usage. Verify licenses for any models and libraries before redistribution.

Maintainer: TheGiat-tech
