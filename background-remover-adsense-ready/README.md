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

License & attribution
- This project bundles third-party model references and CDN usage. Verify licenses for any models and libraries before redistribution.

Maintainer: TheGiat-tech
