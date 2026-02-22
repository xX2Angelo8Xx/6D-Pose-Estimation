#!/usr/bin/env python3
"""
LUT Quality Gallery — Full-Resolution HTML Browser
====================================================
Rendert die schlechtesten N Posen in voller Auflösung (1280×720) und erzeugt
eine interaktive HTML-Galerie mit Zoom, Filter und Sortierung.

Usage:
  python3 tools/lut_quality_gallery.py [--top N] [--out-dir PATH]
"""
from __future__ import annotations
import argparse, base64, csv, json, math
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np

BASE    = Path(__file__).resolve().parent.parent
LUT_NPZ  = BASE / "CAD-Edge-Generation/data/silhouette_lut_roll.npz"
LUT_JSON = BASE / "CAD-Edge-Generation/data/silhouette_lut_roll.lookup.json"
OUT_DIR  = Path(__file__).resolve().parent / "lut_quality_output"
CSV_PATH = OUT_DIR / "lut_quality_report.csv"

LUT_FX = LUT_FY = 800.0
LUT_W, LUT_H = 1280, 720
LUT_CX, LUT_CY = 640.0, 360.0


def project(pts_cam: np.ndarray) -> np.ndarray:
    valid = pts_cam[:, 2] > 0.001
    p = pts_cam[valid]
    u = LUT_FX * p[:, 0] / p[:, 2] + LUT_CX
    v = LUT_FY * p[:, 1] / p[:, 2] + LUT_CY
    return np.column_stack([u, v])


def gap_fraction(pts_cam: np.ndarray) -> float:
    if len(pts_cam) < 4:
        return 1.0
    px = project(pts_cam)
    cx, cy = px[:, 0].mean(), px[:, 1].mean()
    ang = np.sort(np.arctan2(px[:, 1] - cy, px[:, 0] - cx))
    d = np.diff(ang)
    wrap = 2 * math.pi + ang[0] - ang[-1]
    return float(np.append(d, wrap).max()) / (2 * math.pi)


def render_fullres(pts_cam: np.ndarray) -> np.ndarray:
    """Render to RGB 1280×720, white edges on black, with grid overlay."""
    img = np.zeros((LUT_H, LUT_W, 3), np.uint8)
    if len(pts_cam) == 0:
        return img
    px = project(pts_cam)
    xi = np.clip(np.round(px[:, 0]).astype(int), 0, LUT_W - 1)
    yi = np.clip(np.round(px[:, 1]).astype(int), 0, LUT_H - 1)
    img[yi, xi] = (255, 255, 255)
    # Dilate slightly for visibility
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.dilate(img, k)
    # Draw faint grid
    for x in range(0, LUT_W, 128):
        cv2.line(img, (x, 0), (x, LUT_H), (25, 25, 25), 1)
    for y in range(0, LUT_H, 72):
        cv2.line(img, (0, y), (LUT_W, y), (25, 25, 25), 1)
    # Centre crosshair
    cv2.line(img, (int(LUT_CX), 0), (int(LUT_CX), LUT_H), (0, 50, 50), 1)
    cv2.line(img, (0, int(LUT_CY)), (LUT_W, int(LUT_CY)), (0, 50, 50), 1)
    return img


def img_to_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode()


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>LUT Quality Gallery</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #111; color: #ccc; font-family: monospace; }}
  #header {{ background: #1a1a1a; padding: 12px 20px; border-bottom: 1px solid #333;
             display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }}
  #header h1 {{ font-size: 1.1em; color: #8cf; white-space: nowrap; }}
  #stats {{ font-size: 0.82em; color: #888; }}
  .controls {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
  .controls label {{ font-size: 0.8em; color: #aaa; }}
  .controls select, .controls input {{ background: #2a2a2a; color: #ddd;
    border: 1px solid #444; padding: 4px 8px; font-family: monospace; font-size: 0.82em; }}
  #grid {{ display: grid; gap: 8px; padding: 12px;
           grid-template-columns: repeat(var(--cols, 3), 1fr); }}
  .card {{ background: #1c1c1c; border: 2px solid #333; border-radius: 4px;
           cursor: pointer; transition: border-color .15s; overflow: hidden; }}
  .card:hover {{ border-color: #5af; }}
  .card.bad-gap {{ border-color: #c33; }}
  .card.bad-sparse {{ border-color: #c83; }}
  .card.ok {{ border-color: #393; }}
  .card img {{ width: 100%; display: block; image-rendering: pixelated; }}
  .card .meta {{ padding: 6px 8px; font-size: 0.75em; line-height: 1.6; }}
  .meta .key {{ color: #7bf; font-size: 0.85em; }}
  .meta .val {{ color: #fb7; }}
  .badge {{ display: inline-block; padding: 1px 6px; border-radius: 2px;
            font-size: 0.75em; margin-left: 4px; }}
  .badge.gap {{ background: #622; color: #faa; }}
  .badge.sparse {{ background: #642; color: #fca; }}
  .badge.ok  {{ background: #252; color: #afa; }}

  /* Lightbox */
  #lb {{ display: none; position: fixed; inset:0; background: rgba(0,0,0,.9);
         z-index: 999; align-items: center; justify-content: center; flex-direction: column; }}
  #lb.active {{ display: flex; }}
  #lb img {{ max-width: 95vw; max-height: 80vh; image-rendering: auto; border: 1px solid #555; }}
  #lb .lbmeta {{ margin-top: 10px; font-size: 0.85em; text-align: center; color: #aaa; }}
  #lb .close {{ position: absolute; top: 16px; right: 24px; font-size: 2em;
                cursor: pointer; color: #888; line-height: 1; }}
  #lb .nav {{ position: absolute; top: 50%; transform: translateY(-50%);
              font-size: 2.5em; cursor: pointer; color: #888; padding: 10px;
              user-select: none; }}
  #lb .nav:hover {{ color: #5af; }}
  #lb .prev {{ left: 10px; }}
  #lb .next {{ right: 10px; }}
</style>
</head>
<body>
<div id="header">
  <h1>LUT Quality Gallery</h1>
  <div id="stats">Geladen: <span id="nShown">0</span> / {n_total} Posen</div>
  <div class="controls">
    <label>Sortierung:
      <select id="sortSel" onchange="applyFilters()">
        <option value="quality_asc">Quality ↑ (schlechteste zuerst)</option>
        <option value="quality_desc">Quality ↓</option>
        <option value="gap_desc">Gap ↓</option>
        <option value="npts_asc">n_points ↑</option>
        <option value="az_asc">Azimuth ↑</option>
        <option value="el_asc">Elevation ↑</option>
      </select>
    </label>
    <label>Filter:
      <select id="filterSel" onchange="applyFilters()">
        <option value="all">Alle</option>
        <option value="gap">Nur Gap-Fehler</option>
        <option value="sparse">Nur Sparse-Fehler</option>
        <option value="bad">Alle fehlerhaft</option>
        <option value="ok">Nur OK</option>
      </select>
    </label>
    <label>Az:
      <input id="filterAz" type="number" placeholder="z.B. 90" style="width:70px"
             onchange="applyFilters()">
    </label>
    <label>Spalten:
      <input id="colsSel" type="range" min="1" max="6" value="3"
             oninput="document.getElementById('grid').style.setProperty('--cols',this.value)">
    </label>
  </div>
</div>

<div id="grid"></div>

<div id="lb">
  <span class="close" onclick="closeLb()">✕</span>
  <span class="nav prev" onclick="lbNav(-1)">&#9664;</span>
  <img id="lbImg" src="">
  <div class="lbmeta" id="lbMeta"></div>
  <span class="nav next" onclick="lbNav(+1)">&#9654;</span>
</div>

<script>
const DATA = {data_json};

let filtered = [];
let lbIdx = 0;

function classify(d) {{
  if (d.gap_fraction > 0.15) return 'gap';
  if (d.n_points < 2200) return 'sparse';
  return 'ok';
}}

function applyFilters() {{
  const sort   = document.getElementById('sortSel').value;
  const filter = document.getElementById('filterSel').value;
  const azStr  = document.getElementById('filterAz').value.trim();
  const azFilt = azStr !== '' ? parseFloat(azStr) : null;

  filtered = DATA.filter(d => {{
    const cls = classify(d);
    if (filter === 'gap'    && cls !== 'gap')    return false;
    if (filter === 'sparse' && cls !== 'sparse') return false;
    if (filter === 'bad'    && cls === 'ok')     return false;
    if (filter === 'ok'     && cls !== 'ok')     return false;
    if (azFilt !== null && Math.abs(d.az - azFilt) > 1.5) return false;
    return true;
  }});

  const key = sort.replace(/_asc|_desc/, '');
  const asc = sort.endsWith('_asc');
  filtered.sort((a,b) => {{
    const va = a[key] ?? 0, vb = b[key] ?? 0;
    return asc ? va - vb : vb - va;
  }});

  document.getElementById('nShown').textContent = filtered.length;
  renderGrid();
}}

function renderGrid() {{
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  filtered.forEach((d, i) => {{
    const cls = classify(d);
    const card = document.createElement('div');
    card.className = `card ${{cls === 'gap' ? 'bad-gap' : cls === 'sparse' ? 'bad-sparse' : 'ok'}}`;
    const badge = cls === 'gap'
      ? '<span class="badge gap">GAP</span>'
      : cls === 'sparse'
      ? '<span class="badge sparse">SPARSE</span>'
      : '<span class="badge ok">OK</span>';
    card.innerHTML = `
      <img src="data:image/png;base64,${{d.b64}}" loading="lazy">
      <div class="meta">
        <span class="key">${{d.key}}</span>${{badge}}<br>
        <span class="val">n=${{d.n_points}}</span> &nbsp;
        gap=<span class="val">${{d.gap_fraction.toFixed(3)}}</span> &nbsp;
        fill=<span class="val">${{(d.fill_ratio||0).toFixed(3)}}</span> &nbsp;
        Q=<span class="val">${{(d.quality_score||0).toFixed(3)}}</span>
      </div>`;
    card.onclick = () => openLb(i);
    grid.appendChild(card);
  }});
}}

function openLb(i) {{
  lbIdx = i;
  showLb();
  document.getElementById('lb').classList.add('active');
}}
function closeLb() {{
  document.getElementById('lb').classList.remove('active');
}}
function lbNav(dir) {{
  lbIdx = (lbIdx + dir + filtered.length) % filtered.length;
  showLb();
}}
function showLb() {{
  const d = filtered[lbIdx];
  document.getElementById('lbImg').src = `data:image/png;base64,${{d.b64}}`;
  document.getElementById('lbMeta').innerHTML =
    `<b>${{d.key}}</b> &nbsp;|&nbsp; az=${{d.az}}° el=${{d.el}}° roll=${{d.roll}}° &nbsp;|&nbsp; ` +
    `n_points=${{d.n_points}} &nbsp;|&nbsp; gap=${{d.gap_fraction.toFixed(4)}} &nbsp;|&nbsp; ` +
    `fill=${{(d.fill_ratio||0).toFixed(4)}} &nbsp;|&nbsp; Q=${{(d.quality_score||0).toFixed(4)}} &nbsp;` +
    `(${{lbIdx+1}}/${{filtered.length}})`;
}}
document.addEventListener('keydown', e => {{
  if (!document.getElementById('lb').classList.contains('active')) return;
  if (e.key === 'ArrowRight') lbNav(+1);
  if (e.key === 'ArrowLeft')  lbNav(-1);
  if (e.key === 'Escape')     closeLb();
}});

// Init
applyFilters();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=200,
                    help="Render top-N worst poses (default 200)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load existing CSV if available ────────────────────────────────────────
    csv_rows = []
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        print(f"Loaded {len(csv_rows)} rows from CSV")
    else:
        print("No CSV found — will use JSON only")

    print("Loading JSON …")
    with open(LUT_JSON) as f:
        idx = json.load(f)
    poses = idx['poses']

    # Build lookup: key → csv metrics
    csv_by_key = {r['key']: r for r in csv_rows} if csv_rows else {}

    # Select worst N poses
    if csv_rows:
        # Sort by quality_score ascending (worst first), only roll=0
        roll0 = [r for r in csv_rows if abs(float(r.get('roll', 0))) < 0.1]
        def _qs(r):
            v = r.get('quality_score', '')
            return float(v) if v != '' else 1.0
        roll0.sort(key=_qs)
        target_keys = [r['key'] for r in roll0[:args.top]]
    else:
        # Fallback: sort by n_points from JSON
        roll0 = [(k, v) for k, v in poses.items() if v['roll'] == 0.0]
        roll0.sort(key=lambda x: x[1]['n_points'])
        target_keys = [k for k, _ in roll0[:args.top]]

    print(f"Loading NPZ (mmap) …")
    lut = np.load(LUT_NPZ, mmap_mode='r')
    available = set(lut.files)

    print(f"Rendering {len(target_keys)} poses at full resolution …")
    data_records = []

    for i, key in enumerate(target_keys):
        if key not in available:
            print(f"  [{i+1}/{len(target_keys)}] SKIP (not in NPZ): {key}")
            continue

        pts = lut[key].copy()
        img = render_fullres(pts)

        # Metrics
        gf = gap_fraction(pts)
        n  = int(len(pts))
        csv_r = csv_by_key.get(key, {})
        _fill_s = csv_r.get('fill_ratio', '')
        _qs_s   = csv_r.get('quality_score', '')
        fill = float(_fill_s) if _fill_s != '' else 0.0
        qs   = float(_qs_s)   if _qs_s   != '' else fill * (1 - gf)

        pose  = poses.get(key, {})
        _az_s = csv_r.get('az', '')
        _el_s = csv_r.get('el', '')
        az    = float(pose.get('azimuth',   _az_s   if _az_s   != '' else 0))
        el    = float(pose.get('elevation', _el_s   if _el_s   != '' else 0))
        roll  = float(pose.get('roll', 0))

        # Overlay label on image
        label = (f"az={az:.1f} el={el:.1f} roll={roll:.0f} | "
                 f"n={n} gap={gf:.3f} Q={qs:.3f}")
        cv2.putText(img, label, (8, LUT_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

        b64 = img_to_b64(img)

        data_records.append({
            "key": key,
            "az": az, "el": el, "roll": roll,
            "n_points": n,
            "gap_fraction": round(gf, 5),
            "fill_ratio": round(fill, 5),
            "quality_score": round(qs, 5),
            "b64": b64,
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(target_keys)} done …")

    print(f"Building HTML …")
    import json as _json
    data_json = _json.dumps(data_records)

    html = HTML_TEMPLATE.format(
        n_total=len(poses),
        data_json=data_json,
    )

    gallery_path = out / "lut_quality_gallery.html"
    gallery_path.write_text(html, encoding="utf-8")
    size_mb = gallery_path.stat().st_size / 1024 / 1024
    print(f"\n✓ Gallery saved → {gallery_path}  ({size_mb:.1f} MB)")
    print(f"  {len(data_records)} Posen in Galerie")
    print(f"\n  Öffnen mit:  xdg-open {gallery_path}")


if __name__ == "__main__":
    main()
