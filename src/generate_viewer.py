"""
Generate a MapLibre trajectory viewer HTML from a wgs84.json file

Usage:
  python generate_viewer.py                          # reads from default output directory
  python generate_viewer.py -i path/to/wgs84.json    # specify input file

License: MIT License
Author: Toki Hirose
"""

import json
import os
import shutil
import argparse
from pathlib import Path
from config import TRAJECTORY_DIR, VIEWER_DIR, GCP_CONFIG_PATH


def find_wgs84_json(trajectory_dir):
    """Find *_wgs84.json files in trajectory_dir"""
    files = sorted(Path(trajectory_dir).glob("*_wgs84.json"))
    if not files:
        return None
    if len(files) == 1:
        return files[0]
    print("Multiple wgs84.json files found:")
    for i, f in enumerate(files):
        print(f"  {i + 1}. {f.name}")
    idx = input(f"Select a number (1-{len(files)}): ").strip()
    try:
        return files[int(idx) - 1]
    except (ValueError, IndexError):
        return None


def generate_html(data, json_filename):
    """Generate HTML from trajectory data"""
    tracks = [t for t in data.get("tracks", [])
              if t.get("trajectory_wgs84") and len(t["trajectory_wgs84"]) >= 2]

    if not tracks:
        print("Error: no displayable trajectories found")
        return None

    # Use the midpoint of the first trajectory as the map center
    first = tracks[0]["trajectory_wgs84"]
    mid = first[len(first) // 2]
    center_lon = mid["lon"]
    center_lat = mid["lat"]

    # Convert trajectory data for JavaScript (lon / lat / time_sec)
    js_tracks = []
    for track in tracks:
        coords = [[p["lon"], p["lat"], p["time_sec"]] for p in track["trajectory_wgs84"]]
        js_tracks.append({
            "coords": coords,
            "start": coords[0][2],
            "end":   coords[-1][2],
        })

    # Total video duration (latest end time across all tracks)
    video_duration = max(t["end"] for t in js_tracks)
    # MM:SS string
    dur_m = int(video_duration // 60)
    dur_s = int(video_duration % 60)
    dur_str = f"{dur_m:02d}:{dur_s:02d}"

    # Pre-compute per-track colors in Python
    track_colors = []
    n = len(js_tracks)
    for i in range(n):
        hue = 200 if n == 1 else (i * 137.508) % 360  # golden angle to maximize separation between adjacent indices
        # HSL to RGB conversion (s=0.9, l=0.6)
        h, s, l = hue / 360, 0.9, 0.6
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((hue / 60) % 2 - 1))
        m = l - c / 2
        if hue < 60:   r, g, b = c, x, 0
        elif hue < 120: r, g, b = x, c, 0
        elif hue < 180: r, g, b = 0, c, x
        elif hue < 240: r, g, b = 0, x, c
        elif hue < 300: r, g, b = x, 0, c
        else:           r, g, b = c, 0, x
        ri, gi, bi = int((r+m)*255), int((g+m)*255), int((b+m)*255)
        track_colors.append(f"rgb({ri},{gi},{bi})")

    tracks_json    = json.dumps(js_tracks)
    colors_json    = json.dumps(track_colors)

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trajectory Viewer - {json_filename}</title>
<link rel="stylesheet" href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css">
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; background: #1a1a2e; }}
  #map {{ width: 100%; height: 100%; }}
  #info {{
    position: absolute; top: 10px; left: 10px; z-index: 1;
    background: rgba(0,0,0,0.7); color: #fff; padding: 8px 14px;
    border-radius: 6px; font: 13px/1.4 sans-serif;
  }}
  #controls {{
    position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
    z-index: 1; background: rgba(0,0,0,0.7); color: #fff; padding: 10px 16px;
    border-radius: 8px; font: 12px sans-serif;
    display: flex; flex-direction: column; gap: 8px; align-items: center;
    min-width: 340px;
  }}
  #timeline-row {{ display: flex; align-items: center; gap: 8px; width: 100%; }}
  #timeline {{ flex: 1; }}
  #time-display {{ font: bold 13px monospace; min-width: 110px; text-align: right; }}
  #param-row {{ display: flex; gap: 16px; align-items: center; }}
  #param-row label {{ display: flex; align-items: center; gap: 6px; }}
  #param-row input[type=range] {{ width: 80px; }}
  #param-row select {{ background: #333; color: #fff; border: none; padding: 2px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="info">
  <strong>Trajectory Viewer</strong><br>
  Source: {json_filename}<br>
  Tracks: {len(tracks)}
</div>
<div id="controls">
  <div id="timeline-row">
    <input type="range" id="timeline" min="0" max="{video_duration:.2f}" step="0.1" value="0">
    <span id="time-display">00:00 / {dur_str}</span>
  </div>
  <div id="param-row">
    <label>Speed
      <select id="speed">
        <option value="1">1x</option>
        <option value="5">5x</option>
        <option value="10" selected>10x</option>
        <option value="30">30x</option>
        <option value="60">60x</option>
        <option value="120">120x</option>
      </select>
    </label>
    <label>Trail
      <input type="range" id="trail" min="1" max="30" value="5">
      <span id="trail-val">5s</span>
    </label>
  </div>
</div>

<script>
const TRACKS        = {tracks_json};
const TRACK_COLORS  = {colors_json};
const CENTER        = [{center_lon}, {center_lat}];
const VIDEO_DURATION = {video_duration:.2f};

// Background coordinates for all tracks (lon/lat only, static)
const BG_COORDS = TRACKS.map(t => t.coords.map(([lon, lat]) => [lon, lat]));

function fmtTime(s) {{
  return String(Math.floor(s / 60)).padStart(2,'0') + ':' + String(Math.floor(s % 60)).padStart(2,'0');
}}

function upperIdx(coords, t) {{
  let lo = 0, hi = coords.length - 1, res = -1;
  while (lo <= hi) {{
    const mid = (lo + hi) >> 1;
    if (coords[mid][2] <= t) {{ res = mid; lo = mid + 1; }}
    else hi = mid - 1;
  }}
  return res;
}}

function lowerIdx(coords, t) {{
  let lo = 0, hi = coords.length - 1, res = coords.length;
  while (lo <= hi) {{
    const mid = (lo + hi) >> 1;
    if (coords[mid][2] >= t) {{ res = mid; hi = mid - 1; }}
    else lo = mid + 1;
  }}
  return res;
}}

const EMPTY_FC = {{ type: 'FeatureCollection', features: [] }};

const map = new maplibregl.Map({{
  container: 'map',
  style: {{
    version: 8,
    sources: {{
      osm: {{
        type: 'raster',
        tiles: ['https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png'],
        tileSize: 256,
        attribution: '&copy; OpenStreetMap contributors'
      }}
    }},
    layers: [{{ id: 'osm', type: 'raster', source: 'osm' }}]  // no maxzoom = always visible
  }},
  center: CENTER,
  zoom: 19
}});

map.on('load', () => {{
  // ─── Only 2 sources total for all tracks ───────────────────────────
  // Background: full path of each active track (updated only on activate/deactivate)
  map.addSource('bg', {{ type: 'geojson', data: EMPTY_FC }});
  map.addLayer({{
    id: 'bg', type: 'line', source: 'bg',
    paint: {{ 'line-color': ['get', 'color'], 'line-width': 2, 'line-opacity': 0.15 }}
  }});

  // Trail: updated every frame (trajectory slice around current time)
  map.addSource('trail', {{ type: 'geojson', data: EMPTY_FC }});
  map.addLayer({{
    id: 'trail-glow', type: 'line', source: 'trail',
    paint: {{ 'line-color': ['get', 'color'], 'line-width': 8, 'line-blur': 6, 'line-opacity': 0.3 }}
  }});
  map.addLayer({{
    id: 'trail-line', type: 'line', source: 'trail',
    paint: {{ 'line-color': ['get', 'color'], 'line-width': 3 }}
  }});

  const timelineEl  = document.getElementById('timeline');
  const timeDisplay = document.getElementById('time-display');
  const trailEl     = document.getElementById('trail');
  const trailValEl  = document.getElementById('trail-val');
  const durStr      = fmtTime(VIDEO_DURATION);

  trailEl.addEventListener('input', () => {{ trailValEl.textContent = trailEl.value + 's'; }});

  let userScrubbing = false;
  timelineEl.addEventListener('mousedown', () => {{ userScrubbing = true; }});
  timelineEl.addEventListener('mouseup',   () => {{ userScrubbing = false; currentTime = parseFloat(timelineEl.value); }});

  let currentTime = 0;
  let lastTs      = null;
  const activeSet = new Set(); // set of indices of currently active tracks

  function buildBgFC(active) {{
    return {{
      type: 'FeatureCollection',
      features: [...active].map(i => ({{
        type: 'Feature',
        properties: {{ color: TRACK_COLORS[i] }},
        geometry: {{ type: 'LineString', coordinates: BG_COORDS[i] }}
      }}))
    }};
  }}

  function animate(ts) {{
    if (lastTs !== null && !userScrubbing) {{
      currentTime += (ts - lastTs) / 1000 * parseFloat(document.getElementById('speed').value);
      if (currentTime > VIDEO_DURATION) currentTime = 0;
    }}
    lastTs = ts;

    if (!userScrubbing) timelineEl.value = currentTime;
    timeDisplay.textContent = fmtTime(currentTime) + ' / ' + durStr;

    const trailSec   = parseFloat(trailEl.value);
    const trailStart = currentTime - trailSec;

    let bgDirty = false;
    const trailFeatures = [];

    for (let i = 0; i < TRACKS.length; i++) {{
      const track    = TRACKS[i];
      const coords   = track.coords;
      const isActive = currentTime >= track.start && trailStart <= track.end;

      // Set dirty flag only when a track's active state changes
      if (isActive !== activeSet.has(i)) {{
        isActive ? activeSet.add(i) : activeSet.delete(i);
        bgDirty = true;
      }}

      if (!isActive) continue;

      const headIdx = upperIdx(coords, currentTime);
      if (headIdx < 0) continue;
      const tailIdx = lowerIdx(coords, trailStart);
      const slice   = coords.slice(tailIdx, headIdx + 1).map(([lon, lat]) => [lon, lat]);

      if (slice.length >= 2) {{
        trailFeatures.push({{
          type: 'Feature',
          properties: {{ color: TRACK_COLORS[i] }},
          geometry: {{ type: 'LineString', coordinates: slice }}
        }});
      }}
    }}

    // Trail: call setData exactly once per frame
    map.getSource('trail').setData({{ type: 'FeatureCollection', features: trailFeatures }});

    // Background: update only when a track is activated or deactivated
    if (bgDirty) map.getSource('bg').setData(buildBgFC(activeSet));

    requestAnimationFrame(animate);
  }}
  requestAnimationFrame(animate);
}});
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate trajectory viewer HTML")
    parser.add_argument("-i", "--input", type=str, default=None,
                        help="Path to wgs84.json file")
    parser.add_argument("-o", "--output-dir", type=str, default=None,
                        help="Output directory (default: output/viewer)")
    args = parser.parse_args()

    # Input file
    if args.input:
        json_path = Path(args.input)
    else:
        json_path = find_wgs84_json(TRAJECTORY_DIR)

    if json_path is None or not json_path.exists():
        print(f"Error: wgs84.json not found")
        return

    # Output directory
    if args.output_dir:
        viewer_dir = Path(args.output_dir)
    else:
        viewer_dir = Path(VIEWER_DIR)
    viewer_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    html = generate_html(data, json_path.name)
    if html is None:
        return

    out_path = viewer_dir / f"{json_path.stem}_viewer.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Viewer generated: {out_path}")

    # Copy GCP config alongside the viewer file
    script_dir = Path(__file__).parent
    gcp_src = (script_dir / GCP_CONFIG_PATH).resolve()
    if gcp_src.exists():
        gcp_dst = viewer_dir / f"{json_path.stem}_viewer_gcp.json"
        shutil.copy(gcp_src, gcp_dst)
        print(f"GCP config saved: {gcp_dst}")
    else:
        print(f"Warning: GCP config not found: {gcp_src}")

    print(f"Open the HTML file directly in a browser (no server required)")


if __name__ == "__main__":
    main()
