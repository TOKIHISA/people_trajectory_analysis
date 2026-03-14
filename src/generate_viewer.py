"""
wgs84.json から MapLibre 軌跡ビューア HTML を生成する

Usage:
  python generate_viewer.py                          # デフォルトのoutputから読む
  python generate_viewer.py -i path/to/wgs84.json    # 指定ファイル

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
    """trajectory_dir 内の *_wgs84.json を探す"""
    files = sorted(Path(trajectory_dir).glob("*_wgs84.json"))
    if not files:
        return None
    if len(files) == 1:
        return files[0]
    print("複数のwgs84.jsonがあります:")
    for i, f in enumerate(files):
        print(f"  {i + 1}. {f.name}")
    idx = input(f"番号を選択 (1-{len(files)}): ").strip()
    try:
        return files[int(idx) - 1]
    except (ValueError, IndexError):
        return None


def generate_html(data, json_filename):
    """軌跡データからHTMLを生成"""
    tracks = [t for t in data.get("tracks", [])
              if t.get("trajectory_wgs84") and len(t["trajectory_wgs84"]) >= 2]

    if not tracks:
        print("エラー: 表示可能な軌跡がありません")
        return None

    # 最初の軌跡の中間点を地図中心にする
    first = tracks[0]["trajectory_wgs84"]
    mid = first[len(first) // 2]
    center_lon = mid["lon"]
    center_lat = mid["lat"]

    # 軌跡データをJS用に変換（lon / lat / time_sec）
    js_tracks = []
    for track in tracks:
        coords = [[p["lon"], p["lat"], p["time_sec"]] for p in track["trajectory_wgs84"]]
        js_tracks.append({
            "coords": coords,
            "start": coords[0][2],
            "end":   coords[-1][2],
        })

    # 動画全体の時間長（全トラックの最終時刻）
    video_duration = max(t["end"] for t in js_tracks)
    # MM:SS 文字列
    dur_m = int(video_duration // 60)
    dur_s = int(video_duration % 60)
    dur_str = f"{dur_m:02d}:{dur_s:02d}"

    # トラックごとの色を事前計算（Python側で）
    track_colors = []
    n = len(js_tracks)
    for i in range(n):
        hue = 200 if n == 1 else (i * 137.508) % 360  # 黄金角で隣接インデックスを最大分離
        # HSL→RGB 変換（s=0.9, l=0.6）
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

// 全トラックの背景座標（lon/lat のみ・固定）
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
    layers: [{{ id: 'osm', type: 'raster', source: 'osm' }}]  // maxzoom 省略 = 常時表示
  }},
  center: CENTER,
  zoom: 19
}});

map.on('load', () => {{
  // ─── ソースは全トラック合計 2つだけ ───────────────────────────
  // 背景: 活動中トラックの全経路（更新は activate/deactivate 時のみ）
  map.addSource('bg', {{ type: 'geojson', data: EMPTY_FC }});
  map.addLayer({{
    id: 'bg', type: 'line', source: 'bg',
    paint: {{ 'line-color': ['get', 'color'], 'line-width': 2, 'line-opacity': 0.15 }}
  }});

  // トレイル: 毎フレーム更新（現在時刻付近の軌跡スライス）
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
  const activeSet = new Set(); // 活動中トラックのインデックス集合

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

      // 背景の表示状態が変わった時だけ dirty フラグを立てる
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

    // トレイル: 毎フレーム 1回だけ setData
    map.getSource('trail').setData({{ type: 'FeatureCollection', features: trailFeatures }});

    // 背景: activate/deactivate が発生した時だけ更新
    if (bgDirty) map.getSource('bg').setData(buildBgFC(activeSet));

    requestAnimationFrame(animate);
  }}
  requestAnimationFrame(animate);
}});
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="軌跡ビューアHTMLを生成")
    parser.add_argument("-i", "--input", type=str, default=None,
                        help="wgs84.json ファイルパス")
    parser.add_argument("-o", "--output-dir", type=str, default=None,
                        help="出力ディレクトリ（デフォルト: output/viewer）")
    args = parser.parse_args()

    # 入力ファイル
    if args.input:
        json_path = Path(args.input)
    else:
        json_path = find_wgs84_json(TRAJECTORY_DIR)

    if json_path is None or not json_path.exists():
        print(f"エラー: wgs84.json が見つかりません")
        return

    # 出力先
    if args.output_dir:
        viewer_dir = Path(args.output_dir)
    else:
        viewer_dir = Path(VIEWER_DIR)
    viewer_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    html = generate_html(data, json_path.name)
    if html is None:
        return

    out_path = viewer_dir / f"{json_path.stem}_viewer.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"ビューア生成完了: {out_path}")

    # GCPコンフィグをビューアと同名で隣に保存
    script_dir = Path(__file__).parent
    gcp_src = (script_dir / GCP_CONFIG_PATH).resolve()
    if gcp_src.exists():
        gcp_dst = viewer_dir / f"{json_path.stem}_viewer_gcp.json"
        shutil.copy(gcp_src, gcp_dst)
        print(f"GCPコンフィグ保存: {gcp_dst}")
    else:
        print(f"警告: GCPコンフィグが見つかりません: {gcp_src}")

    print(f"ブラウザで直接開けます（サーバー不要）")


if __name__ == "__main__":
    main()
