"""
反応特徴量の情報幾何学的分析メインスクリプト

入力:  people_trajectory/output/trajectory/ 以下の *_wgs84.json
       （trajectory フィールド = 画像座標を使用。wgs84 フィールドは不使用）
出力:  people_trajectory/analysis/output/
         reaction_analysis.json   全人物のスタンディング + 地点サマリ
         reaction_viewer.html     インタラクティブビューア

【2パス処理】
  Pass 1: 地点（動画）ごとに全トラックの速度を収集
          → 地点内パーセンタイル閾値を計算
  Pass 2: その閾値で各トラックの反応分布 θ を計算
          → スタンディングを算出

Usage:
  python analyze_reactions.py
  python analyze_reactions.py --input path/to/wgs84.json [...]
  python analyze_reactions.py --reference uniform

License: MIT License
Author: Toki Hirose
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from reaction_features import (
    compute_pixel_speed_profile,
    compute_location_thresholds,
    extract_invariant_features,
)
from information_geometry import compute_ig_positions, summarize_location


# ── デフォルトパス設定 ────────────────────────────────────────────
_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent.parent
_TRAJECTORY_DIR = _PROJECT_ROOT / 'output' / 'trajectory'
_OUTPUT_DIR = _HERE.parent / 'output'


def load_json(path: Path) -> list:
    """*_wgs84.json を読み込み、トラックリストを返す（video 名付き）"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    video_name = data.get('video_name', path.stem)
    tracks = []
    for track in data.get('tracks', []):
        track['_video'] = video_name
        tracks.append(track)
    return tracks


def find_input_files(trajectory_dir: Path) -> list[Path]:
    """trajectory_dir 内の *_wgs84.json を列挙"""
    return sorted(trajectory_dir.glob('*_wgs84.json'))


def run_analysis(input_paths: list[Path], reference: str) -> dict:
    """
    全入力ファイルから反応特徴量を抽出し、スタンディングを計算する

    Pass 1: 地点ごとに全速度を収集してパーセンタイル閾値を計算
    Pass 2: 閾値を使って各トラックの特徴量・スタンディングを計算

    Returns:
        {
          'all_standings':   全人物のスタンディングリスト,
          'by_location':     地点ごとのサマリ,
          'thresholds':      地点ごとの速度閾値 [px/s],
        }
    """
    # ── 全トラックをロード ───────────────────────────────────────
    all_tracks = []
    for path in input_paths:
        print(f"  読み込み: {path.name}")
        all_tracks.extend(load_json(path))

    if not all_tracks:
        print("エラー: 有効な軌跡データが見つかりません")
        return {}

    # ── Pass 1: 地点ごとに速度収集 → 閾値計算 ───────────────────
    location_speeds: dict[str, list] = defaultdict(list)
    for track in all_tracks:
        traj = track.get('trajectory', [])
        if len(traj) < 2:
            continue
        video = track.get('_video', '')
        speeds = compute_pixel_speed_profile(traj)
        location_speeds[video].extend(speeds.tolist())

    thresholds_by_loc: dict[str, np.ndarray] = {}
    print("\n  地点別速度閾値 [px/s]:")
    for video, speeds in location_speeds.items():
        thr = compute_location_thresholds(np.array(speeds))
        thresholds_by_loc[video] = thr
        print(f"    {video[:40]}")
        print(f"      stopped<{thr[0]:.1f}  slow<{thr[1]:.1f}"
              f"  medium<{thr[2]:.1f}  fast>={thr[2]:.1f}")

    # ── Pass 2: 特徴量抽出 → スタンディング計算 ─────────────────
    all_features = []
    for track in all_tracks:
        video = track.get('_video', '')
        thr = thresholds_by_loc.get(video)
        if thr is None:
            continue
        feat = extract_invariant_features(track, thresholds=thr, video_name=video)
        if feat is not None:
            all_features.append(feat)

    print(f"\n  有効トラック数: {len(all_features)}")

    if not all_features:
        print("エラー: 有効な軌跡データが見つかりません")
        return {}

    standings = compute_ig_positions(all_features, reference=reference)

    # 地点ごとのサマリ
    videos = sorted({s['video'] for s in standings})
    by_location = {}
    for v in videos:
        loc_s = [s for s in standings if s['video'] == v]
        summary = summarize_location(loc_s)
        summary['thresholds_px_s'] = thresholds_by_loc.get(v, np.zeros(3)).tolist()
        by_location[v] = summary

    return {
        'all_standings': standings,
        'by_location':   by_location,
        'thresholds':    {k: v.tolist() for k, v in thresholds_by_loc.items()},
    }


def save_json(result: dict, output_dir: Path) -> Path:
    out_path = output_dir / 'reaction_analysis.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  JSON 保存: {out_path}")
    return out_path


def generate_html(result: dict, output_dir: Path) -> Path:
    standings  = result.get('all_standings', [])
    by_location = result.get('by_location', {})

    if not standings:
        return None

    js_data = json.dumps({
        'standings':  standings,
        'byLocation': by_location,
    }, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>反応特徴量 スタンディングビューア</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: sans-serif; background: #111; color: #eee; padding: 16px; }}
    h1 {{ font-size: 18px; margin-bottom: 12px; color: #7cf; }}
    h2 {{ font-size: 14px; color: #adf; margin: 16px 0 6px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .card {{ background: #1e1e2e; border-radius: 8px; padding: 14px; }}
    .card.full {{ grid-column: 1 / -1; }}
    canvas {{ max-height: 380px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
    th {{ background: #2a2a3e; padding: 5px 8px; text-align: left; white-space: nowrap; }}
    td {{ padding: 4px 8px; border-bottom: 1px solid #333; white-space: nowrap; }}
    tr:hover td {{ background: #2a2a3e; }}
    select {{ background: #2a2a3e; color: #eee; border: 1px solid #444;
              padding: 3px 6px; border-radius: 4px; }}
    .controls {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:10px; }}
    label {{ font-size:12px; color:#aaa; }}
    .note {{ font-size:11px; color:#666; margin-top:6px; }}
  </style>
</head>
<body>
<h1>反応特徴量 スタンディングビューア</h1>
<p class="note">
  速度は画像内ピクセル速度を地点内パーセンタイルで相対分類。
  e-measure = 行動の特化度（KL(ref‖p)）&nbsp;
  m-measure = 行動の希少性（KL(p‖ref)）
</p>

<div class="controls">
  <label>地点フィルタ: <select id="locFilter"><option value="">全地点</option></select></label>
  <label>PGA色分け:
    <select id="colorBy">
      <option value="e_measure">e-測度（特化度）</option>
      <option value="m_measure">m-測度（希少性）</option>
      <option value="j_divergence">J-divergence</option>
      <option value="e_minus_m">e − m</option>
      <option value="sinuosity">迂回率</option>
      <option value="duration_sec">滞在時間</option>
    </select>
  </label>
</div>

<div class="grid">

  <div class="card">
    <h2>多様体スタンディング（PGA空間）</h2>
    <canvas id="pgaChart"></canvas>
  </div>

  <div class="card">
    <h2>e-測度（特化度）vs m-測度（希少性）</h2>
    <canvas id="emChart"></canvas>
  </div>

  <div class="card full">
    <h2>地点別 反応組成（Fréchet 平均）</h2>
    <canvas id="locChart"></canvas>
  </div>

  <div class="card full">
    <h2>個人スタンディング一覧（e-measure 降順）</h2>
    <table>
      <thead><tr>
        <th>動画</th><th>ID</th>
        <th>duration(s)</th><th>迂回率</th><th>停止回数</th>
        <th>e-measure</th><th>m-measure</th><th>e−m</th><th>J-div</th>
        <th>θ_stopped</th><th>θ_slow</th><th>θ_medium</th><th>θ_fast</th>
        <th>閾値[px/s]</th>
      </tr></thead>
      <tbody id="tableBody"></tbody>
    </table>
  </div>

</div>

<script>
const RAW = {js_data};
const standings  = RAW.standings;
const byLocation = RAW.byLocation;
const STATE_LABELS  = ['stopped','slow','medium','fast'];
const STATE_COLORS  = ['#e74c3c','#f39c12','#2ecc71','#3498db'];

function colorScale(v, mn, mx) {{
  const t = mx > mn ? (v - mn) / (mx - mn) : 0.5;
  return `rgba(${{Math.round(255*(1-t))}}, ${{Math.round(200*t)}}, ${{Math.round(255*t)}}, 0.8)`;
}}
function fmt(v, d=3) {{ return typeof v === 'number' ? v.toFixed(d) : (v ?? '-'); }}

// 地点フィルタ
const locFilter = document.getElementById('locFilter');
[...new Set(standings.map(s => s.video))].sort().forEach(v => {{
  const o = document.createElement('option');
  o.value = v; o.textContent = v;
  locFilter.appendChild(o);
}});
function getFiltered() {{
  const v = locFilter.value;
  return v ? standings.filter(s => s.video === v) : standings;
}}

// PGA 散布図
let pgaChart = null;
function buildPGA(data) {{
  const key = document.getElementById('colorBy').value;
  const vals = data.map(s => s[key] ?? 0);
  const mn = Math.min(...vals), mx = Math.max(...vals);
  if (pgaChart) pgaChart.destroy();
  pgaChart = new Chart(document.getElementById('pgaChart'), {{
    type: 'scatter',
    data: {{ datasets: data.map(s => ({{
      label: `${{s.video}} #${{s.track_id}}`,
      data: [{{ x: s.pga_x, y: s.pga_y }}],
      backgroundColor: colorScale(s[key] ?? 0, mn, mx),
      pointRadius: 6, pointHoverRadius: 9, _s: s,
    }})) }},
    options: {{
      plugins: {{ legend: {{ display:false }},
        tooltip: {{ callbacks: {{ label: ctx => {{
          const s = ctx.dataset._s;
          return [`${{s.video}} #${{s.track_id}}`,
                  `e=${{fmt(s.e_measure)}} m=${{fmt(s.m_measure)}}`,
                  `θ=[${{s.theta.map(v=>(v*100).toFixed(0)+'%').join(' ')}}]`];
        }} }} }} }},
      scales: {{
        x: {{ title:{{display:true,text:'PGA-1',color:'#aaa'}}, grid:{{color:'#333'}}, ticks:{{color:'#aaa'}} }},
        y: {{ title:{{display:true,text:'PGA-2',color:'#aaa'}}, grid:{{color:'#333'}}, ticks:{{color:'#aaa'}} }},
      }},
    }},
  }});
}}

// e/m 散布図
let emChart = null;
function buildEM(data) {{
  if (emChart) emChart.destroy();
  emChart = new Chart(document.getElementById('emChart'), {{
    type: 'scatter',
    data: {{ datasets: data.map(s => ({{
      label: `${{s.video}} #${{s.track_id}}`,
      data: [{{ x: s.e_measure, y: s.m_measure }}],
      backgroundColor: s.e_minus_m > 0 ? 'rgba(100,180,255,0.7)' : 'rgba(255,140,80,0.7)',
      pointRadius: 5, pointHoverRadius: 8, _s: s,
    }})) }},
    options: {{
      plugins: {{ legend: {{ display:false }},
        tooltip: {{ callbacks: {{ label: ctx => {{
          const s = ctx.dataset._s;
          return [`${{s.video}} #${{s.track_id}}`,
                  `e=${{fmt(s.e_measure)}} 特化度`,
                  `m=${{fmt(s.m_measure)}} 希少性`,
                  `e−m=${{fmt(s.e_minus_m)}}`];
        }} }} }} }},
      scales: {{
        x: {{ title:{{display:true,text:'e-measure（特化度）',color:'#aaa'}}, grid:{{color:'#333'}}, ticks:{{color:'#aaa'}} }},
        y: {{ title:{{display:true,text:'m-measure（希少性）',color:'#aaa'}}, grid:{{color:'#333'}}, ticks:{{color:'#aaa'}} }},
      }},
    }},
  }});
}}

// 地点別積み上げ棒グラフ
let locChart = null;
function buildLocChart() {{
  const locs = Object.keys(byLocation);
  if (locChart) locChart.destroy();
  locChart = new Chart(document.getElementById('locChart'), {{
    type: 'bar',
    data: {{
      labels: locs.map(v => v.replace('.mp4','')),
      datasets: STATE_LABELS.map((lbl, ki) => ({{
        label: lbl,
        data: locs.map(v => byLocation[v].frechet_mean[ki]),
        backgroundColor: STATE_COLORS[ki],
      }})),
    }},
    options: {{
      plugins: {{
        legend: {{ labels:{{color:'#eee'}} }},
        tooltip: {{ callbacks: {{ label: ctx =>
          ` ${{ctx.dataset.label}}: ${{(ctx.raw*100).toFixed(1)}}%` }} }},
      }},
      scales: {{
        x: {{ stacked:true, ticks:{{color:'#aaa'}}, grid:{{color:'#333'}} }},
        y: {{ stacked:true, max:1,
              ticks:{{color:'#aaa', callback:v=>(v*100).toFixed(0)+'%'}},
              grid:{{color:'#333'}} }},
      }},
    }},
  }});
}}

// テーブル
function buildTable(data) {{
  const tbody = document.getElementById('tableBody');
  tbody.innerHTML = '';
  [...data].sort((a,b) => b.e_measure - a.e_measure).forEach(s => {{
    const th = s.theta;
    const thr = s.thresholds_px_s ? s.thresholds_px_s.map(v=>v.toFixed(1)).join('/') : '-';
    tbody.insertAdjacentHTML('beforeend', `<tr>
      <td>${{s.video}}</td><td>${{s.track_id}}</td>
      <td>${{fmt(s.duration_sec,1)}}</td>
      <td>${{fmt(s.sinuosity,3)}}</td>
      <td>${{s.num_stops}}</td>
      <td style="color:#7cf">${{fmt(s.e_measure,4)}}</td>
      <td style="color:#fc7">${{fmt(s.m_measure,4)}}</td>
      <td style="color:${{s.e_minus_m>0?'#7cf':'#fc7'}}">${{fmt(s.e_minus_m,4)}}</td>
      <td>${{fmt(s.j_divergence,4)}}</td>
      ${{th.map((v,i)=>`<td style="color:${{STATE_COLORS[i]}}">${{(v*100).toFixed(1)}}%</td>`).join('')}}
      <td style="color:#888">${{thr}}</td>
    </tr>`);
  }});
}}

function refresh() {{
  const data = getFiltered();
  buildPGA(data); buildEM(data); buildTable(data);
}}

locFilter.addEventListener('change', refresh);
document.getElementById('colorBy').addEventListener('change', () => buildPGA(getFiltered()));
buildLocChart();
refresh();
</script>
</body>
</html>
"""
    out_path = output_dir / 'reaction_viewer.html'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  HTML 保存: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description='反応特徴量の情報幾何学的分析')
    parser.add_argument('--input', '-i', nargs='*', type=Path)
    parser.add_argument('--reference', '-r',
                        choices=['frechet_mean', 'uniform'], default='frechet_mean')
    parser.add_argument('--output-dir', '-o', type=Path, default=_OUTPUT_DIR)
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input] if args.input \
                  else find_input_files(_TRAJECTORY_DIR)

    if not input_paths:
        print(f"エラー: *_wgs84.json が見つかりません: {_TRAJECTORY_DIR}")
        sys.exit(1)

    print('=' * 60)
    print('反応特徴量 情報幾何学的分析（画像座標 / 相対速度）')
    print('=' * 60)
    print(f"入力ファイル数: {len(input_paths)}  参照分布: {args.reference}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = run_analysis(input_paths, reference=args.reference)
    if not result:
        sys.exit(1)

    save_json(result, args.output_dir)
    generate_html(result, args.output_dir)

    print('\n── 地点別サマリ ──')
    for loc, s in result['by_location'].items():
        label = loc.replace('_analysis_wgs84', '').replace('.mp4', '')
        fm = s.get('frechet_mean', [])
        lbls = s.get('theta_labels', ['stopped', 'slow', 'medium', 'fast'])
        comp = '  '.join(f"{l}:{v*100:.1f}%" for l, v in zip(lbls, fm))
        thr = s.get('thresholds_px_s', [])
        print(f"  {label}")
        print(f"    人数: {s['n_persons']}  閾値[px/s]: {[round(t,1) for t in thr]}")
        print(f"    反応組成: {comp}")
        print(f"    e={s['e_mean']:.4f}  m={s['m_mean']:.4f}  e-m={s['e_minus_m_mean']:.4f}")

    print('\n' + '=' * 60 + '\n完了\n' + '=' * 60)


if __name__ == '__main__':
    main()
