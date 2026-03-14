"""
軌跡データから反応特徴量を抽出するモジュール

【座標空間の選択】
  画像座標（trajectory: ピクセル）を使用する。
  WGS84（trajectory_wgs84）を使わない理由:
    - _simplify_trajectory で中間点が大量に間引かれ、状態滞在時間の推定が不正確
    - ホモグラフィーの歪みが GCP 凸包の端ほど大きい

【透視歪みへの対処】
  画像座標にも透視歪みがある（遠くの人は同じ速度でも小さく動く）。
  これを地点内相対速度で補正する:
    その地点の全トラック速度の 25/50/75 パーセンタイルを閾値として使い
    「遅い・普通・速い」を相対的に分類する。
  → 透視歪みは地点内で一様にかかるため、相対比較で打ち消せる。

【2パス処理】
  Pass 1: 地点（動画）ごとに全速度を収集 → パーセンタイル閾値を計算
  Pass 2: その閾値を使って各トラックの反応分布 θ を計算

【反応特徴量 θ】
  θ = (θ_stopped, θ_slow, θ_medium, θ_fast) ∈ S³（確率単体）
  各成分 = その状態で過ごした時間割合

License: MIT License
Author: Toki Hirose
"""

import math
import numpy as np
from typing import List, Dict, Optional, Tuple


STATE_LABELS = ['stopped', 'slow', 'medium', 'fast']
N_STATES = len(STATE_LABELS)

# Dirichlet smoothing: 確率0を避けるための微小量
DIRICHLET_EPS = 1e-4

# 方向転換と判定する角度閾値 [deg]
DIRECTION_CHANGE_THRESHOLD_DEG = 30.0

# 停止判定: 地点内速度分布の下位このパーセンタイル以下を stopped とみなす
STOPPED_PERCENTILE = 10.0


def compute_pixel_speed_profile(traj: List[Dict]) -> np.ndarray:
    """
    画像座標軌跡から各区間の速度を計算 [px/s]

    Args:
        traj: [{'x', 'y', 'time_sec', ...}, ...]  trajectory フィールド

    Returns:
        speeds: shape=(N-1,)
    """
    if len(traj) < 2:
        return np.array([0.0])

    speeds = []
    for i in range(1, len(traj)):
        p1, p2 = traj[i - 1], traj[i]
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        dist = math.sqrt(dx * dx + dy * dy)
        dt = p2['time_sec'] - p1['time_sec']
        speeds.append(dist / dt if dt > 1e-6 else 0.0)
    return np.array(speeds)


def compute_location_thresholds(all_speeds: np.ndarray) -> np.ndarray:
    """
    地点内の全速度からパーセンタイル閾値を計算

    stopped / slow / medium / fast の境界値を返す。
    stopped 境界は STOPPED_PERCENTILE で固定し、
    残りを 3分割する。

    Args:
        all_speeds: その地点の全トラック・全区間の速度を結合した配列

    Returns:
        thresholds: shape=(3,)  [stopped_upper, slow_upper, medium_upper]
                    fast は medium_upper 以上
    """
    nonzero = all_speeds[all_speeds > 1e-6]
    if len(nonzero) == 0:
        return np.array([1.0, 2.0, 4.0])  # フォールバック

    t_stopped = float(np.percentile(nonzero, STOPPED_PERCENTILE))
    # stopped より上の範囲を 3 等分
    moving = nonzero[nonzero > t_stopped]
    if len(moving) == 0:
        return np.array([t_stopped, t_stopped * 2, t_stopped * 4])

    t_slow   = float(np.percentile(moving, 33.3))
    t_medium = float(np.percentile(moving, 66.7))
    return np.array([t_stopped, t_slow, t_medium])


def classify_speed_states(speeds: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    速度配列を4状態に分類（パーセンタイル閾値使用）

    States (index):
        0: stopped  speed <= thresholds[0]
        1: slow     thresholds[0] < speed <= thresholds[1]
        2: medium   thresholds[1] < speed <= thresholds[2]
        3: fast     speed > thresholds[2]

    Returns:
        states: shape=(N,) int
    """
    s = np.zeros(len(speeds), dtype=int)
    s[speeds >  thresholds[0]] = 1
    s[speeds >  thresholds[1]] = 2
    s[speeds >  thresholds[2]] = 3
    return s


def extract_reaction_distribution(
    traj: List[Dict],
    thresholds: np.ndarray
) -> np.ndarray:
    """
    軌跡から反応特徴量（行動状態分布）を抽出

    各区間の時間長さで重み付けして各状態の時間割合を計算し、
    Dirichlet smoothing で確率単体の境界から離す。

    Args:
        traj:       trajectory フィールド（画像座標）
        thresholds: compute_location_thresholds() の出力

    Returns:
        theta: shape=(4,)  [stopped, slow, medium, fast]
    """
    uniform = np.ones(N_STATES) / N_STATES
    if len(traj) < 2:
        return uniform

    speeds = compute_pixel_speed_profile(traj)
    states = classify_speed_states(speeds, thresholds)
    dt = np.array([
        max(traj[i + 1]['time_sec'] - traj[i]['time_sec'], 0.0)
        for i in range(len(speeds))
    ])
    total = dt.sum()
    if total < 1e-9:
        return uniform

    theta = np.array([dt[states == k].sum() for k in range(N_STATES)])
    theta /= total
    theta = (theta + DIRICHLET_EPS) / (theta + DIRICHLET_EPS).sum()
    return theta


def _count_stops(speeds: np.ndarray, thresholds: np.ndarray) -> int:
    """stopped 状態への遷移回数（停止開始回数）"""
    states = classify_speed_states(speeds, thresholds)
    count, in_stop = 0, False
    for s in states:
        if s == 0 and not in_stop:
            count += 1
            in_stop = True
        elif s != 0:
            in_stop = False
    return count


def _count_direction_changes(traj: List[Dict]) -> int:
    """DIRECTION_CHANGE_THRESHOLD_DEG 以上の方向転換回数（ピクセル座標）"""
    count = 0
    for i in range(1, len(traj) - 1):
        p0, p1, p2 = traj[i - 1], traj[i], traj[i + 1]
        v1 = np.array([p1['x'] - p0['x'], p1['y'] - p0['y']], dtype=float)
        v2 = np.array([p2['x'] - p1['x'], p2['y'] - p1['y']], dtype=float)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 0 and n2 > 0:
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            if math.degrees(math.acos(cos_a)) > DIRECTION_CHANGE_THRESHOLD_DEG:
                count += 1
    return count


def extract_invariant_features(
    track: Dict,
    thresholds: np.ndarray,
    video_name: str = ''
) -> Optional[Dict]:
    """
    1トラックから不変特徴量を抽出

    Args:
        track:      detects_people.py 出力のトラック辞書
                    必要なキー: 'trajectory', 'id'
        thresholds: compute_location_thresholds() で計算した地点閾値
        video_name: ソース動画名（複数地点比較用ラベル）

    Returns:
        features辞書  or  None（データ不足の場合）
    """
    traj = track.get('trajectory', [])
    if len(traj) < 3:
        return None

    duration = traj[-1]['time_sec'] - traj[0]['time_sec']
    if duration < 0.5:
        return None

    speeds = compute_pixel_speed_profile(traj)

    # ピクセル総移動距離
    total_dist = sum(
        math.sqrt((traj[i]['x'] - traj[i-1]['x'])**2 +
                  (traj[i]['y'] - traj[i-1]['y'])**2)
        for i in range(1, len(traj))
    )
    # 始点→終点の直線距離（ピクセル）
    straight_dist = math.sqrt(
        (traj[-1]['x'] - traj[0]['x'])**2 +
        (traj[-1]['y'] - traj[0]['y'])**2
    )
    # 迂回率: 実際経路 / 直線距離（1.0 = 最短経路）
    sinuosity = total_dist / straight_dist if straight_dist > 1.0 else 1.0

    theta = extract_reaction_distribution(traj, thresholds)

    return {
        'track_id':              track.get('id'),
        'video':                 video_name,
        'duration_sec':          round(float(duration), 3),
        'total_distance_px':     round(float(total_dist), 2),
        'straight_distance_px':  round(float(straight_dist), 2),
        'sinuosity':             round(float(sinuosity), 4),
        'speed_mean_px_s':       round(float(speeds.mean()), 3),
        'speed_std_px_s':        round(float(speeds.std()), 3),
        'num_stops':             _count_stops(speeds, thresholds),
        'num_direction_changes': _count_direction_changes(traj),
        'theta':                 theta.tolist(),
        'theta_labels':          STATE_LABELS,
        'thresholds_px_s':       thresholds.tolist(),  # 使用した閾値（記録用）
    }
