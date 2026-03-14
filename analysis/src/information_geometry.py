"""
情報幾何学的分析モジュール

確率単体（カテゴリ分布の多様体）上で以下を計算する:

  Fisher-Rao 計量
    カテゴリ分布に対するリーマン計量。
    距離: d(p, q) = 2 arccos(Σ √(p_i q_i))  (球面上の測地距離に対応)

  e-測度 (exponential measure)
    KL(ref ‖ p)
    「参照分布 ref を、個人 p の分布で近似したときの情報損失」
    → 分布 p の「特化度」: 特定の状態に集中しているほど大きい
    → 指数型接続 (e-connection) 方向の乖離を測る

  m-測度 (mixture measure)
    KL(p ‖ ref)
    「個人 p の分布を、参照分布 ref で近似したときの情報損失」
    → 分布 p の「希少性」: 参照（平均）と異なるほど大きい
    → 混合型接続 (m-connection) 方向の乖離を測る

  e と m の差異が示すもの:
    e > m: p は「集中型」分布 → 少数状態に強く偏る
    m > e: p は「混合型」分布 → 複数状態が均等に混在

  スタンディング (standing)
    グルーピング = 離散的なクラスタ所属
    スタンディング = (e-measure, m-measure, PGA座標) による
                    多様体上の連続的な位置付け
    → 同一の空間で異なる人々の反応組成の変化を比較できる

  PGA (Principal Geodesic Analysis)
    Fisher-Rao 計量の球面近似のもとで
    切線空間での PCA を行い、多様体の主成分方向を求める。

License: MIT License
Author: Toki Hirose
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    KL(p ‖ q) = Σ p_i log(p_i / q_i)

    p, q は Dirichlet smoothing 済みを想定（ゼロなし）。
    """
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


def e_measure(p: np.ndarray, ref: np.ndarray) -> float:
    """
    e-測度: KL(ref ‖ p)

    解釈:
        ref（平均的な人）を p という分布でエンコードしたときの情報損失。
        p が特定状態に集中（偏在）しているほど大きくなる。
        → 「その空間に対する行動の特化度」
    """
    return kl_divergence(ref, p)


def m_measure(p: np.ndarray, ref: np.ndarray) -> float:
    """
    m-測度: KL(p ‖ ref)

    解釈:
        p（個人）を ref（平均）でエンコードしたときの情報損失。
        p が ref から離れているほど大きくなる。
        → 「その空間での行動の希少性（平均からの逸脱）」
    """
    return kl_divergence(p, ref)


def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Fisher-Rao 測地距離（カテゴリ分布）

    d(p, q) = 2 arccos(Σ √(p_i · q_i))

    カテゴリ分布の多様体は球面 S^(K-1) に等長埋め込まれるため
    (x_i = √p_i) これが真の測地距離になる。
    """
    inner = np.sum(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None)))
    return 2.0 * float(np.arccos(np.clip(inner, -1.0, 1.0)))


def compute_frechet_mean(thetas: np.ndarray) -> np.ndarray:
    """
    確率単体上の Fréchet 平均（Fisher-Rao 計量）

    球面近似: x_i = √(θ_i) として S^(K-1) 上の点とみなし、
    球面上の Fréchet 平均 ≈ 正規化したユークリッド平均の二乗 を返す。

    Args:
        thetas: shape=(N, K)

    Returns:
        mean: shape=(K,)  確率単体上の点
    """
    sqrt_th = np.sqrt(np.clip(thetas, 0, None))
    mean_sqrt = sqrt_th.mean(axis=0)
    norm = np.linalg.norm(mean_sqrt)
    if norm < 1e-12:
        return np.ones(thetas.shape[1]) / thetas.shape[1]
    mean_sqrt /= norm
    mean = mean_sqrt ** 2
    return mean / mean.sum()


def principal_geodesic_analysis(
    thetas: np.ndarray,
    n_components: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    確率単体上の主測地線分析 (PGA)

    球面近似のもとで切線空間に射影し、そこで PCA を行う。
    対数写像: log_μ(x) ≈ x - <x, μ> μ

    Args:
        thetas:       shape=(N, K)
        n_components: 返す主成分数

    Returns:
        projections:              shape=(N, n_components)  各点の PGA 座標
        explained_variance_ratio: shape=(n_components,)
    """
    # 球面上の点 x_i = √θ_i / ‖√θ_i‖
    sqrt_th = np.sqrt(np.clip(thetas, 1e-12, None))
    norms = np.linalg.norm(sqrt_th, axis=1, keepdims=True)
    x = sqrt_th / norms

    # Fréchet 平均（球面上）
    mu = x.mean(axis=0)
    mu /= np.linalg.norm(mu)

    # 切線空間への射影（対数写像）
    inner = np.clip(np.sum(x * mu, axis=1, keepdims=True), -1.0 + 1e-8, 1.0 - 1e-8)
    tangent = x - inner * mu   # shape=(N, K)

    # 切線空間で PCA
    _, S, Vt = np.linalg.svd(tangent, full_matrices=False)
    n_components = min(n_components, len(S))

    projections = tangent @ Vt[:n_components].T   # shape=(N, n_components)

    variance = S ** 2 / max(len(thetas) - 1, 1)
    total_var = variance.sum()
    evr = variance[:n_components] / (total_var + 1e-12)

    return projections, evr


def compute_ig_positions(
    features: List[Dict],
    reference: str = 'frechet_mean'
) -> List[Dict]:
    """
    各個人の情報幾何的位置を計算してfeatureに追加する

    (e-measure, m-measure, Fisher-Rao距離, PGA座標) の組で
    統計多様体上での位置を連続量として表現する。

    Args:
        features:  extract_invariant_features() の出力リスト
        reference: 参照分布の種類
                     'frechet_mean' → データのFréchet平均
                     'uniform'      → 均一分布（完全ランダムウォーカー）

    Returns:
        各 feature に以下のキーを追加したリスト:

        e_measure         KL(ref ‖ p)  行動の特化度
        m_measure         KL(p ‖ ref)  行動の希少性
        j_divergence      (e + m) / 2  対称的な乖離度
        e_minus_m         e - m        特化型(+) vs 混合型(-)
        fisher_rao_dist   真の測地距離
        pga_x, pga_y      主測地線座標（多様体上の位置）
        reference_dist    使用した参照分布
        pga_evr           PGA の説明分散比
    """
    valid = [f for f in features if f is not None]
    if not valid:
        return features

    thetas = np.array([f['theta'] for f in valid])   # shape=(N, 4)

    # 参照分布
    if reference == 'frechet_mean':
        ref = compute_frechet_mean(thetas)
    else:
        ref = np.ones(thetas.shape[1]) / thetas.shape[1]

    # PGA
    projections, evr = principal_geodesic_analysis(thetas, n_components=2)

    result = []
    for i, feat in enumerate(valid):
        p = np.array(feat['theta'])
        em = e_measure(p, ref)
        mm = m_measure(p, ref)

        result.append({
            **feat,
            'e_measure':        round(float(em), 6),
            'm_measure':        round(float(mm), 6),
            'j_divergence':     round(float((em + mm) / 2.0), 6),
            'e_minus_m':        round(float(em - mm), 6),
            'fisher_rao_dist':  round(float(fisher_rao_distance(p, ref)), 6),
            'pga_x':            round(float(projections[i, 0]), 6),
            'pga_y':            round(float(projections[i, 1]) if projections.shape[1] > 1 else 0.0, 6),
            'reference_dist':   ref.tolist(),
            'pga_evr':          evr.tolist(),
        })

    return result


def summarize_location(standings: List[Dict]) -> Dict:
    """
    1地点（1動画）分のスタンディングからサマリを生成

    複数地点を比較するための集計値を返す。

    Returns:
        {
          'n_persons':       人数
          'frechet_mean':    反応組成のFréchet平均
          'e_mean', 'e_std': e-測度の平均・標準偏差
          'm_mean', 'm_std': m-測度の平均・標準偏差
          'j_mean':          J-divergence の平均
          'e_minus_m_mean':  e - m の平均（全体の特化傾向）
        }
    """
    if not standings:
        return {}

    thetas = np.array([s['theta'] for s in standings])
    e_vals = np.array([s['e_measure'] for s in standings])
    m_vals = np.array([s['m_measure'] for s in standings])
    em_diff = np.array([s['e_minus_m'] for s in standings])

    return {
        'n_persons':      len(standings),
        'frechet_mean':   compute_frechet_mean(thetas).tolist(),
        'theta_labels':   standings[0].get('theta_labels', []),
        'e_mean':         round(float(e_vals.mean()), 6),
        'e_std':          round(float(e_vals.std()), 6),
        'm_mean':         round(float(m_vals.mean()), 6),
        'm_std':          round(float(m_vals.std()), 6),
        'j_mean':         round(float(((e_vals + m_vals) / 2).mean()), 6),
        'e_minus_m_mean': round(float(em_diff.mean()), 6),
    }
