"""
ホモグラフィー変換を使って画像座標をWGS84に変換するモジュール

ワークフロー:
1. gcp_selector_map.py で地図上のGCP（WGS84座標）を取得
2. gcp_selector_video.py で動画フレームの対応点（画像座標）を取得
3. このモジュールでホモグラフィー行列を計算
4. detects_people.pyの出力JSONに変換を適用

License: MIT License
Author: Toki Hirose
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from config import SIMPLIFY_ANGLE_THRESHOLD, SIMPLIFY_SPEED_RATIO_THRESHOLD


class HomographyTransformer:
    """ホモグラフィー変換を管理するクラス"""

    def __init__(self):
        self.H: Optional[np.ndarray] = None  # ホモグラフィー行列
        self.H_inv: Optional[np.ndarray] = None  # 逆行列
        self.src_points: Optional[np.ndarray] = None  # 画像座標
        self.dst_points: Optional[np.ndarray] = None  # WGS84座標
        self.reprojection_error: float = 0.0

    def compute_from_gcp(self, gcp_image: List[Dict], gcp_wgs84: List[Dict]) -> np.ndarray:
        """
        GCPからホモグラフィー行列を計算

        Args:
            gcp_image: [{'x': int, 'y': int}, ...] 画像座標
            gcp_wgs84: [{'lat': float, 'lon': float}, ...] WGS84座標

        Returns:
            np.ndarray: 3x3 ホモグラフィー行列
        """
        if len(gcp_image) != len(gcp_wgs84):
            raise ValueError(f"GCPの数が一致しません: 画像={len(gcp_image)}, WGS84={len(gcp_wgs84)}")

        if len(gcp_image) < 4:
            raise ValueError(f"4点以上必要です（現在: {len(gcp_image)}点）")

        # NumPy配列に変換
        self.src_points = np.array(
            [[pt['x'], pt['y']] for pt in gcp_image],
            dtype=np.float32
        )

        # WGS84座標: 経度をX、緯度をYとして扱う（メートル単位ではないことに注意）
        self.dst_points = np.array(
            [[pt['lon'], pt['lat']] for pt in gcp_wgs84],
            dtype=np.float64
        )

        # ホモグラフィー行列を計算（RANSACを使用）
        self.H, _ = cv2.findHomography(
            self.src_points,
            self.dst_points,
            cv2.RANSAC,
            ransacReprojThreshold=3.0
        )

        if self.H is None:
            raise ValueError("ホモグラフィー行列を計算できません")

        # 逆行列を計算
        self.H_inv = np.linalg.inv(self.H)

        # 再投影誤差を計算
        self._compute_reprojection_error()

        print(f"ホモグラフィー行列を計算しました")
        print(f"  使用GCP: {len(gcp_image)}点")
        print(f"  再投影誤差: {self.reprojection_error:.6f}度")

        return self.H

    def _compute_reprojection_error(self):
        """再投影誤差を計算"""
        if self.H is None or self.src_points is None:
            return

        # 画像座標をWGS84に変換
        src_reshaped = self.src_points.reshape(-1, 1, 2).astype(np.float64)
        projected = cv2.perspectiveTransform(src_reshaped, self.H)
        projected = projected.reshape(-1, 2)

        # 誤差を計算
        errors = np.sqrt(np.sum((projected - self.dst_points) ** 2, axis=1))
        self.reprojection_error = float(np.mean(errors))

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        1点を変換

        Args:
            x, y: 画像座標

        Returns:
            (lon, lat): WGS84座標
        """
        if self.H is None:
            raise ValueError("ホモグラフィー行列が設定されていません")

        pt = np.array([[[x, y]]], dtype=np.float64)
        transformed = cv2.perspectiveTransform(pt, self.H)

        return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        複数点を変換

        Args:
            points: (N, 2) 画像座標

        Returns:
            np.ndarray: (N, 2) WGS84座標 [lon, lat]
        """
        if self.H is None:
            raise ValueError("ホモグラフィー行列が設定されていません")

        pts = points.reshape(-1, 1, 2).astype(np.float64)
        transformed = cv2.perspectiveTransform(pts, self.H)

        return transformed.reshape(-1, 2)

    def _build_valid_hull(self):
        """GCP画像座標の凸包を構築（キャッシュ）"""
        if not hasattr(self, '_hull'):
            self._hull = cv2.convexHull(self.src_points.astype(np.float32))
        return self._hull

    def is_in_valid_region(self, x: float, y: float) -> bool:
        """
        画像座標がGCP凸包の内側かどうかを判定

        Args:
            x, y: 画像座標（足元座標を渡すこと）

        Returns:
            True: 有効領域内
        """
        if self.src_points is None:
            return True
        hull = self._build_valid_hull()
        return cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0

    def _simplify_trajectory(
        self,
        trajectory: List[Dict],
        angle_threshold_deg: float = SIMPLIFY_ANGLE_THRESHOLD,
        speed_ratio_threshold: float = SIMPLIFY_SPEED_RATIO_THRESHOLD,
    ) -> List[Dict]:
        """
        方向変化・速度変化に基づく軌跡の間引き

        以下の点を保持する:
        - 方向が angle_threshold_deg 以上変化した点（転換点）
        - 速度が speed_ratio_threshold 以上の比率で変化した点（停止・発進・急加減速）

        停止の扱い:
        - 停止開始: speed が 0 に近づく → 速度変化率 ≈ 1.0 → 保持
        - 停止中:   両側の speed ≈ 0   → 変化率 ≈ 0     → 除外（重複除去）
        - 発進:     speed が 0 から増加 → 速度変化率 ≈ 1.0 → 保持
        """
        if len(trajectory) <= 2:
            return trajectory

        result = [trajectory[0]]

        for i in range(1, len(trajectory) - 1):
            prev = trajectory[i - 1]
            curr = trajectory[i]
            nxt  = trajectory[i + 1]

            # 方向ベクトル（lon/lat 差分）
            v1 = np.array([curr['lon'] - prev['lon'], curr['lat'] - prev['lat']])
            v2 = np.array([nxt['lon']  - curr['lon'], nxt['lat']  - curr['lat']])

            # 方向変化角度
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angle = float(np.degrees(np.arccos(cos_a)))
            else:
                angle = 0.0

            # 速度（距離 / 時間）
            dt1 = curr['time_sec'] - prev['time_sec']
            dt2 = nxt['time_sec']  - curr['time_sec']
            speed1 = n1 / dt1 if dt1 > 0 else 0.0
            speed2 = n2 / dt2 if dt2 > 0 else 0.0

            # 速度変化比率（max を分母にして停止→発進も対称に扱う）
            max_speed = max(speed1, speed2)
            speed_change = abs(speed2 - speed1) / max_speed if max_speed > 1e-12 else 0.0

            if angle > angle_threshold_deg or speed_change > speed_ratio_threshold:
                result.append(curr)

        result.append(trajectory[-1])
        return result

    def transform_trajectory(self, trajectory: List[Dict]) -> List[Dict]:
        """
        軌跡データを変換（GCP領域外の点は除外）

        Args:
            trajectory: [{'x': int, 'y': int, 'frame': int, 'time_sec': float}, ...]

        Returns:
            [{'lon': float, 'lat': float, 'frame': int, 'time_sec': float}, ...]
        """
        if not trajectory:
            return []

        # GCP凸包内の点だけを残す
        valid = [pt for pt in trajectory if self.is_in_valid_region(pt['x'], pt['y'])]
        if not valid:
            return []

        points = np.array([[pt['x'], pt['y']] for pt in valid], dtype=np.float64)
        transformed = self.transform_points(points)

        result = []
        for i, pt in enumerate(valid):
            result.append({
                'lon': float(transformed[i, 0]),
                'lat': float(transformed[i, 1]),
                'frame': pt['frame'],
                'time_sec': pt['time_sec']
            })

        return self._simplify_trajectory(result)

    def save(self, path: str):
        """ホモグラフィー行列を保存"""
        data = {
            'H': self.H.tolist() if self.H is not None else None,
            'H_inv': self.H_inv.tolist() if self.H_inv is not None else None,
            'src_points': self.src_points.tolist() if self.src_points is not None else None,
            'dst_points': self.dst_points.tolist() if self.dst_points is not None else None,
            'reprojection_error': self.reprojection_error
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"ホモグラフィー行列を保存: {path}")

    def load(self, path: str):
        """ホモグラフィー行列を読み込み"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.H = np.array(data['H']) if data.get('H') else None
        self.H_inv = np.array(data['H_inv']) if data.get('H_inv') else None
        self.src_points = np.array(data['src_points']) if data.get('src_points') else None
        self.dst_points = np.array(data['dst_points']) if data.get('dst_points') else None
        self.reprojection_error = data.get('reprojection_error', 0.0)

        print(f"ホモグラフィー行列を読み込み: {path}")


def transform_tracking_json(
    input_json_path: str,
    output_json_path: str,
    transformer: HomographyTransformer
) -> Dict:
    """
    detects_people.pyの出力JSONにホモグラフィー変換を適用

    Args:
        input_json_path: 入力JSONパス（detects_people.pyの出力）
        output_json_path: 出力JSONパス
        transformer: HomographyTransformerインスタンス

    Returns:
        変換後のデータ
    """
    # 入力を読み込み
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 各トラックを変換
    for track in data.get('tracks', []):
        # 軌跡を変換
        if 'trajectory' in track:
            track['trajectory_wgs84'] = transformer.transform_trajectory(track['trajectory'])

            # GeoJSON LineStringも変換
            track['geometry_wgs84'] = {
                'type': 'LineString',
                'coordinates': [
                    [pt['lon'], pt['lat']]
                    for pt in track['trajectory_wgs84']
                ]
            }

    # メタデータを追加
    data['coordinate_system'] = 'WGS84'
    data['homography_applied'] = True
    data['reprojection_error_deg'] = transformer.reprojection_error

    # 保存
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"変換完了: {output_json_path}")
    print(f"  トラック数: {len(data.get('tracks', []))}")

    return data


def main():
    """メイン処理"""
    import argparse

    parser = argparse.ArgumentParser(description='軌跡JSONにホモグラフィー変換を適用')
    parser.add_argument('--gcp', '-g', type=str, required=True, help='GCP設定ファイル')
    parser.add_argument('--input', '-i', type=str, required=True, help='入力JSON（detects_people.pyの出力）')
    parser.add_argument('--output', '-o', type=str, default=None, help='出力JSON')
    parser.add_argument('--save-h', type=str, default=None, help='ホモグラフィー行列の保存先')

    args = parser.parse_args()

    # GCP設定を読み込み
    with open(args.gcp, 'r', encoding='utf-8') as f:
        gcp_config = json.load(f)

    # ホモグラフィー変換器を作成
    transformer = HomographyTransformer()
    transformer.compute_from_gcp(
        gcp_config['gcp_image'],
        gcp_config['gcp_wgs84']
    )

    # ホモグラフィー行列を保存
    if args.save_h:
        transformer.save(args.save_h)

    # 出力パスを決定
    output_path = args.output
    if output_path is None:
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}_wgs84.json")

    # 変換を適用
    transform_tracking_json(args.input, output_path, transformer)


if __name__ == '__main__':
    main()
