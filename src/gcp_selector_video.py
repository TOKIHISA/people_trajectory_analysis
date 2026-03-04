"""
動画フレームからGCP（Ground Control Points）の画像座標を選択するツール

使い方:
1. python gcp_selector_video.py --video path/to/video.mp4 --gcp gcp_config.json
2. 動画の10秒後のフレームが表示される
3. 地図で指定した順番に対応する点をクリック
4. 4点以上指定したらEnterで確定
5. gcp_config.jsonに画像座標が追加される

License: MIT License
Author: Toki Hirose
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path


class GCPVideoSelector:
    """動画フレームからGCPを選択するクラス"""

    def __init__(self, video_path: str, start_sec: float = 10.0):
        """
        Args:
            video_path: 動画ファイルパス
            start_sec: 開始秒数（安定したフレームを使用）
        """
        self.video_path = video_path
        self.start_sec = start_sec
        self.points = []
        self.frame = None
        self.display_frame = None
        self.window_name = "GCP Selector - Click points in order"

    def get_stable_frame(self) -> np.ndarray:
        """動画から安定したフレームを取得"""
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError(f"動画を開けません: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = int(self.start_sec * fps)

        # フレーム数が足りない場合は中間フレームを使用
        if target_frame >= total_frames:
            target_frame = total_frames // 2
            print(f"警告: 動画が短いため、フレーム {target_frame} を使用")

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("フレームを取得できません")

        print(f"フレーム {target_frame} を取得 (fps={fps:.1f}, {self.start_sec}秒後)")
        return frame

    def mouse_callback(self, event, x, y, flags, param):
        """マウスクリックイベント"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append({'x': x, 'y': y})
            self._draw_points()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右クリックで最後の点を削除
            if self.points:
                self.points.pop()
                self._draw_points()

    def _draw_points(self):
        """フレーム上にポイントを描画"""
        self.display_frame = self.frame.copy()

        for i, pt in enumerate(self.points):
            # 点を描画
            cv2.circle(self.display_frame, (pt['x'], pt['y']), 8, (0, 0, 255), -1)
            cv2.circle(self.display_frame, (pt['x'], pt['y']), 10, (255, 255, 255), 2)

            # 番号を描画
            cv2.putText(
                self.display_frame,
                str(i + 1),
                (pt['x'] + 15, pt['y'] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            cv2.putText(
                self.display_frame,
                str(i + 1),
                (pt['x'] + 15, pt['y'] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                1
            )

        # 線で結ぶ
        if len(self.points) > 1:
            pts = np.array([[pt['x'], pt['y']] for pt in self.points], np.int32)
            cv2.polylines(self.display_frame, [pts], False, (0, 255, 0), 2)

        # ステータス表示
        status = f"Points: {len(self.points)} | Left-click: Add | Right-click: Remove | Enter: Confirm | ESC: Cancel"
        cv2.putText(
            self.display_frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.imshow(self.window_name, self.display_frame)

    def select_points(self) -> list:
        """GCPポイントを選択"""
        self.frame = self.get_stable_frame()
        self.display_frame = self.frame.copy()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # ウィンドウサイズを調整
        h, w = self.frame.shape[:2]
        scale = min(1920 / w, 1080 / h, 1.0)
        cv2.resizeWindow(self.window_name, int(w * scale), int(h * scale))

        self._draw_points()

        print("\n操作方法:")
        print("  左クリック: ポイント追加")
        print("  右クリック: 最後のポイント削除")
        print("  Enter: 確定")
        print("  ESC: キャンセル")
        print("\n地図で指定した順番と同じ順番でクリックしてください\n")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("キャンセルしました")
                cv2.destroyAllWindows()
                return []

            elif key == 13:  # Enter
                if len(self.points) >= 4:
                    print(f"\n{len(self.points)}点を確定しました")
                    cv2.destroyAllWindows()
                    return self.points
                else:
                    print(f"4点以上必要です（現在: {len(self.points)}点）")

        cv2.destroyAllWindows()
        return self.points

    def save_frame(self, output_path: str):
        """選択したポイント付きのフレームを保存"""
        if self.display_frame is not None:
            cv2.imwrite(output_path, self.display_frame)
            print(f"フレームを保存: {output_path}")


def update_gcp_config(config_path: str, image_points: list, video_path: str = None):
    """GCP設定ファイルに画像座標を追加"""

    # 既存の設定を読み込み
    if Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {
            "gcp_wgs84": [],
            "gcp_image": [],
        }

    # 画像座標を更新
    config['gcp_image'] = image_points

    if video_path:
        config['video_path'] = video_path

    # 座標数の確認（WGS84が既にある場合のみ）
    n_wgs84 = len(config.get('gcp_wgs84', []))
    if n_wgs84 > 0 and n_wgs84 != len(image_points):
        print(f"警告: WGS84座標({n_wgs84}点)と"
              f"画像座標({len(image_points)}点)の数が一致しません")

    # 保存
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"GCP設定を更新: {config_path}")
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='動画フレームからGCPを選択')
    parser.add_argument('--video', '-v', type=str, required=True, help='動画ファイルパス')
    parser.add_argument('--gcp', '-g', type=str, default='gcp_config.json', help='GCP設定ファイル')
    parser.add_argument('--start', '-s', type=float, default=10.0, help='開始秒数')
    parser.add_argument('--save-frame', type=str, default=None, help='フレーム画像の保存先')

    args = parser.parse_args()

    # GCPを選択
    selector = GCPVideoSelector(args.video, args.start)
    points = selector.select_points()

    if points:
        # 設定を更新
        config = update_gcp_config(args.gcp, points, args.video)

        # フレームを保存
        if args.save_frame:
            selector.save_frame(args.save_frame)

        print("\n選択した画像座標:")
        for i, pt in enumerate(points):
            print(f"  {i+1}: ({pt['x']}, {pt['y']})")
