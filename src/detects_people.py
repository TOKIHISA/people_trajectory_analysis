"""
OpenCV を使用して動画内の通勤者を検出・追跡するスクリプト
- YOLOX-Tiny (OpenCV DNN + ONNX) で人物検出(CentroidTrackerで追跡)
- 動線分析・滞在時間測定

必要なライブラリ:
pip install opencv-python numpy

License: MIT License
Author: Toki Hirose

"""

import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import onnxruntime as ort

# 設定をインポート
from config import (
    VIDEO_DIR,
    OUTPUT_DIR,
    YOLOX_ONNX,
    YOLOX_PADDING_VALUE,
    CONFIDENCE_THRESHOLD,
    NMS_THRESHOLD,
    INPUT_WIDTH,
    INPUT_HEIGHT,
    MAX_DISAPPEARED,
    MIN_TRACK_LENGTH,
)


class CentroidTracker:
    """
    重心ベースの物体追跡クラス
    各検出された物体の中心座標を追跡し、IDを割り当てる
    """

    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = {}  # ID: 重心座標
        self.disappeared = {}  # ID: 消失フレーム数
        self.trajectories = defaultdict(list)  # ID: [(x, y, frame), ...]
        self.first_seen = {}  # ID: 最初に検出されたフレーム
        self.last_seen = {}  # ID: 最後に検出されたフレーム
        self.max_disappeared = max_disappeared

    def register(self, centroid, foot_point, frame_num):
        """新しい物体を登録"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.trajectories[self.next_object_id].append(
            (foot_point[0], foot_point[1], frame_num)
        )
        self.first_seen[self.next_object_id] = frame_num
        self.last_seen[self.next_object_id] = frame_num
        self.next_object_id += 1

    def deregister(self, object_id):
        """物体の追跡を終了"""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects, frame_num):
        """
        検出結果で追跡を更新

        Args:
            rects: [(x1, y1, x2, y2), ...] のリスト
            frame_num: 現在のフレーム番号

        Returns:
            objects: {ID: (cx, cy)} の辞書
        """
        # 検出がない場合
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.objects

        # 重心（マッチング用）と足元座標（軌跡記録用・地面接地点）を計算
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_feet = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            input_centroids[i] = (cx, int((y1 + y2) / 2.0))
            input_feet[i] = (cx, y2)  # バウンディングボックス底辺中心 = 地面接地点

        # 既存の追跡物体がない場合
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_feet[i], frame_num)

        # 既存物体と新規検出をマッチング
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # 距離行列を計算
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i, oc in enumerate(object_centroids):
                for j, ic in enumerate(input_centroids):
                    D[i, j] = np.linalg.norm(oc - ic)

            # 最小距離でマッチング
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                # 距離が近い場合のみマッチング
                if D[row, col] > 100:  # ピクセル単位での最大距離
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]  # マッチング用は重心を維持
                self.disappeared[object_id] = 0
                self.trajectories[object_id].append(
                    (input_feet[col][0], input_feet[col][1], frame_num)  # 軌跡は足元座標
                )
                self.last_seen[object_id] = frame_num

                used_rows.add(row)
                used_cols.add(col)

            # マッチしなかった既存物体
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # マッチしなかった新規検出
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], input_feet[col], frame_num)

        return self.objects


def download_yolox_files():
    """YOLOXモデルファイルのダウンロード・エクスポート手順を表示"""
    print("\n" + "="*60)
    print("YOLOX ONNXモデルファイルが必要です")
    print("="*60)
    print("\n以下の手順でONNXファイルを作成してください:\n")
    print("1. 一時環境でPyTorch + YOLOXをインストール:")
    print("   pip install torch yolox onnx")
    print("\n2. 学習済みモデルをダウンロード:")
    print("   https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth")
    print("\n3. ONNXにエクスポート:")
    print("   python -m yolox.tools.export_onnx --output-name yolox_tiny.onnx -n yolox-tiny -c yolox_tiny.pth --decode_in_inference")
    print("\n4. yolox_tiny.onnx をこのスクリプトと同じディレクトリに配置")
    print("="*60 + "\n")


def load_yolox_model(onnx_path):
    """YOLOX ONNXモデルをONNX Runtimeでロード"""
    if not os.path.exists(onnx_path):
        print(f"エラー: {onnx_path} が見つかりません")
        download_yolox_files()
        return None

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    return session


def preprocess_yolox(frame, input_w, input_h):
    """
    YOLOXのレターボックス前処理

    Returns:
        blob: shape [1, 3, input_h, input_w], float32
        ratio: リサイズ比率（元画像座標への逆変換に使用）
    """
    img_h, img_w = frame.shape[:2]
    ratio = min(input_w / img_w, input_h / img_h)

    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((input_h, input_w, 3), YOLOX_PADDING_VALUE, dtype=np.float32)
    padded[:new_h, :new_w, :] = resized.astype(np.float32)

    # HWC -> CHW, バッチ次元を追加
    # YOLOXは1/255正規化なし、swapRBなし（BGRのまま）
    blob = padded.transpose(2, 0, 1)[np.newaxis, ...]

    return blob, ratio


def demo_postprocess(outputs, img_size):
    """
    --decode_in_inference なしでエクスポートされたYOLOX ONNXモデル用の
    グリッドデコード処理（フォールバック）
    """
    grids = []
    expanded_strides = []
    strides = [8, 16, 32]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


def detect_persons(frame, session):
    """
    YOLOX-Tinyでフレーム内の人物を検出

    Returns:
        boxes: [(x1, y1, x2, y2), ...] のリスト（元画像座標）
        confidences: [conf, ...] のリスト
    """
    img_h, img_w = frame.shape[:2]

    # 前処理（レターボックス）
    blob, ratio = preprocess_yolox(frame, INPUT_WIDTH, INPUT_HEIGHT)

    # ONNX Runtime で推論
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: blob})[0]
    predictions = output[0]  # [N, 85]

    # デコード済みかどうかを自動判定（座標が小さすぎる場合は未デコード）
    if predictions.shape[0] > 0 and np.max(predictions[:, :4]) < 2.0:
        predictions = demo_postprocess(output, (INPUT_HEIGHT, INPUT_WIDTH))[0]

    # objectness * person_class_score で最終信頼度を計算
    objectness = predictions[:, 4]
    person_scores = predictions[:, 5]  # クラス0 = person
    scores = objectness * person_scores

    # 信頼度フィルタ
    mask = scores > CONFIDENCE_THRESHOLD
    filtered = predictions[mask]
    filtered_scores = scores[mask]

    if len(filtered) == 0:
        return [], []

    # cx, cy, w, h → x1, y1, x2, y2（パディング画像座標）
    cx = filtered[:, 0]
    cy = filtered[:, 1]
    w = filtered[:, 2]
    h = filtered[:, 3]

    x1 = (cx - w / 2.0) / ratio
    y1 = (cy - h / 2.0) / ratio
    x2 = (cx + w / 2.0) / ratio
    y2 = (cy + h / 2.0) / ratio

    # 元画像の範囲にクリップ
    x1 = np.clip(x1, 0, img_w).astype(int)
    y1 = np.clip(y1, 0, img_h).astype(int)
    x2 = np.clip(x2, 0, img_w).astype(int)
    y2 = np.clip(y2, 0, img_h).astype(int)

    boxes = [[int(a), int(b), int(c), int(d)] for a, b, c, d in zip(x1, y1, x2, y2)]
    confidences = [float(s) for s in filtered_scores]

    # Non-Maximum Suppression
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(
            [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes],
            confidences,
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD
        )

        if len(indices) > 0:
            idx = indices.flatten()
            return [boxes[i] for i in idx], [confidences[i] for i in idx]

    return [], []


def analyze_trajectory(trajectory, fps, frame_width, frame_height):
    """
    軌跡を分析

    Returns:
        dict: 分析結果
    """
    if len(trajectory) < 2:
        return None

    # 座標とフレーム番号を取得
    points = np.array([(x, y) for x, y, _ in trajectory])
    frames = np.array([f for _, _, f in trajectory])

    # 移動距離（ピクセル）
    total_distance = 0
    for i in range(1, len(points)):
        total_distance += np.linalg.norm(points[i] - points[i-1])

    # 滞在時間（秒）
    duration = (frames[-1] - frames[0]) / fps

    # 開始・終了位置
    start_pos = points[0]
    end_pos = points[-1]

    # 画面境界との距離
    def distance_to_edge(point):
        """画面端までの最短距離"""
        x, y = point
        return min(x, y, frame_width - x, frame_height - y)

    start_edge_dist = distance_to_edge(start_pos)
    end_edge_dist = distance_to_edge(end_pos)

    # 画面外に出たかどうか（画面端から近い位置で終了）
    exited = bool(end_edge_dist < 50)  # 50ピクセル以内

    # 移動方向
    direction_vector = end_pos - start_pos
    angle = float(np.arctan2(direction_vector[1], direction_vector[0]) * 180 / np.pi)

    return {
        'duration': float(duration),
        'total_distance': float(total_distance),
        'start_pos': start_pos.tolist(),
        'end_pos': end_pos.tolist(),
        'exited': exited,
        'direction_angle': angle,
        'num_frames': len(trajectory)
    }


def track_video(video_path, net, output_dir):
    """動画内の人物を追跡"""
    print(f"\n処理中: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"エラー: 動画を開けません: {video_path}")
        return

    # 動画情報
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps}, 解像度: {width}x{height}, 総フレーム数: {total_frames}")

    # 出力設定
    output_video_path = os.path.join(output_dir, f"{video_path.stem}_tracked.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # トラッカー初期化
    tracker = CentroidTracker(max_disappeared=MAX_DISAPPEARED)

    frame_count = 0
    colors = {}  # ID ごとの色

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 人物検出
        boxes, confidences = detect_persons(frame, net)

        # 追跡更新
        objects = tracker.update(boxes, frame_count)

        # 描画
        for object_id, centroid in objects.items():
            # IDごとに色を生成
            if object_id not in colors:
                colors[object_id] = tuple(map(int, np.random.randint(0, 255, 3)))

            color = colors[object_id]

            # 軌跡を描画
            trajectory = tracker.trajectories[object_id]
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    cv2.line(frame,
                            (trajectory[i-1][0], trajectory[i-1][1]),
                            (trajectory[i][0], trajectory[i][1]),
                            color, 2)

            # 現在位置
            cv2.circle(frame, tuple(centroid), 5, color, -1)

            # ID表示
            text = f"ID: {object_id}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 情報表示
        info_text = f"Frame: {frame_count}/{total_frames} | Active: {len(objects)}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(frame)

        if frame_count % 10 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"進捗: {progress:.1f}%", end='\r')

    cap.release()
    out.release()

    # 分析結果を保存
    save_analysis(video_path, tracker, fps, width, height, output_dir)

    print(f"\n完了!")
    print(f"出力動画: {output_video_path}")


def save_analysis(video_path, tracker, fps, width, height, output_dir):
    """分析結果を保存"""
    analysis_path = os.path.join(output_dir, f"{video_path.stem}_analysis.json")
    txt_path = os.path.join(output_dir, f"{video_path.stem}_analysis.txt")

    results = {
        'video_name': video_path.name,
        'fps': fps,
        'resolution': f"{width}x{height}",
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tracks': []
    }

    # 各追跡IDを分析
    for object_id, trajectory in tracker.trajectories.items():
        if len(trajectory) < MIN_TRACK_LENGTH:
            continue

        analysis = analyze_trajectory(trajectory, fps, width, height)
        if analysis:
            analysis['id'] = object_id
            analysis['first_frame'] = tracker.first_seen[object_id]
            analysis['last_frame'] = tracker.last_seen[object_id]

            # 軌跡データを追加（ホモグラフィー変換用）
            analysis['trajectory'] = [
                {
                    'x': int(x),
                    'y': int(y),
                    'frame': frame,
                    'time_sec': round(frame / fps, 3)
                }
                for x, y, frame in trajectory
            ]

            # GeoJSON互換のLineString形式
            analysis['geometry'] = {
                'type': 'LineString',
                'coordinates': [[int(x), int(y)] for x, y, _ in trajectory]
            }

            results['tracks'].append(analysis)

    # JSON保存
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # テキスト保存
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"動画: {video_path.name}\n")
        f.write(f"処理日時: {results['timestamp']}\n")
        f.write(f"FPS: {fps}, 解像度: {width}x{height}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"追跡結果 (最小フレーム数: {MIN_TRACK_LENGTH})\n")
        f.write(f"{'='*60}\n\n")

        for track in results['tracks']:
            f.write(f"ID {track['id']}:\n")
            f.write(f"  滞在時間: {track['duration']:.2f}秒\n")
            f.write(f"  移動距離: {track['total_distance']:.1f}ピクセル\n")
            f.write(f"  開始位置: ({track['start_pos'][0]:.0f}, {track['start_pos'][1]:.0f})\n")
            f.write(f"  終了位置: ({track['end_pos'][0]:.0f}, {track['end_pos'][1]:.0f})\n")
            f.write(f"  画面外退出: {'はい' if track['exited'] else 'いいえ'}\n")
            f.write(f"  移動方向: {track['direction_angle']:.1f}度\n")
            f.write(f"  フレーム数: {track['num_frames']}\n\n")

    print(f"分析結果: {txt_path}")


def main():
    """メイン処理"""
    print("=" * 60)
    print("OpenCV 通勤者追跡・動線分析システム")
    print("=" * 60)

    # 出力ディレクトリ
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # YOLOXモデルロード
    script_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(script_dir, YOLOX_ONNX)

    net = load_yolox_model(onnx_path)
    if net is None:
        return

    print("YOLOXモデルロード完了")

    # 動画ファイル取得
    video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
    video_files = []

    video_path = Path(VIDEO_DIR)
    if video_path.exists():
        for ext in video_extensions:
            video_files.extend(video_path.glob(f'*{ext}'))

    if not video_files:
        print(f"\nエラー: 動画ファイルが見つかりません: {VIDEO_DIR}")
        return

    print(f"\n{len(video_files)} 個の動画ファイルが見つかりました")

    # 処理
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}]")
        track_video(video_file, net, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("すべての処理が完了しました")
    print("=" * 60)


if __name__ == "__main__":
    main()
