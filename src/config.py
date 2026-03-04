"""
人物検出・追跡の設定ファイル

License: MIT License
Author: Toki Hirose
"""

# ディレクトリ設定
VIDEO_DIR = r"..\input"
OUTPUT_DIR = r"..\output"
TRAJECTORY_DIR = r"..\output\trajectory"
VIEWER_DIR = r"..\output\viewer"

# YOLOX モデル設定（OpenCV DNN + ONNX）
YOLOX_ONNX = "yolox_tiny.onnx"
YOLOX_PADDING_VALUE = 114.0  # レターボックスのパディング値

# 検出パラメータ
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
INPUT_WIDTH = 416
INPUT_HEIGHT = 416

# 追跡パラメータ
MAX_DISAPPEARED = 30  # この数のフレーム後に追跡を終了
MIN_TRACK_LENGTH = 10  # 最小追跡フレーム数（ノイズ除去）

# GCP / ホモグラフィー設定
GCP_CONFIG_PATH = r"..\output\gcp_config.json"  # GCP設定ファイル
GCP_FRAME_SEC = 2.0  # GCP取得に使うフレームの秒数
MAP_CENTER_LAT = 35.6812  # 地図の中心緯度
MAP_CENTER_LON = 139.7671  # 地図の中心経度
MAP_ZOOM = 18  # 地図の初期ズーム

# 軌跡間引き設定（WGS84変換後に適用）
SIMPLIFY_ANGLE_THRESHOLD = 10.0   # 方向変化の閾値（度）: これ以上の転換点を保持
SIMPLIFY_SPEED_RATIO_THRESHOLD = 0.3  # 速度変化の閾値（比率）: 停止・発進・急加減速を保持
