"""
Configuration file for person detection and tracking

License: MIT License
Author: Toki Hirose
"""

# Directory settings
VIDEO_DIR = r"..\input"
OUTPUT_DIR = r"..\output"
TRAJECTORY_DIR = r"..\output\trajectory"
VIEWER_DIR = r"..\output\viewer"

# YOLOX model settings (OpenCV DNN + ONNX)
YOLOX_ONNX = "yolox_tiny.onnx"
YOLOX_PADDING_VALUE = 114.0  # letterbox padding value

# Detection parameters
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
INPUT_WIDTH = 416
INPUT_HEIGHT = 416

# Tracking parameters
MAX_DISAPPEARED = 30  # deregister after this many missing frames
MIN_TRACK_LENGTH = 10  # minimum track length (noise filter)

# GCP / homography settings
GCP_CONFIG_PATH = r"..\output\gcp_config.json"  # GCP config file
GCP_FRAME_SEC = 2.0  # timestamp (seconds) of the frame used for GCP selection
MAP_CENTER_LAT = 35.6812  # map center latitude
MAP_CENTER_LON = 139.7671  # map center longitude
MAP_ZOOM = 18  # map initial zoom

# Trajectory simplification settings (applied after WGS84 conversion)
SIMPLIFY_ANGLE_THRESHOLD = 10.0   # direction change threshold (degrees): keep turning points above this angle
SIMPLIFY_SPEED_RATIO_THRESHOLD = 0.3  # speed change threshold (ratio): keep stops, starts, and rapid acceleration/deceleration
