# Pedestrian Trajectory Extraction Module

This module extracts pedestrian trajectories from street video footage using YOLOX-Tiny for detection and a custom CentroidTracker for tracking. The output is structured JSON data containing trajectory information for each detected pedestrian.

## Overview

The pipeline consists of:
1. **Person Detection**: YOLOX-Tiny ONNX model detects persons in video frames
2. **Tracking**: Centroid-based tracking assigns consistent IDs to detected persons across frames
3. **Trajectory Analysis**: Extracts movement patterns, duration, distance, and spatial features
4. **JSON Output**: Saves trajectory data in structured JSON format compatible with geospatial analysis

## Files

- `detects_people.py`: Main detection and tracking script
- `config.py`: Configuration parameters
- `run_all.py`: Full pipeline runner (includes GCP selection and homography)
- `gcp_selector_map.py`: GCP selection on map
- `gcp_selector_video.py`: GCP selection on video frame
- `project_v2wgs84.py`: Coordinate transformation to WGS84
- `generate_viewer.py`: Creates interactive trajectory viewer
- `yolox_tiny.onnx`: YOLOX-Tiny model (ONNX format)

## Dependencies

- opencv-python
- numpy
- onnxruntime
- pathlib
- json

## Usage

### Basic Trajectory Extraction

```python
from detects_people import track_video, load_yolox_model
import os

# Load YOLOX model
script_dir = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(script_dir, "yolox_tiny.onnx")
net = load_yolox_model(onnx_path)

# Process video
from pathlib import Path
video_path = Path("../input/sample_video.mp4")
output_dir = "../output"
track_video(video_path, net, output_dir)
```

### Full Pipeline (with Georeferencing)

```bash
python run_all.py
```

This runs the complete pipeline:
1. Select GCP points on video frame
2. Select corresponding GCP points on map
3. Detect and track pedestrians
4. Transform trajectories to WGS84 coordinates
5. Generate interactive viewer

## Configuration

Key parameters in `config.py`:

- `CONFIDENCE_THRESHOLD`: Detection confidence (0.5)
- `NMS_THRESHOLD`: Non-maximum suppression threshold (0.45)
- `MAX_DISAPPEARED`: Frames to wait before deregistering track (30)
- `MIN_TRACK_LENGTH`: Minimum frames for valid trajectory (10)

## Output Format

The JSON output contains:

```json
{
  "video_name": "sample.mp4",
  "fps": 30,
  "resolution": "1920x1080",
  "timestamp": "2024-01-01 12:00:00",
  "tracks": [
    {
      "id": 1,
      "duration": 15.2,
      "total_distance": 245.8,
      "start_pos": [100, 200],
      "end_pos": [300, 250],
      "exited": false,
      "direction_angle": 45.2,
      "num_frames": 456,
      "first_frame": 10,
      "last_frame": 465,
      "trajectory": [
        {"x": 100, "y": 200, "frame": 10, "time_sec": 0.333},
        {"x": 105, "y": 205, "frame": 11, "time_sec": 0.367}
      ],
      "geometry": {
        "type": "LineString",
        "coordinates": [[100, 200], [105, 205]]
      }
    }
  ]
}
```

## Algorithm Details

### Detection
- YOLOX-Tiny ONNX model for real-time person detection
- Letterbox preprocessing to maintain aspect ratio
- Confidence thresholding and NMS for filtering

### Tracking
- Centroid-based tracking using Euclidean distance
- Maximum displacement threshold (100 pixels)
- Automatic track deregistration after `MAX_DISAPPEARED` frames

### Analysis
- Trajectory features: distance, duration, direction, screen exit detection
- GeoJSON-compatible LineString format for mapping applications

## Limitations

- Assumes static camera (no camera motion compensation)
- Simple centroid tracking may fail with occlusions
- Requires manual GCP selection for georeferencing
- Performance depends on video resolution and frame rate

## License

MIT License - Toki Hirose</content>
<parameter name="filePath">h:\01_development\demogracy_map\people_trajectory\src\README.md