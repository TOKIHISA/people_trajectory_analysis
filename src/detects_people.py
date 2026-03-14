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

# import settings
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
    centroid-based tracking algorithm for associating detected bounding boxes across frames.
    In addition to tracking centroids, it also maintains trajectories based on the foot point of the bounding box (the point where the person touches the ground), which is more stable for movement analysis.
    """

    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = {}  # ID: (centroid_x, centroid_y)
        self.disappeared = {}  # ID: disappeared_frame_count
        self.trajectories = defaultdict(list)  # ID: [(x, y, frame), ...]
        self.first_seen = {}  # ID: first frame detected
        self.last_seen = {}  # ID: last frame detected
        self.max_disappeared = max_disappeared

    def register(self, centroid, foot_point, frame_num):
        """register a new object with a unique ID"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.trajectories[self.next_object_id].append(
            (foot_point[0], foot_point[1], frame_num)
        )
        self.first_seen[self.next_object_id] = frame_num
        self.last_seen[self.next_object_id] = frame_num
        self.next_object_id += 1

    def deregister(self, object_id):
        """deregister an object and remove it from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects, frame_num):
        """
        update the tracker with new bounding box detections

        Args:
            rects: the list of detected bounding boxes [(x1, y1, x2, y2), ...]
            frame_num: the current frame number

        Returns:
            objects: a dictionary mapping object IDs to their current centroids {(cx, cy)}
        """
        # when no detections are present, mark existing objects as disappeared
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.objects

        # conpute centroids and foot points for the current detections
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_feet = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            input_centroids[i] = (cx, int((y1 + y2) / 2.0))
            input_feet[i] = (cx, y2)  # foot point is the bottom center of the bounding box

        # if no existing objects, register all input centroids
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_feet[i], frame_num)

        # existing objects are present, match input centroids to existing object centroids
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # conpute distance matrix between existing object centroids and input centroids
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i, oc in enumerate(object_centroids):
                for j, ic in enumerate(input_centroids):
                    D[i, j] = np.linalg.norm(oc - ic)

            # find the smallest distance pairs (existing object to input centroid)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                # when distance is lower than a threshold, consider it a match
                if D[row, col] > 100:  # if the distance is too large, ignore the match (this threshold can be tuned)
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]  # using the centroid for tracking
                self.disappeared[object_id] = 0
                self.trajectories[object_id].append(
                    (input_feet[col][0], input_feet[col][1], frame_num)  # using the foot point for trajectory analysis
                )
                self.last_seen[object_id] = frame_num

                used_rows.add(row)
                used_cols.add(col)

            # not matched existing objects
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # not matched input centroids
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], input_feet[col], frame_num)

        return self.objects


def download_yolox_files():
    """Display the steps to download and export YOLOX model files"""
    print("\n" + "="*60)
    print("YOLOX ONNX model file is required for detection.")
    print("="*60)
    print("\nPlease follow these steps to create the ONNX file:\n")
    print("1. Install PyTorch + YOLOX in a temporary environment:")
    print("   pip install torch yolox onnx")
    print("\n2. Download the pre-trained model:")
    print("   https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth")
    print("\n3. Export to ONNX:")
    print("   python -m yolox.tools.export_onnx --output-name yolox_tiny.onnx -n yolox-tiny -c yolox_tiny.pth --decode_in_inference")
    print("\n4. Place yolox_tiny.onnx in the same directory as this script or specify the path in config.py")
    print("="*60 + "\n")


def load_yolox_model(onnx_path):
    """Load YOLOX ONNX model with ONNX Runtime"""
    if not os.path.exists(onnx_path):
        print(f"Error: {onnx_path} not found")
        download_yolox_files()
        return None

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    return session


def preprocess_yolox(frame, input_w, input_h):
    """
    preprocess the input frame for YOLOX ONNX model (letterbox resize)

    Returns:
        blob: shape [1, 3, input_h, input_w], float32
        ratio: resize ratio (used for inverse transformation to original image coordinates)
    """
    img_h, img_w = frame.shape[:2]
    ratio = min(input_w / img_w, input_h / img_h)

    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((input_h, input_w, 3), YOLOX_PADDING_VALUE, dtype=np.float32)
    padded[:new_h, :new_w, :] = resized.astype(np.float32)

    # HWC -> CHW, add batch dimension, and convert to float32
    # Note: the model expects BGR order (OpenCV default), so we keep it as is without converting to RGB
    blob = padded.transpose(2, 0, 1)[np.newaxis, ...]

    return blob, ratio


def demo_postprocess(outputs, img_size):
    """
    --decode_in_inference 
    decode the raw output of the model to bounding box coordinates in the padded image space
    This is needed because the ONNX model may output raw predictions that require decoding (if not already decoded during export).
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
    Detect persons in the frame using YOLOX-Tiny

    Returns:
        boxes: list of [(x1, y1, x2, y2), ...]
        confidences: list of [conf, ...]
    """
    img_h, img_w = frame.shape[:2]

    # preprocess the frame for YOLOX
    blob, ratio = preprocess_yolox(frame, INPUT_WIDTH, INPUT_HEIGHT)

    # predict with ONNX Runtime
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: blob})[0]
    predictions = output[0]  # [N, 85]

    # if the output is not already decoded (i.e., if the max coordinate value is small), decode it
    if predictions.shape[0] > 0 and np.max(predictions[:, :4]) < 2.0:
        predictions = demo_postprocess(output, (INPUT_HEIGHT, INPUT_WIDTH))[0]

    # objectness * person_class_score = confidence score for person class
    objectness = predictions[:, 4]
    person_scores = predictions[:, 5]  # クラス0 = person
    scores = objectness * person_scores

    # confidence thresholding
    mask = scores > CONFIDENCE_THRESHOLD
    filtered = predictions[mask]
    filtered_scores = scores[mask]

    if len(filtered) == 0:
        return [], []

    # cx, cy, w, h → x1, y1, x2, y2 in the original image coordinates
    cx = filtered[:, 0]
    cy = filtered[:, 1]
    w = filtered[:, 2]
    h = filtered[:, 3]

    x1 = (cx - w / 2.0) / ratio
    y1 = (cy - h / 2.0) / ratio
    x2 = (cx + w / 2.0) / ratio
    y2 = (cy + h / 2.0) / ratio

    # clip coordinates to be within the original image size
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
    analysis of a single trajectory

    Returns:
        dict: trajectory_details {
            'duration': duration in seconds,
            'total_distance': total movement distance in pixels,
            'start_pos': (x, y),
            'end_pos': (x, y),
            'exited': boolean indicating if the person exited the frame (based on proximity to edges),
            'direction_angle': movement direction in degrees (0-360, where 0 is to the right, 90 is down, etc.),
            'num_frames': number of frames in the trajectory
        }
    """
    if len(trajectory) < 2:
        return None

    # trajectory is a list of (x, y, frame_num)
    points = np.array([(x, y) for x, y, _ in trajectory])
    frames = np.array([f for _, _, f in trajectory])

    # movement distance (sum of distances between consecutive points)
    total_distance = 0
    for i in range(1, len(points)):
        total_distance += np.linalg.norm(points[i] - points[i-1])

    # duration in seconds
    duration = (frames[-1] - frames[0]) / fps

    # start and end positions
    start_pos = points[0]
    end_pos = points[-1]

    # distance to edges (for exit detection)
    def distance_to_edge(point):
        """minimum distance from the point to the edges of the frame"""
        x, y = point
        return min(x, y, frame_width - x, frame_height - y)

    start_edge_dist = distance_to_edge(start_pos)
    end_edge_dist = distance_to_edge(end_pos)

    # if the end position is close to the edge, consider it as exited
    exited = bool(end_edge_dist < 50)  # just a threshold (can be tuned)

    # movement direction
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
    """track people in a single video and save the output video and analysis results"""
    print(f"\nTracking people in: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"error: Failed to open video: {video_path}")
        return

    # Info about the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps}, Resolution: {width}x{height}, Total Frames: {total_frames}")

    # Output settings - two videos: one with original footage + tracks, one with tracks only on white background
    output_video_path = os.path.join(output_dir, f"{video_path.stem}_tracked.mp4")
    output_tracks_path = os.path.join(output_dir, f"{video_path.stem}_tracks_only.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    out_tracks = cv2.VideoWriter(output_tracks_path, fourcc, fps, (width, height))

    # Create white background frame for tracks-only video
    white_frame = np.full((height, width, 3), 255, dtype=np.uint8)

    # Init tracker
    tracker = CentroidTracker(max_disappeared=MAX_DISAPPEARED)

    frame_count = 0
    colors = {}  # color for each object ID

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # person detection
        boxes, confidences = detect_persons(frame, net)

        # tracker update
        objects = tracker.update(boxes, frame_count)

        # visualization
        for object_id, centroid in objects.items():
            # generate a random color for new object IDs
            if object_id not in colors:
                colors[object_id] = tuple(map(int, np.random.randint(0, 255, 3)))

            color = colors[object_id]

            # visualize trajectory (using foot points) for this object ID
            trajectory = tracker.trajectories[object_id]
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    # video frame
                    cv2.line(frame,
                            (trajectory[i-1][0], trajectory[i-1][1]),
                            (trajectory[i][0], trajectory[i][1]),
                            color, 2)
                    # tracks-only frame
                    cv2.line(white_frame,
                            (trajectory[i-1][0], trajectory[i-1][1]),
                            (trajectory[i][0], trajectory[i][1]),
                            color, 2)

            # set circle at the current centroid position (using foot point for better stability)
            cv2.circle(frame, tuple(centroid), 5, color, -1)
            cv2.circle(white_frame, tuple(centroid), 5, color, -1)

            # ID
            text = f"ID: {object_id}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(white_frame, text, (centroid[0] - 10, centroid[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # info text
        info_text = f"Frame: {frame_count}/{total_frames} | Active: {len(objects)}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # write frames to output videos
        out.write(frame)
        out_tracks.write(white_frame)

        if frame_count % 10 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}%", end='\r')

    cap.release()
    out.release()
    out_tracks.release()

    # save analysis results
    save_analysis(video_path, tracker, fps, width, height, output_dir)

    print(f"\ncompleted tracking for: {video_path.name}")
    print(f"Output video (original + trajectories): {output_video_path}")
    print(f"Output tracks-only video: {output_tracks_path}")


def save_analysis(video_path, tracker, fps, width, height, output_dir):
    """save trajectory analysis results to JSON and text files"""
    analysis_path = os.path.join(output_dir, f"{video_path.stem}_analysis.json")
    txt_path = os.path.join(output_dir, f"{video_path.stem}_analysis.txt")

    results = {
        'video_name': video_path.name,
        'fps': fps,
        'resolution': f"{width}x{height}",
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tracks': []
    }

    # analyze each trajectory and compile results
    for object_id, trajectory in tracker.trajectories.items():
        if len(trajectory) < MIN_TRACK_LENGTH:
            continue

        analysis = analyze_trajectory(trajectory, fps, width, height)
        if analysis:
            analysis['id'] = object_id
            analysis['first_frame'] = tracker.first_seen[object_id]
            analysis['last_frame'] = tracker.last_seen[object_id]

            # add trajectory points to the analysis (for potential visualization or further analysis)
            analysis['trajectory'] = [
                {
                    'x': int(x),
                    'y': int(y),
                    'frame': frame,
                    'time_sec': round(frame / fps, 3)
                }
                for x, y, frame in trajectory
            ]

            # LineString as geometry for potential GIS visualization (after coordinate transformation)
            analysis['geometry'] = {
                'type': 'LineString',
                'coordinates': [[int(x), int(y)] for x, y, _ in trajectory]
            }

            results['tracks'].append(analysis)

    # save JSON
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # save text summary
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"video: {video_path.name}\n")
        f.write(f"processing time: {results['timestamp']}\n")
        f.write(f"FPS: {fps}, resolution: {width}x{height}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"Tracking Results (Minimum Frames: {MIN_TRACK_LENGTH})\n")
        f.write(f"{'='*60}\n\n")

        for track in results['tracks']:
            f.write(f"ID {track['id']}:\n")
            f.write(f"  Dwell Time: {track['duration']:.2f} seconds\n")
            f.write(f"  Travel Distance: {track['total_distance']:.1f} pixels\n")
            f.write(f"  Start Position: ({track['start_pos'][0]:.0f}, {track['start_pos'][1]:.0f})\n")
            f.write(f"  End Position: ({track['end_pos'][0]:.0f}, {track['end_pos'][1]:.0f})\n")
            f.write(f"  Exited Screen: {'Yes' if track['exited'] else 'No'}\n")
            f.write(f"  Movement Direction: {track['direction_angle']:.1f} degrees\n")
            f.write(f"  Number of Frames: {track['num_frames']}\n\n")

    print(f"Analysis Results: {txt_path}")


def main():
    """main function to process all videos in the input directory"""
    print("=" * 60)
    print("people_trajectory - People Detection and Tracking with OpenCV and YOLOX")
    print("=" * 60)

    # output directory setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # load YOLOX model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(script_dir, YOLOX_ONNX)

    net = load_yolox_model(onnx_path)
    if net is None:
        return

    print("load YOLOX model successful")

    # get video files from the input directory
    video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
    video_files = []

    video_path = Path(VIDEO_DIR)
    if video_path.exists():
        for ext in video_extensions:
            video_files.extend(video_path.glob(f'*{ext}'))

    if not video_files:
        print(f"\nNo video files found in: {VIDEO_DIR}")
        return

    print(f"\n{len(video_files)} video files found")

    # Process each video file
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}]")
        track_video(video_file, net, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("All processing completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
