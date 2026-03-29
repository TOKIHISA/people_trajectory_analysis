"""
Run the full pipeline: person detection → tracking → homography transformation

Steps:
  1. GCP map selection    - Open a browser map to collect WGS84 coordinates
  2. GCP video selection  - Click corresponding image coordinates on a video frame
  3. Person detection     - Detect and track people with YOLO, output trajectories
  4. Homography transform - Convert trajectories to WGS84 coordinates

License: MIT License
Author: Toki Hirose
"""

import os
import sys
import json
import webbrowser
from pathlib import Path

from config import (
    VIDEO_DIR,
    OUTPUT_DIR,
    TRAJECTORY_DIR,
    VIEWER_DIR,
    GCP_CONFIG_PATH,
    GCP_FRAME_SEC,
    MAP_CENTER_LAT,
    MAP_CENTER_LON,
    MAP_ZOOM,
)


def step2_select_gcp_map(gcp_config_path: str) -> bool:
    """Step 2: Select GCP WGS84 coordinates on a map"""
    print("\n" + "=" * 60)
    print("Step 2: Select GCPs on the map (map points corresponding to video points)")
    print("=" * 60)

    # Skip if WGS84 coordinates already exist
    if os.path.exists(gcp_config_path):
        with open(gcp_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if len(config.get('gcp_wgs84', [])) >= 4:
            print(f"  GCP config already exists ({len(config['gcp_wgs84'])} points)")
            ans = input("  Re-select? (y/N): ").strip().lower()
            if ans != 'y':
                return True

    from gcp_selector_map import create_gcp_selector_map

    # Show frame image with GCP points in the map panel if available
    frame_image_path = os.path.join(
        os.path.dirname(gcp_config_path), "gcp_frame.png"
    )
    if not os.path.exists(frame_image_path):
        frame_image_path = None

    html_path = create_gcp_selector_map(
        center_lat=MAP_CENTER_LAT,
        center_lon=MAP_CENTER_LON,
        zoom_start=MAP_ZOOM,
        frame_image_path=frame_image_path,
    )

    webbrowser.open('file://' + os.path.abspath(html_path))

    print("\n  Map opened in browser")
    print("  1. Click 4 or more points to select GCPs")
    print("  2. Click 'Export JSON' to save gcp_config.json")
    print(f"  3. Save location: {os.path.abspath(gcp_config_path)}")

    input("\n  Press Enter when done...")

    # Verify
    if not os.path.exists(gcp_config_path):
        print("  Error: gcp_config.json not found")
        return False

    with open(gcp_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    n_points = len(config.get('gcp_wgs84', []))
    if n_points < 4:
        print(f"  Error: only {n_points} GCPs found (4 or more required)")
        return False

    print(f"  OK: {n_points} GCPs confirmed")
    return True


def step1_select_gcp_video(video_path: str, gcp_config_path: str) -> bool:
    """Step 1: Select GCP image coordinates from a video frame"""
    print("\n" + "=" * 60)
    print("Step 1: Select GCP image coordinates from a video frame")
    print("=" * 60)

    # Skip if image coordinates already exist
    if os.path.exists(gcp_config_path):
        with open(gcp_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if len(config.get('gcp_image', [])) >= 4:
            print(f"  Image coordinates already exist ({len(config['gcp_image'])} points)")
            ans = input("  Re-select? (y/N): ").strip().lower()
            if ans != 'y':
                return True

    from gcp_selector_video import GCPVideoSelector, update_gcp_config

    print(f"  Video: {video_path}")
    print(f"  Frame: {GCP_FRAME_SEC}s")

    selector = GCPVideoSelector(video_path, GCP_FRAME_SEC)
    points = selector.select_points()

    if not points:
        print("  Cancelled")
        return False

    update_gcp_config(gcp_config_path, points, video_path)

    # Save annotated frame image for reference in Step 2 map panel
    frame_path = os.path.join(
        os.path.dirname(gcp_config_path), "gcp_frame.png"
    )
    os.makedirs(os.path.dirname(frame_path), exist_ok=True)
    selector.save_frame(frame_path)

    print(f"  OK: {len(points)} image coordinates saved")
    print(f"  → In the next step, select {len(points)} corresponding points on the map")
    return True


def step3_detect_and_track(video_path: str, output_dir: str) -> str:
    """Step 3: Person detection and tracking"""
    print("\n" + "=" * 60)
    print("Step 3: Person detection and tracking")
    print("=" * 60)

    video_name = Path(video_path).stem
    analysis_json = os.path.join(output_dir, f"{video_name}_analysis.json")

    # Skip if results already exist
    if os.path.exists(analysis_json):
        print(f"  Detection results already exist: {analysis_json}")
        ans = input("  Re-run? (y/N): ").strip().lower()
        if ans != 'y':
            return analysis_json

    from detects_people import load_yolox_model, track_video
    from config import YOLOX_ONNX

    script_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(script_dir, YOLOX_ONNX)
    net = load_yolox_model(onnx_path)
    if net is None:
        return None

    print(f"  Video: {video_path}")
    print(f"  Output: {output_dir}")

    track_video(Path(video_path), net, output_dir)

    if not os.path.exists(analysis_json):
        print("  Error: detection results were not generated")
        return None

    print(f"  OK: {analysis_json}")
    return analysis_json


def step4_homography_transform(
    analysis_json: str,
    gcp_config_path: str,
    output_dir: str
) -> str:
    """Step 4: Homography transformation"""
    print("\n" + "=" * 60)
    print("Step 4: Homography transformation")
    print("=" * 60)

    from project_v2wgs84 import HomographyTransformer, transform_tracking_json

    # Load GCP config
    with open(gcp_config_path, 'r', encoding='utf-8') as f:
        gcp_config = json.load(f)

    # Save / validate per-video GCP config for reproducibility
    video_stem = Path(analysis_json).stem.replace('_analysis', '')
    saved_gcp_path = Path(output_dir) / f"{video_stem}_gcp_config.json"

    if saved_gcp_path.exists():
        with open(saved_gcp_path, 'r', encoding='utf-8') as f:
            saved_gcp = json.load(f)
        if saved_gcp != gcp_config:
            print(f"\n  WARNING: gcp_config differs from previously saved calibration!")
            print(f"  Saved   : {saved_gcp_path}")
            print(f"  Current : {gcp_config_path}")
            answer = input("  Continue with current gcp_config? [y/N]: ").strip().lower()
            if answer != 'y':
                print("  Aborted.")
                return None
        else:
            print(f"  GCP config matches saved calibration: {saved_gcp_path.name}")
    else:
        with open(saved_gcp_path, 'w', encoding='utf-8') as f:
            json.dump(gcp_config, f, indent=2, ensure_ascii=False)
        print(f"  GCP config saved: {saved_gcp_path.name}")

    # Compute homography matrix
    transformer = HomographyTransformer()
    transformer.compute_from_gcp(
        gcp_config['gcp_image'],
        gcp_config['gcp_wgs84']
    )

    # Save homography matrix
    h_path = os.path.join(output_dir, "homography.json")
    transformer.save(h_path)

    # Output path
    input_path = Path(analysis_json)
    output_json = str(input_path.parent / f"{input_path.stem}_wgs84.json")

    # Apply transformation
    transform_tracking_json(analysis_json, output_json, transformer)

    print(f"  OK: {output_json}")
    return output_json


def step5_generate_viewer(wgs84_json: str, viewer_dir: str) -> str:
    """Step 5: Generate trajectory viewer HTML"""
    print("\n" + "=" * 60)
    print("Step 5: Generate trajectory viewer")
    print("=" * 60)

    from generate_viewer import generate_html

    with open(wgs84_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    html = generate_html(data, Path(wgs84_json).name)
    if html is None:
        return None

    os.makedirs(viewer_dir, exist_ok=True)

    stem = Path(wgs84_json).stem
    out_path = os.path.join(viewer_dir, f"{stem}_viewer.html")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  OK: {out_path}")
    return out_path


def run_all(video_path: str = None, output_dir: str = None, gcp_config_path: str = None):
    """Run the full pipeline"""
    print("=" * 60)
    print("  Person tracking → WGS84 conversion pipeline")
    print("=" * 60)

    # Set defaults
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if gcp_config_path is None:
        gcp_config_path = GCP_CONFIG_PATH

    trajectory_dir = TRAJECTORY_DIR
    viewer_dir = VIEWER_DIR

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(trajectory_dir, exist_ok=True)
    os.makedirs(viewer_dir, exist_ok=True)

    # Select video file
    if video_path is None:
        video_dir = Path(VIDEO_DIR)
        videos = sorted(video_dir.glob("*.mp4")) + sorted(video_dir.glob("*.avi"))

        if not videos:
            print(f"Error: no videos found in {VIDEO_DIR}")
            return

        print(f"\nVideo files ({video_dir}):")
        for i, v in enumerate(videos):
            print(f"  {i + 1}. {v.name}")

        idx = input(f"\nSelect a number (1-{len(videos)}): ").strip()
        try:
            video_path = str(videos[int(idx) - 1])
        except (ValueError, IndexError):
            print("Invalid selection")
            return

    print(f"\nTarget video: {video_path}")

    # Step 1: Select GCP image coordinates from the video (identify features first)
    if not step1_select_gcp_video(video_path, gcp_config_path):
        print("\nError in Step 1")
        return

    # Step 2: Select GCP WGS84 coordinates on the map (corresponding to video points)
    if not step2_select_gcp_map(gcp_config_path):
        print("\nError in Step 2")
        return

    # Step 3: Person detection and tracking
    analysis_json = step3_detect_and_track(video_path, trajectory_dir)
    if analysis_json is None:
        print("\nError in Step 3")
        return

    # Step 4: Homography transformation
    output_json = step4_homography_transform(
        analysis_json, gcp_config_path, trajectory_dir
    )
    if output_json is None:
        print("\nError in Step 4")
        return

    # Step 5: Generate viewer
    viewer_path = step5_generate_viewer(output_json, viewer_dir)

    # Done
    print("\n" + "=" * 60)
    print("  All steps complete!")
    print("=" * 60)
    print(f"  Input video:       {video_path}")
    print(f"  GCP config:        {gcp_config_path}")
    print(f"  Detection results: {analysis_json}")
    print(f"  WGS84 output:      {output_json}")
    if viewer_path:
        print(f"  Viewer:            {viewer_path}")
    print("=" * 60)

    # Open viewer in browser
    if viewer_path:
        ans = input("\nOpen viewer in browser? (Y/n): ").strip().lower()
        if ans != 'n':
            webbrowser.open(Path(viewer_path).absolute().as_uri())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run the full pipeline')
    parser.add_argument('--video', '-v', type=str, default=None, help='Path to video file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output directory')
    parser.add_argument('--gcp', '-g', type=str, default=None, help='GCP config file')

    args = parser.parse_args()

    run_all(
        video_path=args.video,
        output_dir=args.output,
        gcp_config_path=args.gcp
    )
