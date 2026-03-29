"""
Tool for selecting image coordinates of GCPs (Ground Control Points) from a video frame

Usage:
1. python gcp_selector_video.py --video path/to/video.mp4 --gcp gcp_config.json
2. A frame from 10 seconds into the video will be displayed
3. Click the points corresponding to the map GCPs in the same order
4. Press Enter to confirm after selecting 4 or more points
5. Image coordinates will be added to gcp_config.json

License: MIT License
Author: Toki Hirose
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path


class GCPVideoSelector:
    """Class for selecting GCPs from a video frame"""

    def __init__(self, video_path: str, start_sec: float = 10.0):
        """
        Args:
            video_path: Path to video file
            start_sec: Timestamp in seconds to extract a stable frame from
        """
        self.video_path = video_path
        self.start_sec = start_sec
        self.points = []
        self.frame = None
        self.display_frame = None
        self.window_name = "GCP Selector - Click points in order"

    def get_stable_frame(self) -> np.ndarray:
        """Extract a stable frame from the video"""
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = int(self.start_sec * fps)

        # Fall back to the middle frame if the video is too short
        if target_frame >= total_frames:
            target_frame = total_frames // 2
            print(f"Warning: video is short, using frame {target_frame}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Failed to read frame")

        print(f"Frame {target_frame} extracted (fps={fps:.1f}, {self.start_sec}s)")
        return frame

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse click event handler"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append({'x': x, 'y': y})
            self._draw_points()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click removes the last point
            if self.points:
                self.points.pop()
                self._draw_points()

    def _draw_points(self):
        """Draw points on the frame"""
        self.display_frame = self.frame.copy()

        for i, pt in enumerate(self.points):
            # Draw point
            cv2.circle(self.display_frame, (pt['x'], pt['y']), 8, (0, 0, 255), -1)
            cv2.circle(self.display_frame, (pt['x'], pt['y']), 10, (255, 255, 255), 2)

            # Draw number label
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

        # Connect points with a line
        if len(self.points) > 1:
            pts = np.array([[pt['x'], pt['y']] for pt in self.points], np.int32)
            cv2.polylines(self.display_frame, [pts], False, (0, 255, 0), 2)

        # Status text
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
        """Select GCP points interactively"""
        self.frame = self.get_stable_frame()
        self.display_frame = self.frame.copy()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Fit window to screen
        h, w = self.frame.shape[:2]
        scale = min(1920 / w, 1080 / h, 1.0)
        cv2.resizeWindow(self.window_name, int(w * scale), int(h * scale))

        self._draw_points()

        print("\nControls:")
        print("  Left-click: Add point")
        print("  Right-click: Remove last point")
        print("  Enter: Confirm")
        print("  ESC: Cancel")
        print("\nClick points in the same order as on the map\n")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("Cancelled")
                cv2.destroyAllWindows()
                return []

            elif key == 13:  # Enter
                if len(self.points) >= 4:
                    print(f"\n{len(self.points)} points confirmed")
                    cv2.destroyAllWindows()
                    return self.points
                else:
                    print(f"4 or more points required (current: {len(self.points)})")

        cv2.destroyAllWindows()
        return self.points

    def save_frame(self, output_path: str):
        """Save the frame with selected points drawn on it"""
        if self.display_frame is not None:
            cv2.imwrite(output_path, self.display_frame)
            print(f"Frame saved: {output_path}")


def update_gcp_config(config_path: str, image_points: list, video_path: str = None):
    """Add image coordinates to the GCP config file"""

    # Load existing config
    if Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {
            "gcp_wgs84": [],
            "gcp_image": [],
        }

    # Update image coordinates
    config['gcp_image'] = image_points

    if video_path:
        config['video_path'] = video_path

    # Verify point count matches WGS84 entries (only if WGS84 points already exist)
    n_wgs84 = len(config.get('gcp_wgs84', []))
    if n_wgs84 > 0 and n_wgs84 != len(image_points):
        print(f"Warning: WGS84 point count ({n_wgs84}) does not match "
              f"image point count ({len(image_points)})")

    # Save
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"GCP config updated: {config_path}")
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select GCPs from a video frame')
    parser.add_argument('--video', '-v', type=str, required=True, help='Path to video file')
    parser.add_argument('--gcp', '-g', type=str, default='gcp_config.json', help='GCP config file')
    parser.add_argument('--start', '-s', type=float, default=10.0, help='Frame timestamp in seconds')
    parser.add_argument('--save-frame', type=str, default=None, help='Path to save the annotated frame image')

    args = parser.parse_args()

    # Select GCPs
    selector = GCPVideoSelector(args.video, args.start)
    points = selector.select_points()

    if points:
        # Update config
        config = update_gcp_config(args.gcp, points, args.video)

        # Save frame
        if args.save_frame:
            selector.save_frame(args.save_frame)

        print("\nSelected image coordinates:")
        for i, pt in enumerate(points):
            print(f"  {i+1}: ({pt['x']}, {pt['y']})")
