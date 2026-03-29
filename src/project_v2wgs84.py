"""
Module for converting image coordinates to WGS84 using homography transformation

Workflow:
1. Use gcp_selector_map.py to collect GCP coordinates (WGS84) on a map
2. Use gcp_selector_video.py to collect corresponding points (image coordinates) from a video frame
3. Use this module to compute the homography matrix
4. Apply the transformation to the JSON output from detects_people.py

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
    """Manages homography transformation between image and WGS84 coordinates"""

    def __init__(self):
        self.H: Optional[np.ndarray] = None  # homography matrix
        self.H_inv: Optional[np.ndarray] = None  # inverse matrix
        self.src_points: Optional[np.ndarray] = None  # image coordinates
        self.dst_points: Optional[np.ndarray] = None  # WGS84 coordinates
        self.reprojection_error: float = 0.0

    def compute_from_gcp(self, gcp_image: List[Dict], gcp_wgs84: List[Dict]) -> np.ndarray:
        """
        Compute the homography matrix from GCP pairs

        Args:
            gcp_image: [{'x': int, 'y': int}, ...] image coordinates
            gcp_wgs84: [{'lat': float, 'lon': float}, ...] WGS84 coordinates

        Returns:
            np.ndarray: 3x3 homography matrix
        """
        if len(gcp_image) != len(gcp_wgs84):
            raise ValueError(f"GCP count mismatch: image={len(gcp_image)}, WGS84={len(gcp_wgs84)}")

        if len(gcp_image) < 4:
            raise ValueError(f"At least 4 points required (current: {len(gcp_image)})")

        # Convert to NumPy arrays
        self.src_points = np.array(
            [[pt['x'], pt['y']] for pt in gcp_image],
            dtype=np.float32
        )

        # WGS84: longitude as X, latitude as Y (note: not in metres)
        self.dst_points = np.array(
            [[pt['lon'], pt['lat']] for pt in gcp_wgs84],
            dtype=np.float64
        )

        # Compute homography matrix with RANSAC
        self.H, _ = cv2.findHomography(
            self.src_points,
            self.dst_points,
            cv2.RANSAC,
            ransacReprojThreshold=3.0
        )

        if self.H is None:
            raise ValueError("Failed to compute homography matrix")

        # Compute inverse matrix
        self.H_inv = np.linalg.inv(self.H)

        # Compute reprojection error
        self._compute_reprojection_error()

        print(f"Homography matrix computed")
        print(f"  GCPs used: {len(gcp_image)}")
        print(f"  Reprojection error: {self.reprojection_error:.6f} degrees")

        return self.H

    def _compute_reprojection_error(self):
        """Compute reprojection error"""
        if self.H is None or self.src_points is None:
            return

        # Project image coordinates to WGS84
        src_reshaped = self.src_points.reshape(-1, 1, 2).astype(np.float64)
        projected = cv2.perspectiveTransform(src_reshaped, self.H)
        projected = projected.reshape(-1, 2)

        # Compute mean error
        errors = np.sqrt(np.sum((projected - self.dst_points) ** 2, axis=1))
        self.reprojection_error = float(np.mean(errors))

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transform a single point

        Args:
            x, y: image coordinates

        Returns:
            (lon, lat): WGS84 coordinates
        """
        if self.H is None:
            raise ValueError("Homography matrix is not set")

        pt = np.array([[[x, y]]], dtype=np.float64)
        transformed = cv2.perspectiveTransform(pt, self.H)

        return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform multiple points

        Args:
            points: (N, 2) image coordinates

        Returns:
            np.ndarray: (N, 2) WGS84 coordinates [lon, lat]
        """
        if self.H is None:
            raise ValueError("Homography matrix is not set")

        pts = points.reshape(-1, 1, 2).astype(np.float64)
        transformed = cv2.perspectiveTransform(pts, self.H)

        return transformed.reshape(-1, 2)

    def _build_valid_hull(self):
        """Build and cache the convex hull of GCP image coordinates"""
        if not hasattr(self, '_hull'):
            self._hull = cv2.convexHull(self.src_points.astype(np.float32))
        return self._hull

    def is_in_valid_region(self, x: float, y: float) -> bool:
        """
        Check whether an image coordinate is inside the GCP convex hull

        Args:
            x, y: image coordinates (pass the foot point of the bounding box)

        Returns:
            True if inside the valid region
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
        Simplify a trajectory by removing redundant points based on direction and speed changes

        Points are retained if:
        - Direction changes by more than angle_threshold_deg (turning points)
        - Speed changes by more than speed_ratio_threshold (stops, starts, rapid acceleration)

        Stop handling:
        - Stop onset:  speed approaches 0 → speed change ratio ≈ 1.0 → retained
        - Stopped:     speed ≈ 0 on both sides → change ratio ≈ 0 → removed (deduplication)
        - Start:       speed increases from 0 → speed change ratio ≈ 1.0 → retained
        """
        if len(trajectory) <= 2:
            return trajectory

        result = [trajectory[0]]

        for i in range(1, len(trajectory) - 1):
            prev = trajectory[i - 1]
            curr = trajectory[i]
            nxt  = trajectory[i + 1]

            # Direction vectors (lon/lat deltas)
            v1 = np.array([curr['lon'] - prev['lon'], curr['lat'] - prev['lat']])
            v2 = np.array([nxt['lon']  - curr['lon'], nxt['lat']  - curr['lat']])

            # Direction change angle
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angle = float(np.degrees(np.arccos(cos_a)))
            else:
                angle = 0.0

            # Speed (distance / time)
            dt1 = curr['time_sec'] - prev['time_sec']
            dt2 = nxt['time_sec']  - curr['time_sec']
            speed1 = n1 / dt1 if dt1 > 0 else 0.0
            speed2 = n2 / dt2 if dt2 > 0 else 0.0

            # Speed change ratio (use max as denominator so stop→start is symmetric)
            max_speed = max(speed1, speed2)
            speed_change = abs(speed2 - speed1) / max_speed if max_speed > 1e-12 else 0.0

            if angle > angle_threshold_deg or speed_change > speed_ratio_threshold:
                result.append(curr)

        result.append(trajectory[-1])
        return result

    def transform_trajectory(self, trajectory: List[Dict]) -> List[Dict]:
        """
        Transform a trajectory (points outside the GCP region are discarded)

        Args:
            trajectory: [{'x': int, 'y': int, 'frame': int, 'time_sec': float}, ...]

        Returns:
            [{'lon': float, 'lat': float, 'frame': int, 'time_sec': float}, ...]
        """
        if not trajectory:
            return []

        # Keep only points inside the GCP convex hull
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
        """Save the homography matrix to a file"""
        data = {
            'H': self.H.tolist() if self.H is not None else None,
            'H_inv': self.H_inv.tolist() if self.H_inv is not None else None,
            'src_points': self.src_points.tolist() if self.src_points is not None else None,
            'dst_points': self.dst_points.tolist() if self.dst_points is not None else None,
            'reprojection_error': self.reprojection_error
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Homography matrix saved: {path}")

    def load(self, path: str):
        """Load the homography matrix from a file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.H = np.array(data['H']) if data.get('H') else None
        self.H_inv = np.array(data['H_inv']) if data.get('H_inv') else None
        self.src_points = np.array(data['src_points']) if data.get('src_points') else None
        self.dst_points = np.array(data['dst_points']) if data.get('dst_points') else None
        self.reprojection_error = data.get('reprojection_error', 0.0)

        print(f"Homography matrix loaded: {path}")


def transform_tracking_json(
    input_json_path: str,
    output_json_path: str,
    transformer: HomographyTransformer
) -> Dict:
    """
    Apply homography transformation to the JSON output from detects_people.py

    Args:
        input_json_path: Input JSON path (output from detects_people.py)
        output_json_path: Output JSON path
        transformer: HomographyTransformer instance

    Returns:
        Transformed data dict
    """
    # Load input
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Transform each track
    for track in data.get('tracks', []):
        # Transform trajectory
        if 'trajectory' in track:
            track['trajectory_wgs84'] = transformer.transform_trajectory(track['trajectory'])

            # Also convert GeoJSON LineString
            track['geometry_wgs84'] = {
                'type': 'LineString',
                'coordinates': [
                    [pt['lon'], pt['lat']]
                    for pt in track['trajectory_wgs84']
                ]
            }

    # Add metadata
    data['coordinate_system'] = 'WGS84'
    data['homography_applied'] = True
    data['reprojection_error_deg'] = transformer.reprojection_error

    # Save
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Transformation complete: {output_json_path}")
    print(f"  Tracks: {len(data.get('tracks', []))}")

    return data


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Apply homography transformation to trajectory JSON')
    parser.add_argument('--gcp', '-g', type=str, required=True, help='GCP config file')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input JSON (output from detects_people.py)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output JSON')
    parser.add_argument('--save-h', type=str, default=None, help='Path to save the homography matrix')

    args = parser.parse_args()

    # Load GCP config
    with open(args.gcp, 'r', encoding='utf-8') as f:
        gcp_config = json.load(f)

    # Create transformer and compute homography
    transformer = HomographyTransformer()
    transformer.compute_from_gcp(
        gcp_config['gcp_image'],
        gcp_config['gcp_wgs84']
    )

    # Save homography matrix if requested
    if args.save_h:
        transformer.save(args.save_h)

    # Resolve output path
    output_path = args.output
    if output_path is None:
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}_wgs84.json")

    # Apply transformation
    transform_tracking_json(args.input, output_path, transformer)


if __name__ == '__main__':
    main()
