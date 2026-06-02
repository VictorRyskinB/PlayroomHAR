# backend/homography_config.py
# Camera homography: map pixel coordinates → real-world floor coordinates (cm).
#
# The user clicks 4 known floor points in the video frame and enters their
# real-world (X, Y) positions in cm.  cv2.findHomography() computes the 3×3
# perspective transform; from that point all TrackPoint pixel centroids can
# be projected onto the flat floor plan.
#
# Coordinate convention (world / floor space):
#   Origin (0, 0) = front-left corner of the room (camera side, left)
#   +X = right across the room width
#   +Y = toward the far wall (away from camera)
#   Room dimensions default: 250 cm × 600 cm  (2.5 m × 6 m)
#
# File format:
#   {
#     "pixel_points":    [[px1,py1], [px2,py2], [px3,py3], [px4,py4]],
#     "world_points_cm": [[X1,Y1],  [X2,Y2],  [X3,Y3],  [X4,Y4]],
#     "room_size_cm":    [width, depth],
#     "matrix":          [[...3×3 floats...]]
#   }

from __future__ import annotations
import json
from pathlib import Path

import cv2
import numpy as np

# Default room dimensions (cm) — user should adjust to measured values
DEFAULT_ROOM_WIDTH_CM = 250
DEFAULT_ROOM_DEPTH_CM = 600


def compute_matrix(
    pixel_points: list[list[float]],
    world_points_cm: list[list[float]],
) -> np.ndarray:
    """
    Compute a 3×3 homography matrix mapping pixel coords → world coords (cm).
    Both inputs are lists of 4 [x, y] pairs.
    Raises ValueError if the points are collinear or degenerate.
    """
    src = np.float32(pixel_points)
    dst = np.float32(world_points_cm)
    mat, status = cv2.findHomography(src, dst)
    if mat is None or status is None:
        raise ValueError("findHomography failed — check that the 4 points "
                         "are not collinear.")
    return mat


def save_homography(
    path: str | Path,
    pixel_points: list[list[float]],
    world_points_cm: list[list[float]],
    room_size_cm: list[float],
) -> np.ndarray:
    """
    Compute and persist the homography.  Returns the matrix so the caller
    can use it immediately without a round-trip load.
    """
    matrix = compute_matrix(pixel_points, world_points_cm)
    path   = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "pixel_points":    pixel_points,
            "world_points_cm": world_points_cm,
            "room_size_cm":    room_size_cm,
            "matrix":          matrix.tolist(),
        }, f, indent=2)
    return matrix


def load_homography(path: str | Path) -> dict:
    """
    Load a saved homography file.
    Returns a dict with keys: pixel_points, world_points_cm, room_size_cm,
    matrix (as a numpy float32 3×3 array).
    Raises FileNotFoundError or ValueError on bad data.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    required = {"pixel_points", "world_points_cm", "room_size_cm", "matrix"}
    missing  = required - data.keys()
    if missing:
        raise ValueError(f"Homography file missing keys: {missing}")
    data["matrix"] = np.float32(data["matrix"])
    return data


def transform_point(
    matrix: np.ndarray,
    cx_px: float,
    cy_px: float,
) -> tuple[float, float]:
    """
    Apply a homography matrix to a single pixel point.
    Returns (x_cm, y_cm) in world / floor coordinates.
    """
    pt     = np.float32([[[cx_px, cy_px]]])
    result = cv2.perspectiveTransform(pt, matrix)
    return float(result[0][0][0]), float(result[0][0][1])


def transform_points(
    matrix: np.ndarray,
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Batch-transform a list of (cx_px, cy_px) tuples → (x_cm, y_cm)."""
    if not points:
        return []
    arr    = np.float32([[[p[0], p[1]] for p in points]])
    result = cv2.perspectiveTransform(arr, matrix)
    return [(float(p[0]), float(p[1])) for p in result[0]]


def default_homography_path(video_path: str | Path) -> Path:
    """Return the default .homography.json path next to the video."""
    p = Path(video_path)
    return p.parent / (p.stem + ".homography.json")
