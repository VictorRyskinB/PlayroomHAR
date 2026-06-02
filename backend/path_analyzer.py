# backend/path_analyzer.py
# Converts raw TrackPoints into world-space PathPoints, computes statistics,
# and builds a heatmap grid for the PathMapWidget.

from __future__ import annotations
import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PathPoint:
    """A single position sample with both pixel and world coordinates."""
    frame_index:  int
    timestamp_ms: float
    cx_px:        float          # original video pixel centroid x
    cy_px:        float          # original video pixel centroid y
    x_cm:         float | None   # None when homography not available
    y_cm:         float | None
    interpolated: bool = False


@dataclass
class PathStats:
    """Summary statistics for a session path."""
    total_distance_m:  float   # sum of straight-line segments between samples
    avg_speed_m_s:     float   # total distance / total time
    duration_s:        float
    n_points:          int
    n_interpolated:    int
    calibrated:        bool    # False = pixel-only, no real-world scale


# ---------------------------------------------------------------------------
# Conversion from raw track → path points
# ---------------------------------------------------------------------------

def build_path_points(
    track_points,                         # list[TrackPoint]
    homography_matrix = None,             # np.ndarray 3×3 or None
) -> list[PathPoint]:
    """
    Apply homography (if available) to convert TrackPoints to PathPoints.
    When homography is None, x_cm / y_cm are left as None.
    """
    from backend.homography_config import transform_points

    if homography_matrix is not None and len(track_points) > 0:
        pixel_pairs = [(tp.cx_px, tp.cy_px) for tp in track_points]
        world_pairs = transform_points(homography_matrix, pixel_pairs)
    else:
        world_pairs = [None] * len(track_points)

    result = []
    for tp, wp in zip(track_points, world_pairs):
        x_cm = wp[0] if wp is not None else None
        y_cm = wp[1] if wp is not None else None
        result.append(PathPoint(
            frame_index=tp.frame_index,
            timestamp_ms=tp.timestamp_ms,
            cx_px=tp.cx_px,
            cy_px=tp.cy_px,
            x_cm=x_cm,
            y_cm=y_cm,
            interpolated=getattr(tp, "interpolated", False),
        ))
    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(path_points: list[PathPoint]) -> PathStats:
    """Compute distance, speed, and interpolation count for a path."""
    n        = len(path_points)
    n_interp = sum(1 for p in path_points if p.interpolated)
    calibrated = any(p.x_cm is not None for p in path_points)

    if n < 2:
        return PathStats(0.0, 0.0, 0.0, n, n_interp, calibrated)

    duration_s = (path_points[-1].timestamp_ms
                  - path_points[0].timestamp_ms) / 1000.0

    dist_cm = 0.0
    for a, b in zip(path_points, path_points[1:]):
        if calibrated and a.x_cm is not None and b.x_cm is not None:
            dx = b.x_cm - a.x_cm
            dy = b.y_cm - a.y_cm
        else:
            # Fall back to pixel distance (not physically meaningful)
            dx = b.cx_px - a.cx_px
            dy = b.cy_px - a.cy_px
        dist_cm += math.hypot(dx, dy)

    dist_m   = dist_cm / 100.0 if calibrated else dist_cm
    avg_spd  = dist_m / duration_s if duration_s > 0 else 0.0

    return PathStats(
        total_distance_m=dist_m,
        avg_speed_m_s=avg_spd,
        duration_s=duration_s,
        n_points=n,
        n_interpolated=n_interp,
        calibrated=calibrated,
    )


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def compute_heatmap(
    path_points: list[PathPoint],
    room_size_cm: tuple[float, float],   # (width, depth) in cm
    grid_cols: int = 50,
    grid_rows: int = 120,
    blur_sigma: float = 2.0,
) -> np.ndarray:
    """
    Build a normalised (0–1) heatmap grid from world-space path points.

    Returns a float32 array of shape (grid_rows, grid_cols).
    Row 0 corresponds to y_cm = room_depth (far wall).
    Row grid_rows-1 corresponds to y_cm = 0 (camera side).
    This matches the top-down view orientation (camera at bottom).

    Returns a zero array if no calibrated points are available.
    """
    room_w, room_d = room_size_cm
    grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)

    calibrated_pts = [p for p in path_points
                      if p.x_cm is not None and p.y_cm is not None]
    if not calibrated_pts:
        return grid

    for p in calibrated_pts:
        col = int(np.clip(p.x_cm / room_w * grid_cols, 0, grid_cols - 1))
        # Flip y so row 0 = far wall (top of display)
        row = int(np.clip((1.0 - p.y_cm / room_d) * grid_rows, 0, grid_rows - 1))
        grid[row, col] += 1.0

    # Smooth with a Gaussian kernel
    ksize = max(3, int(blur_sigma * 4) | 1)   # must be odd
    grid  = cv2.GaussianBlur(grid, (ksize, ksize), blur_sigma)

    # Normalise to 0–1
    mx = grid.max()
    if mx > 0:
        grid /= mx

    return grid


def heatmap_to_rgba(
    grid: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Convert a normalised (0–1) float32 grid to an RGBA uint8 image.
    Uses a blue→cyan→green→yellow→red colormap (JET-like).
    Returns shape (rows, cols, 4).
    """
    h, w    = grid.shape
    grid_u8 = (grid * 255).astype(np.uint8)
    colored = cv2.applyColorMap(grid_u8, cv2.COLORMAP_JET)   # BGR uint8
    rgba    = cv2.cvtColor(colored, cv2.COLOR_BGR2RGBA)

    # Set alpha proportional to density; zero-density cells are transparent
    rgba[:, :, 3] = (grid * alpha * 255).astype(np.uint8)
    return rgba
