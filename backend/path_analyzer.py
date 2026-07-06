# backend/path_analyzer.py
# Converts raw TrackPoints into world-space PathPoints, computes statistics,
# and builds a heatmap grid for the PathMapWidget.
#
# Public API
# ──────────
# Data types:
#   PathPoint, PathStats       — existing, unchanged
#   SpeedSample                — (timestamp_ms, speed_m_s, speed_cm_s)
#   MovementEpisode            — one active or stationary run
#   RegionVisit                — one entry→exit event for a named region
#   RegionStats                — aggregated region metrics
#
# Functions (existing, unchanged):
#   build_path_points(track_points, homography_matrix) → list[PathPoint]
#   compute_stats(path_points)                         → PathStats
#   compute_heatmap(path_points, room_size_cm, ...)    → np.ndarray
#   heatmap_to_rgba(grid, alpha)                       → np.ndarray
#
# Functions (new):
#   smooth_path(path_points, window)                   → list[PathPoint]
#   compute_speed_series(path_points)                  → list[SpeedSample]
#   compute_episodes(path_points, ...)                 → list[MovementEpisode]
#   compute_region_stats(path_points, regions, fw, fh) → list[RegionStats]
#   compute_coverage_pct(path_points, room_size_cm)    → float

from __future__ import annotations
import math
from dataclasses import dataclass, field

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

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


@dataclass
class SpeedSample:
    """Instantaneous speed tied to one path point."""
    timestamp_ms: float
    speed_m_s:    float
    speed_cm_s:   float


@dataclass
class MovementEpisode:
    """A continuous period of active movement or stillness."""
    kind:          str     # "active" | "stationary"
    start_ms:      float
    end_ms:        float
    distance_m:    float   # 0.0 for stationary episodes
    avg_speed_m_s: float   # 0.0 for stationary episodes

    @property
    def duration_s(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0

    def as_table_row(self) -> tuple:
        """6-tuple for the episodes table."""
        return (
            _fmt_ms(self.start_ms),
            _fmt_ms(self.end_ms),
            f"{self.duration_s:.1f} s",
            self.kind.capitalize(),
            f"{self.distance_m:.2f} m" if self.distance_m > 0 else "—",
            f"{self.avg_speed_m_s:.2f} m/s" if self.avg_speed_m_s > 0 else "—",
        )


@dataclass
class RegionVisit:
    """One continuous entry-to-exit event for a named region."""
    region_name: str
    entry_ms:    float
    exit_ms:     float

    @property
    def duration_s(self) -> float:
        return (self.exit_ms - self.entry_ms) / 1000.0


@dataclass
class RegionStats:
    """Aggregated metrics for one named region."""
    region_name:    str
    total_dwell_s:  float
    dwell_pct:      float          # % of session duration
    visit_count:    int
    first_visit_ms: float | None   # latency from session start; None if never visited
    visits:         list[RegionVisit] = field(default_factory=list)

    def as_summary_row(self) -> tuple:
        """5-tuple for the region summary table."""
        lat = (f"{self.first_visit_ms / 1000:.1f} s"
               if self.first_visit_ms is not None else "—")
        return (
            self.region_name,
            f"{self.total_dwell_s:.1f} s",
            f"{self.dwell_pct:.1f} %",
            str(self.visit_count),
            lat,
        )


# ---------------------------------------------------------------------------
# Conversion from raw track → path points  (unchanged)
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
            frame_index  = tp.frame_index,
            timestamp_ms = tp.timestamp_ms,
            cx_px        = tp.cx_px,
            cy_px        = tp.cy_px,
            x_cm         = x_cm,
            y_cm         = y_cm,
            interpolated = getattr(tp, "interpolated", False),
        ))
    return result


# ---------------------------------------------------------------------------
# Path smoothing
# ---------------------------------------------------------------------------

def smooth_path(
    path_points: list[PathPoint],
    window: int = 5,
) -> list[PathPoint]:
    """
    Apply a centred rolling-average to reduce tracker jitter.
    Smooths both pixel (cx_px, cy_px) and world (x_cm, y_cm) coordinates.
    Timestamps and frame indices are preserved exactly.
    """
    if len(path_points) < 3 or window < 2:
        return path_points

    half = window // 2
    n    = len(path_points)
    result = []

    for i, p in enumerate(path_points):
        lo    = max(0, i - half)
        hi    = min(n, i + half + 1)
        chunk = path_points[lo:hi]

        avg_cx = sum(q.cx_px for q in chunk) / len(chunk)
        avg_cy = sum(q.cy_px for q in chunk) / len(chunk)

        cal = [q for q in chunk if q.x_cm is not None and q.y_cm is not None]
        avg_x = sum(q.x_cm for q in cal) / len(cal) if cal else None
        avg_y = sum(q.y_cm for q in cal) / len(cal) if cal else None

        result.append(PathPoint(
            frame_index  = p.frame_index,
            timestamp_ms = p.timestamp_ms,
            cx_px        = avg_cx,
            cy_px        = avg_cy,
            x_cm         = avg_x,
            y_cm         = avg_y,
            interpolated = p.interpolated,
        ))

    return result


# ---------------------------------------------------------------------------
# Speed series
# ---------------------------------------------------------------------------

def compute_speed_series(
    path_points: list[PathPoint],
) -> list[SpeedSample]:
    """
    Compute instantaneous speed at each path point using centred differences
    (forward difference at the first point, backward at the last).

    Uses world coords when calibrated; falls back to pixel coords otherwise
    (pixel-based values have no physical unit and are not exposed in the UI).
    Returns speed_m_s = 0.0 for uncalibrated points.
    """
    n = len(path_points)
    if n == 0:
        return []

    calibrated = any(p.x_cm is not None for p in path_points)
    speeds_cm_s: list[float] = [0.0] * n

    for i in range(n):
        if i == 0:
            a, b = path_points[0], path_points[min(1, n - 1)]
        elif i == n - 1:
            a, b = path_points[max(0, n - 2)], path_points[n - 1]
        else:
            a, b = path_points[i - 1], path_points[i + 1]

        dt_ms = b.timestamp_ms - a.timestamp_ms
        if dt_ms <= 0:
            speeds_cm_s[i] = speeds_cm_s[i - 1] if i > 0 else 0.0
            continue

        if calibrated and a.x_cm is not None and b.x_cm is not None:
            dist = math.hypot(b.x_cm - a.x_cm, b.y_cm - a.y_cm)   # cm
        else:
            dist = math.hypot(b.cx_px - a.cx_px, b.cy_px - a.cy_px)  # px

        speeds_cm_s[i] = dist / (dt_ms / 1000.0)

    return [
        SpeedSample(
            timestamp_ms = path_points[i].timestamp_ms,
            speed_cm_s   = speeds_cm_s[i] if calibrated else 0.0,
            speed_m_s    = speeds_cm_s[i] / 100.0 if calibrated else 0.0,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(path_points: list[PathPoint]) -> PathStats:
    """Compute distance, speed, and interpolation count for a path."""
    n          = len(path_points)
    n_interp   = sum(1 for p in path_points if p.interpolated)
    calibrated = any(p.x_cm is not None for p in path_points)

    if n < 2:
        return PathStats(0.0, 0.0, 0.0, n, n_interp, calibrated)

    duration_s = (path_points[-1].timestamp_ms
                  - path_points[0].timestamp_ms) / 1000.0

    dist_cm = 0.0
    for a, b in zip(path_points, path_points[1:]):
        if calibrated and a.x_cm is not None and b.x_cm is not None:
            dist_cm += math.hypot(b.x_cm - a.x_cm, b.y_cm - a.y_cm)
        else:
            dist_cm += math.hypot(b.cx_px - a.cx_px, b.cy_px - a.cy_px)

    dist_m   = dist_cm / 100.0 if calibrated else dist_cm
    avg_spd  = dist_m / duration_s if duration_s > 0 else 0.0

    return PathStats(
        total_distance_m = dist_m,
        avg_speed_m_s    = avg_spd,
        duration_s       = duration_s,
        n_points         = n,
        n_interpolated   = n_interp,
        calibrated       = calibrated,
    )


# ---------------------------------------------------------------------------
# Episode segmentation
# ---------------------------------------------------------------------------

def compute_episodes(
    path_points: list[PathPoint],
    stationary_threshold_m_s: float = 0.10,
    min_episode_ms: float = 500.0,
) -> list[MovementEpisode]:
    """
    Segment the path into alternating active / stationary episodes.

    stationary_threshold_m_s
        Speed below this value → point labelled "stationary".
        Default 0.10 m/s (10 cm/s).  Adjust in the UI.

    min_episode_ms
        Episodes shorter than this are absorbed into their neighbours.
        Prevents single noisy frames from fragmenting the sequence.
    """
    if len(path_points) < 2:
        return []

    calibrated   = any(p.x_cm is not None for p in path_points)
    speed_series = compute_speed_series(path_points)
    threshold_cm = stationary_threshold_m_s * 100.0

    # Label each point
    labels: list[str] = [
        "stationary" if s.speed_cm_s < threshold_cm else "active"
        for s in speed_series
    ]

    # Enforce minimum episode length — iterate until stable
    for _ in range(20):           # safety cap
        runs    = _run_length_encode(labels)
        changed = False
        new_labels = list(labels)
        for kind, si, ei in runs:
            dur_ms = (path_points[ei].timestamp_ms
                      - path_points[si].timestamp_ms)
            if dur_ms < min_episode_ms:
                # Absorb into the previous run's kind (or next if at start)
                absorb = (labels[si - 1] if si > 0
                          else labels[ei + 1] if ei + 1 < len(labels)
                          else kind)
                if absorb != kind:
                    for k in range(si, ei + 1):
                        new_labels[k] = absorb
                    changed = True
        labels = new_labels
        if not changed:
            break

    # Build MovementEpisode objects from the final labelling
    episodes: list[MovementEpisode] = []
    for kind, si, ei in _run_length_encode(labels):
        pts   = path_points[si: ei + 1]
        start = pts[0].timestamp_ms
        end   = pts[-1].timestamp_ms

        if kind == "active" and calibrated:
            dist_cm = sum(
                math.hypot(
                    (b.x_cm or 0.0) - (a.x_cm or 0.0),
                    (b.y_cm or 0.0) - (a.y_cm or 0.0),
                )
                for a, b in zip(pts, pts[1:])
                if a.x_cm is not None and b.x_cm is not None
            )
            dist_m  = dist_cm / 100.0
            dur_s   = (end - start) / 1000.0
            avg_spd = dist_m / dur_s if dur_s > 0 else 0.0
        else:
            dist_m  = 0.0
            avg_spd = 0.0

        episodes.append(MovementEpisode(
            kind          = kind,
            start_ms      = start,
            end_ms        = end,
            distance_m    = dist_m,
            avg_speed_m_s = avg_spd,
        ))

    return episodes


def episode_summary(episodes: list[MovementEpisode]) -> dict:
    """
    Return a dict of headline stats derived from the episode list.
    Useful for the summary panel in the Analysis tab.
    """
    active_eps      = [e for e in episodes if e.kind == "active"]
    stationary_eps  = [e for e in episodes if e.kind == "stationary"]
    total_s         = sum(e.duration_s for e in episodes)

    active_s        = sum(e.duration_s for e in active_eps)
    stationary_s    = sum(e.duration_s for e in stationary_eps)

    return {
        "n_active_episodes":     len(active_eps),
        "n_stationary_episodes": len(stationary_eps),
        "active_s":              active_s,
        "stationary_s":          stationary_s,
        "active_ratio":          active_s / total_s if total_s > 0 else 0.0,
        "avg_active_dur_s":      (active_s / len(active_eps)
                                  if active_eps else 0.0),
        "avg_stationary_dur_s":  (stationary_s / len(stationary_eps)
                                  if stationary_eps else 0.0),
    }


# ---------------------------------------------------------------------------
# Region metrics
# ---------------------------------------------------------------------------

def compute_region_stats(
    path_points: list[PathPoint],
    regions: list[dict],     # [{name, nx, ny, nw, nh}, ...]
    frame_w: int,
    frame_h: int,
) -> list[RegionStats]:
    """
    Compute dwell time, visit count, and first-visit latency for every region.

    Containment is tested in pixel space (cx_px, cy_px) against the normalised
    region rectangles — no homography required.

    first_visit_ms is the latency from the first path point (session start).
    """
    if not path_points or not regions:
        return []

    t0         = path_points[0].timestamp_ms
    session_ms = path_points[-1].timestamp_ms - t0
    results    = []

    for region in regions:
        name = region["name"]
        rx0  = region["nx"]               * frame_w
        ry0  = region["ny"]               * frame_h
        rx1  = (region["nx"] + region["nw"]) * frame_w
        ry1  = (region["ny"] + region["nh"]) * frame_h

        visits:    list[RegionVisit] = []
        in_region: bool  = False
        entry_ms:  float = 0.0
        first_ms:  float | None = None

        for p in path_points:
            inside = (rx0 <= p.cx_px <= rx1 and ry0 <= p.cy_px <= ry1)
            if inside and not in_region:
                in_region = True
                entry_ms  = p.timestamp_ms
                if first_ms is None:
                    first_ms = p.timestamp_ms
            elif not inside and in_region:
                in_region = False
                visits.append(RegionVisit(name, entry_ms, p.timestamp_ms))

        if in_region:
            visits.append(RegionVisit(name, entry_ms, path_points[-1].timestamp_ms))

        total_dwell_ms = sum(v.exit_ms - v.entry_ms for v in visits)
        dwell_pct      = (total_dwell_ms / session_ms * 100.0
                          if session_ms > 0 else 0.0)
        first_lat_ms   = (first_ms - t0) if first_ms is not None else None

        results.append(RegionStats(
            region_name    = name,
            total_dwell_s  = total_dwell_ms / 1000.0,
            dwell_pct      = dwell_pct,
            visit_count    = len(visits),
            first_visit_ms = first_lat_ms,
            visits         = visits,
        ))

    return results


# ---------------------------------------------------------------------------
# Region floor-footprint projection (shared by the map widget and Excel export)
# ---------------------------------------------------------------------------

def project_region_shapes(
    regions: list[dict],          # [{name, nx, ny, nw, nh}, ...]
    homography_matrix,            # np.ndarray 3×3 or None
    frame_w: int,
    frame_h: int,
    room_size_cm: tuple[float, float],
) -> list[dict]:
    """
    Project each region's video-space rectangle onto the floor plane.

    Returns one dict per region:
        {"name": str,
         "polygon_cm": [(x, y) × 4] | None,   # floor quadrilateral
         "point_cm":   (x, y) | None}         # bottom-center fallback

    polygon_cm is None when the projection is degenerate — regions with a lot
    of vertical extent throw their top edge toward the image horizon, which
    maps to absurd floor coordinates.  In that case point_cm (the projected
    bottom-center, i.e. the floor contact point) is provided instead.
    Returns [] when homography is None.
    """
    if homography_matrix is None or not regions or frame_w == 0:
        return []

    room_w, room_d = room_size_cm
    bound_x = (-0.5 * room_w, 1.5 * room_w)
    bound_y = (-0.5 * room_d, 1.5 * room_d)
    result  = []

    for region in regions:
        x0 = region["nx"] * frame_w
        y0 = region["ny"] * frame_h
        x1 = (region["nx"] + region["nw"]) * frame_w
        y1 = (region["ny"] + region["nh"]) * frame_h

        corners_px = np.float32([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]])
        corners_w  = cv2.perspectiveTransform(corners_px, homography_matrix)[0]

        sane = all(
            bound_x[0] <= float(cx) <= bound_x[1]
            and bound_y[0] <= float(cy) <= bound_y[1]
            for cx, cy in corners_w
        )

        if sane:
            result.append({
                "name":       region["name"],
                "polygon_cm": [(float(cx), float(cy)) for cx, cy in corners_w],
                "point_cm":   None,
            })
        else:
            pt = cv2.perspectiveTransform(
                np.float32([[[(x0 + x1) / 2.0, y1]]]), homography_matrix
            )[0][0]
            result.append({
                "name":       region["name"],
                "polygon_cm": None,
                "point_cm":   (float(pt[0]), float(pt[1])),
            })

    return result


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def compute_coverage_pct(
    path_points: list[PathPoint],
    room_size_cm: tuple[float, float],
    grid_cols: int = 50,
    grid_rows: int = 120,
) -> float:
    """
    Estimate what percentage of the room floor was visited.
    Uses the same grid as the heatmap.  Returns 0.0 if not calibrated.
    """
    calibrated_pts = [p for p in path_points
                      if p.x_cm is not None and p.y_cm is not None]
    if not calibrated_pts:
        return 0.0

    room_w, room_d = room_size_cm
    visited = set()
    for p in calibrated_pts:
        col = int(np.clip(p.x_cm / room_w * grid_cols, 0, grid_cols - 1))
        row = int(np.clip((1.0 - p.y_cm / room_d) * grid_rows, 0, grid_rows - 1))
        visited.add((row, col))

    return len(visited) / (grid_rows * grid_cols) * 100.0


# ---------------------------------------------------------------------------
# Heatmap  (unchanged)
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
        row = int(np.clip((1.0 - p.y_cm / room_d) * grid_rows, 0, grid_rows - 1))
        grid[row, col] += 1.0

    ksize = max(3, int(blur_sigma * 4) | 1)
    grid  = cv2.GaussianBlur(grid, (ksize, ksize), blur_sigma)

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
    colored = cv2.applyColorMap(grid_u8, cv2.COLORMAP_JET)
    rgba    = cv2.cvtColor(colored, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = (grid * alpha * 255).astype(np.uint8)
    return rgba


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_length_encode(labels: list[str]) -> list[tuple[str, int, int]]:
    """Return [(kind, start_idx, end_idx), ...] for a flat label list."""
    if not labels:
        return []
    runs: list[tuple[str, int, int]] = []
    cur_kind = labels[0]
    cur_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur_kind:
            runs.append((cur_kind, cur_start, i - 1))
            cur_kind  = labels[i]
            cur_start = i
    runs.append((cur_kind, cur_start, len(labels) - 1))
    return runs


def _fmt_ms(ms: float) -> str:
    """Format milliseconds as H:MM:SS.d (decisecond precision) for table display.
    Tenths-of-second suffix prevents sub-second episodes from all reading 0:00:00."""
    total_ms = int(ms)
    total_s  = total_ms // 1000
    h        = total_s // 3600
    m        = (total_s % 3600) // 60
    s        = total_s % 60
    d        = (total_ms % 1000) // 100   # 0-9  (tenths of a second)
    return f"{h}:{m:02d}:{s:02d}.{d}"
