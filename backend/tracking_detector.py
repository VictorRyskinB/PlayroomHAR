# backend/tracking_detector.py
# Sequential person tracker using ultralytics ByteTrack.
#
# Unlike YoloDetector (which batches frames for throughput), tracking is
# inherently sequential — the ByteTrack state must persist frame-by-frame.
# This module processes the video with model.track(stream=True) and returns
# a list of TrackPoint objects for the primary person track (most frames seen).
#
# Occlusion handling: brief gaps (up to GAP_TOLERANCE_MS) are bridged by
# linear interpolation so the path line stays continuous on screen.

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import cv2

# Maximum gap (ms) to bridge with interpolation
GAP_TOLERANCE_MS = 1500   # ~45 frames at 30 fps

# Minimum number of points required to count as the "primary" track
MIN_PRIMARY_FRAMES = 5


@dataclass
class TrackPoint:
    """One position sample for the tracked person."""
    frame_index: int
    timestamp_ms: float
    cx_px: float    # horizontal center of bbox (floor contact x)
    cy_px: float    # BOTTOM of bbox (closest pixel to feet on the floor plane)
    track_id: int
    interpolated: bool = False   # True if this point was filled in


class TrackingDetector:
    """
    Runs ByteTrack on a video and returns the primary person's path.

    Usage:
        det = TrackingDetector(model_name="yolov8n.pt")
        points = det.run("session.mp4", progress_cb=lambda p: ...)
        # points: list[TrackPoint], sorted by frame_index
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
    ):
        self._model_name = model_name
        self._conf       = conf_threshold
        self._model      = None

    def _load_model(self):
        from ultralytics import YOLO
        print(f"[TrackingDetector] Loading {self._model_name}…")
        self._model = YOLO(self._model_name)

    def run(
        self,
        video_path: str,
        progress_cb: Callable[[int], None] | None = None,
    ) -> list[TrackPoint]:
        if self._model is None:
            self._load_model()

        # Get total frame count for progress
        cap = cv2.VideoCapture(video_path)
        fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Collect raw per-track data: {track_id: [TrackPoint, ...]}
        tracks: dict[int, list[TrackPoint]] = {}
        frame_idx = 0
        last_pct  = -1

        # stream=True → generator; persist=True → tracker keeps state
        stream = self._model.track(
            source=video_path,
            stream=True,
            persist=True,
            conf=self._conf,
            classes=[0],      # 0 = person in COCO
            verbose=False,
        )

        for result in stream:
            ts_ms = frame_idx / fps * 1000.0
            boxes = result.boxes

            if boxes is not None and boxes.id is not None:
                for i in range(len(boxes)):
                    tid = int(boxes.id[i])
                    x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i])
                    cx = (x1 + x2) / 2.0
                    cy = y2   # bottom-center = feet; correct reference for floor homography
                    pt = TrackPoint(
                        frame_index=frame_idx,
                        timestamp_ms=ts_ms,
                        cx_px=cx,
                        cy_px=cy,
                        track_id=tid,
                    )
                    tracks.setdefault(tid, []).append(pt)

            frame_idx += 1
            if progress_cb and total > 0:
                pct = int(frame_idx / total * 100)
                if pct != last_pct:
                    progress_cb(pct)
                    last_pct = pct

        if not tracks:
            print("[TrackingDetector] No person tracks found.")
            return []

        # Pick the track with the most points as the primary person
        primary_id = max(tracks, key=lambda tid: len(tracks[tid]))
        primary    = sorted(tracks[primary_id], key=lambda p: p.frame_index)
        print(f"[TrackingDetector] Primary track id={primary_id}, "
              f"{len(primary)} points "
              f"({len(tracks)} total track(s) detected)")

        # Bridge short gaps with linear interpolation
        primary = _interpolate_gaps(primary, fps, GAP_TOLERANCE_MS)
        return primary


# ---------------------------------------------------------------------------
# Gap interpolation
# ---------------------------------------------------------------------------

def _interpolate_gaps(
    points: list[TrackPoint],
    fps: float,
    gap_tolerance_ms: float,
) -> list[TrackPoint]:
    """
    Fill gaps shorter than gap_tolerance_ms with linearly interpolated points
    at the native frame rate.  Longer gaps are left as breaks.
    """
    if len(points) < 2:
        return points

    filled: list[TrackPoint] = [points[0]]
    for prev, curr in zip(points, points[1:]):
        gap_ms = curr.timestamp_ms - prev.timestamp_ms
        if 0 < gap_ms <= gap_tolerance_ms:
            # Interpolate
            n_frames = int(round(gap_ms / (1000.0 / fps)))
            for step in range(1, n_frames):
                t = step / n_frames
                ms = prev.timestamp_ms + gap_ms * t
                fi = int(prev.frame_index + (curr.frame_index - prev.frame_index) * t)
                filled.append(TrackPoint(
                    frame_index=fi,
                    timestamp_ms=ms,
                    cx_px=prev.cx_px + (curr.cx_px - prev.cx_px) * t,
                    cy_px=prev.cy_px + (curr.cy_px - prev.cy_px) * t,
                    track_id=prev.track_id,
                    interpolated=True,
                ))
        filled.append(curr)

    return filled
