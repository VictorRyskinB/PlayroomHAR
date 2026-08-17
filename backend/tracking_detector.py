# backend/tracking_detector.py
# Sequential person tracker using ultralytics ByteTrack.
#
# Unlike YoloDetector (which batches frames for throughput), tracking is
# inherently sequential — the ByteTrack state must persist frame-by-frame.
# This module processes the video with model.track(stream=True) and returns
# a list of TrackPoint objects for the primary person, stitched together
# across the multiple ByteTrack ids that one person can accumulate.
#
# Occlusion handling: brief gaps (up to GAP_TOLERANCE_MS) are bridged by
# linear interpolation so the path line stays continuous on screen.

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import cv2

# Maximum gap (ms) to bridge with interpolation
GAP_TOLERANCE_MS = 1500   # ~45 frames at 30 fps

# Fragments shorter than this are treated as detection noise, not a person
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
        frame_stride: int = 1,
    ):
        self._model_name = model_name
        self._conf       = conf_threshold
        self._stride     = max(1, frame_stride)
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
        # vid_stride=N → ultralytics reads every Nth frame; the generator only
        # yields sampled frames, so the true frame index is yield_count * N.
        stream = self._model.track(
            source=video_path,
            stream=True,
            persist=True,
            conf=self._conf,
            classes=[0],      # 0 = person in COCO
            vid_stride=self._stride,
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

            frame_idx += self._stride
            if progress_cb and total > 0:
                pct = min(100, int(frame_idx / total * 100))
                if pct != last_pct:
                    progress_cb(pct)
                    last_pct = pct

        if not tracks:
            print("[TrackingDetector] No person tracks found.")
            return []

        # ByteTrack assigns a fresh id after losing someone (occlusion, odd
        # pose, leaving frame), so one child often spans several ids. Stitch
        # temporally non-overlapping fragments into one path; fragments that
        # overlap the child's timeline are a different person and are dropped.
        primary = _stitch_fragments(tracks)

        # Bridge short gaps with linear interpolation
        primary = _interpolate_gaps(primary, fps, GAP_TOLERANCE_MS)
        return primary


# ---------------------------------------------------------------------------
# Fragment stitching
# ---------------------------------------------------------------------------

# A fragment may share at most this fraction of its frames with the track
# built so far and still be considered the same person re-identified.
MAX_OVERLAP_FRACTION = 0.2


def _stitch_fragments(tracks: dict[int, list[TrackPoint]]) -> list[TrackPoint]:
    """
    Merge per-id track fragments into a single path for the primary person.

    Starts from the longest fragment, then greedily absorbs other fragments
    (longest first) whose frames barely overlap what is already covered —
    those are the same person under a new ByteTrack id. Fragments that
    substantially overlap in time are concurrent people and are skipped.
    """
    fragments = sorted(tracks.values(), key=len, reverse=True)
    fragments = [f for f in fragments if len(f) >= MIN_PRIMARY_FRAMES]
    if not fragments:
        # Everything was noise-length; fall back to the longest raw fragment.
        fragments = [max(tracks.values(), key=len)]

    # The longest single fragment can belong to the *other* person in frame,
    # so try each fragment as the seed and keep whichever merged set covers
    # the most frames — the child, present for the whole session, wins.
    best: tuple[list[TrackPoint], list[int]] = ([], [])
    for seed in range(len(fragments)):
        covered: set[int] = set()
        merged: list[TrackPoint] = []
        used_ids: list[int] = []
        order = [fragments[seed]] + fragments[:seed] + fragments[seed + 1:]
        for frag in order:
            frames = {p.frame_index for p in frag}
            overlap = len(frames & covered) / len(frames)
            if merged and overlap > MAX_OVERLAP_FRACTION:
                continue
            merged.extend(p for p in frag if p.frame_index not in covered)
            covered |= frames
            used_ids.append(frag[0].track_id)
        if len(merged) > len(best[0]):
            best = (merged, used_ids)

    merged, used_ids = best
    merged.sort(key=lambda p: p.frame_index)
    print(f"[TrackingDetector] Stitched {len(used_ids)} fragment(s) "
          f"(ids {used_ids}) into {len(merged)} points "
          f"({len(tracks)} total track(s) detected)")
    return merged


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
