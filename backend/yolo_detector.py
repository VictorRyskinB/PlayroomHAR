# backend/yolo_detector.py
# Runs YOLOv8/YOLOv11 inference on every frame of a video and returns
# structured per-frame detections.
#
# MMAction2 INJECTION POINT: This module provides the visual object
# detections only. Once MMAction2 is integrated, its per-clip action
# labels will be merged with these detections in segment_builder.py.

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import cv2

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL       = "yolo11n.pt"    # auto-downloads on first run (~5 MB)
DEFAULT_FRAME_STRIDE = 1              # process every Nth frame (1 = every frame, the default)
DEFAULT_BATCH_SIZE  = 32             # frames sent to YOLO per GPU call
                                     # T4/A100: 32–64; CPU-only: 8–16

# CAPTURE_CONF_FLOOR is the minimum confidence used when running YOLO in UI mode.
# All detections above this floor are stored in the JSON so that the display
# threshold slider can be adjusted post-hoc without re-running YOLO.
# In CLI mode, --yolo-confidence acts as a pre-mapper filter applied to the
# captured detections (so the CLI JSON also stores everything ≥ 0.05).
CAPTURE_CONF_FLOOR = 0.05

# Default for the UI confidence spinbox initial value (display threshold).
DEFAULT_CONF_THRESHOLD = 0.25

# Classes the detector will keep when the class filter is active.
# Pass allowed_classes=None to YoloDetector to accept every detected class —
# that is the recommended mode for real playroom videos where toys are unlikely
# to match standard COCO class names.
#
# "person" must stay — the interaction mapper depends on it.
DEFAULT_CLASSES: set[str] = {
    "person",
    "sports ball",
    "teddy bear",
    "bottle",
    "cup",
    "bowl",
    "book",
    "scissors",
    "remote",
    "cell phone",
    "backpack",
    "handbag",
    "suitcase",
    "umbrella",
    "tie",
    "frisbee",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "cat",
    "dog",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "keyboard",
    "laptop",
    "mouse",
    "toy",           # present in some fine-tuned COCO variants
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single object detection for one frame."""
    label: str
    confidence: float
    # Absolute pixel coords in the original video frame: (x1, y1, x2, y2)
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class FrameDetections:
    """All detections for one video frame."""
    frame_index: int
    timestamp_ms: float
    detections: list[Detection] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class YoloDetector:
    """
    Runs YOLO inference on a video file frame-by-frame.

    Usage:
        detector = YoloDetector()
        results = detector.run(
            video_path="session.mp4",
            progress_cb=lambda pct: print(f"{pct}%")
        )
        # results: list[FrameDetections], one entry per processed frame
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        allowed_classes: set[str] | None = DEFAULT_CLASSES,
        frame_stride: int = DEFAULT_FRAME_STRIDE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        allowed_classes:
            DEFAULT_CLASSES  — keep only the curated playroom class list (default).
            None             — accept every class the model detects. Use this for
                               real playroom videos where toys are unlikely to match
                               standard COCO class names (recommended).

        batch_size:
            Number of frames sent to YOLO in a single GPU call.  Larger batches
            keep the GPU fed and dramatically reduce overhead vs single-frame mode.
            Default 32 is safe for T4/16 GB.  Reduce to 8–16 on CPU-only systems.
        """
        self._model_name = model_name
        self._conf       = conf_threshold
        self._allowed    = allowed_classes   # None = no filter (all classes)
        self._stride     = max(1, frame_stride)
        self._batch_size = max(1, batch_size)
        self._model      = None   # lazy-loaded on first run() call

    def _load_model(self):
        from ultralytics import YOLO
        print(f"[YoloDetector] Loading model: {self._model_name}")
        self._model = YOLO(self._model_name)
        names = self._model.names  # dict {int: str}
        if self._allowed is None:
            print(f"[YoloDetector] Class filter: DISABLED — accepting all "
                  f"{len(names)} model classes.")
        else:
            print(f"[YoloDetector] Class filter active — keeping "
                  f"{len(self._allowed)} of {len(names)} classes: "
                  f"{sorted(self._allowed)}")
            present = {n for n in names.values() if n in self._allowed}
            absent  = self._allowed - present
            if absent:
                print(f"[YoloDetector] WARNING — these filter classes are NOT in "
                      f"the model vocabulary: {sorted(absent)}")

    def run(
        self,
        video_path: str | Path,
        progress_cb: Callable[[int], None] | None = None,
    ) -> list[FrameDetections]:
        """
        Process the video and return per-frame detections.

        Frames are collected into batches of `batch_size` and sent to YOLO
        together, keeping the GPU saturated instead of doing one frame at a time.

        progress_cb receives integers 0–100 as processing advances.
        Raises RuntimeError if the video cannot be opened.
        """
        if self._model is None:
            self._load_model()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        last_pct  = -1

        all_results: list[FrameDetections] = []
        batch_bgrs: list    = []   # accumulated BGR frames
        batch_indices: list = []   # corresponding frame indices

        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            if frame_idx % self._stride == 0:
                batch_bgrs.append(bgr)
                batch_indices.append(frame_idx)

                # Flush when batch is full
                if len(batch_bgrs) >= self._batch_size:
                    all_results.extend(
                        self._process_batch(batch_bgrs, batch_indices, fps)
                    )
                    batch_bgrs   = []
                    batch_indices = []

            frame_idx += 1

            if progress_cb and total > 0:
                pct = int(frame_idx / total * 100)
                if pct != last_pct:
                    progress_cb(pct)
                    last_pct = pct

        # Flush any remaining frames
        if batch_bgrs:
            all_results.extend(
                self._process_batch(batch_bgrs, batch_indices, fps)
            )

        cap.release()

        # ── Summary diagnostics ──
        n_frames       = len(all_results)
        n_with_person  = sum(1 for fd in all_results
                             if any(d.label == "person" for d in fd.detections))
        n_with_objects = sum(1 for fd in all_results
                             if any(d.label != "person" for d in fd.detections))
        class_counts: dict[str, int] = {}
        for fd in all_results:
            for d in fd.detections:
                if d.label != "person":
                    class_counts[d.label] = class_counts.get(d.label, 0) + 1

        print(f"[YoloDetector] Processed {n_frames} frames "
              f"({frame_idx} total, stride={self._stride}, "
              f"batch={self._batch_size}).")
        print(f"[YoloDetector]   Frames with person  : {n_with_person}")
        print(f"[YoloDetector]   Frames with objects : {n_with_objects}")
        if class_counts:
            top = sorted(class_counts.items(), key=lambda x: -x[1])[:10]
            print(f"[YoloDetector]   Top object classes : "
                  + ", ".join(f"{k}({v})" for k, v in top))
        else:
            print("[YoloDetector]   *** No objects detected above threshold. ***")
            if self._allowed is not None:
                print("[YoloDetector]   Tip: try --all-classes to bypass the "
                      "class filter.")
        return all_results

    # ------------------------------------------------------------------ batch

    def _process_batch(
        self,
        bgrs: list,
        frame_indices: list[int],
        fps: float,
    ) -> list[FrameDetections]:
        """
        Run YOLO on a list of BGR frames in one GPU call and return
        one FrameDetections per frame.
        """
        # Single predict call for the entire batch — keeps GPU saturated
        preds = self._model.predict(bgrs, conf=self._conf, verbose=False)

        fd_list: list[FrameDetections] = []
        for pred, frame_idx in zip(preds, frame_indices):
            ts_ms = frame_idx / fps * 1000.0
            fd    = FrameDetections(frame_index=frame_idx, timestamp_ms=ts_ms)
            names = pred.names  # {int: str}

            for box in pred.boxes:
                cls_id     = int(box.cls[0])
                label      = names.get(cls_id, str(cls_id))
                confidence = float(box.conf[0])

                if self._allowed is not None and label not in self._allowed:
                    continue

                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                fd.detections.append(
                    Detection(
                        label=label,
                        confidence=confidence,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                    )
                )
            fd_list.append(fd)

        return fd_list
