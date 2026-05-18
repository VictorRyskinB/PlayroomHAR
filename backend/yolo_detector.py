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

DEFAULT_MODEL = "yolo11n.pt"          # auto-downloads on first run (~5 MB)
DEFAULT_CONF_THRESHOLD = 0.40         # minimum detection confidence
DEFAULT_FRAME_STRIDE = 2              # process every Nth frame (1 = every frame)

# Classes the detector will keep; all others are silently dropped.
# Tune this list to match objects found in your playroom sessions.
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
    "cat",
    "dog",
    "chair",
    "couch",
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
        allowed_classes: set[str] | None = None,
        frame_stride: int = DEFAULT_FRAME_STRIDE,
    ):
        self._model_name = model_name
        self._conf = conf_threshold
        self._allowed = allowed_classes if allowed_classes is not None else DEFAULT_CLASSES
        self._stride = max(1, frame_stride)
        self._model = None   # lazy-loaded on first run() call

    def _load_model(self):
        from ultralytics import YOLO
        print(f"[YoloDetector] Loading model: {self._model_name}")
        self._model = YOLO(self._model_name)
        # Log all class names so researchers can tune the filter list
        names = self._model.names  # dict {int: str}
        print(f"[YoloDetector] Model has {len(names)} classes. "
              f"Active filter keeps: {sorted(self._allowed)}")
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
        Process the entire video and return per-frame detections.

        progress_cb receives integers 0–100 as processing advances.
        Raises RuntimeError if the video cannot be opened.
        """
        if self._model is None:
            self._load_model()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx  = 0
        results: list[FrameDetections] = []
        last_pct   = -1

        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            if frame_idx % self._stride == 0:
                ts_ms = frame_idx / fps * 1000.0
                fd = self._process_frame(bgr, frame_idx, ts_ms)
                results.append(fd)

            frame_idx += 1

            if progress_cb and total > 0:
                pct = int(frame_idx / total * 100)
                if pct != last_pct:
                    progress_cb(pct)
                    last_pct = pct

        cap.release()
        print(f"[YoloDetector] Processed {len(results)} frames "
              f"({frame_idx} total, stride={self._stride}).")
        return results

    def _process_frame(
        self, bgr, frame_index: int, timestamp_ms: float
    ) -> FrameDetections:
        fd = FrameDetections(frame_index=frame_index, timestamp_ms=timestamp_ms)

        # Run inference — verbose=False suppresses per-frame console spam
        preds = self._model.predict(bgr, conf=self._conf, verbose=False)
        if not preds:
            return fd

        result = preds[0]
        names  = result.names   # {int: str}

        for box in result.boxes:
            cls_id     = int(box.cls[0])
            label      = names.get(cls_id, str(cls_id))
            confidence = float(box.conf[0])

            if label not in self._allowed:
                continue

            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            fd.detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                )
            )

        return fd
