# backend/yolo_detector.py
# Runs YOLOv8 or YOLO-World inference on every frame of a video and returns
# structured per-frame detections.
#
# Standard models (yolov8n/s/m/l/x):
#   Detect the 80 COCO classes.  Use allowed_classes to keep only the ones
#   relevant to a playroom, or pass None to keep everything.
#
# YOLO-World (yolov8s-worldv2):
#   Open-vocabulary detector.  Pass world_classes=["toy", "slinky", ...] and
#   the model will detect exactly those objects (no COCO limit).
#   allowed_classes is ignored — the world_classes list IS the filter.

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import cv2

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL        = "yolov8n.pt"    # auto-downloads on first run (~6 MB)
DEFAULT_FRAME_STRIDE = 1
DEFAULT_BATCH_SIZE   = 32

# Map short UI keys → actual ultralytics model filenames
YOLO_MODEL_OPTIONS: dict[str, str] = {
    "yolov8n  (nano)":    "yolov8n.pt",
    "yolov8s  (small)":   "yolov8s.pt",
    "yolov8m  (medium)":  "yolov8m.pt",
    "yolov8l  (large)":   "yolov8l.pt",
    "yolov8x  (extra)":   "yolov8x.pt",
    "YOLO-World":         "yolov8s-worldv2.pt",
}

CAPTURE_CONF_FLOOR     = 0.05
DEFAULT_CONF_THRESHOLD = 0.25

# Curated COCO subset for standard models when "All objects" is OFF.
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
    "toy",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single object detection for one frame."""
    label: str
    confidence: float
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
    Runs YOLO inference (standard or YOLO-World) on a video file.

    Standard usage (COCO classes):
        detector = YoloDetector(model_name="yolov8m.pt")

    YOLO-World usage (open vocabulary):
        detector = YoloDetector(
            model_name="yolov8s-worldv2.pt",
            world_classes=["toy", "slinky", "ball", "book", ...],
        )
        # allowed_classes is ignored for YOLO-World — world_classes IS the filter.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        allowed_classes: set[str] | None = DEFAULT_CLASSES,
        frame_stride: int = DEFAULT_FRAME_STRIDE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        world_classes: list[str] | None = None,
    ):
        self._model_name   = model_name
        self._conf         = conf_threshold
        self._allowed      = allowed_classes
        self._stride       = max(1, frame_stride)
        self._batch_size   = max(1, batch_size)
        self._world_classes = world_classes   # non-None → YOLO-World mode
        self._is_world     = "world" in model_name.lower()
        self._model        = None   # lazy-loaded

    def _load_model(self):
        from ultralytics import YOLO
        print(f"[YoloDetector] Loading model: {self._model_name}")
        self._model = YOLO(self._model_name)

        if self._is_world:
            from backend.yolo_classes_config import DEFAULT_PLAYROOM_CLASSES
            classes = self._world_classes or DEFAULT_PLAYROOM_CLASSES
            self._model.set_classes(classes)
            print(f"[YoloDetector] YOLO-World mode — {len(classes)} custom classes: "
                  f"{classes[:8]}{'…' if len(classes) > 8 else ''}")
            # No allowed_classes filter needed; the model already limits output
            self._allowed = None
        else:
            names = self._model.names
            if self._allowed is None:
                print(f"[YoloDetector] Class filter: DISABLED — "
                      f"accepting all {len(names)} COCO classes.")
            else:
                present = {n for n in names.values() if n in self._allowed}
                absent  = self._allowed - present
                print(f"[YoloDetector] Class filter: keeping "
                      f"{len(present)} of {len(names)} classes.")
                if absent:
                    print(f"[YoloDetector] WARNING — not in model vocab: "
                          f"{sorted(absent)}")

    def run(
        self,
        video_path: str | Path,
        progress_cb: Callable[[int], None] | None = None,
    ) -> list[FrameDetections]:
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
        batch_bgrs: list    = []
        batch_indices: list = []

        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            if frame_idx % self._stride == 0:
                batch_bgrs.append(bgr)
                batch_indices.append(frame_idx)

                if len(batch_bgrs) >= self._batch_size:
                    all_results.extend(
                        self._process_batch(batch_bgrs, batch_indices, fps)
                    )
                    batch_bgrs    = []
                    batch_indices = []

            frame_idx += 1

            if progress_cb and total > 0:
                pct = int(frame_idx / total * 100)
                if pct != last_pct:
                    progress_cb(pct)
                    last_pct = pct

        if batch_bgrs:
            all_results.extend(
                self._process_batch(batch_bgrs, batch_indices, fps)
            )

        cap.release()

        n_frames      = len(all_results)
        n_with_person = sum(1 for fd in all_results
                            if any(d.label == "person" for d in fd.detections))
        n_with_objects = sum(1 for fd in all_results
                             if any(d.label != "person" for d in fd.detections))
        class_counts: dict[str, int] = {}
        for fd in all_results:
            for d in fd.detections:
                if d.label != "person":
                    class_counts[d.label] = class_counts.get(d.label, 0) + 1

        print(f"[YoloDetector] Processed {n_frames} frames "
              f"(stride={self._stride}, batch={self._batch_size}).")
        print(f"[YoloDetector]   Frames with person  : {n_with_person}")
        print(f"[YoloDetector]   Frames with objects : {n_with_objects}")
        if class_counts:
            top = sorted(class_counts.items(), key=lambda x: -x[1])[:10]
            print(f"[YoloDetector]   Top classes: "
                  + ", ".join(f"{k}({v})" for k, v in top))
        else:
            print("[YoloDetector]   *** No objects detected above threshold. ***")
        return all_results

    def _process_batch(
        self,
        bgrs: list,
        frame_indices: list[int],
        fps: float,
    ) -> list[FrameDetections]:
        preds = self._model.predict(bgrs, conf=self._conf, verbose=False)

        fd_list: list[FrameDetections] = []
        for pred, frame_idx in zip(preds, frame_indices):
            ts_ms = frame_idx / fps * 1000.0
            fd    = FrameDetections(frame_index=frame_idx, timestamp_ms=ts_ms)
            names = pred.names

            for box in pred.boxes:
                cls_id     = int(box.cls[0])
                label      = names.get(cls_id, str(cls_id))
                confidence = float(box.conf[0])

                # Standard model with class filter
                if not self._is_world and self._allowed is not None:
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
            fd_list.append(fd)

        return fd_list
