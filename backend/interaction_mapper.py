# backend/interaction_mapper.py
# Determines child-object interactions from per-frame YOLO detections.
#
# Two detection methods (applied in priority order):
#   1. IoU  — bounding boxes physically overlap
#   2. Proximity  — boxes are close but not overlapping
#
# REGION COUNTING INJECTION POINT: A third method will be added here once
# region-of-interest (ROI) counts from YOLO are available. The mapper will
# check whether the child and an object both appear inside a named region
# (e.g. "play mat", "toy shelf") as an additional interaction signal.
#
# MMAction2 INJECTION POINT: After this mapper produces interaction labels
# such as "Interacting with ball", action recognition results from MMAction2
# (e.g. "throwing", "rolling", "stacking") will override the generic verb in
# segment_builder.py so the final label reads "Throwing ball" instead.

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto

from backend.yolo_detector import FrameDetections, Detection

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------

IOU_THRESHOLD       = 0.05   # minimum IoU to count as overlapping interaction
PROXIMITY_THRESHOLD = 150    # pixel distance between edges for proximity fallback
                             # (increased from 80 — HD video makes 80 px too tight)
CHILD_LABEL         = "person"

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class InteractionKind(Enum):
    NONE      = auto()
    OVERLAP   = auto()   # IoU-based
    PROXIMITY = auto()   # edge-distance-based
    # REGION  = auto()   # future: region-counting-based


@dataclass
class MappedDetection:
    """Enriched detection for one object in one frame."""
    detection: Detection
    is_child: bool
    interaction: InteractionKind = InteractionKind.NONE
    # iou_score is set when interaction == OVERLAP
    iou_score: float = 0.0


@dataclass
class MappedFrame:
    """All mapped detections for one frame, plus the child bbox if found."""
    frame_index: int
    timestamp_ms: float
    objects: list[MappedDetection] = field(default_factory=list)

    @property
    def child(self) -> MappedDetection | None:
        for o in self.objects:
            if o.is_child:
                return o
        return None

    def interacting_objects(self) -> list[MappedDetection]:
        return [
            o for o in self.objects
            if not o.is_child and o.interaction != InteractionKind.NONE
        ]


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------

class InteractionMapper:
    """
    Maps per-frame YOLO detections to child-object interactions.

    Usage:
        mapper = InteractionMapper()
        mapped_frames = mapper.map(frame_detections_list)
    """

    def __init__(
        self,
        iou_threshold: float = IOU_THRESHOLD,
        proximity_px: int = PROXIMITY_THRESHOLD,
        child_label: str = CHILD_LABEL,
    ):
        self._iou_thr = iou_threshold
        self._prox_px = proximity_px
        self._child   = child_label

    def map(self, frames: list[FrameDetections]) -> list[MappedFrame]:
        return [self._map_frame(fd) for fd in frames]

    # ------------------------------------------------------------------ frame

    def _map_frame(self, fd: FrameDetections) -> MappedFrame:
        mf = MappedFrame(frame_index=fd.frame_index, timestamp_ms=fd.timestamp_ms)

        # Separate the child (person) detection from objects.
        # If multiple "person" boxes exist, use the largest one as the child.
        persons = [d for d in fd.detections if d.label == self._child]
        objects = [d for d in fd.detections if d.label != self._child]

        if not persons:
            # No child visible — record objects without interaction
            for obj in objects:
                mf.objects.append(MappedDetection(detection=obj, is_child=False))
            return mf

        child_det = max(persons, key=lambda d: d.area)
        mf.objects.append(MappedDetection(detection=child_det, is_child=True))

        for obj in objects:
            kind, iou = self._classify_interaction(child_det, obj)
            mf.objects.append(
                MappedDetection(detection=obj, is_child=False,
                                interaction=kind, iou_score=iou)
            )

        return mf

    # ------------------------------------------------------------------ geometry

    def _classify_interaction(
        self, child: Detection, obj: Detection
    ) -> tuple[InteractionKind, float]:
        iou = _iou(child.bbox, obj.bbox)
        if iou >= self._iou_thr:
            return InteractionKind.OVERLAP, iou

        dist = _edge_distance(child.bbox, obj.bbox)
        if dist <= self._prox_px:
            return InteractionKind.PROXIMITY, 0.0

        # REGION COUNTING INJECTION POINT:
        #   Add a third branch here that checks whether both child and obj
        #   share the same named region polygon.  Example:
        #
        #     if self._same_region(child.bbox, obj.bbox):
        #         return InteractionKind.REGION, 0.0

        return InteractionKind.NONE, 0.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over Union for two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter   = inter_w * inter_h
    if inter == 0:
        return 0.0

    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _edge_distance(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> float:
    """
    Minimum distance between the outer edges of two boxes.
    Returns 0 if they overlap (caller should check IoU first).
    """
    # Horizontal gap
    gap_x = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    # Vertical gap
    gap_y = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return (gap_x ** 2 + gap_y ** 2) ** 0.5
