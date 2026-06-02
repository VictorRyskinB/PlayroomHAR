# backend/interaction_mapper.py
# Determines child-object interactions from per-frame YOLO detections.
#
# Two detection methods (applied in priority order):
#   1. IoU  — bounding boxes physically overlap
#   2. Proximity  — boxes are close but not overlapping
#
# Subject/Object model
# ──────────────────────────────────────────────────────────────────────────────
# "Subjects" are the active agents in an interaction (person, hand, foot …).
# "Objects"  are the passive items they interact with (ball, toy, table …).
#
# Valid interactions  : Subject ↔ Object
# Ignored             : Subject ↔ Subject  (person↔hand → constant spam)
#                       Object  ↔ Object   (ball↔table  → not meaningful)
#
# When multiple subjects are present in a frame, each object receives the
# BEST interaction across all subjects (OVERLAP beats PROXIMITY).
# Tie-break priority: hand > foot > person  (more specific beats more general).
#
# The subject that caused an interaction is recorded in
# MappedDetection.interacting_subject, and propagated to Segment.subject_label
# by SegmentBuilder.

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto

from backend.yolo_detector import FrameDetections, Detection

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------

IOU_THRESHOLD       = 0.05   # minimum IoU to count as overlapping interaction
PROXIMITY_THRESHOLD = 150    # pixel distance between edges for proximity fallback
DEFAULT_SUBJECTS    = frozenset({"person"})

# Subject tie-break priority (lower value = higher priority = more specific)
_SUBJECT_PRIORITY: dict[str, int] = {
    "hand":   0,
    "hands":  0,
    "foot":   1,
    "feet":   1,
    "person": 2,
}

def _subj_prio(label: str) -> int:
    return _SUBJECT_PRIORITY.get(label.lower(), 99)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class InteractionKind(Enum):
    NONE      = auto()
    OVERLAP   = auto()   # IoU-based
    PROXIMITY = auto()   # edge-distance-based


@dataclass
class MappedDetection:
    """Enriched detection for one box in one frame."""
    detection:           Detection
    is_child:            bool            = False   # True = this IS a subject
    subject_label:       str             = ""      # e.g. "person"/"hand" when is_child
    interaction:         InteractionKind = InteractionKind.NONE
    interacting_subject: str             = ""      # which subject caused this (on objects)
    iou_score:           float           = 0.0


@dataclass
class MappedFrame:
    """All mapped detections for one frame."""
    frame_index:  int
    timestamp_ms: float
    objects:      list[MappedDetection] = field(default_factory=list)

    @property
    def child(self) -> MappedDetection | None:
        """
        Primary subject for backward-compat usage (UI box colouring etc.).
        Returns the person detection if present, else the first subject.
        """
        subjects = [o for o in self.objects if o.is_child]
        if not subjects:
            return None
        person_subjs = [s for s in subjects if s.detection.label == "person"]
        return person_subjs[0] if person_subjs else subjects[0]

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
    Maps per-frame YOLO detections to subject-object interactions.

    Usage:
        mapper = InteractionMapper(subject_labels={"person", "hand"})
        mapped_frames = mapper.map(frame_detections_list)
    """

    def __init__(
        self,
        iou_threshold:  float                = IOU_THRESHOLD,
        proximity_px:   int                  = PROXIMITY_THRESHOLD,
        # Kept for backward compat — use subject_labels instead
        child_label:    str                  = "person",
        subject_labels: frozenset[str] | set[str] = DEFAULT_SUBJECTS,
    ):
        self._iou_thr  = iou_threshold
        self._prox_px  = proximity_px
        # subject_labels wins over child_label when both are supplied
        self._subjects = frozenset(subject_labels)

    def map(self, frames: list[FrameDetections]) -> list[MappedFrame]:
        return [self._map_frame(fd) for fd in frames]

    # ------------------------------------------------------------------ frame

    def _map_frame(self, fd: FrameDetections) -> MappedFrame:
        mf = MappedFrame(frame_index=fd.frame_index, timestamp_ms=fd.timestamp_ms)

        subjects = [d for d in fd.detections if d.label in self._subjects]
        objects  = [d for d in fd.detections if d.label not in self._subjects]

        # Add all subject detections (largest person is still the visual anchor,
        # but we keep all subjects for interaction checking)
        for subj_det in subjects:
            mf.objects.append(MappedDetection(
                detection=subj_det,
                is_child=True,
                subject_label=subj_det.label,
            ))

        if not subjects:
            # No subjects visible — record objects without interaction
            for obj in objects:
                mf.objects.append(MappedDetection(detection=obj, is_child=False))
            return mf

        # For each object, find the best interaction across all subjects
        for obj in objects:
            best_kind  = InteractionKind.NONE
            best_subj  = ""
            best_iou   = 0.0

            for subj_det in subjects:
                kind, iou = self._classify_interaction(subj_det, obj)
                # Higher interaction kind wins (OVERLAP=3 > PROXIMITY=2 > NONE=1)
                if kind.value > best_kind.value:
                    best_kind = kind
                    best_subj = subj_det.label
                    best_iou  = iou
                elif kind == best_kind and kind != InteractionKind.NONE:
                    # Tie → prefer the more specific subject (lower priority number)
                    if _subj_prio(subj_det.label) < _subj_prio(best_subj):
                        best_subj = subj_det.label
                        best_iou  = iou

            mf.objects.append(MappedDetection(
                detection=obj,
                is_child=False,
                interaction=best_kind,
                interacting_subject=best_subj if best_kind != InteractionKind.NONE else "",
                iou_score=best_iou,
            ))

        return mf

    # ------------------------------------------------------------------ geometry

    def _classify_interaction(
        self, subj: Detection, obj: Detection
    ) -> tuple[InteractionKind, float]:
        iou = _iou(subj.bbox, obj.bbox)
        if iou >= self._iou_thr:
            return InteractionKind.OVERLAP, iou

        dist = _edge_distance(subj.bbox, obj.bbox)
        if dist <= self._prox_px:
            return InteractionKind.PROXIMITY, 0.0

        return InteractionKind.NONE, 0.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over Union for two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
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
    """Minimum edge-to-edge distance.  Returns 0 if boxes overlap."""
    gap_x = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    gap_y = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return (gap_x ** 2 + gap_y ** 2) ** 0.5
