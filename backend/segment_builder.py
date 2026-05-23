# backend/segment_builder.py
# Converts per-frame interaction data into time-stamped segments and
# into the normalized bounding-box format consumed by VideoPlayerWidget.
#
# The build() method optionally accepts MMAction2 action clips; when
# provided, apply_action_labels() overlays specific action verbs onto
# the generic YOLO-derived "Interacting with X" labels.

from __future__ import annotations
from dataclasses import dataclass, field

from backend.interaction_mapper import MappedFrame, InteractionKind

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------

MIN_SEGMENT_DURATION_MS = 200    # ignore interactions shorter than this
                                 # (lowered from 500 ms — catches brief contacts)
SAME_OBJECT_GAP_MS      = 2000   # merge consecutive same-object segments
                                 # if the gap between them is ≤ this value
                                 # (raised from 1000 ms — joins near-miss runs)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """One child-object interaction segment."""
    start_ms: float
    end_ms: float
    object_label: str
    interaction_type: str  = "Interacting with"  # overwritten by MMAction2
    yolo_confidence: float = 0.0   # average YOLO detection confidence
    action_confidence: float = 0.0 # MMAction2 confidence (0 when not used)
    action_source: str = "yolo_spatial"  # "yolo_spatial" | "mmaction2"

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def as_table_row(self) -> tuple[str, str, str, str, str, str]:
        """
        Return a 6-tuple for the results table:
          (start, end, action, object, yolo_conf, action_conf)
        The last two columns are shown only in debug mode.
        """
        yolo_str   = f"{self.yolo_confidence:.2f}"   if self.yolo_confidence > 0 else "—"
        action_str = f"{self.action_confidence:.2f}" if self.action_confidence > 0 else "—"
        return (
            _ms_to_hms(self.start_ms),
            _ms_to_hms(self.end_ms),
            f"{self.interaction_type} {self.object_label}",
            self.object_label,
            yolo_str,
            action_str,
        )


@dataclass
class UiBox:
    """
    One bounding-box entry for VideoPlayerWidget.
    Coordinates normalized [0.0, 1.0] relative to frame dimensions.
    """
    frame_index: int
    label: str
    color: tuple[int, int, int]
    nx: float
    ny: float
    nw: float
    nh: float


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class SegmentBuilder:
    """
    Converts MappedFrame objects into:
      - list[Segment]             — for the results table
      - dict[int, list[UiBox]]    — for the video overlay

    Usage (YOLO-only):
        builder = SegmentBuilder(frame_w=1280, frame_h=720)
        segments, ui_boxes = builder.build(mapped_frames)

    Usage (YOLO + MMAction2):
        from backend.action_recognizer import apply_action_labels
        segments, ui_boxes = builder.build(mapped_frames, action_clips=clips)
    """

    COLOR_CHILD       = (255, 130,  50)   # orange — child
    COLOR_INTERACTING = ( 60, 220,  90)   # green  — interacting object
    COLOR_PASSIVE     = ( 80, 160, 255)   # blue   — passive object

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        min_duration_ms: float = MIN_SEGMENT_DURATION_MS,
        merge_gap_ms: float    = SAME_OBJECT_GAP_MS,
    ):
        self._fw      = frame_w
        self._fh      = frame_h
        self._min_dur = min_duration_ms
        self._gap     = merge_gap_ms

    def build(
        self,
        mapped_frames: list[MappedFrame],
        action_clips: list | None = None,   # list[ActionClip] or None
    ) -> tuple[list[Segment], dict[int, list[UiBox]]]:
        ui_boxes = self._build_ui_boxes(mapped_frames)
        raw_segs = self._extract_raw_segments(mapped_frames)
        merged   = self._merge_and_filter(raw_segs)

        # ACTION_OVERRIDE: overlay MMAction2 labels when clips are provided
        if action_clips:
            from backend.action_recognizer import apply_action_labels
            merged = apply_action_labels(merged, action_clips)

        # REGION COUNTING INJECTION POINT:
        #   Once region-of-interest counts are available from interaction_mapper,
        #   add a second override pass here:
        #     from backend.region_mapper import apply_region_context
        #     merged = apply_region_context(merged, region_data)

        return merged, ui_boxes

    # ------------------------------------------------------------------ ui boxes

    def _build_ui_boxes(
        self, mapped_frames: list[MappedFrame]
    ) -> dict[int, list[UiBox]]:
        result: dict[int, list[UiBox]] = {}
        for mf in mapped_frames:
            boxes: list[UiBox] = []
            for md in mf.objects:
                d = md.detection
                boxes.append(UiBox(
                    frame_index=mf.frame_index,
                    label=d.label,
                    color=self._pick_color(md),
                    nx=d.x1 / self._fw,
                    ny=d.y1 / self._fh,
                    nw=(d.x2 - d.x1) / self._fw,
                    nh=(d.y2 - d.y1) / self._fh,
                ))
            if boxes:
                result[mf.frame_index] = boxes
        return result

    def _pick_color(self, md) -> tuple[int, int, int]:
        if md.is_child:
            return self.COLOR_CHILD
        if md.interaction != InteractionKind.NONE:
            return self.COLOR_INTERACTING
        return self.COLOR_PASSIVE

    # ------------------------------------------------------------------ segments

    def _extract_raw_segments(
        self, mapped_frames: list[MappedFrame]
    ) -> list[Segment]:
        """
        Walk frames in order and group consecutive interacting-object runs
        into Segment instances, accumulating YOLO detection confidence.
        """
        # {label: (start_ms, last_ms, [confidence_values])}
        active: dict[str, tuple[float, float, list[float]]] = {}
        raw: list[Segment] = []

        for mf in mapped_frames:
            interacting = {
                md.detection.label: md.detection.confidence
                for md in mf.interacting_objects()
            }

            # Close runs for objects no longer interacting
            for label in set(active) - set(interacting):
                start_ms, last_ms, confs = active.pop(label)
                avg_conf = sum(confs) / len(confs) if confs else 0.0
                raw.append(Segment(
                    start_ms=start_ms, end_ms=last_ms,
                    object_label=label,
                    yolo_confidence=avg_conf,
                ))

            # Open or extend runs
            for label, conf in interacting.items():
                if label not in active:
                    active[label] = (mf.timestamp_ms, mf.timestamp_ms, [conf])
                else:
                    s, _, cs = active[label]
                    cs.append(conf)
                    active[label] = (s, mf.timestamp_ms, cs)

        # Close still-open runs at end of video
        for label, (start_ms, last_ms, confs) in active.items():
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            raw.append(Segment(
                start_ms=start_ms, end_ms=last_ms,
                object_label=label,
                yolo_confidence=avg_conf,
            ))

        raw.sort(key=lambda s: s.start_ms)
        return raw

    def _merge_and_filter(self, segments: list[Segment]) -> list[Segment]:
        """Merge nearby same-object segments, then drop too-short ones."""
        by_label: dict[str, list[Segment]] = {}
        for s in segments:
            by_label.setdefault(s.object_label, []).append(s)

        merged: list[Segment] = []
        for label, segs in by_label.items():
            segs.sort(key=lambda s: s.start_ms)
            cur = segs[0]
            for nxt in segs[1:]:
                if nxt.start_ms - cur.end_ms <= self._gap:
                    # Weighted average confidence
                    d1, d2 = cur.duration_ms, nxt.duration_ms
                    total = d1 + d2 or 1
                    avg_conf = (cur.yolo_confidence * d1 + nxt.yolo_confidence * d2) / total
                    cur = Segment(
                        start_ms=cur.start_ms, end_ms=nxt.end_ms,
                        object_label=label,
                        interaction_type=cur.interaction_type,
                        yolo_confidence=avg_conf,
                    )
                else:
                    merged.append(cur)
                    cur = nxt
            merged.append(cur)

        merged.sort(key=lambda s: s.start_ms)
        return [s for s in merged if s.duration_ms >= self._min_dur]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms_to_hms(ms: float) -> str:
    total_s = int(ms) // 1000
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
