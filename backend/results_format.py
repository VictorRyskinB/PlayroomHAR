# backend/results_format.py
# Defines the JSON results schema used by both CLI output and UI loading.
# No dependency on PyQt6 — pure processing code safe to import on Colab.
#
# Schema overview:
#   {
#     "schema_version": "1.0",
#     "video_info":   { filename, frame_width, frame_height, fps, total_frames },
#     "processing":   { date, settings, modules_used },
#     "segments":     [ { start_time, end_time, label, object, yolo_confidence,
#                         action_confidence, action_source }, ... ],
#     "frame_detections": {
#       "frames": {
#         "<frame_index>": [
#           { bbox:[x1,y1,x2,y2], label, confidence, is_child, interaction,
#             is_interacting_with }, ...
#         ], ...
#       }
#     }
#   }

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "1.0"

# Colors for converting JSON detections back to the UI overlay format.
# Must stay in sync with SegmentBuilder color constants.
_COLOR_CHILD       = (255, 130,  50)
_COLOR_INTERACTING = ( 60, 220,  90)
_COLOR_PASSIVE     = ( 80, 160, 255)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(
    output_path: str | Path,
    video_path: str | Path,
    settings: dict,
    modules_used: dict,
    segments: list,       # list[Segment] from segment_builder
    mapped_frames: list,  # list[MappedFrame] from interaction_mapper
    fps: float,
    frame_w: int,
    frame_h: int,
    total_frames: int,
) -> None:
    """
    Serialize analysis results to a JSON file.

    segments      — list[backend.segment_builder.Segment]
    mapped_frames — list[backend.interaction_mapper.MappedFrame]
    """
    doc = {
        "schema_version": SCHEMA_VERSION,
        "video_info": {
            "filename": Path(video_path).name,
            "frame_width": frame_w,
            "frame_height": frame_h,
            "fps": fps,
            "total_frames": total_frames,
        },
        "processing": {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "settings": settings,
            "modules_used": modules_used,
        },
        "segments": _serialize_segments(segments),
        "frame_detections": {
            "frames": _serialize_mapped_frames(mapped_frames),
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def _serialize_segments(segments: list) -> list:
    out = []
    for s in segments:
        out.append({
            "start_time": _ms_to_hms_ms(s.start_ms),
            "end_time":   _ms_to_hms_ms(s.end_ms),
            "label":      f"{s.interaction_type} {s.object_label}",
            "action":     s.interaction_type,
            "object":     s.object_label,
            "yolo_confidence":   round(s.yolo_confidence, 4),
            "action_confidence": round(s.action_confidence, 4),
            "action_source":     s.action_source,
        })
    return out


def _serialize_mapped_frames(mapped_frames: list) -> dict:
    frames: dict[str, list] = {}
    for mf in mapped_frames:
        detections = []
        for md in mf.objects:
            d = md.detection
            det: dict = {
                "bbox":       [d.x1, d.y1, d.x2, d.y2],
                "label":      d.label,
                "confidence": round(d.confidence, 4),
                "is_child":   md.is_child,
                "interaction": md.interaction.name,  # "NONE"|"OVERLAP"|"PROXIMITY"
            }
            # is_interacting_with mirrors what the UI uses for colouring
            if not md.is_child and md.interaction.name != "NONE":
                det["is_interacting_with"] = "person"
            else:
                det["is_interacting_with"] = None
            detections.append(det)
        if detections:
            frames[str(mf.frame_index)] = detections
    return frames


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_results(json_path: str | Path) -> dict:
    """
    Load and lightly validate a results JSON file.
    Raises ValueError if the file is missing required keys.
    Returns the raw parsed dict.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required = ("schema_version", "video_info", "processing", "segments",
                "frame_detections")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"Results JSON is missing required keys: {missing}\n"
            f"File: {json_path}"
        )

    return data


# ---------------------------------------------------------------------------
# Converters for the UI
# ---------------------------------------------------------------------------

def segments_to_table_rows(segments_json: list) -> list[tuple]:
    """
    Convert the JSON segments list → list of 6-tuples for ResultsTableWidget.
    Format: (start_str, end_str, label, object, yolo_conf_str, action_conf_str)
    Times are trimmed to HH:MM:SS (no milliseconds) to match the UI column width.
    """
    rows = []
    for s in segments_json:
        start = _trim_ms(s.get("start_time", "00:00:00.000"))
        end   = _trim_ms(s.get("end_time",   "00:00:00.000"))
        label = s.get("label", "")
        obj   = s.get("object", "")
        yc    = s.get("yolo_confidence", 0)
        ac    = s.get("action_confidence", 0)
        yc_s  = f"{yc:.2f}" if yc else "—"
        ac_s  = f"{ac:.2f}" if ac else "—"
        rows.append((start, end, label, obj, yc_s, ac_s))
    return rows


def frame_detections_to_ui_boxes(
    frames_json: dict,
    frame_w: int,
    frame_h: int,
) -> dict[int, list]:
    """
    Convert JSON frame_detections["frames"] →
    {frame_index: [(label, color_rgb, nx, ny, nw, nh), ...]}
    as expected by VideoPlayerWidget.set_frame_detections().
    """
    result: dict[int, list] = {}
    for frame_str, detections in frames_json.items():
        frame_idx = int(frame_str)
        boxes = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            is_child = det.get("is_child", False)
            interacting = det.get("is_interacting_with") is not None

            if is_child:
                color = _COLOR_CHILD
            elif interacting:
                color = _COLOR_INTERACTING
            else:
                color = _COLOR_PASSIVE

            nx = x1 / frame_w if frame_w else 0.0
            ny = y1 / frame_h if frame_h else 0.0
            nw = (x2 - x1) / frame_w if frame_w else 0.0
            nh = (y2 - y1) / frame_h if frame_h else 0.0
            boxes.append((det["label"], color, nx, ny, nw, nh))

        if boxes:
            result[frame_idx] = boxes
    return result


def default_output_path(video_path: str | Path) -> Path:
    """Return <video_stem>_results.json next to the video file."""
    p = Path(video_path)
    return p.parent / (p.stem + "_results.json")


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _ms_to_hms_ms(ms: float) -> str:
    """Format milliseconds as HH:MM:SS.mmm — used in JSON output."""
    total_ms = int(ms)
    h  = total_ms // 3_600_000
    m  = (total_ms % 3_600_000) // 60_000
    s  = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _trim_ms(hms_ms: str) -> str:
    """'00:00:01.200' → '00:00:01'  (for UI table display)."""
    return hms_ms.split(".")[0]
