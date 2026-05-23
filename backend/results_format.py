# backend/results_format.py
# Defines the JSON results schema used by both CLI output and UI loading.
# No dependency on PyQt6 — pure processing code safe to import on Colab.
#
# Schema v1.1 changes from v1.0:
#   • frame_detections now stores RAW FrameDetections (bbox / label / confidence only).
#     The mapper-derived fields (is_child, interaction, is_interacting_with) are no
#     longer stored — they are recomputed on load / whenever a threshold slider moves.
#   • frame_detections["capture_conf_floor"] records the YOLO floor used at capture
#     time so the UI can warn when a slider is set below that floor.
#
# Schema overview (v1.1):
#   {
#     "schema_version": "1.1",
#     "video_info":   { filename, frame_width, frame_height, fps, total_frames },
#     "processing":   { date, settings, modules_used },
#     "segments":     [ { start_time, end_time, label, object, yolo_confidence,
#                         action_confidence, action_source }, ... ],
#     "frame_detections": {
#       "capture_conf_floor": 0.05,
#       "frames": {
#         "<frame_index>": [
#           { "bbox": [x1,y1,x2,y2], "label": str, "confidence": float }, ...
#         ], ...
#       }
#     }
#   }

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION    = "1.1"
CAPTURE_CONF_FLOOR = 0.05   # YOLO floor used when producing new JSONs


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(
    output_path: str | Path,
    video_path: str | Path,
    settings: dict,
    modules_used: dict,
    segments: list,     # list[Segment] from segment_builder
    raw_frames: list,   # list[FrameDetections] from yolo_detector — at capture floor
    fps: float,
    frame_w: int,
    frame_h: int,
    total_frames: int,
) -> None:
    """
    Serialize analysis results to a JSON file.

    segments   — list[backend.segment_builder.Segment]
    raw_frames — list[backend.yolo_detector.FrameDetections] captured at
                 CAPTURE_CONF_FLOOR (0.05).  Storing raw detections lets the
                 UI recompute interactions and segments live when thresholds change.
    """
    doc = {
        "schema_version": SCHEMA_VERSION,
        "video_info": {
            "filename":     Path(video_path).name,
            "frame_width":  frame_w,
            "frame_height": frame_h,
            "fps":          fps,
            "total_frames": total_frames,
        },
        "processing": {
            "date":         datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "settings":     settings,
            "modules_used": modules_used,
        },
        "segments": _serialize_segments(segments),
        "frame_detections": {
            "capture_conf_floor": settings.get("capture_conf_floor", CAPTURE_CONF_FLOOR),
            "frames": _serialize_raw_frames(raw_frames),
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
            "start_time":        _ms_to_hms_ms(s.start_ms),
            "end_time":          _ms_to_hms_ms(s.end_ms),
            "label":             f"{s.interaction_type} {s.object_label}",
            "action":            s.interaction_type,
            "object":            s.object_label,
            "yolo_confidence":   round(s.yolo_confidence, 4),
            "action_confidence": round(s.action_confidence, 4),
            "action_source":     s.action_source,
        })
    return out


def _serialize_raw_frames(raw_frames: list) -> dict:
    """
    Serialize raw FrameDetections (YOLO output) to JSON.
    Only bbox / label / confidence are stored — mapper-derived fields are
    recomputed on demand, so they are not persisted.
    """
    frames: dict[str, list] = {}
    for fd in raw_frames:
        if not fd.detections:
            continue
        frames[str(fd.frame_index)] = [
            {
                "bbox":       [d.x1, d.y1, d.x2, d.y2],
                "label":      d.label,
                "confidence": round(d.confidence, 4),
            }
            for d in fd.detections
        ]
    return frames


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_results(json_path: str | Path) -> dict:
    """
    Load and lightly validate a results JSON file.
    Raises ValueError if required keys are missing.
    Returns the raw parsed dict; supports both schema v1.0 and v1.1.
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
# Reconstruction — raw frames → FrameDetections (for live refilter)
# ---------------------------------------------------------------------------

def raw_frames_to_frame_detections(frames_json: dict, fps: float) -> list:
    """
    Reconstruct a list[FrameDetections] from the JSON frame_detections["frames"].

    Compatible with both schema v1.0 (has extra mapper-derived keys which are
    silently ignored) and v1.1 (stores only bbox/label/confidence).

    Returns frames sorted by frame_index.
    """
    from backend.yolo_detector import FrameDetections, Detection

    result = []
    for frame_str, dets in frames_json.items():
        frame_idx = int(frame_str)
        ts_ms     = frame_idx / fps * 1000.0
        detections = [
            Detection(
                label=d["label"],
                confidence=d["confidence"],
                x1=d["bbox"][0], y1=d["bbox"][1],
                x2=d["bbox"][2], y2=d["bbox"][3],
            )
            for d in dets
        ]
        result.append(FrameDetections(
            frame_index=frame_idx,
            timestamp_ms=ts_ms,
            detections=detections,
        ))

    result.sort(key=lambda fd: fd.frame_index)
    return result


# ---------------------------------------------------------------------------
# Converters for legacy / export use
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
