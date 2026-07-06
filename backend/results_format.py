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

SCHEMA_VERSION    = "1.3"
CAPTURE_CONF_FLOOR = 0.05   # YOLO floor used when producing new JSONs


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(
    output_path: str | Path,
    video_path: str | Path,
    settings: dict,
    modules_used: dict,
    segments: list,       # list[Segment] from segment_builder
    raw_frames: list,     # list[FrameDetections] from yolo_detector — at capture floor
    fps: float,
    frame_w: int,
    frame_h: int,
    total_frames: int,
    action_clips: list | None = None,  # list[ActionClip] — ALL clips, unfiltered
    yolo_run_settings: dict | None = None,  # run-time settings for same-settings check
) -> None:
    """
    Serialize analysis results to a JSON file.

    segments   — list[backend.segment_builder.Segment]
    raw_frames — list[backend.yolo_detector.FrameDetections] captured at
                 CAPTURE_CONF_FLOOR (0.05).  Storing raw detections lets the
                 UI recompute interactions and segments live when thresholds change.
    """
    _now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
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
            "date":         _now,
            "settings":     settings,
            "modules_used": modules_used,
            "yolo":         {"date": _now, **(yolo_run_settings or {})},
            "action":       {},
            "path":         {},
        },
        "segments": _serialize_segments(segments),
        "frame_detections": {
            "capture_conf_floor": settings.get("capture_conf_floor", CAPTURE_CONF_FLOOR),
            "frames": _serialize_raw_frames(raw_frames),
        },
        "action_recognition": _serialize_action_clips(action_clips or []),
    }

    output_path = Path(output_path)

    # Preserve sections written by other run types (path_tracking, etc.)
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for section in ("path_tracking",):
                if section in existing:
                    doc[section] = existing[section]
            # Preserve per-run-type processing sub-dicts we're not overwriting
            for key in ("path",):
                if existing.get("processing", {}).get(key):
                    doc["processing"][key] = existing["processing"][key]
        except Exception:
            pass  # corrupt or missing existing file — just overwrite cleanly

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Compact separators — these files hold 10k+ detections; pretty-printing
    # roughly doubles the size for no benefit.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))


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


def _serialize_action_clips(clips: list) -> dict:
    """
    Serialize ALL action clips (unfiltered) so the UI can apply the threshold live.
    """
    return {
        "capture_conf_floor": 0.0,   # all clips stored regardless of confidence
        "clips": [
            {
                "start_ms":    round(c.start_ms, 1),
                "end_ms":      round(c.end_ms, 1),
                "action_label": c.action_label,
                "raw_label":   c.raw_label,
                "confidence":  round(c.confidence, 4),
                "model_name":  c.model_name,
            }
            for c in clips
        ],
    }


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


def action_clips_from_json(action_rec_json: dict) -> list:
    """
    Reconstruct list[ActionClip] from the JSON "action_recognition" section.
    Returns [] if the section is absent (v1.0/v1.1 JSONs or YOLO-only runs).
    """
    from backend.action_recognizer import ActionClip
    clips = []
    for c in action_rec_json.get("clips", []):
        try:
            clips.append(ActionClip(
                start_ms=float(c["start_ms"]),
                end_ms=float(c["end_ms"]),
                action_label=c["action_label"],
                raw_label=c.get("raw_label", c["action_label"]),
                confidence=float(c["confidence"]),
                model_name=c.get("model_name", "unknown"),
            ))
        except (KeyError, TypeError):
            continue
    return clips


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


# ---------------------------------------------------------------------------
# Upsert helpers — update a single section without overwriting the rest
# ---------------------------------------------------------------------------

def _load_or_create_base(
    json_path: str | Path,
    video_path: str | Path,
    fps: float,
    fw: int,
    fh: int,
    total_frames: int,
) -> dict:
    """
    Load an existing results JSON, or create a minimal skeleton.
    Ensures processing.yolo / .action / .path sub-dicts exist.
    """
    path = Path(json_path)
    doc: dict = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            doc = {}

    doc.setdefault("schema_version", SCHEMA_VERSION)
    doc.setdefault("video_info", {
        "filename":     Path(video_path).name,
        "frame_width":  fw,
        "frame_height": fh,
        "fps":          fps,
        "total_frames": total_frames,
    })
    doc.setdefault("processing", {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    for key in ("yolo", "action", "path"):
        doc["processing"].setdefault(key, {})
    return doc


def _write_json(doc: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))


def _serialize_track_points(points: list) -> list:
    """Serialize list[PathPoint] to a JSON-safe list of dicts."""
    return [
        {
            "frame_index":  p.frame_index,
            "timestamp_ms": round(p.timestamp_ms, 1),
            "cx_px":        round(p.cx_px, 1),
            "cy_px":        round(p.cy_px, 1),
            "x_cm":         round(p.x_cm, 2) if p.x_cm is not None else None,
            "y_cm":         round(p.y_cm, 2) if p.y_cm is not None else None,
            "interpolated": p.interpolated,
        }
        for p in points
    ]


def load_path_section(path_section: dict) -> list:
    """
    Reconstruct list[PathPoint] from the JSON path_tracking section.
    Returns [] if the section is absent or malformed.
    """
    from backend.path_analyzer import PathPoint
    pts = []
    for d in path_section.get("track_points", []):
        try:
            pts.append(PathPoint(
                frame_index  = int(d["frame_index"]),
                timestamp_ms = float(d["timestamp_ms"]),
                cx_px        = float(d["cx_px"]),
                cy_px        = float(d["cy_px"]),
                x_cm         = float(d["x_cm"]) if d.get("x_cm") is not None else None,
                y_cm         = float(d["y_cm"]) if d.get("y_cm") is not None else None,
                interpolated = bool(d.get("interpolated", False)),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return pts


def upsert_action_section(
    output_path: str | Path,
    video_path: str | Path,
    fps: float,
    fw: int,
    fh: int,
    total_frames: int,
    action_clips: list,
    run_settings: dict | None = None,
) -> None:
    """Update only the action_recognition section; leave all other sections intact."""
    doc = _load_or_create_base(output_path, video_path, fps, fw, fh, total_frames)
    doc["action_recognition"] = _serialize_action_clips(action_clips)
    if run_settings:
        doc["processing"]["action"] = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **run_settings,
        }
    _write_json(doc, output_path)


def upsert_path_section(
    output_path: str | Path,
    video_path: str | Path,
    fps: float,
    fw: int,
    fh: int,
    total_frames: int,
    track_points: list,
    room_size_cm: tuple,
    run_settings: dict | None = None,
    smooth_window: int = 1,
) -> None:
    """
    Update only the path_tracking section; leave all other sections intact.

    track_points are stored RAW (unsmoothed) so smoothing can be re-applied
    with any window later.  The stats are computed on the smoothed path so the
    stored numbers match what the UI displays; smooth_window is recorded.
    """
    from backend.path_analyzer import compute_stats, smooth_path

    doc = _load_or_create_base(output_path, video_path, fps, fw, fh, total_frames)

    stats = compute_stats(smooth_path(track_points, smooth_window))
    stats_dict: dict = {
        "n_points":       stats.n_points,
        "n_interpolated": stats.n_interpolated,
        "duration_s":     round(stats.duration_s, 2),
        "calibrated":     stats.calibrated,
        "smooth_window":  smooth_window,
    }
    if stats.calibrated:
        stats_dict["total_distance_m"] = round(stats.total_distance_m, 3)
        stats_dict["avg_speed_m_s"]    = round(stats.avg_speed_m_s, 3)

    doc["path_tracking"] = {
        "room_size_cm": list(room_size_cm),
        "calibrated":   stats.calibrated,
        "stats":        stats_dict,
        "track_points": _serialize_track_points(track_points),
    }
    if run_settings:
        doc["processing"]["path"] = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **run_settings,
        }
    _write_json(doc, output_path)


# ---------------------------------------------------------------------------
# Run-settings readers  (for same-settings warning)
# ---------------------------------------------------------------------------

def get_yolo_run_settings(json_path: str | Path) -> dict | None:
    """Return stored YOLO run settings from processing.yolo, or None."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        s = doc.get("processing", {}).get("yolo", {})
        return s if s else None
    except Exception:
        return None


def get_action_run_settings(json_path: str | Path) -> dict | None:
    """Return stored action run settings from processing.action, or None."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        s = doc.get("processing", {}).get("action", {})
        return s if s else None
    except Exception:
        return None


def get_path_run_settings(json_path: str | Path) -> dict | None:
    """Return stored path run settings from processing.path, or None."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        s = doc.get("processing", {}).get("path", {})
        return s if s else None
    except Exception:
        return None
