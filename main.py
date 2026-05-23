"""
Playroom Action Annotator — unified entry point.

UI mode  (default):
    python main.py

CLI mode (headless, no PyQt6 needed — works on Colab):
    python main.py --cli --input video.mp4 [options]

Options:
    --output               Path for results JSON  (default: <video>_results.json)
    --sample-rate          Process every Nth frame (default: 1)
    --all-classes          Detect every YOLO class (bypass curated filter)
    --yolo-only            Skip MMAction2 even if available
    --yolo-confidence      Display/mapper threshold applied AFTER capture (default: 0.25)
                           YOLO always captures at CAPTURE_CONF_FLOOR (0.05) so the JSON
                           stores all raw detections and thresholds can be adjusted live
                           in the desktop UI without re-running YOLO.
    --iou-threshold        IoU threshold for interactions (default: 0.05)
    --proximity            Proximity threshold in pixels (default: 150)
    --min-duration         Minimum segment duration in seconds (default: 0.2)
    --mmaction2-confidence MMAction2 label threshold (default: 0.50)
"""

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Playroom Action Annotator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cli", action="store_true",
                   help="Run in headless CLI mode (no GUI)")
    p.add_argument("--input",  type=str, default=None,
                   help="(CLI) Path to input video file")
    p.add_argument("--output", type=str, default=None,
                   help="(CLI) Path for output JSON "
                        "(default: <video>_results.json next to the video)")
    p.add_argument("--sample-rate", type=int, default=1,
                   help="(CLI) Process every Nth frame")
    p.add_argument("--yolo-only", action="store_true",
                   help="(CLI) Skip MMAction2 even if available")
    p.add_argument("--all-classes", action="store_true",
                   help="(CLI) Detect every YOLO class, bypassing the curated filter")
    p.add_argument("--yolo-confidence", type=float, default=0.25,
                   help="(CLI) Display/mapper threshold; YOLO always captures at 0.05")
    p.add_argument("--iou-threshold", type=float, default=0.05,
                   help="(CLI) IoU threshold for child-object interactions")
    p.add_argument("--proximity", type=int, default=150,
                   help="(CLI) Proximity threshold in pixels for interactions")
    p.add_argument("--min-duration", type=float, default=0.2,
                   help="(CLI) Minimum segment duration in seconds")
    p.add_argument("--mmaction2-confidence", type=float, default=0.50,
                   help="(CLI) MMAction2 minimum confidence to accept a label")
    return p


# ---------------------------------------------------------------------------
# CLI mode
# ---------------------------------------------------------------------------

def run_cli(args) -> None:
    """Headless pipeline: YOLO → mapper → segments → (optional MMAction2) → JSON."""
    import cv2
    from pathlib import Path

    from backend.yolo_detector      import YoloDetector
    from backend.interaction_mapper  import InteractionMapper
    from backend.segment_builder    import SegmentBuilder
    from backend.results_format     import (
        save_results, default_output_path,
    )

    video_path = Path(args.input)
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else default_output_path(video_path)
    print(f"Input  : {video_path}")
    print(f"Output : {output_path}")
    print(f"Sample rate : every {args.sample_rate} frame(s)")
    print(f"Class filter: {'DISABLED (all classes)' if args.all_classes else 'curated list'}")
    print(f"Conf threshold: capture at 0.05, mapper/display at {args.yolo_confidence}")

    # ---- get video metadata ----
    cap = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Video  : {frame_w}×{frame_h}  {fps:.2f} fps  {total_frames} frames")

    # ---- YOLO — always run at capture floor ----
    print("\nPhase 1/3 — YOLO detection (capture floor 0.05)")

    def yolo_progress(pct: int):
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%", end="", flush=True)

    from backend.yolo_detector import (
        DEFAULT_CLASSES as _DEFAULT_CLASSES,
        CAPTURE_CONF_FLOOR,
        FrameDetections as _FD,
    )
    allowed_classes = None if args.all_classes else _DEFAULT_CLASSES
    detector = YoloDetector(
        conf_threshold=CAPTURE_CONF_FLOOR,   # always capture at floor
        frame_stride=args.sample_rate,
        allowed_classes=allowed_classes,
    )
    raw_frames = detector.run(str(video_path), progress_cb=yolo_progress)
    print(f"\n  → {len(raw_frames)} frames captured at floor {CAPTURE_CONF_FLOOR}")

    # Apply display/mapper threshold before passing to mapper
    frames = [
        _FD(fd.frame_index, fd.timestamp_ms,
            [d for d in fd.detections if d.confidence >= args.yolo_confidence])
        for fd in raw_frames
        if any(d.confidence >= args.yolo_confidence for d in fd.detections)
    ]
    print(f"  → {len(frames)} frames kept at display threshold {args.yolo_confidence}")

    # ---- Interaction mapper ----
    print("\nPhase 2/3 — Interaction mapping")
    mapper = InteractionMapper(
        iou_threshold=args.iou_threshold,
        proximity_px=args.proximity,
    )
    mapped = mapper.map(frames)
    n_interacting = sum(
        1 for mf in mapped if mf.interacting_objects()
    )
    print(f"  → {n_interacting}/{len(mapped)} frames have child-object interactions")

    # ---- MMAction2 (optional) ----
    action_clips = []
    mmaction2_used = False

    if not args.yolo_only:
        from backend.action_recognizer import ActionRecognizer
        recognizer = ActionRecognizer(conf_threshold=args.mmaction2_confidence)
        if recognizer.is_available():
            print("\nPhase 3a/3 — MMAction2 action recognition")

            def action_progress(pct: int):
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                print(f"\r  [{bar}] {pct:3d}%", end="", flush=True)

            action_clips = recognizer.recognize(
                str(video_path), progress_cb=action_progress
            )
            print(f"\n  → {len(action_clips)} action clips above threshold")
            mmaction2_used = True
        else:
            print(f"\n[WARNING] MMAction2 unavailable — using YOLO spatial labels only.")
            print(f"  Reason: {recognizer.unavailable_reason()}")

    # ---- Segment builder ----
    print("\nPhase 3/3 — Building segments")
    builder = SegmentBuilder(
        frame_w=frame_w,
        frame_h=frame_h,
        min_duration_ms=int(args.min_duration * 1000),
    )
    segments, _ = builder.build(mapped, action_clips=action_clips)
    print(f"  → {len(segments)} interaction segments")

    if segments:
        print("\n  Segments:")
        for s in segments:
            row = s.as_table_row()
            print(f"    {row[0]} → {row[1]}  {row[2]}")

    # ---- Save JSON ----
    settings = {
        "frame_sample_rate":              args.sample_rate,
        "capture_conf_floor":             CAPTURE_CONF_FLOOR,
        "display_conf_threshold":         args.yolo_confidence,
        "all_classes_mode":               args.all_classes,
        "iou_threshold":                  args.iou_threshold,
        "proximity_threshold_px":         args.proximity,
        "mmaction2_confidence_threshold": args.mmaction2_confidence,
        "min_segment_duration_sec":       args.min_duration,
    }
    modules_used = {
        "yolo":             True,
        "mmaction2":        mmaction2_used,
        "region_counting":  False,
    }

    save_results(
        output_path=output_path,
        video_path=video_path,
        settings=settings,
        modules_used=modules_used,
        segments=segments,
        raw_frames=raw_frames,   # full capture at 0.05 floor
        fps=fps,
        frame_w=frame_w,
        frame_h=frame_h,
        total_frames=total_frames,
    )
    print(f"\nDone. Results saved to: {output_path}\n")


# ---------------------------------------------------------------------------
# UI mode
# ---------------------------------------------------------------------------

def run_ui() -> None:
    # PyQt6 is only imported here — CLI mode never touches it
    from PyQt6.QtWidgets import QApplication
    from ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Playroom Action Annotator")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    if args.cli:
        if not args.input:
            parser.error("--input is required in CLI mode")
        run_cli(args)
    else:
        run_ui()
