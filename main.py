"""
Playroom Action Annotator — unified entry point.

UI mode  (default):
    python main.py

CLI mode (headless, no PyQt6 needed — works on Colab):
    python main.py --cli --input video.mp4 [options]

Options:
    --output          Path for results JSON  (default: <video>_results.json)
    --sample-rate     Process every Nth frame (default: 1)
    --yolo-only       Skip MMAction2 even if available
    --yolo-confidence YOLO detection threshold (default: 0.40)
    --iou-threshold   IoU threshold for interactions (default: 0.05)
    --mmaction2-confidence  MMAction2 label threshold (default: 0.50)
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
    p.add_argument("--yolo-confidence", type=float, default=0.40,
                   help="(CLI) YOLO detection confidence threshold")
    p.add_argument("--iou-threshold", type=float, default=0.05,
                   help="(CLI) IoU threshold for child-object interactions")
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
    print(f"Sample rate: every {args.sample_rate} frame(s)")

    # ---- get video metadata ----
    cap = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Video  : {frame_w}×{frame_h}  {fps:.2f} fps  {total_frames} frames")

    # ---- YOLO ----
    print("\nPhase 1/3 — YOLO detection")

    def yolo_progress(pct: int):
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%", end="", flush=True)

    detector = YoloDetector(
        conf_threshold=args.yolo_confidence,
        frame_stride=args.sample_rate,
    )
    frames = detector.run(str(video_path), progress_cb=yolo_progress)
    print(f"\n  → {len(frames)} frames processed")

    # ---- Interaction mapper ----
    print("\nPhase 2/3 — Interaction mapping")
    mapper = InteractionMapper(iou_threshold=args.iou_threshold)
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
    builder = SegmentBuilder(frame_w=frame_w, frame_h=frame_h)
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
        "yolo_confidence_threshold":      args.yolo_confidence,
        "iou_threshold":                  args.iou_threshold,
        "proximity_threshold_px":         80,
        "mmaction2_confidence_threshold": args.mmaction2_confidence,
        "min_segment_duration_sec":       0.5,
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
        mapped_frames=mapped,
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
