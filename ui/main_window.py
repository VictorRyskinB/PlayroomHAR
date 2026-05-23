# ui/main_window.py
# MainWindow: top-level window.
#
# Live-threshold architecture
# ───────────────────────────
# YOLO always runs at CAPTURE_CONF_FLOOR (0.05) and every detection above
# that floor is stored in the JSON.  Three spinboxes (confidence, proximity,
# min-duration) control how those raw detections are filtered and assembled
# into segments — without re-running YOLO.  Changing any spinbox triggers a
# lightweight refilter (InteractionMapper + SegmentBuilder, ~50–150 ms for a
# 10-min video) via a 250 ms debounce timer, updating both the bounding-box
# overlay and the results table in real time.
#
# Workers:
#   _YoloWorker         — YOLO-only pipeline (fast, for iteration)
#   _FullAnalysisWorker — YOLO + MMAction2 pipeline (production)

from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSplitter, QStatusBar,
    QProgressBar, QCheckBox, QFrame, QSpinBox, QDoubleSpinBox, QFileDialog,
)

from ui.video_player import VideoPlayerWidget
from ui.results_table import ResultsTableWidget
from mock_data import MOCK_BOUNDING_BOXES, MOCK_RESULTS


# ---------------------------------------------------------------------------
# Worker: YOLO-only
# ---------------------------------------------------------------------------

class _YoloWorker(QObject):
    progress = pyqtSignal(int)
    status   = pyqtSignal(str)
    # raw_frames (list[FrameDetections]), diagnostic string
    finished = pyqtSignal(object, str)
    error    = pyqtSignal(str)

    def __init__(self, video_path: str, frame_w: int, frame_h: int,
                 frame_stride: int = 1, detect_all_classes: bool = False,
                 display_conf: float = 0.25,
                 fps: float = 30.0, total_frames: int = 0):
        super().__init__()
        self._path         = video_path
        self._fw           = frame_w
        self._fh           = frame_h
        self._stride       = frame_stride
        self._all_classes  = detect_all_classes
        self._display_conf = display_conf   # used for the JSON snapshot only
        self._fps          = fps
        self._total_frames = total_frames

    def run(self):
        try:
            from backend.yolo_detector import (
                YoloDetector, DEFAULT_CLASSES, CAPTURE_CONF_FLOOR,
                FrameDetections,
            )
            from backend.interaction_mapper import InteractionMapper
            from backend.segment_builder    import SegmentBuilder

            allowed = None if self._all_classes else DEFAULT_CLASSES

            # ── Phase 1: YOLO at capture floor ──
            self.status.emit("Phase 1/3 — YOLO detection…")
            detector = YoloDetector(
                frame_stride=self._stride,
                allowed_classes=allowed,
                conf_threshold=CAPTURE_CONF_FLOOR,   # always capture at floor
            )
            raw_frames = detector.run(
                self._path,
                progress_cb=lambda p: self.progress.emit(p),
            )

            # ── Diagnostics ──
            n_frames  = len(raw_frames)
            n_person  = sum(1 for f in raw_frames
                            if any(d.label == "person" for d in f.detections))
            n_objects = sum(1 for f in raw_frames
                            if any(d.label != "person" for d in f.detections))
            self.status.emit(
                f"Phase 2/3 — Mapping interactions…  "
                f"({n_person}/{n_frames} frames with person, "
                f"{n_objects} frames with objects)"
            )

            # ── Phase 2: filter at display conf → mapper → builder ──
            # (used only for the JSON snapshot; the UI refilters live)
            filtered = _filter_frames(raw_frames, self._display_conf)
            mapped   = InteractionMapper().map(filtered)
            n_inter  = sum(1 for mf in mapped if mf.interacting_objects())
            self.status.emit(
                f"Phase 3/3 — Building segments…  "
                f"({n_inter}/{n_frames} frames with interactions)"
            )
            segs, _ = SegmentBuilder(self._fw, self._fh).build(mapped)

            # ── Save JSON ──
            _save_results_quietly(
                video_path=self._path,
                segments=segs,
                raw_frames=raw_frames,
                fps=self._fps,
                frame_w=self._fw,
                frame_h=self._fh,
                total_frames=self._total_frames,
                stride=self._stride,
                display_conf=self._display_conf,
                all_classes=self._all_classes,
                mmaction2_used=False,
            )

            diag = (
                f"YOLO: {n_frames} frames  |  "
                f"Person: {n_person}  |  "
                f"Objects: {n_objects}  |  "
                f"Interaction frames (@{self._display_conf:.2f}): {n_inter}"
            )
            self.progress.emit(100)
            self.finished.emit(raw_frames, diag)

        except Exception as exc:
            import traceback; traceback.print_exc()
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Worker: YOLO + MMAction2
# ---------------------------------------------------------------------------

class _FullAnalysisWorker(QObject):
    progress = pyqtSignal(int)
    status   = pyqtSignal(str)
    finished = pyqtSignal(object, str)
    error    = pyqtSignal(str)

    def __init__(self, video_path: str, frame_w: int, frame_h: int,
                 frame_stride: int = 1, detect_all_classes: bool = False,
                 display_conf: float = 0.25,
                 fps: float = 30.0, total_frames: int = 0):
        super().__init__()
        self._path         = video_path
        self._fw           = frame_w
        self._fh           = frame_h
        self._stride       = frame_stride
        self._all_classes  = detect_all_classes
        self._display_conf = display_conf
        self._fps          = fps
        self._total_frames = total_frames

    def run(self):
        try:
            from backend.yolo_detector import (
                YoloDetector, DEFAULT_CLASSES, CAPTURE_CONF_FLOOR,
            )
            from backend.interaction_mapper import InteractionMapper
            from backend.segment_builder    import SegmentBuilder
            from backend.action_recognizer  import ActionRecognizer

            allowed = None if self._all_classes else DEFAULT_CLASSES

            # ── Phase 1: YOLO at capture floor  (0 → 50 %) ──
            self.status.emit("Phase 1/3 — YOLO detection…")
            detector = YoloDetector(
                frame_stride=self._stride,
                allowed_classes=allowed,
                conf_threshold=CAPTURE_CONF_FLOOR,
            )
            raw_frames = detector.run(
                self._path,
                progress_cb=lambda p: self.progress.emit(p // 2),
            )

            n_frames  = len(raw_frames)
            n_person  = sum(1 for f in raw_frames
                            if any(d.label == "person" for d in f.detections))
            n_objects = sum(1 for f in raw_frames
                            if any(d.label != "person" for d in f.detections))

            filtered = _filter_frames(raw_frames, self._display_conf)
            self.status.emit(
                f"Phase 2/3 — Mapping interactions…  "
                f"({n_person}/{n_frames} frames with person, "
                f"{n_objects} frames with objects)"
            )
            mapped = InteractionMapper().map(filtered)
            n_inter = sum(1 for mf in mapped if mf.interacting_objects())

            # ── Phase 2: MMAction2  (50 → 90 %) ──
            recognizer     = ActionRecognizer()
            action_clips   = []
            mmaction2_used = False
            if recognizer.is_available():
                self.status.emit(
                    f"Phase 2/3 — MMAction2 action recognition…  "
                    f"({n_inter} interaction frames)"
                )
                action_clips = recognizer.recognize(
                    self._path,
                    progress_cb=lambda p: self.progress.emit(50 + p * 40 // 100),
                )
                mmaction2_used = bool(action_clips)
            else:
                self.status.emit(
                    f"Phase 2/3 — MMAction2 unavailable, using YOLO labels…  "
                    f"({n_inter}/{n_frames} interaction frames)"
                )

            # ── Phase 3: build + save  (90 → 100 %) ──
            self.status.emit("Phase 3/3 — Building segments…")
            self.progress.emit(90)
            segs, _ = SegmentBuilder(self._fw, self._fh).build(
                mapped, action_clips=action_clips
            )

            _save_results_quietly(
                video_path=self._path,
                segments=segs,
                raw_frames=raw_frames,
                fps=self._fps,
                frame_w=self._fw,
                frame_h=self._fh,
                total_frames=self._total_frames,
                stride=self._stride,
                display_conf=self._display_conf,
                all_classes=self._all_classes,
                mmaction2_used=mmaction2_used,
            )

            diag = (
                f"YOLO: {n_frames} frames  |  "
                f"Person: {n_person}  |  "
                f"Objects: {n_objects}  |  "
                f"Interaction frames (@{self._display_conf:.2f}): {n_inter}"
            )
            self.progress.emit(100)
            self.finished.emit(raw_frames, diag)

        except Exception as exc:
            import traceback; traceback.print_exc()
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Playroom Action Annotator")
        self.resize(1320, 820)
        self._video_path: str | None = None
        self._worker_thread: QThread | None = None
        self._result_time_ranges: list = []

        # Raw FrameDetections cached for live refilter
        self._raw_frames: list | None = None
        # Frame dimensions used by refilter (may differ from video player if no video)
        self._analysis_fw: int = 0
        self._analysis_fh: int = 0
        # Capture floor recorded in the loaded JSON
        self._capture_conf_floor: float = 0.05

        # Debounce timer — refilter fires 250 ms after the last slider change
        self._refilter_timer = QTimer(self)
        self._refilter_timer.setSingleShot(True)
        self._refilter_timer.setInterval(250)
        self._refilter_timer.timeout.connect(self._refilter)

        self._build_ui()
        self._show_mmaction2_banner()
        self._inject_mock_data()

    # ------------------------------------------------------------------ layout

    def _build_ui(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #12122a; color: #dde; }
            QPushButton {
                background: #2e2e5e; color: #dde; border: 1px solid #4e4e8e;
                border-radius: 4px; padding: 5px 10px;
            }
            QPushButton:hover    { background: #3e3e7e; }
            QPushButton:pressed  { background: #1e1e4e; }
            QPushButton:disabled { color: #555; border-color: #333; }
            QCheckBox { spacing: 5px; }
            QCheckBox::indicator {
                width: 14px; height: 14px; border: 1px solid #4e4e8e;
                border-radius: 3px; background: #1e1e3e;
            }
            QCheckBox::indicator:checked { background: #5555cc; }
            QSpinBox, QDoubleSpinBox {
                background: #1e1e3e; color: #dde; border: 1px solid #4e4e8e;
                border-radius: 4px; padding: 2px 4px;
            }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background: #2e2e5e; border: none; width: 16px;
            }
            QSlider::groove:horizontal { height: 4px; background: #2e2e5e; border-radius: 2px; }
            QSlider::handle:horizontal {
                background: #7070cc; border-radius: 6px;
                width: 12px; height: 12px; margin: -4px 0;
            }
            QProgressBar {
                border: 1px solid #3e3e6e; border-radius: 4px;
                background: #1a1a3a; color: #dde; text-align: center;
                font-size: 11px; max-height: 16px;
            }
            QProgressBar::chunk { background: #5555cc; border-radius: 3px; }
            QHeaderView::section {
                background: #1e1e3e; color: #aab; border: none;
                padding: 4px; font-size: 12px;
            }
            QTableView { background: #1a1a30; alternate-background-color: #1e1e38; color: #dde; }
            QFrame#banner {
                background: #2a1a00; border: 1px solid #664400;
                border-radius: 4px; padding: 2px;
            }
            QFrame#parambar {
                background: #0e0e22; border-top: 1px solid #2a2a4a;
                border-bottom: 1px solid #2a2a4a;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 4)
        root.setSpacing(0)

        # ── MMAction2 banner ──
        self._banner = QFrame()
        self._banner.setObjectName("banner")
        bl = QHBoxLayout(self._banner)
        bl.setContentsMargins(8, 4, 8, 4)
        self._banner_label = QLabel()
        self._banner_label.setStyleSheet("color:#ffbb55; font-size:11px;")
        self._banner_label.setWordWrap(True)
        bl.addWidget(self._banner_label)
        root.addWidget(self._banner)

        # ── Toolbar row 1: title + action buttons ──
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 4, 0, 4)
        toolbar.setSpacing(6)

        app_title = QLabel("Playroom Action Annotator")
        app_title.setStyleSheet("font-size:14px; font-weight:bold; color:#aad;")

        self._mock_toggle  = QCheckBox("Mock data")
        self._mock_toggle.setChecked(True)
        self._mock_toggle.stateChanged.connect(self._on_mock_toggled)

        self._debug_toggle = QCheckBox("Debug cols")
        self._debug_toggle.setChecked(False)
        self._debug_toggle.stateChanged.connect(self._on_debug_toggled)

        self._yolo_btn = QPushButton("Run YOLO")
        self._yolo_btn.setEnabled(False)
        self._yolo_btn.setToolTip("YOLO detection only")
        self._yolo_btn.clicked.connect(self._run_yolo)

        self._full_btn = QPushButton("Run Full Analysis")
        self._full_btn.setEnabled(False)
        self._full_btn.setToolTip("YOLO + MMAction2")
        self._full_btn.clicked.connect(self._run_full)

        self._load_json_btn = QPushButton("Load JSON…")
        self._load_json_btn.setToolTip("Load a previously saved _results.json")
        self._load_json_btn.clicked.connect(self._load_json_dialog)

        self._progress    = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setFixedWidth(160)
        self._progress.setVisible(False)

        self._phase_label = QLabel("")
        self._phase_label.setStyleSheet("color:#aab; font-size:10px;")
        self._phase_label.setVisible(False)

        toolbar.addWidget(app_title)
        toolbar.addStretch()
        toolbar.addWidget(self._mock_toggle)
        toolbar.addWidget(self._debug_toggle)
        toolbar.addSpacing(8)
        toolbar.addWidget(self._yolo_btn)
        toolbar.addWidget(self._full_btn)
        toolbar.addWidget(self._load_json_btn)
        toolbar.addWidget(self._phase_label)
        toolbar.addWidget(self._progress)
        root.addLayout(toolbar)

        # ── Toolbar row 2: live filter parameters ──
        parambar_frame = QFrame()
        parambar_frame.setObjectName("parambar")
        parambar = QHBoxLayout(parambar_frame)
        parambar.setContentsMargins(6, 5, 6, 5)
        parambar.setSpacing(6)

        def _param_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-size:11px; color:#99aacc;")
            return lbl

        # All-objects checkbox
        self._all_classes_chk = QCheckBox("All objects")
        self._all_classes_chk.setChecked(True)
        self._all_classes_chk.setToolTip(
            "Checked: detect every YOLO class (recommended for real playroom videos).\n"
            "Unchecked: curated COCO filter only (sports ball, teddy bear, …).\n"
            "Changes take effect on next Run — not a live filter."
        )
        self._all_classes_chk.setStyleSheet("font-size:11px;")

        # Sample rate  (run-time setting, not a live filter)
        self._sample_spin = QSpinBox()
        self._sample_spin.setRange(1, 60)
        self._sample_spin.setValue(1)
        self._sample_spin.setFixedWidth(50)
        self._sample_spin.setToolTip(
            "Process every Nth frame (1 = every frame).\n"
            "Higher values are faster but may miss brief interactions.\n"
            "Changes take effect on next Run — not a live filter."
        )

        # ── Live filters ──
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.25)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setFixedWidth(62)
        self._conf_spin.setToolTip(
            "YOLO detection confidence threshold — LIVE.\n"
            "Lower → more (noisier) detections.\n"
            "Higher → fewer but more certain detections.\n"
            f"Note: detections below the capture floor (0.05) are never stored."
        )
        self._conf_spin.valueChanged.connect(self._schedule_refilter)

        self._prox_spin = QSpinBox()
        self._prox_spin.setRange(20, 500)
        self._prox_spin.setSingleStep(10)
        self._prox_spin.setValue(150)
        self._prox_spin.setFixedWidth(62)
        self._prox_spin.setToolTip(
            "Proximity threshold in pixels — LIVE.\n"
            "Objects within this edge-to-edge distance of the child\n"
            "are counted as interactions even without overlap.\n"
            "Increase if you're missing nearby objects."
        )
        self._prox_spin.valueChanged.connect(self._schedule_refilter)

        self._min_dur_spin = QDoubleSpinBox()
        self._min_dur_spin.setRange(0.1, 10.0)
        self._min_dur_spin.setSingleStep(0.1)
        self._min_dur_spin.setValue(0.2)
        self._min_dur_spin.setDecimals(1)
        self._min_dur_spin.setFixedWidth(58)
        self._min_dur_spin.setToolTip(
            "Minimum interaction segment duration (seconds) — LIVE.\n"
            "Segments shorter than this are discarded.\n"
            "Lower to catch brief touches; higher to suppress noise."
        )
        self._min_dur_spin.valueChanged.connect(self._schedule_refilter)

        parambar.addWidget(self._all_classes_chk)
        parambar.addSpacing(8)
        parambar.addWidget(_param_label("Sample every"))
        parambar.addWidget(self._sample_spin)
        parambar.addWidget(_param_label("frame(s)"))
        parambar.addSpacing(16)
        parambar.addWidget(_param_label("Conf ≥"))
        parambar.addWidget(self._conf_spin)
        parambar.addSpacing(8)
        parambar.addWidget(_param_label("Proximity ≤"))
        parambar.addWidget(self._prox_spin)
        parambar.addWidget(_param_label("px"))
        parambar.addSpacing(8)
        parambar.addWidget(_param_label("Min duration ≥"))
        parambar.addWidget(self._min_dur_spin)
        parambar.addWidget(_param_label("s"))
        parambar.addStretch()
        root.addWidget(parambar_frame)
        root.addSpacing(4)

        # ── Results source label ──
        self._source_label = QLabel("")
        self._source_label.setStyleSheet(
            "color:#7799cc; font-size:10px; padding: 2px 2px;"
        )
        root.addWidget(self._source_label)

        # ── Content splitter ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)

        self._video = VideoPlayerWidget()
        self._video.position_changed.connect(self._on_position_changed)
        self._video.video_loaded.connect(self._on_video_loaded)

        self._results = ResultsTableWidget()

        splitter.addWidget(self._video)
        splitter.addWidget(self._results)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

        self._status = QStatusBar()
        self._status.setStyleSheet("color:#888; font-size:11px;")
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — open a video file to begin.")

    # ------------------------------------------------------------------ banner

    def _show_mmaction2_banner(self):
        from backend.action_recognizer import ActionRecognizer
        if ActionRecognizer.is_available():
            self._banner.setVisible(False)
        else:
            short = ActionRecognizer.unavailable_reason().split(".")[0] + "."
            self._banner_label.setText(
                f"MMAction2 unavailable — {short}  "
                f"'Run Full Analysis' will use YOLO spatial labels only."
            )

    # ------------------------------------------------------------------ video loaded

    def _on_video_loaded(self, path: str):
        self._video_path = path
        self._yolo_btn.setEnabled(True)
        self._full_btn.setEnabled(True)

        # Auto-load results JSON if it exists beside the video
        from backend.results_format import default_output_path
        json_path = default_output_path(path)
        if json_path.exists():
            self._load_json(str(json_path), auto=True)
        else:
            self._status.showMessage(f"Loaded: {path}")
            if self._mock_toggle.isChecked():
                self._inject_mock_data()

    # ------------------------------------------------------------------ toggles

    def _on_mock_toggled(self, _state: int):
        if self._mock_toggle.isChecked():
            self._inject_mock_data()
            self._status.showMessage("Mock data mode active.")
            self._source_label.setText("")
        else:
            self._video.clear_detections()
            self._results.set_results([])
            self._result_time_ranges = []
            self._source_label.setText("")
            self._status.showMessage(
                "Mock data cleared — run analysis or load a JSON file."
            )

    def _on_debug_toggled(self, _state: int):
        self._results.set_debug_mode(self._debug_toggle.isChecked())

    # ------------------------------------------------------------------ mock data

    def _inject_mock_data(self):
        self._video.set_bounding_boxes(MOCK_BOUNDING_BOXES)
        self._results.set_results(MOCK_RESULTS)
        self._result_time_ranges = [
            (_hms_to_ms(r[0]), _hms_to_ms(r[1])) for r in MOCK_RESULTS
        ]

    # ------------------------------------------------------------------ JSON loading

    def _load_json_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Results JSON", "",
            "JSON results (*.json);;All files (*)"
        )
        if path:
            self._load_json(path, auto=False)

    def _load_json(self, json_path: str, auto: bool = False):
        try:
            from backend.results_format import (
                load_results, raw_frames_to_frame_detections,
            )
            data = load_results(json_path)

            fps = data["video_info"]["fps"]
            fw  = data["video_info"]["frame_width"]
            fh  = data["video_info"]["frame_height"]

            # Reconstruct raw FrameDetections — works with both v1.0 and v1.1
            raw_frames = raw_frames_to_frame_detections(
                data["frame_detections"]["frames"], fps
            )
            self._raw_frames       = raw_frames
            self._analysis_fw      = fw
            self._analysis_fh      = fh
            self._capture_conf_floor = float(
                data["frame_detections"].get("capture_conf_floor", 0.05)
            )

            self._mock_toggle.setChecked(False)

            date_str   = data["processing"].get("date", "")[:10]
            prefix     = "Auto-loaded" if auto else "Loaded"
            self._source_label.setText(
                f"{prefix} JSON (captured {date_str})  ·  "
                f"floor {self._capture_conf_floor:.2f}  ·  adjusting thresholds below…"
            )
            self._status.showMessage(
                f"{prefix}: {Path(json_path).name}  —  refiltering…"
            )

            # Run initial refilter using the current spinbox values
            self._refilter()

        except Exception as exc:
            self._status.showMessage(f"Failed to load JSON: {exc}")
            print(f"[LoadJSON] Error: {exc}")

    # ------------------------------------------------------------------ live refilter

    def _schedule_refilter(self):
        """Called by any parameter spinbox change — debounced 250 ms."""
        self._refilter_timer.start()   # restart if already running

    def _refilter(self):
        """
        Apply current conf / proximity / min-duration thresholds to the cached
        raw FrameDetections and update the bounding-box overlay + results table.
        Fast enough (~50–150 ms) to run on the main thread with a debounce.
        """
        if not self._raw_frames:
            return

        from backend.yolo_detector     import FrameDetections
        from backend.interaction_mapper import InteractionMapper
        from backend.segment_builder   import SegmentBuilder

        conf    = self._conf_spin.value()
        prox    = self._prox_spin.value()
        min_dur = int(self._min_dur_spin.value() * 1000)
        fw      = self._analysis_fw
        fh      = self._analysis_fh

        if fw == 0 or fh == 0:
            # Try to get dimensions from the video player
            fw = self._video.frame_width
            fh = self._video.frame_height
            if fw == 0 or fh == 0:
                return
            self._analysis_fw = fw
            self._analysis_fh = fh

        # 1. Filter raw detections by confidence threshold
        filtered = _filter_frames(self._raw_frames, conf)

        # 2. Re-run mapper + builder
        mapped = InteractionMapper(proximity_px=prox).map(filtered)
        segs, ui_boxes = SegmentBuilder(fw, fh, min_duration_ms=min_dur).build(mapped)

        # 3. Update display
        table_rows = [s.as_table_row() for s in segs]
        self._video.set_frame_detections(_convert_boxes(ui_boxes))
        self._results.set_results(table_rows)
        self._result_time_ranges = [
            (_hms_to_ms(r[0]), _hms_to_ms(r[1])) for r in table_rows
        ]

        # 4. Update labels
        n     = len(segs)
        floor = self._capture_conf_floor
        floor_warn = (
            f"  ⚠ conf below capture floor ({floor:.2f})"
            if conf < floor else ""
        )
        self._source_label.setText(
            f"conf ≥ {conf:.2f}  ·  prox ≤ {prox} px  ·  "
            f"min {self._min_dur_spin.value():.1f} s  —  "
            f"{n} segment{'s' if n != 1 else ''}{floor_warn}"
        )

        if n == 0 and self._raw_frames:
            n_total = sum(len(fd.detections) for fd in self._raw_frames)
            n_above = sum(
                1 for fd in self._raw_frames
                for d in fd.detections
                if d.confidence >= conf and d.label != "person"
            )
            hint = (
                f"⚠ No segments at current thresholds  "
                f"({n_above} non-person detections ≥ {conf:.2f}).  "
                f"Try lowering Conf or increasing Proximity."
            )
            self._status.showMessage(hint)
        else:
            self._status.showMessage(
                f"{n} segment{'s' if n != 1 else ''}  "
                f"(conf ≥ {conf:.2f}, prox ≤ {prox} px, "
                f"min {self._min_dur_spin.value():.1f} s)"
            )

    # ------------------------------------------------------------------ pipeline launchers

    def _run_yolo(self):
        self._launch_worker(_YoloWorker)

    def _run_full(self):
        self._launch_worker(_FullAnalysisWorker)

    def _launch_worker(self, worker_cls):
        if not self._video_path:
            self._status.showMessage("No video loaded.")
            return
        if self._worker_thread and self._worker_thread.isRunning():
            return

        fw, fh = self._video.frame_width, self._video.frame_height
        if fw == 0 or fh == 0:
            self._status.showMessage("Cannot read video dimensions.")
            return

        stride       = self._sample_spin.value()
        all_cls      = self._all_classes_chk.isChecked()
        display_conf = self._conf_spin.value()

        self._set_running(True)
        self._worker = worker_cls(
            self._video_path, fw, fh,
            frame_stride=stride,
            detect_all_classes=all_cls,
            display_conf=display_conf,
            fps=self._video.fps,
            total_frames=self._video.total_frames,
        )
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.status.connect(self._on_worker_status)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(lambda *_: self._worker_thread.quit())
        self._worker.error.connect(lambda *_: self._worker_thread.quit())

        self._worker_thread.start()

    # ------------------------------------------------------------------ worker slots

    def _on_worker_status(self, msg: str):
        self._phase_label.setText(msg)
        self._status.showMessage(msg)

    def _on_finished(self, raw_frames, diagnostic: str):
        self._set_running(False)
        self._mock_toggle.setChecked(False)

        # Cache raw frames and dimensions for live refilter
        self._raw_frames      = raw_frames
        self._analysis_fw     = self._video.frame_width
        self._analysis_fh     = self._video.frame_height
        self._capture_conf_floor = 0.05   # workers always use CAPTURE_CONF_FLOOR

        # Show JSON save location
        from backend.results_format import default_output_path
        json_path = default_output_path(self._video_path)
        self._source_label.setText(
            f"Run complete  ·  auto-saved → {json_path.name}  ·  refiltering…"
        )
        self._status.showMessage(f"YOLO complete — {diagnostic}")

        # Run initial refilter using current spinbox values
        self._refilter()

    def _on_error(self, message: str):
        self._set_running(False)
        self._status.showMessage(f"Error: {message}")
        print(f"[Worker] Error: {message}")

    def _set_running(self, running: bool):
        self._yolo_btn.setEnabled(not running)
        self._full_btn.setEnabled(not running)
        self._load_json_btn.setEnabled(not running)
        self._mock_toggle.setEnabled(not running)
        self._sample_spin.setEnabled(not running)
        self._all_classes_chk.setEnabled(not running)
        # Live-filter spinboxes stay enabled during a run so you can
        # adjust them for the next refilter immediately after completion.
        self._progress.setValue(0)
        self._progress.setVisible(running)
        self._phase_label.setVisible(running)
        if not running:
            self._phase_label.setText("")

    # ------------------------------------------------------------------ seek sync

    def _on_position_changed(self, position_ms: int):
        if self._result_time_ranges:
            self._results.highlight_row_at(position_ms, self._result_time_ranges)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_frames(raw_frames: list, conf_threshold: float) -> list:
    """
    Return a new list of FrameDetections keeping only detections ≥ conf_threshold.
    Frames that become empty after filtering are omitted.
    """
    from backend.yolo_detector import FrameDetections
    result = []
    for fd in raw_frames:
        kept = [d for d in fd.detections if d.confidence >= conf_threshold]
        if kept:
            result.append(FrameDetections(
                frame_index=fd.frame_index,
                timestamp_ms=fd.timestamp_ms,
                detections=kept,
            ))
    return result


def _save_results_quietly(
    video_path, segments, raw_frames,
    fps, frame_w, frame_h, total_frames,
    stride, display_conf, all_classes, mmaction2_used,
) -> "Path | None":
    """Save _results.json next to the video. Never raises — logs on failure."""
    try:
        from backend.results_format import (
            save_results, default_output_path, CAPTURE_CONF_FLOOR,
        )
        output_path = default_output_path(video_path)
        settings = {
            "frame_sample_rate":              stride,
            "capture_conf_floor":             CAPTURE_CONF_FLOOR,
            "display_conf_threshold":         display_conf,
            "all_classes_mode":               all_classes,
            "iou_threshold":                  0.05,
            "proximity_threshold_px":         150,
            "mmaction2_confidence_threshold":  0.50,
            "min_segment_duration_sec":        0.2,
        }
        modules_used = {
            "yolo":            True,
            "mmaction2":       mmaction2_used,
            "region_counting": False,
        }
        save_results(
            output_path=output_path,
            video_path=video_path,
            settings=settings,
            modules_used=modules_used,
            segments=segments,
            raw_frames=raw_frames,
            fps=fps,
            frame_w=frame_w,
            frame_h=frame_h,
            total_frames=total_frames,
        )
        print(f"[UI] Results saved → {output_path}")
        return output_path
    except Exception as exc:
        print(f"[UI] WARNING: could not save results JSON: {exc}")
        return None


def _convert_boxes(ui_boxes_by_frame: dict) -> dict:
    return {
        fi: [(b.label, b.color, b.nx, b.ny, b.nw, b.nh) for b in box_list]
        for fi, box_list in ui_boxes_by_frame.items()
    }


def _hms_to_ms(hms: str) -> int:
    parts = hms.split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return int((h * 3600 + m * 60 + s) * 1000)
