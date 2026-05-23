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
#   _MMAction2Worker    — MMAction2 only (skips YOLO; uses existing video)

from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSplitter, QStatusBar,
    QProgressBar, QCheckBox, QFrame, QSpinBox, QDoubleSpinBox,
    QFileDialog, QInputDialog, QTabWidget, QComboBox, QLineEdit, QMenu,
)

from ui.video_player import VideoPlayerWidget
from ui.results_table import ResultsTableWidget, GenericTableWidget
from mock_data import MOCK_BOUNDING_BOXES, MOCK_RESULTS


# ---------------------------------------------------------------------------
# Worker: YOLO-only
# ---------------------------------------------------------------------------

class _YoloWorker(QObject):
    progress = pyqtSignal(int)
    status   = pyqtSignal(str)
    # raw_frames (list[FrameDetections]), action_clips (list[ActionClip]), diagnostic
    finished = pyqtSignal(object, object, str)
    error    = pyqtSignal(str)

    def __init__(self, video_path: str, frame_w: int, frame_h: int,
                 frame_stride: int = 1, detect_all_classes: bool = False,
                 display_conf: float = 0.25,
                 fps: float = 30.0, total_frames: int = 0,
                 yolo_model: str = "yolov8n.pt",
                 world_classes: list | None = None,
                 enabled_mm_models: list | None = None):
        super().__init__()
        self._path           = video_path
        self._fw             = frame_w
        self._fh             = frame_h
        self._stride         = frame_stride
        self._all_classes    = detect_all_classes
        self._display_conf   = display_conf
        self._fps            = fps
        self._total_frames   = total_frames
        self._yolo_model     = yolo_model
        self._world_classes  = world_classes
        self._enabled_mm     = enabled_mm_models   # unused in YOLO-only, kept for symmetry

    def run(self):
        try:
            from backend.yolo_detector import (
                YoloDetector, DEFAULT_CLASSES, CAPTURE_CONF_FLOOR,
                FrameDetections,
            )
            from backend.interaction_mapper import InteractionMapper
            from backend.segment_builder    import SegmentBuilder

            is_world = "world" in self._yolo_model.lower()
            allowed  = None if (self._all_classes or is_world) else DEFAULT_CLASSES

            # ── Phase 1: YOLO at capture floor ──
            self.status.emit("Phase 1/3 — YOLO detection…")
            detector = YoloDetector(
                model_name=self._yolo_model,
                frame_stride=self._stride,
                allowed_classes=allowed,
                conf_threshold=CAPTURE_CONF_FLOOR,
                world_classes=self._world_classes,
            )
            raw_frames = detector.run(
                self._path,
                progress_cb=lambda p: self.progress.emit(p),
            )

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

            filtered = _filter_frames(raw_frames, self._display_conf)
            mapped   = InteractionMapper().map(filtered)
            n_inter  = sum(1 for mf in mapped if mf.interacting_objects())
            self.status.emit(
                f"Phase 3/3 — Building segments…  "
                f"({n_inter}/{n_frames} frames with interactions)"
            )
            segs, _ = SegmentBuilder(self._fw, self._fh).build(mapped)

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
                action_clips=[],
            )

            diag = (
                f"YOLO: {n_frames} frames  |  "
                f"Person: {n_person}  |  "
                f"Objects: {n_objects}  |  "
                f"Interaction frames (@{self._display_conf:.2f}): {n_inter}"
            )
            self.progress.emit(100)
            self.finished.emit(raw_frames, [], diag)

        except Exception as exc:
            import traceback; traceback.print_exc()
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Worker: YOLO + MMAction2
# ---------------------------------------------------------------------------

class _FullAnalysisWorker(QObject):
    progress = pyqtSignal(int)
    status   = pyqtSignal(str)
    finished = pyqtSignal(object, object, str)   # (raw_frames, action_clips, diag)
    error    = pyqtSignal(str)

    def __init__(self, video_path: str, frame_w: int, frame_h: int,
                 frame_stride: int = 1, detect_all_classes: bool = False,
                 display_conf: float = 0.25,
                 fps: float = 30.0, total_frames: int = 0,
                 yolo_model: str = "yolov8n.pt",
                 world_classes: list | None = None,
                 enabled_mm_models: list | None = None):
        super().__init__()
        self._path           = video_path
        self._fw             = frame_w
        self._fh             = frame_h
        self._stride         = frame_stride
        self._all_classes    = detect_all_classes
        self._display_conf   = display_conf
        self._fps            = fps
        self._total_frames   = total_frames
        self._yolo_model     = yolo_model
        self._world_classes  = world_classes
        self._enabled_mm     = enabled_mm_models

    def run(self):
        try:
            from backend.yolo_detector import (
                YoloDetector, DEFAULT_CLASSES, CAPTURE_CONF_FLOOR,
            )
            from backend.interaction_mapper import InteractionMapper
            from backend.segment_builder    import SegmentBuilder
            from backend.action_recognizer  import ActionRecognizer

            is_world = "world" in self._yolo_model.lower()
            allowed  = None if (self._all_classes or is_world) else DEFAULT_CLASSES

            # ── Phase 1: YOLO  (0 → 50 %) ──
            self.status.emit("Phase 1/3 — YOLO detection…")
            detector = YoloDetector(
                model_name=self._yolo_model,
                frame_stride=self._stride,
                allowed_classes=allowed,
                conf_threshold=CAPTURE_CONF_FLOOR,
                world_classes=self._world_classes,
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
                f"Phase 2/3 — MMAction2 action recognition…  "
                f"({n_person}/{n_frames} frames with person)"
            )
            mapped = InteractionMapper().map(filtered)
            n_inter = sum(1 for mf in mapped if mf.interacting_objects())

            # ── Phase 2: MMAction2  (50 → 90 %) ──
            recognizer     = ActionRecognizer(
                conf_threshold=0.0,
                enabled_models=self._enabled_mm,
            )
            action_clips   = []
            mmaction2_used = False
            if recognizer.is_available():
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
                action_clips=action_clips,
            )

            diag = (
                f"YOLO: {n_frames} frames  |  "
                f"Person: {n_person}  |  "
                f"Objects: {n_objects}  |  "
                f"Interaction frames (@{self._display_conf:.2f}): {n_inter}"
            )
            self.progress.emit(100)
            self.finished.emit(raw_frames, action_clips, diag)

        except Exception as exc:
            import traceback; traceback.print_exc()
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Worker: MMAction2 only (no YOLO)
# ---------------------------------------------------------------------------

class _MMAction2Worker(QObject):
    progress = pyqtSignal(int)
    status   = pyqtSignal(str)
    finished = pyqtSignal(object, str)   # (action_clips, diagnostic)
    error    = pyqtSignal(str)

    def __init__(self, video_path: str, enabled_models: list | None = None):
        super().__init__()
        self._path         = video_path
        self._enabled_mm   = enabled_models

    def run(self):
        try:
            from backend.action_recognizer import ActionRecognizer

            recognizer = ActionRecognizer(
                conf_threshold=0.0,
                enabled_models=self._enabled_mm,
            )
            if not recognizer.is_available():
                self.error.emit(
                    f"MMAction2 is not available: {recognizer.unavailable_reason()}"
                )
                return

            self.status.emit("MMAction2 — running action recognition…")
            action_clips = recognizer.recognize(
                self._path,
                progress_cb=lambda p: self.progress.emit(p),
            )
            diag = f"MMAction2: {len(action_clips)} clips captured from {self._path}"
            self.progress.emit(100)
            self.finished.emit(action_clips, diag)

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
        self.resize(1320, 860)
        self._video_path: str | None = None
        self._worker_thread: QThread | None = None
        self._result_time_ranges: list = []
        self._region_time_ranges: list = []
        self._action_time_ranges: list = []

        # Raw FrameDetections cached for live refilter
        self._raw_frames: list | None = None
        # ALL MMAction2 clips (unfiltered) — UI applies threshold live
        self._action_clips: list = []
        # Named spatial regions
        self._regions: list[dict] = []
        # YOLO-World custom class list (populated from default or loaded file)
        from backend.yolo_classes_config import DEFAULT_PLAYROOM_CLASSES
        self._world_classes: list[str] = list(DEFAULT_PLAYROOM_CLASSES)
        # Frame dimensions used by refilter
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
            QPushButton#editRegionsBtn:checked {
                background: #5e3e1e; border-color: #cc8833; color: #ffcc66;
            }
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
            QFrame#regionbar {
                background: #0a1a0a; border-top: 1px solid #1a3a1a;
                border-bottom: 1px solid #1a3a1a;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 4)
        root.setSpacing(0)

        # ── MMAction2 availability banner ──
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
        self._mock_toggle.setChecked(False)
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

        self._mmaction2_btn = QPushButton("Run MMAction2")
        self._mmaction2_btn.setEnabled(False)
        self._mmaction2_btn.setToolTip(
            "Run MMAction2 action recognition only (no YOLO).\n"
            "Useful for testing action labels without re-running detection."
        )
        self._mmaction2_btn.clicked.connect(self._run_mmaction2)

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
        toolbar.addWidget(self._mmaction2_btn)
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

        self._all_classes_chk = QCheckBox("All objects")
        self._all_classes_chk.setChecked(True)
        self._all_classes_chk.setToolTip(
            "Checked: detect every YOLO class (recommended for real playroom videos).\n"
            "Unchecked: curated COCO filter only (sports ball, teddy bear, …).\n"
            "Changes take effect on next Run — not a live filter."
        )
        self._all_classes_chk.setStyleSheet("font-size:11px;")

        self._sample_spin = QSpinBox()
        self._sample_spin.setRange(1, 60)
        self._sample_spin.setValue(1)
        self._sample_spin.setFixedWidth(50)
        self._sample_spin.setToolTip(
            "Process every Nth frame (1 = every frame).\n"
            "Changes take effect on next Run — not a live filter."
        )

        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.25)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setFixedWidth(62)
        self._conf_spin.setToolTip(
            "YOLO detection confidence threshold — LIVE.\n"
            f"Detections below the capture floor (0.05) are never stored."
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
            "are counted as interactions even without overlap."
        )
        self._prox_spin.valueChanged.connect(self._schedule_refilter)

        self._min_dur_spin = QDoubleSpinBox()
        self._min_dur_spin.setRange(0.1, 10.0)
        self._min_dur_spin.setSingleStep(0.1)
        self._min_dur_spin.setValue(0.2)
        self._min_dur_spin.setDecimals(1)
        self._min_dur_spin.setFixedWidth(58)
        self._min_dur_spin.setToolTip(
            "Minimum interaction segment duration (seconds) — LIVE."
        )
        self._min_dur_spin.valueChanged.connect(self._schedule_refilter)

        self._mmaction2_conf_spin = QDoubleSpinBox()
        self._mmaction2_conf_spin.setRange(0.05, 0.95)
        self._mmaction2_conf_spin.setSingleStep(0.05)
        self._mmaction2_conf_spin.setValue(0.50)
        self._mmaction2_conf_spin.setDecimals(2)
        self._mmaction2_conf_spin.setFixedWidth(62)
        self._mmaction2_conf_spin.setToolTip(
            "MMAction2 action confidence threshold — LIVE.\n"
            "All action clips are stored; only clips ≥ this value are shown."
        )
        self._mmaction2_conf_spin.valueChanged.connect(self._schedule_refilter)

        self._yolo_vis_chk = QCheckBox("YOLO boxes")
        self._yolo_vis_chk.setChecked(True)
        self._yolo_vis_chk.setStyleSheet("font-size:11px;")
        self._yolo_vis_chk.stateChanged.connect(
            lambda s: self._video.set_yolo_visible(bool(s))
        )

        self._action_vis_chk = QCheckBox("Action labels")
        self._action_vis_chk.setChecked(True)
        self._action_vis_chk.setStyleSheet("font-size:11px;")
        self._action_vis_chk.stateChanged.connect(
            lambda s: self._video.set_mmaction2_visible(bool(s))
        )

        parambar.addWidget(self._all_classes_chk)
        parambar.addSpacing(8)
        parambar.addWidget(_param_label("Sample every"))
        parambar.addWidget(self._sample_spin)
        parambar.addWidget(_param_label("frame(s)"))
        parambar.addSpacing(16)
        parambar.addWidget(_param_label("YOLO conf ≥"))
        parambar.addWidget(self._conf_spin)
        parambar.addSpacing(8)
        parambar.addWidget(_param_label("Action conf ≥"))
        parambar.addWidget(self._mmaction2_conf_spin)
        parambar.addSpacing(8)
        parambar.addWidget(_param_label("Proximity ≤"))
        parambar.addWidget(self._prox_spin)
        parambar.addWidget(_param_label("px"))
        parambar.addSpacing(8)
        parambar.addWidget(_param_label("Min duration ≥"))
        parambar.addWidget(self._min_dur_spin)
        parambar.addWidget(_param_label("s"))
        parambar.addSpacing(12)
        parambar.addWidget(self._yolo_vis_chk)
        parambar.addWidget(self._action_vis_chk)
        parambar.addStretch()
        root.addWidget(parambar_frame)

        # ── Toolbar row 3: model selection ──
        modelbar_frame = QFrame()
        modelbar_frame.setObjectName("parambar")
        modelbar = QHBoxLayout(modelbar_frame)
        modelbar.setContentsMargins(6, 4, 6, 4)
        modelbar.setSpacing(8)

        def _model_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-size:11px; color:#99aacc;")
            return lbl

        # ── YOLO model combo ──
        self._yolo_model_combo = QComboBox()
        from backend.yolo_detector import YOLO_MODEL_OPTIONS
        self._yolo_model_display = list(YOLO_MODEL_OPTIONS.keys())   # ordered
        for display_name in self._yolo_model_display:
            self._yolo_model_combo.addItem(display_name)
        self._yolo_model_combo.setCurrentIndex(0)
        self._yolo_model_combo.setFixedWidth(170)
        self._yolo_model_combo.setToolTip(
            "YOLO model used by Run YOLO and Run Full Analysis.\n"
            "Larger models are slower but detect more accurately.\n"
            "YOLO-World uses an open vocabulary — set custom class names below."
        )
        self._yolo_model_combo.currentIndexChanged.connect(
            self._on_yolo_model_changed
        )

        # ── MMAction2 model checkable dropdown ──
        from backend.action_recognizer import _CANDIDATE_MODELS as _MM_MODELS
        self._mmaction2_model_btn = QPushButton()
        self._mmaction2_model_btn.setFixedWidth(200)
        self._mmaction2_model_btn.setToolTip(
            "Choose which MMAction2 model(s) to try.\n"
            "Models are tried top-to-bottom; the first one that loads is used.\n"
            "Future: ensemble mode (both run, highest confidence wins)."
        )
        mm_menu = QMenu(self._mmaction2_model_btn)
        self._mm_model_actions: list[QAction] = []
        for m in _MM_MODELS:
            act = QAction(f"{m['name']}  —  {m['desc']}", mm_menu)
            act.setCheckable(True)
            act.setChecked(True)
            act.setData(m["dataset"])
            act.changed.connect(self._update_mmaction2_btn_label)
            mm_menu.addAction(act)
            self._mm_model_actions.append(act)
        self._mmaction2_model_btn.setMenu(mm_menu)
        self._update_mmaction2_btn_label()   # set initial label

        modelbar.addWidget(_model_label("YOLO model:"))
        modelbar.addWidget(self._yolo_model_combo)
        modelbar.addSpacing(24)
        modelbar.addWidget(_model_label("MMAction2 model:"))
        modelbar.addWidget(self._mmaction2_model_btn)
        modelbar.addStretch()
        root.addWidget(modelbar_frame)

        # ── YOLO-World classes row (hidden unless YOLO-World is selected) ──
        self._world_frame = QFrame()
        self._world_frame.setObjectName("regionbar")
        world_layout = QHBoxLayout(self._world_frame)
        world_layout.setContentsMargins(6, 4, 6, 4)
        world_layout.setSpacing(6)

        world_lbl = QLabel("YOLO-World classes:")
        world_lbl.setStyleSheet("font-size:11px; color:#88bb88;")

        self._world_classes_edit = QLineEdit()
        self._world_classes_edit.setPlaceholderText(
            "e.g.  toy, ball, slinky, book, crayon, …"
        )
        self._world_classes_edit.setToolTip(
            "Comma-separated list of objects YOLO-World should detect.\n"
            "Plain English nouns work best (e.g. 'building block', not 'block_toy').\n"
            "Changes take effect on the next Run."
        )
        self._world_classes_edit.setText(", ".join(self._world_classes))
        self._world_classes_edit.textChanged.connect(self._on_world_classes_changed)

        self._save_classes_btn = QPushButton("Save Classes…")
        self._save_classes_btn.clicked.connect(self._save_classes_dialog)

        self._load_classes_btn = QPushButton("Load Classes…")
        self._load_classes_btn.clicked.connect(self._load_classes_dialog)

        world_layout.addWidget(world_lbl)
        world_layout.addWidget(self._world_classes_edit, stretch=1)
        world_layout.addWidget(self._save_classes_btn)
        world_layout.addWidget(self._load_classes_btn)
        self._world_frame.setVisible(False)   # hidden until YOLO-World selected
        root.addWidget(self._world_frame)

        # ── Toolbar row 4: region capture ──
        regionbar_frame = QFrame()
        regionbar_frame.setObjectName("regionbar")
        regionbar = QHBoxLayout(regionbar_frame)
        regionbar.setContentsMargins(6, 4, 6, 4)
        regionbar.setSpacing(6)

        def _region_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-size:11px; color:#88bb88;")
            return lbl

        self._edit_regions_btn = QPushButton("✏  Edit Regions")
        self._edit_regions_btn.setObjectName("editRegionsBtn")
        self._edit_regions_btn.setCheckable(True)
        self._edit_regions_btn.setChecked(False)
        self._edit_regions_btn.setEnabled(False)
        self._edit_regions_btn.setToolTip(
            "Toggle region drawing mode.\n"
            "Drag a rectangle on the video frame to add a named region.\n"
            "Regions are saved per camera/room in a .regions.json file."
        )
        self._edit_regions_btn.toggled.connect(self._on_edit_regions_toggled)

        self._region_count_lbl = QLabel("0 regions")
        self._region_count_lbl.setStyleSheet("font-size:11px; color:#88bb88;")

        self._regions_vis_chk = QCheckBox("Show")
        self._regions_vis_chk.setChecked(True)
        self._regions_vis_chk.setStyleSheet("font-size:11px;")
        self._regions_vis_chk.setToolTip("Toggle region overlay visibility")
        self._regions_vis_chk.stateChanged.connect(
            lambda s: self._video.set_regions_visible(bool(s))
        )

        self._clear_regions_btn = QPushButton("Clear")
        self._clear_regions_btn.setEnabled(False)
        self._clear_regions_btn.setToolTip("Remove all regions")
        self._clear_regions_btn.clicked.connect(self._clear_regions)

        self._save_regions_btn = QPushButton("Save Regions…")
        self._save_regions_btn.setEnabled(False)
        self._save_regions_btn.setToolTip(
            "Save current regions to a .regions.json file.\n"
            "Each file is specific to a camera/room setup."
        )
        self._save_regions_btn.clicked.connect(self._save_regions_dialog)

        self._load_regions_btn = QPushButton("Load Regions…")
        self._load_regions_btn.setEnabled(False)
        self._load_regions_btn.setToolTip(
            "Load a previously saved .regions.json file."
        )
        self._load_regions_btn.clicked.connect(self._load_regions_dialog)

        regionbar.addWidget(_region_label("Regions:"))
        regionbar.addWidget(self._edit_regions_btn)
        regionbar.addWidget(self._region_count_lbl)
        regionbar.addWidget(self._regions_vis_chk)
        regionbar.addSpacing(8)
        regionbar.addWidget(self._clear_regions_btn)
        regionbar.addWidget(self._save_regions_btn)
        regionbar.addWidget(self._load_regions_btn)
        regionbar.addStretch()
        root.addWidget(regionbar_frame)
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
        self._video.region_drawn.connect(self._on_region_drawn)

        # ── Three-tab results panel ──
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #2a2a4a; background: #12122a;
            }
            QTabBar::tab {
                background: #1e1e3e; color: #99aacc;
                border: 1px solid #2a2a4a; border-bottom: none;
                padding: 5px 12px; font-size: 11px;
            }
            QTabBar::tab:selected { background: #2e2e5e; color: #dde; }
            QTabBar::tab:hover    { background: #262650; }
        """)

        # Tab 0 — YOLO Object Interactions (existing schema)
        self._results = ResultsTableWidget()
        self._tabs.addTab(self._results, "Object Interactions")

        # Tab 1 — Region Presence
        self._region_table = GenericTableWidget(
            ["Start", "End", "Duration", "Region", "Frames"],
            stretch_col=3,
        )
        self._tabs.addTab(self._region_table, "Regions")

        # Tab 2 — MMAction2 Action Clips
        self._action_table = GenericTableWidget(
            ["Start", "End", "Duration", "Action", "Confidence", "Model"],
            stretch_col=3,
        )
        self._tabs.addTab(self._action_table, "Actions (MMAction2)")

        splitter.addWidget(self._video)
        splitter.addWidget(self._tabs)
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
        self._mmaction2_btn.setEnabled(True)
        self._edit_regions_btn.setEnabled(True)
        self._load_regions_btn.setEnabled(True)

        # Auto-load results JSON if it exists beside the video
        from backend.results_format import default_output_path
        json_path = default_output_path(path)
        if json_path.exists():
            self._load_json(str(json_path), auto=True)
        else:
            self._status.showMessage(f"Loaded: {path}")

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

    # ------------------------------------------------------------------ region editing

    # ------------------------------------------------------------------ model selection

    def _update_mmaction2_btn_label(self):
        """Refresh the MMAction2 model button label based on checked actions."""
        from backend.action_recognizer import MODEL_DISPLAY_NAMES
        checked = [a.data() for a in self._mm_model_actions if a.isChecked()]
        if not checked:
            label = "⚠ none selected"
        elif len(checked) == len(self._mm_model_actions):
            label = " + ".join(
                MODEL_DISPLAY_NAMES.get(k, k).replace("TSN ", "")
                for k in checked
            )
        else:
            label = " + ".join(
                MODEL_DISPLAY_NAMES.get(k, k).replace("TSN ", "")
                for k in checked
            )
        self._mmaction2_model_btn.setText(f"{label}  ▾")

    def _get_enabled_mmaction2_models(self) -> list[str]:
        """Return dataset keys for checked MMAction2 models (fallback: all)."""
        enabled = [a.data() for a in self._mm_model_actions if a.isChecked()]
        return enabled if enabled else [m["dataset"]
                                        for m in self._mm_model_actions]

    def _on_yolo_model_changed(self, _index: int):
        """Show/hide YOLO-World classes row when YOLO-World is selected."""
        is_world = "world" in self._yolo_model_combo.currentText().lower()
        self._world_frame.setVisible(is_world)
        # "All objects" checkbox is irrelevant for YOLO-World
        self._all_classes_chk.setEnabled(not is_world)
        if is_world:
            self._all_classes_chk.setToolTip(
                "Disabled — YOLO-World uses its own class list above."
            )
        else:
            self._all_classes_chk.setToolTip(
                "Checked: detect every YOLO class.\n"
                "Unchecked: curated COCO filter only."
            )

    def _on_world_classes_changed(self, text: str):
        """Parse the classes text field into self._world_classes."""
        self._world_classes = [
            c.strip() for c in text.split(",") if c.strip()
        ]

    def _get_yolo_model_name(self) -> str:
        """Return the actual model filename for the currently selected YOLO model."""
        from backend.yolo_detector import YOLO_MODEL_OPTIONS
        display = self._yolo_model_combo.currentText()
        return YOLO_MODEL_OPTIONS.get(display, "yolov8n.pt")

    def _save_classes_dialog(self):
        default_dir = (str(Path(self._video_path).parent)
                       if self._video_path else "")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Class List", default_dir,
            "Class list (*.classes.json);;All files (*)"
        )
        if not path:
            return
        if not path.endswith(".classes.json") and not path.endswith(".json"):
            path += ".classes.json"
        config_name, ok = QInputDialog.getText(
            self, "Config Name",
            "Name for this class list config\n(e.g. 'Playroom – Room A'):",
            text=Path(path).stem.replace(".classes", ""),
        )
        if not ok or not config_name.strip():
            return
        try:
            from backend.yolo_classes_config import save_class_list
            save_class_list(path, config_name.strip(), self._world_classes)
            self._status.showMessage(
                f"Class list saved → {Path(path).name}  "
                f"({len(self._world_classes)} classes)"
            )
        except Exception as exc:
            self._status.showMessage(f"Failed to save class list: {exc}")

    def _load_classes_dialog(self):
        default_dir = (str(Path(self._video_path).parent)
                       if self._video_path else "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Class List", default_dir,
            "Class list (*.classes.json *.json);;All files (*)"
        )
        if not path:
            return
        try:
            from backend.yolo_classes_config import load_class_list
            config_name, classes = load_class_list(path)
            self._world_classes = classes
            self._world_classes_edit.setText(", ".join(classes))
            self._status.showMessage(
                f"Class list loaded: '{config_name}'  "
                f"({len(classes)} class{'es' if len(classes) != 1 else ''})"
            )
        except Exception as exc:
            self._status.showMessage(f"Failed to load class list: {exc}")

    # ------------------------------------------------------------------ region editing

    def _on_edit_regions_toggled(self, checked: bool):
        self._video.set_region_edit_mode(checked)
        if checked:
            self._status.showMessage(
                "Region edit mode — drag a rectangle on the video to add a region."
            )
        else:
            self._status.showMessage("Region edit mode off.")

    def _on_region_drawn(self, nx: float, ny: float, nw: float, nh: float):
        """Called when the user finishes drawing a rubber-band rectangle."""
        default_name = f"Region {len(self._regions) + 1}"
        name, ok = QInputDialog.getText(
            self, "Name Region",
            "Enter a name for this region\n(e.g. 'Toy shelf', 'Play mat'):",
            text=default_name,
        )
        if ok and name.strip():
            self._regions.append({
                "name": name.strip(),
                "nx": round(nx, 4),
                "ny": round(ny, 4),
                "nw": round(nw, 4),
                "nh": round(nh, 4),
            })
            self._video.set_regions(self._regions)
            self._update_region_ui()
            self._status.showMessage(
                f"Region '{name.strip()}' added  ({len(self._regions)} total)."
            )

    def _clear_regions(self):
        self._regions = []
        self._video.clear_regions()
        self._update_region_ui()
        self._status.showMessage("All regions cleared.")

    def _save_regions_dialog(self):
        default_dir = str(Path(self._video_path).parent) if self._video_path else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Region Config", default_dir,
            "Region config (*.regions.json);;All files (*)"
        )
        if not path:
            return
        if not path.endswith(".regions.json") and not path.endswith(".json"):
            path += ".regions.json"

        config_name, ok = QInputDialog.getText(
            self, "Config Name",
            "Enter a name for this region config\n(e.g. 'Room A – Camera 1'):",
            text=Path(path).stem.replace(".regions", ""),
        )
        if not ok or not config_name.strip():
            return

        try:
            from backend.region_config import save_region_config
            save_region_config(path, config_name.strip(), self._regions)
            self._status.showMessage(
                f"Regions saved → {Path(path).name}  ({len(self._regions)} regions)"
            )
        except Exception as exc:
            self._status.showMessage(f"Failed to save regions: {exc}")

    def _load_regions_dialog(self):
        default_dir = str(Path(self._video_path).parent) if self._video_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Region Config", default_dir,
            "Region config (*.regions.json *.json);;All files (*)"
        )
        if not path:
            return
        try:
            from backend.region_config import load_region_config
            config_name, regions = load_region_config(path)
            self._regions = regions
            self._video.set_regions(self._regions)
            self._update_region_ui()
            self._status.showMessage(
                f"Regions loaded: '{config_name}'  "
                f"({len(regions)} region{'s' if len(regions) != 1 else ''})"
            )
        except Exception as exc:
            self._status.showMessage(f"Failed to load regions: {exc}")

    def _update_region_ui(self):
        n = len(self._regions)
        self._region_count_lbl.setText(f"{n} region{'s' if n != 1 else ''}")
        has = n > 0
        self._clear_regions_btn.setEnabled(has)
        self._save_regions_btn.setEnabled(has)

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
                action_clips_from_json,
            )
            data = load_results(json_path)

            fps = data["video_info"]["fps"]
            fw  = data["video_info"]["frame_width"]
            fh  = data["video_info"]["frame_height"]

            raw_frames = raw_frames_to_frame_detections(
                data["frame_detections"]["frames"], fps
            )
            action_clips = action_clips_from_json(
                data.get("action_recognition", {})
            )

            self._raw_frames         = raw_frames
            self._action_clips       = action_clips
            self._analysis_fw        = fw
            self._analysis_fh        = fh
            self._capture_conf_floor = float(
                data["frame_detections"].get("capture_conf_floor", 0.05)
            )

            self._mock_toggle.setChecked(False)

            date_str = data["processing"].get("date", "")[:10]
            prefix   = "Auto-loaded" if auto else "Loaded"
            self._source_label.setText(
                f"{prefix} JSON (captured {date_str})  ·  "
                f"floor {self._capture_conf_floor:.2f}  ·  adjusting thresholds below…"
            )
            self._status.showMessage(
                f"{prefix}: {Path(json_path).name}  —  refiltering…"
            )
            self._refilter()

        except Exception as exc:
            self._status.showMessage(f"Failed to load JSON: {exc}")
            print(f"[LoadJSON] Error: {exc}")

    # ------------------------------------------------------------------ live refilter

    def _schedule_refilter(self):
        self._refilter_timer.start()

    def _refilter(self):
        """
        Apply current thresholds to all three result tabs.
        Safe to call when _raw_frames is None (MMAction2-only mode).
        """
        mm_conf        = self._mmaction2_conf_spin.value()
        min_dur        = int(self._min_dur_spin.value() * 1000)
        filtered_clips = [c for c in self._action_clips
                          if c.confidence >= mm_conf]

        # ── Tab 2: MMAction2 actions (always available) ──
        self._action_table.set_rows([
            (_fmt_ms(c.start_ms),
             _fmt_ms(c.end_ms),
             _fmt_dur(c.end_ms - c.start_ms),
             c.action_label,
             f"{c.confidence:.3f}",
             c.model_name)
            for c in filtered_clips
        ])
        self._action_time_ranges = [
            (int(c.start_ms), int(c.end_ms)) for c in filtered_clips
        ]

        # ── Video overlay: action-label banner ──
        self._video.set_action_clips([
            {"start_ms":     c.start_ms,
             "end_ms":       c.end_ms,
             "action_label": c.action_label,
             "confidence":   c.confidence}
            for c in filtered_clips
        ])

        if not self._raw_frames:
            n_clips = len(filtered_clips)
            self._source_label.setText(
                f"Action conf ≥ {mm_conf:.2f}  —  "
                f"{n_clips} action clip{'s' if n_clips != 1 else ''}"
                + ("  (no YOLO data loaded)" if not self._action_clips else "")
            )
            # Clear YOLO-dependent tabs so stale data isn't shown
            self._results.set_results([])
            self._region_table.set_rows([])
            return

        from backend.interaction_mapper import InteractionMapper
        from backend.segment_builder   import SegmentBuilder

        conf = self._conf_spin.value()
        prox = self._prox_spin.value()
        fw   = self._analysis_fw
        fh   = self._analysis_fh

        if fw == 0 or fh == 0:
            fw = self._video.frame_width
            fh = self._video.frame_height
            if fw == 0 or fh == 0:
                return
            self._analysis_fw = fw
            self._analysis_fh = fh

        filtered = _filter_frames(self._raw_frames, conf)

        # ── Tab 0: Object Interactions ──
        mapped = InteractionMapper(proximity_px=prox).map(filtered)
        segs, ui_boxes = SegmentBuilder(fw, fh, min_duration_ms=min_dur).build(
            mapped, action_clips=filtered_clips
        )
        self._video.set_frame_detections(_convert_boxes(ui_boxes))

        table_rows = [s.as_table_row() for s in segs]
        self._results.set_results(table_rows)
        self._result_time_ranges = [
            (_hms_to_ms(r[0]), _hms_to_ms(r[1])) for r in table_rows
        ]

        # ── Tab 1: Region Presence ──
        region_segs = _compute_region_segments(
            self._raw_frames, conf, self._regions, fw, fh, min_dur
        )
        self._region_table.set_rows([
            (_fmt_ms(s["start_ms"]),
             _fmt_ms(s["end_ms"]),
             _fmt_dur(s["end_ms"] - s["start_ms"]),
             s["region"],
             str(s["frames"]))
            for s in region_segs
        ])
        self._region_time_ranges = [
            (int(s["start_ms"]), int(s["end_ms"])) for s in region_segs
        ]

        # ── Update tab labels with row counts ──
        n      = len(segs)
        n_reg  = len(region_segs)
        n_clip = len(filtered_clips)
        self._tabs.setTabText(0, f"Object Interactions ({n})")
        self._tabs.setTabText(1, f"Regions ({n_reg})")
        self._tabs.setTabText(2, f"Actions ({n_clip})")

        floor     = self._capture_conf_floor
        floor_warn = (
            f"  ⚠ YOLO conf below capture floor ({floor:.2f})"
            if conf < floor else ""
        )
        self._source_label.setText(
            f"YOLO conf ≥ {conf:.2f}  ·  Action conf ≥ {mm_conf:.2f}  ·  "
            f"prox ≤ {prox} px  ·  min {self._min_dur_spin.value():.1f} s  —  "
            f"{n} interaction{'s' if n != 1 else ''}  ·  "
            f"{n_reg} region segment{'s' if n_reg != 1 else ''}  ·  "
            f"{n_clip} action{'s' if n_clip != 1 else ''}{floor_warn}"
        )

        if n == 0 and not n_reg and not n_clip:
            n_above = sum(
                1 for fd in self._raw_frames
                for d in fd.detections
                if d.confidence >= conf and d.label != "person"
            )
            self._status.showMessage(
                f"⚠ No results at current thresholds  "
                f"({n_above} non-person detections ≥ {conf:.2f}).  "
                f"Try lowering Conf or increasing Proximity."
            )
        else:
            self._status.showMessage(
                f"{n} interaction{'s' if n != 1 else ''}  ·  "
                f"{n_reg} region segment{'s' if n_reg != 1 else ''}  ·  "
                f"{n_clip} action{'s' if n_clip != 1 else ''}  "
                f"(YOLO ≥ {conf:.2f}, action ≥ {mm_conf:.2f}, "
                f"prox ≤ {prox} px, min {self._min_dur_spin.value():.1f} s)"
            )

    # ------------------------------------------------------------------ pipeline launchers

    def _run_yolo(self):
        self._launch_worker(_YoloWorker)

    def _run_full(self):
        self._launch_worker(_FullAnalysisWorker)

    def _run_mmaction2(self):
        """Launch MMAction2 only, without re-running YOLO."""
        if not self._video_path:
            self._status.showMessage("No video loaded.")
            return
        if self._worker_thread and self._worker_thread.isRunning():
            return

        self._set_running(True)
        self._worker = _MMAction2Worker(
            self._video_path,
            enabled_models=self._get_enabled_mmaction2_models(),
        )
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.status.connect(self._on_worker_status)
        self._worker.finished.connect(self._on_mmaction2_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(lambda *_: self._worker_thread.quit())
        self._worker.error.connect(   lambda *_: self._worker_thread.quit())

        self._worker_thread.start()

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
        yolo_model   = self._get_yolo_model_name()
        is_world     = "world" in yolo_model.lower()
        world_cls    = self._world_classes if is_world else None
        mm_models    = self._get_enabled_mmaction2_models()

        self._set_running(True)
        self._worker = worker_cls(
            self._video_path, fw, fh,
            frame_stride=stride,
            detect_all_classes=all_cls,
            display_conf=display_conf,
            fps=self._video.fps,
            total_frames=self._video.total_frames,
            yolo_model=yolo_model,
            world_classes=world_cls,
            enabled_mm_models=mm_models,
        )
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.status.connect(self._on_worker_status)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(lambda *_: self._worker_thread.quit())
        self._worker.error.connect(   lambda *_: self._worker_thread.quit())

        self._worker_thread.start()

    # ------------------------------------------------------------------ worker slots

    def _on_worker_status(self, msg: str):
        self._phase_label.setText(msg)
        self._status.showMessage(msg)

    def _on_finished(self, raw_frames, action_clips, diagnostic: str):
        self._set_running(False)
        self._mock_toggle.setChecked(False)

        self._raw_frames         = raw_frames
        self._action_clips       = action_clips
        self._analysis_fw        = self._video.frame_width
        self._analysis_fh        = self._video.frame_height
        self._capture_conf_floor = 0.05

        from backend.results_format import default_output_path
        json_path = default_output_path(self._video_path)
        self._source_label.setText(
            f"Run complete  ·  auto-saved → {json_path.name}  ·  refiltering…"
        )
        self._status.showMessage(f"Analysis complete — {diagnostic}")
        self._refilter()

    def _on_mmaction2_finished(self, action_clips, diagnostic: str):
        """Slot for standalone MMAction2 run (no YOLO data involved)."""
        self._set_running(False)
        self._action_clips = action_clips

        n = len(action_clips)
        self._source_label.setText(
            f"MMAction2 complete  ·  {n} clip{'s' if n != 1 else ''} captured  ·  "
            f"use 'Action conf ≥' slider to filter"
        )
        self._status.showMessage(f"MMAction2 complete — {diagnostic}")
        # refilter updates the banner and source label (YOLO part skipped if no raw_frames)
        self._refilter()

    def _on_error(self, message: str):
        self._set_running(False)
        self._status.showMessage(f"Error: {message}")
        print(f"[Worker] Error: {message}")

    def _set_running(self, running: bool):
        self._yolo_btn.setEnabled(not running and self._video_path is not None)
        self._full_btn.setEnabled(not running and self._video_path is not None)
        self._mmaction2_btn.setEnabled(not running and self._video_path is not None)
        self._load_json_btn.setEnabled(not running)
        self._mock_toggle.setEnabled(not running)
        self._sample_spin.setEnabled(not running)
        self._all_classes_chk.setEnabled(not running)
        self._progress.setValue(0)
        self._progress.setVisible(running)
        self._phase_label.setVisible(running)
        if not running:
            self._phase_label.setText("")

    # ------------------------------------------------------------------ seek sync

    def _on_position_changed(self, position_ms: int):
        if self._result_time_ranges:
            self._results.highlight_row_at(position_ms, self._result_time_ranges)
        if self._region_time_ranges:
            self._region_table.highlight_row_at(position_ms, self._region_time_ranges)
        if self._action_time_ranges:
            self._action_table.highlight_row_at(position_ms, self._action_time_ranges)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_frames(raw_frames: list, conf_threshold: float) -> list:
    """Return FrameDetections list keeping only detections ≥ conf_threshold."""
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
    action_clips=None,
) -> "Path | None":
    """Save _results.json next to the video.  Never raises — logs on failure."""
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
            action_clips=action_clips or [],
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


def _fmt_ms(ms: float) -> str:
    """Format milliseconds as H:MM:SS.mmm for display in tables."""
    total_s = ms / 1000.0
    h = int(total_s) // 3600
    m = (int(total_s) % 3600) // 60
    s = total_s % 60
    return f"{h}:{m:02d}:{s:06.3f}"


def _fmt_dur(dur_ms: float) -> str:
    """Format a duration in ms as a readable string (e.g. '4.2 s')."""
    s = dur_ms / 1000.0
    if s < 60:
        return f"{s:.1f} s"
    m = int(s) // 60
    s = s % 60
    return f"{m}m {s:.0f}s"


def _compute_region_segments(
    raw_frames: list,
    conf: float,
    regions: list[dict],
    fw: int,
    fh: int,
    min_dur_ms: int,
    gap_tolerance_ms: int = 1000,
) -> list[dict]:
    """
    For each named region, find continuous time spans where a person
    (YOLO conf ≥ conf) was detected inside that region's bounding box.

    Person overlap uses the same IoU-free test as InteractionMapper:
    the person bbox (absolute pixels) vs. the region bbox (denormalised).
    Two frames are considered continuous if the time gap between them is
    ≤ gap_tolerance_ms (default 1 s — covers up to ~30-frame strides).

    Returns a list of dicts sorted by start_ms:
        {"start_ms", "end_ms", "region", "frames"}
    """
    if not regions or not raw_frames:
        return []

    segments: list[dict] = []

    for region in regions:
        rx1 = region["nx"] * fw
        ry1 = region["ny"] * fh
        rx2 = rx1 + region["nw"] * fw
        ry2 = ry1 + region["nh"] * fh
        rname = region["name"]

        cur_start   = None
        cur_end     = None
        cur_frames  = 0

        for fd in raw_frames:
            persons = [
                d for d in fd.detections
                if d.label == "person" and d.confidence >= conf
            ]
            in_region = any(
                d.x1 < rx2 and d.x2 > rx1 and d.y1 < ry2 and d.y2 > ry1
                for d in persons
            )

            if in_region:
                if cur_start is None:
                    cur_start  = fd.timestamp_ms
                    cur_frames = 0
                cur_end     = fd.timestamp_ms
                cur_frames += 1
            else:
                # Close segment if person left AND gap exceeded tolerance
                if cur_start is not None:
                    gap = fd.timestamp_ms - cur_end
                    if gap > gap_tolerance_ms:
                        dur = cur_end - cur_start
                        if dur >= min_dur_ms:
                            segments.append({
                                "start_ms": cur_start,
                                "end_ms":   cur_end,
                                "region":   rname,
                                "frames":   cur_frames,
                            })
                        cur_start  = None
                        cur_end    = None
                        cur_frames = 0

        # Close any segment still open at end of video
        if cur_start is not None:
            dur = cur_end - cur_start
            if dur >= min_dur_ms:
                segments.append({
                    "start_ms": cur_start,
                    "end_ms":   cur_end,
                    "region":   rname,
                    "frames":   cur_frames,
                })

    return sorted(segments, key=lambda s: s["start_ms"])
