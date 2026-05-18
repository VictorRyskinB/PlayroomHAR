# ui/main_window.py
# MainWindow: top-level window.
# Owns two pipeline workers:
#   _YoloWorker       — YOLO only (fast, for testing/iteration)
#   _FullAnalysisWorker — YOLO + MMAction2 (production)

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSplitter, QStatusBar,
    QProgressBar, QCheckBox, QFrame,
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
    finished = pyqtSignal(dict, list)
    error    = pyqtSignal(str)

    def __init__(self, video_path: str, frame_w: int, frame_h: int):
        super().__init__()
        self._path = video_path
        self._fw   = frame_w
        self._fh   = frame_h

    def run(self):
        try:
            from backend.yolo_detector     import YoloDetector
            from backend.interaction_mapper import InteractionMapper
            from backend.segment_builder   import SegmentBuilder

            self.status.emit("Running YOLO detection…")
            detector = YoloDetector()
            frames   = detector.run(self._path,
                                    progress_cb=lambda p: self.progress.emit(p))

            self.status.emit("Mapping interactions…")
            mapped = InteractionMapper().map(frames)

            self.status.emit("Building segments…")
            segs, ui_boxes = SegmentBuilder(self._fw, self._fh).build(mapped)

            self.progress.emit(100)
            self.finished.emit(_convert_boxes(ui_boxes),
                               [s.as_table_row() for s in segs])
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Worker: YOLO + MMAction2
# ---------------------------------------------------------------------------

class _FullAnalysisWorker(QObject):
    progress = pyqtSignal(int)
    status   = pyqtSignal(str)
    finished = pyqtSignal(dict, list)
    error    = pyqtSignal(str)

    def __init__(self, video_path: str, frame_w: int, frame_h: int):
        super().__init__()
        self._path = video_path
        self._fw   = frame_w
        self._fh   = frame_h

    def run(self):
        try:
            from backend.yolo_detector      import YoloDetector
            from backend.interaction_mapper  import InteractionMapper
            from backend.segment_builder    import SegmentBuilder
            from backend.action_recognizer  import ActionRecognizer

            # Phase 1: YOLO (0 → 50%)
            self.status.emit("Phase 1/2 — YOLO detection…")
            detector = YoloDetector()
            frames   = detector.run(
                self._path,
                progress_cb=lambda p: self.progress.emit(p // 2),
            )

            mapped   = InteractionMapper().map(frames)

            # Phase 2: MMAction2 (50 → 90%)
            recognizer = ActionRecognizer()
            action_clips = []
            if recognizer.is_available():
                self.status.emit("Phase 2/2 — Action recognition (MMAction2)…")
                action_clips = recognizer.recognize(
                    self._path,
                    progress_cb=lambda p: self.progress.emit(50 + p * 40 // 100),
                )
            else:
                self.status.emit(
                    "Phase 2/2 — MMAction2 unavailable, using YOLO labels…"
                )
                print(f"[FullAnalysis] {recognizer.unavailable_reason()}")

            # Phase 3: build segments + overlay action labels (90 → 100%)
            self.status.emit("Merging results…")
            self.progress.emit(90)
            segs, ui_boxes = SegmentBuilder(self._fw, self._fh).build(
                mapped, action_clips=action_clips
            )

            self.progress.emit(100)
            self.finished.emit(_convert_boxes(ui_boxes),
                               [s.as_table_row() for s in segs])
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Playroom Action Annotator")
        self.resize(1300, 780)
        self._video_path: str | None = None
        self._worker_thread: QThread | None = None
        self._result_time_ranges: list = []
        self._build_ui()
        self._show_mmaction2_banner()
        self._inject_mock_data()

    # ------------------------------------------------------------------ layout

    def _build_ui(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #12122a; color: #dde; }
            QPushButton {
                background: #2e2e5e; color: #dde; border: 1px solid #4e4e8e;
                border-radius: 4px; padding: 5px 12px;
            }
            QPushButton:hover    { background: #3e3e7e; }
            QPushButton:pressed  { background: #1e1e4e; }
            QPushButton:disabled { color: #555; border-color: #333; }
            QCheckBox { spacing: 6px; }
            QCheckBox::indicator {
                width: 14px; height: 14px; border: 1px solid #4e4e8e;
                border-radius: 3px; background: #1e1e3e;
            }
            QCheckBox::indicator:checked { background: #5555cc; }
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
        """)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 4)
        root.setSpacing(4)

        # ---- MMAction2 availability banner (hidden when available) ----
        self._banner = QFrame()
        self._banner.setObjectName("banner")
        banner_layout = QHBoxLayout(self._banner)
        banner_layout.setContentsMargins(8, 4, 8, 4)
        self._banner_label = QLabel()
        self._banner_label.setStyleSheet("color:#ffbb55; font-size:11px;")
        self._banner_label.setWordWrap(True)
        banner_layout.addWidget(self._banner_label)
        root.addWidget(self._banner)

        # ---- toolbar ----
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        app_title = QLabel("Playroom Action Annotator")
        app_title.setStyleSheet("font-size:15px; font-weight:bold; color:#aad;")

        self._mock_toggle = QCheckBox("Mock data")
        self._mock_toggle.setChecked(True)
        self._mock_toggle.stateChanged.connect(self._on_mock_toggled)

        self._debug_toggle = QCheckBox("Debug columns")
        self._debug_toggle.setChecked(False)
        self._debug_toggle.stateChanged.connect(self._on_debug_toggled)

        self._yolo_btn = QPushButton("Run YOLO")
        self._yolo_btn.setEnabled(False)
        self._yolo_btn.setToolTip("YOLO detection only (fast, for iteration)")
        self._yolo_btn.clicked.connect(self._run_yolo)

        self._full_btn = QPushButton("Run Full Analysis")
        self._full_btn.setEnabled(False)
        self._full_btn.setToolTip("YOLO + MMAction2 action recognition")
        self._full_btn.clicked.connect(self._run_full)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setFixedWidth(180)
        self._progress.setVisible(False)

        self._phase_label = QLabel("")
        self._phase_label.setStyleSheet("color:#aab; font-size:10px;")
        self._phase_label.setVisible(False)

        toolbar.addWidget(app_title)
        toolbar.addStretch()
        toolbar.addWidget(self._mock_toggle)
        toolbar.addWidget(self._debug_toggle)
        toolbar.addWidget(self._yolo_btn)
        toolbar.addWidget(self._full_btn)
        toolbar.addWidget(self._phase_label)
        toolbar.addWidget(self._progress)
        root.addLayout(toolbar)

        # ---- content splitter ----
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
            reason = ActionRecognizer.unavailable_reason()
            # Show just the first sentence (the "what") in the banner
            short = reason.split(".")[0] + "."
            self._banner_label.setText(
                f"MMAction2 unavailable — {short}  "
                f"'Run Full Analysis' will use YOLO spatial labels only."
            )

    # ------------------------------------------------------------------ video loaded

    def _on_video_loaded(self, path: str):
        self._video_path = path
        self._yolo_btn.setEnabled(True)
        self._full_btn.setEnabled(True)
        self._status.showMessage(f"Loaded: {path}")
        if self._mock_toggle.isChecked():
            self._inject_mock_data()

    # ------------------------------------------------------------------ toggles

    def _on_mock_toggled(self, _state: int):
        if self._mock_toggle.isChecked():
            self._inject_mock_data()
            self._status.showMessage("Mock data mode active.")
        else:
            self._video.clear_detections()
            self._results.set_results([])
            self._result_time_ranges = []
            self._status.showMessage(
                "Mock data cleared — run YOLO or Full Analysis."
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

        self._set_running(True)
        self._worker = worker_cls(self._video_path, fw, fh)
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.status.connect(self._on_worker_status)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)

        self._worker_thread.start()

    # ------------------------------------------------------------------ worker slots

    def _on_worker_status(self, msg: str):
        self._phase_label.setText(msg)
        self._status.showMessage(msg)

    def _on_finished(self, box_dict: dict, table_rows: list):
        self._set_running(False)
        self._mock_toggle.setChecked(False)

        self._video.set_frame_detections(box_dict)
        self._results.set_results(table_rows)
        self._result_time_ranges = [
            (_hms_to_ms(r[0]), _hms_to_ms(r[1])) for r in table_rows
        ]

        n = len(table_rows)
        self._status.showMessage(
            f"Complete — {n} interaction segment{'s' if n != 1 else ''} found."
        )

    def _on_error(self, message: str):
        self._set_running(False)
        self._status.showMessage(f"Error: {message}")
        print(f"[Worker] Error: {message}")

    def _set_running(self, running: bool):
        self._yolo_btn.setEnabled(not running)
        self._full_btn.setEnabled(not running)
        self._mock_toggle.setEnabled(not running)
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

def _convert_boxes(ui_boxes_by_frame: dict) -> dict:
    """Convert {int: [UiBox]} → {int: [(label, color, nx, ny, nw, nh)]}."""
    return {
        fi: [(b.label, b.color, b.nx, b.ny, b.nw, b.nh) for b in box_list]
        for fi, box_list in ui_boxes_by_frame.items()
    }


def _hms_to_ms(hms: str) -> int:
    parts = hms.split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return int((h * 3600 + m * 60 + s) * 1000)
