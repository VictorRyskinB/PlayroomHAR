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

from PyQt6.QtCore import Qt, QThread, QTimer, QSettings, pyqtSignal, QObject
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSplitter, QStatusBar,
    QProgressBar, QCheckBox, QFrame, QSpinBox, QDoubleSpinBox,
    QFileDialog, QInputDialog, QTabWidget, QComboBox, QLineEdit, QMenu,
    QDialog, QDialogButtonBox, QFormLayout, QMessageBox,
    QListWidget, QListWidgetItem, QScrollArea,
    QButtonGroup, QRadioButton,
    QStackedWidget, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QGridLayout, QSizePolicy,
)

from ui.video_player import VideoPlayerWidget
from ui.results_table import ResultsTableWidget, GenericTableWidget
from mock_data import MOCK_BOUNDING_BOXES, MOCK_RESULTS


# ---------------------------------------------------------------------------
# Themes and parametrized stylesheet
# ---------------------------------------------------------------------------

_THEMES: dict[str, dict[str, str]] = {
    "Dark Blue": {
        "bg":        "#12122a",
        "bg_alt":    "#1e1e3e",
        "bg_w":      "#1a1a30",
        "bg_btn":    "#2e2e5e",
        "bg_hover":  "#3e3e7e",
        "bg_press":  "#1e1e4e",
        "bg_table":  "#1a1a30",
        "bg_pb":     "#1a1a3a",
        "text":      "#ddeeff",
        "text2":     "#aabbcc",
        "text_dis":  "#555577",
        "border":    "#4e4e8e",
        "border2":   "#2a2a4a",
        "border3":   "#1a3a1a",
        "accent":    "#5555cc",
        "accent2":   "#7070cc",
        "sel":       "#2e2e5e",
        "parambar":  "#0e0e22",
        "regionbar": "#0a1a0a",
    },
    "Dark Neutral": {
        "bg":        "#1a1a1a",
        "bg_alt":    "#242424",
        "bg_w":      "#1e1e1e",
        "bg_btn":    "#333333",
        "bg_hover":  "#404040",
        "bg_press":  "#222222",
        "bg_table":  "#1c1c1c",
        "bg_pb":     "#1e1e1e",
        "text":      "#dddddd",
        "text2":     "#999999",
        "text_dis":  "#555555",
        "border":    "#555555",
        "border2":   "#333333",
        "border3":   "#2a3a2a",
        "accent":    "#7777aa",
        "accent2":   "#9999bb",
        "sel":       "#3a3a3a",
        "parambar":  "#111111",
        "regionbar": "#0d1a0d",
    },
    "Warm Dark": {
        "bg":        "#1a1410",
        "bg_alt":    "#261e16",
        "bg_w":      "#1e1810",
        "bg_btn":    "#3a2e1e",
        "bg_hover":  "#4a3e2e",
        "bg_press":  "#2a1e10",
        "bg_table":  "#1c1810",
        "bg_pb":     "#1e1a10",
        "text":      "#eeddc8",
        "text2":     "#bb9977",
        "text_dis":  "#554433",
        "border":    "#7a5a3a",
        "border2":   "#3a2a1a",
        "border3":   "#1a2a1a",
        "accent":    "#aa7733",
        "accent2":   "#cc9955",
        "sel":       "#3a2e1e",
        "parambar":  "#100e08",
        "regionbar": "#0a1408",
    },
}


def _make_qss(font_pt: int, theme_name: str) -> str:
    """Build the complete application stylesheet parametrized by font size and theme."""
    t = _THEMES.get(theme_name, _THEMES["Dark Blue"])
    return f"""
        QMainWindow, QWidget {{
            background: {t['bg']}; color: {t['text']};
            font-size: {font_pt}pt;
        }}
        QPushButton {{
            background: {t['bg_btn']}; color: {t['text']};
            border: 1px solid {t['border']};
            border-radius: 4px; padding: 5px 10px;
            font-size: {font_pt}pt;
        }}
        QPushButton:hover    {{ background: {t['bg_hover']}; }}
        QPushButton:pressed  {{ background: {t['bg_press']}; }}
        QPushButton:disabled {{ color: {t['text_dis']}; border-color: {t['border2']}; }}
        QPushButton#editRegionsBtn:checked {{
            background: #5e3e1e; border-color: #cc8833; color: #ffcc66;
        }}
        QCheckBox {{ spacing: 5px; font-size: {font_pt}pt; }}
        QCheckBox::indicator {{
            width: 14px; height: 14px; border: 1px solid {t['border']};
            border-radius: 3px; background: {t['bg_alt']};
        }}
        QCheckBox::indicator:checked {{ background: {t['accent']}; }}
        QSpinBox, QDoubleSpinBox {{
            background: {t['bg_alt']}; color: {t['text']};
            border: 1px solid {t['border']};
            border-radius: 4px; padding: 2px 4px;
            font-size: {font_pt}pt;
        }}
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            background: {t['bg_btn']}; border: none; width: 16px;
        }}
        QComboBox {{
            background: {t['bg_alt']}; color: {t['text']};
            border: 1px solid {t['border']};
            border-radius: 4px; padding: 2px 4px;
            font-size: {font_pt}pt;
        }}
        QComboBox QAbstractItemView {{
            background: {t['bg_alt']}; color: {t['text']};
            border: 1px solid {t['border']};
            selection-background-color: {t['sel']};
        }}
        QLineEdit {{
            background: {t['bg_alt']}; color: {t['text']};
            border: 1px solid {t['border']};
            border-radius: 4px; padding: 2px 4px;
            font-size: {font_pt}pt;
        }}
        QLabel {{ font-size: {font_pt}pt; }}
        QSlider::groove:horizontal {{
            height: 4px; background: {t['bg_btn']}; border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: {t['accent2']}; border-radius: 6px;
            width: 12px; height: 12px; margin: -4px 0;
        }}
        QProgressBar {{
            border: 1px solid {t['border']}; border-radius: 4px;
            background: {t['bg_pb']}; color: {t['text']}; text-align: center;
            font-size: {font_pt}pt; max-height: 16px;
        }}
        QProgressBar::chunk {{ background: {t['accent']}; border-radius: 3px; }}
        QHeaderView::section {{
            background: {t['bg_alt']}; color: {t['text2']}; border: none;
            padding: 4px; font-size: {font_pt}pt;
        }}
        QTableView {{
            background: {t['bg_table']};
            alternate-background-color: {t['bg_alt']};
            color: {t['text']};
        }}
        QTabWidget::pane {{
            border: 1px solid {t['border2']}; background: {t['bg']};
        }}
        QTabBar::tab {{
            background: {t['bg_alt']}; color: {t['text2']};
            border: 1px solid {t['border2']}; border-bottom: none;
            padding: 5px 12px; font-size: {font_pt}pt;
        }}
        QTabBar::tab:selected {{ background: {t['sel']}; color: {t['text']}; }}
        QTabBar::tab:hover    {{ background: {t['bg_hover']}; }}
        QRadioButton {{ font-size: {font_pt}pt; spacing: 5px; }}
        QScrollBar:vertical {{
            background: {t['bg']}; width: 10px; border: none;
        }}
        QScrollBar::handle:vertical {{
            background: {t['bg_btn']}; border-radius: 4px; min-height: 20px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QFrame#banner {{
            background: #2a1a00; border: 1px solid #664400;
            border-radius: 4px; padding: 2px;
        }}
        QFrame#parambar {{
            background: {t['parambar']};
            border-top: 1px solid {t['border2']};
            border-bottom: 1px solid {t['border2']};
        }}
        QFrame#regionbar {{
            background: {t['regionbar']};
            border-top: 1px solid {t['border3']};
            border-bottom: 1px solid {t['border3']};
        }}
    """


# ---------------------------------------------------------------------------
# Config dialog  (font size, color theme, recognition window)
# ---------------------------------------------------------------------------

class _ConfigDialog(QDialog):
    """App-wide appearance and MMAction2 recognition window settings."""

    def __init__(
        self,
        current_size: int,
        current_theme: str,
        current_clip_len: int,
        current_clip_stride: int,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(360)

        vbox = QVBoxLayout(self)
        vbox.setSpacing(14)

        # ── Font size ──
        size_lbl = QLabel("Font size")
        size_lbl.setStyleSheet("font-weight: bold;")
        vbox.addWidget(size_lbl)

        self._size_group = QButtonGroup(self)
        size_row = QHBoxLayout()
        for label, pt in [("Small", 10), ("Medium", 12), ("Large", 14)]:
            rb = QRadioButton(f"{label}  ({pt} pt)")
            rb.setProperty("pt", pt)
            self._size_group.addButton(rb)
            size_row.addWidget(rb)
            if pt == current_size:
                rb.setChecked(True)
        vbox.addLayout(size_row)

        # ── Color theme ──
        theme_lbl = QLabel("Color theme")
        theme_lbl.setStyleSheet("font-weight: bold;")
        vbox.addWidget(theme_lbl)

        self._theme_group = QButtonGroup(self)
        theme_row = QHBoxLayout()
        for name in _THEMES:
            rb = QRadioButton(name)
            self._theme_group.addButton(rb)
            theme_row.addWidget(rb)
            if name == current_theme:
                rb.setChecked(True)
        vbox.addLayout(theme_row)

        # ── Recognition window ──
        win_lbl = QLabel("Recognition window  (MMAction2)")
        win_lbl.setStyleSheet("font-weight: bold;")
        vbox.addWidget(win_lbl)

        hint = QLabel(
            "The model reads a sliding window of frames to predict each action.\n"
            "Window: frames per prediction.  Step: how far to advance each time.\n"
            "Example — window 32 / step 16 at 30 fps:\n"
            "  → a prediction every 0.5 s, each covering ~1.1 s of video.\n"
            "Smaller step = more predictions but slower run.\n"
            "Smaller window = faster but may miss slow or sustained actions."
        )
        hint.setStyleSheet("color: #8899aa; font-size: 10pt;")
        hint.setWordWrap(True)
        vbox.addWidget(hint)

        form = QFormLayout()
        form.setSpacing(8)

        self._clip_len_spin = QSpinBox()
        self._clip_len_spin.setRange(8, 128)
        self._clip_len_spin.setSingleStep(8)
        self._clip_len_spin.setValue(current_clip_len)
        self._clip_len_spin.setToolTip(
            "How many video frames the model analyses per prediction.\n"
            "Larger = more temporal context, slower inference."
        )
        form.addRow("Window (frames):", self._clip_len_spin)

        self._clip_stride_spin = QSpinBox()
        self._clip_stride_spin.setRange(1, 64)
        self._clip_stride_spin.setSingleStep(4)
        self._clip_stride_spin.setValue(current_clip_stride)
        self._clip_stride_spin.setToolTip(
            "How far to advance the window between predictions.\n"
            "Step = window → no overlap (fastest).\n"
            "Step = window / 2 → 50 % overlap (more thorough)."
        )
        form.addRow("Step (frames):", self._clip_stride_spin)
        vbox.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        vbox.addWidget(buttons)

    def selected_size(self) -> int:
        btn = self._size_group.checkedButton()
        return btn.property("pt") if btn else 12

    def selected_theme(self) -> str:
        btn = self._theme_group.checkedButton()
        return btn.text() if btn else "Dark Blue"

    def clip_len(self) -> int:
        return self._clip_len_spin.value()

    def clip_stride(self) -> int:
        return self._clip_stride_spin.value()


# ---------------------------------------------------------------------------
# Interaction filter dialog  (non-modal; covers subjects AND object labels)
# ---------------------------------------------------------------------------

_LIST_STYLE = (
    "QListWidget { background:#1a1a30; border: 1px solid #2e2e4e; }"
    "QListWidget::item { color:#dde; padding: 3px; }"
    "QListWidget::item:alternate { background:#1e1e38; }"
)

def _make_list_widget() -> "QListWidget":
    w = QListWidget()
    w.setAlternatingRowColors(True)
    w.setStyleSheet(_LIST_STYLE)
    return w

def _populate_list(
    lst: "QListWidget",
    items: list[str],
    excluded: set[str],
) -> None:
    """Fill *lst* with checkable items; items in *excluded* are unchecked."""
    for label in sorted(items):
        item = QListWidgetItem(label)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(
            Qt.CheckState.Unchecked if label in excluded else Qt.CheckState.Checked
        )
        lst.addItem(item)
    rows = min(lst.count(), 12)
    lst.setFixedHeight(max(60, rows * 24 + 8))

def _get_excluded(lst: "QListWidget") -> set[str]:
    return {
        lst.item(i).text()
        for i in range(lst.count())
        if lst.item(i).checkState() == Qt.CheckState.Unchecked
    }

def _set_excluded_silent(lst: "QListWidget", excluded: set[str]) -> None:
    lst.itemChanged.disconnect()   # caller re-connects after
    for i in range(lst.count()):
        item = lst.item(i)
        item.setCheckState(
            Qt.CheckState.Unchecked if item.text() in excluded
            else Qt.CheckState.Checked
        )


class _InteractionFilterDialog(QDialog):
    """
    Non-modal filter window for the Object Interactions tab.

    Top section  — Subject types  (person / hand / foot / …)
                   Only shows subjects actually detected in the current run.
    Bottom section — Object labels (ball / table / crayon / …)
                   One checkbox per unique detected non-subject label.

    Callback on_change(excluded_subjects: set[str], excluded_labels: set[str])
    is called whenever any checkbox changes.
    """

    def __init__(
        self,
        all_subjects: list[str],
        all_labels:   list[str],
        excl_subjects: set[str],
        excl_labels:   set[str],
        on_change,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Interaction Filter")
        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint
        )
        self.setMinimumWidth(300)
        self._on_change = on_change

        vbox = QVBoxLayout(self)
        vbox.setSpacing(8)

        # ── Subject section ──
        subj_lbl = QLabel("Subject types")
        subj_lbl.setStyleSheet("font-size:11px; font-weight:bold; color:#99ccbb;")
        vbox.addWidget(subj_lbl)

        hint_s = QLabel(
            "Person = broad proximity.  Hand/Foot = specific contact.\n"
            "Subject↔Subject interactions (e.g. person↔hand) are never shown."
        )
        hint_s.setStyleSheet("font-size:10px; color:#8899aa;")
        hint_s.setWordWrap(True)
        vbox.addWidget(hint_s)

        self._subj_list = _make_list_widget()
        _populate_list(self._subj_list, all_subjects, excl_subjects)
        self._subj_list.itemChanged.connect(self._changed)
        vbox.addWidget(self._subj_list)

        # ── Divider ──
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #2e2e4e;")
        vbox.addWidget(line)

        # ── Object label section ──
        obj_lbl = QLabel("Object labels")
        obj_lbl.setStyleSheet("font-size:11px; font-weight:bold; color:#99aacc;")
        vbox.addWidget(obj_lbl)

        hint_o = QLabel("Uncheck a label to hide all interactions with that object.")
        hint_o.setStyleSheet("font-size:10px; color:#8899aa;")
        hint_o.setWordWrap(True)
        vbox.addWidget(hint_o)

        self._obj_list = _make_list_widget()
        _populate_list(self._obj_list, all_labels, excl_labels)
        self._obj_list.itemChanged.connect(self._changed)
        vbox.addWidget(self._obj_list)

        btn_row = QHBoxLayout()
        show_btn = QPushButton("Show All")
        show_btn.clicked.connect(self._show_all)
        hide_btn = QPushButton("Hide All Objects")
        hide_btn.clicked.connect(self._hide_all_objects)
        btn_row.addWidget(show_btn)
        btn_row.addWidget(hide_btn)
        vbox.addLayout(btn_row)

        self.adjustSize()

    # ------------------------------------------------------------------ public

    def populate(
        self,
        all_subjects: list[str],
        all_labels:   list[str],
        excl_subjects: set[str],
        excl_labels:   set[str],
    ):
        """Rebuild both lists (call after a new YOLO run)."""
        for lst in (self._subj_list, self._obj_list):
            lst.itemChanged.disconnect(self._changed)
            lst.clear()
        _populate_list(self._subj_list, all_subjects, excl_subjects)
        _populate_list(self._obj_list,  all_labels,   excl_labels)
        for lst in (self._subj_list, self._obj_list):
            lst.itemChanged.connect(self._changed)
        self.adjustSize()

    def set_excluded_labels(self, excluded: set[str]):
        """Push updated label exclusions without triggering the callback."""
        _set_excluded_silent(self._obj_list, excluded)
        self._obj_list.itemChanged.connect(self._changed)

    def get_excluded_subjects(self) -> set[str]:
        return _get_excluded(self._subj_list)

    def get_excluded_labels(self) -> set[str]:
        return _get_excluded(self._obj_list)

    # ------------------------------------------------------------------ private

    def _changed(self, _item=None):
        self._on_change(self.get_excluded_subjects(), self.get_excluded_labels())

    def _show_all(self):
        for lst in (self._subj_list, self._obj_list):
            lst.itemChanged.disconnect(self._changed)
            for i in range(lst.count()):
                lst.item(i).setCheckState(Qt.CheckState.Checked)
            lst.itemChanged.connect(self._changed)
        self._changed()

    def _hide_all_objects(self):
        self._obj_list.itemChanged.disconnect(self._changed)
        for i in range(self._obj_list.count()):
            self._obj_list.item(i).setCheckState(Qt.CheckState.Unchecked)
        self._obj_list.itemChanged.connect(self._changed)
        self._changed()


# ---------------------------------------------------------------------------
# Calibration point dialog  (single popup collects both X and Y)
# ---------------------------------------------------------------------------

class _CalibPointDialog(QDialog):
    """
    Collects the real-world floor coordinates for one calibration click.
    Shows a single window with two fields (X cm, Y cm) so the user is
    never confused by back-to-back identical-looking prompts.
    """

    def __init__(
        self,
        step: int,
        px: float,
        py: float,
        room_size_cm: tuple,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Calibration — Point {step} / 4")
        self.setModal(True)
        self.setMinimumWidth(360)

        vbox = QVBoxLayout(self)

        hint = QLabel(
            f"<b>Point {step} of 4</b> &nbsp;—&nbsp; pixel ({px:.0f}, {py:.0f})<br><br>"
            "Enter the real-world floor position for this point.<br><br>"
            "<b>Coordinate system:</b><br>"
            "&nbsp; Origin (0, 0) = front-left of room (camera side, left)<br>"
            "&nbsp; +X = right across the room width<br>"
            "&nbsp; +Y = toward the far wall<br><br>"
            f"Room: {room_size_cm[0]:.0f} cm wide &times; "
            f"{room_size_cm[1]:.0f} cm deep"
        )
        hint.setStyleSheet("font-size:11px; color:#aabbcc;")
        hint.setWordWrap(True)
        vbox.addWidget(hint)

        form = QFormLayout()
        form.setSpacing(8)
        self._x_edit = QLineEdit()
        self._x_edit.setPlaceholderText("e.g.  0   or   250")
        self._y_edit = QLineEdit()
        self._y_edit.setPlaceholderText("e.g.  0   or   600")
        form.addRow("X  (cm, across width):", self._x_edit)
        form.addRow("Y  (cm, depth from camera):", self._y_edit)
        vbox.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        vbox.addWidget(buttons)

        self._result: tuple[float, float] | None = None
        self._x_edit.setFocus()

    def _on_accept(self):
        try:
            x = float(self._x_edit.text().strip())
            y = float(self._y_edit.text().strip())
            self._result = (x, y)
            self.accept()
        except ValueError:
            self._x_edit.setStyleSheet("border: 1px solid red;")
            self._y_edit.setStyleSheet("border: 1px solid red;")

    def result_values(self) -> tuple[float, float] | None:
        """Returns (x_cm, y_cm) if accepted, else None."""
        return self._result


# ---------------------------------------------------------------------------
# Worker cancellation sentinel
# ---------------------------------------------------------------------------

class _WorkerCancelled(Exception):
    """Raised inside a progress callback to abort a running worker cleanly."""


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
                 enabled_mm_models: list | None = None,
                 yolo_run_settings: dict | None = None):
        super().__init__()
        self._path             = video_path
        self._fw               = frame_w
        self._fh               = frame_h
        self._stride           = frame_stride
        self._all_classes      = detect_all_classes
        self._display_conf     = display_conf
        self._fps              = fps
        self._total_frames     = total_frames
        self._yolo_model       = yolo_model
        self._world_classes    = world_classes
        self._enabled_mm       = enabled_mm_models   # unused in YOLO-only, kept for symmetry
        self._yolo_run_settings = yolo_run_settings
        self._cancelled        = False

    def cancel(self):
        self._cancelled = True

    def _progress_cb(self, pct: int):
        if self._cancelled:
            raise _WorkerCancelled()
        self.progress.emit(pct)

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
                progress_cb=self._progress_cb,
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
                yolo_run_settings=self._yolo_run_settings,
            )

            diag = (
                f"YOLO: {n_frames} frames  |  "
                f"Person: {n_person}  |  "
                f"Objects: {n_objects}  |  "
                f"Interaction frames (@{self._display_conf:.2f}): {n_inter}"
            )
            self.progress.emit(100)
            self.finished.emit(raw_frames, [], diag)

        except _WorkerCancelled:
            self.status.emit("Cancelled.")
            self.finished.emit([], [], "Cancelled")
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
                 enabled_mm_models: list | None = None,
                 yolo_run_settings: dict | None = None):
        super().__init__()
        self._path              = video_path
        self._fw                = frame_w
        self._fh                = frame_h
        self._stride            = frame_stride
        self._all_classes       = detect_all_classes
        self._display_conf      = display_conf
        self._fps               = fps
        self._total_frames      = total_frames
        self._yolo_model        = yolo_model
        self._world_classes     = world_classes
        self._enabled_mm        = enabled_mm_models
        self._yolo_run_settings = yolo_run_settings
        self._cancelled         = False

    def cancel(self):
        self._cancelled = True

    def _progress_yolo(self, pct: int):
        if self._cancelled:
            raise _WorkerCancelled()
        self.progress.emit(pct // 2)

    def _progress_mm(self, pct: int):
        if self._cancelled:
            raise _WorkerCancelled()
        self.progress.emit(50 + pct * 40 // 100)

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
                progress_cb=self._progress_yolo,
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
                    progress_cb=self._progress_mm,
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
                yolo_run_settings=self._yolo_run_settings,
            )

            diag = (
                f"YOLO: {n_frames} frames  |  "
                f"Person: {n_person}  |  "
                f"Objects: {n_objects}  |  "
                f"Interaction frames (@{self._display_conf:.2f}): {n_inter}"
            )
            self.progress.emit(100)
            self.finished.emit(raw_frames, action_clips, diag)

        except _WorkerCancelled:
            self.status.emit("Cancelled.")
            self.finished.emit([], [], "Cancelled")
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
        self._cancelled    = False

    def cancel(self):
        self._cancelled = True

    def _progress_cb(self, pct: int):
        if self._cancelled:
            raise _WorkerCancelled()
        self.progress.emit(pct)

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
                progress_cb=self._progress_cb,
            )
            diag = f"MMAction2: {len(action_clips)} clips captured from {self._path}"
            self.progress.emit(100)
            self.finished.emit(action_clips, diag)

        except _WorkerCancelled:
            self.status.emit("Cancelled.")
            self.finished.emit([], "Cancelled")
        except Exception as exc:
            import traceback; traceback.print_exc()
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Worker: path tracking (ByteTrack person tracker)
# ---------------------------------------------------------------------------

class _PathTrackingWorker(QObject):
    progress = pyqtSignal(int)
    status   = pyqtSignal(str)
    finished = pyqtSignal(object, str)   # (list[PathPoint], diagnostic)
    error    = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        yolo_model: str = "yolov8n.pt",
        homography_matrix = None,        # np.ndarray or None
        room_size_cm: tuple = (250, 600),
    ):
        super().__init__()
        self._path     = video_path
        self._model    = yolo_model
        self._matrix   = homography_matrix
        self._room     = room_size_cm
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _progress_cb(self, pct: int):
        if self._cancelled:
            raise _WorkerCancelled()
        self.progress.emit(pct)

    def run(self):
        try:
            from backend.tracking_detector import TrackingDetector
            from backend.path_analyzer     import build_path_points, compute_stats

            self.status.emit("Path tracking — running ByteTrack…")
            det = TrackingDetector(
                model_name=self._model,
                conf_threshold=0.25,
            )
            track_pts = det.run(
                self._path,
                progress_cb=self._progress_cb,
            )

            path_pts = build_path_points(track_pts, self._matrix)
            stats    = compute_stats(path_pts)

            diag = (
                f"{len(path_pts)} track points  |  "
                + (f"distance ≈ {stats.total_distance_m:.1f} m  |  "
                   f"avg speed ≈ {stats.avg_speed_m_s:.2f} m/s"
                   if stats.calibrated
                   else "calibrate camera for real-world distances")
            )
            self.progress.emit(100)
            self.finished.emit(path_pts, diag)

        except _WorkerCancelled:
            self.status.emit("Cancelled.")
            self.finished.emit([], "Cancelled")
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
        # YOLO-World custom class list
        from backend.yolo_classes_config import DEFAULT_PLAYROOM_CLASSES
        self._world_classes: list[str] = list(DEFAULT_PLAYROOM_CLASSES)

        # Interaction filter (Object Interactions tab)
        self._label_filter_excluded:   set[str] = set()   # excluded object labels
        self._subject_filter_excluded: set[str] = set()   # excluded subject types
        self._label_filter_dlg: "_InteractionFilterDialog | None" = None

        # Subject labels: which detected classes are treated as subjects
        # (always includes "person"; add "hand"/"foot" via YOLO-World bar)
        self._subject_labels: set[str] = {"person"}

        # Path tracking state
        self._path_points: list = []          # list[PathPoint]
        self._path_stats = None               # PathStats | None
        self._path_live_mode: bool = False    # True = path follows video playhead
        self._homography_matrix = None        # np.ndarray 3×3 or None
        self._room_size_cm: tuple = (250, 600)

        # Camera calibration state machine
        # _cal_step = 0 (idle), 1–4 (waiting for click N)
        self._cal_step: int = 0
        self._cal_pixel_pts: list = []        # [(px, py), ...]  in pixel coords
        self._cal_world_pts: list = []        # [(X_cm, Y_cm), ...]
        self._cal_dialog_active: bool = False # re-entry guard
        # Frame dimensions used by refilter
        self._analysis_fw: int = 0
        self._analysis_fh: int = 0
        # Capture floor recorded in the loaded JSON
        self._capture_conf_floor: float = 0.05

        # Seekbar highlight state (which table row is currently highlighted)
        self._hl_tab: int = -1          # 0/1/2 = which tab; -1 = none
        self._hl_src_row: int = -1      # source-model row index
        self._last_position_ms: int = 0 # last known video position (live path)

        # Appearance / recognition-window settings (persisted via QSettings)
        self._font_size: int          = 12
        self._theme_name: str         = "Dark Blue"
        self._clip_len_frames: int    = 32
        self._clip_stride_frames: int = 16

        # Analysis panel state
        self._analysis_smooth_window: int   = 5     # smoothing window (frames)
        self._analysis_stat_thresh:   float = 0.10  # stationary threshold (m/s)
        self._analysis_min_ep_ms:     float = 500.0 # min episode duration (ms)
        self._analysis_episodes:      list  = []    # cached list[MovementEpisode]
        self._analysis_reg_stats:     list  = []    # cached list[RegionStats]
        self._analysis_speed_s:       list  = []    # cached list[SpeedSample]

        # Debounce timer — refilter fires 250 ms after the last slider change
        self._refilter_timer = QTimer(self)
        self._refilter_timer.setSingleShot(True)
        self._refilter_timer.setInterval(250)
        self._refilter_timer.timeout.connect(self._refilter)

        self._build_ui()
        self._show_mmaction2_banner()
        self._load_settings()
        # Re-apply stylesheet so saved font/theme preferences take effect on startup
        self.setStyleSheet(_make_qss(self._font_size, self._theme_name))

    # ------------------------------------------------------------------ layout

    def _build_ui(self):
        self.setStyleSheet(_make_qss(self._font_size, self._theme_name))

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

        self._track_path_btn = QPushButton("Track Path")
        self._track_path_btn.setEnabled(False)
        self._track_path_btn.setToolTip(
            "Run ByteTrack to extract the child's movement path.\n"
            "Calibrate the camera first for real-world floor coordinates."
        )
        self._track_path_btn.clicked.connect(self._run_path_tracking)

        self._calibrate_btn = QPushButton("Calibrate Camera…")
        self._calibrate_btn.setEnabled(False)
        self._calibrate_btn.setToolTip(
            "Click 4 known floor points to set up the perspective transform.\n"
            "Required for the top-down path map and real-world distances."
        )
        self._calibrate_btn.clicked.connect(self._start_calibration)

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
        toolbar.addWidget(self._track_path_btn)
        toolbar.addWidget(self._calibrate_btn)
        toolbar.addWidget(self._load_json_btn)
        self._analysis_btn = QPushButton("📊 Analysis")
        self._analysis_btn.setEnabled(False)
        self._analysis_btn.setToolTip(
            "Open full-window trajectory analysis:\n"
            "Summary stats, speed graph, episodes, region metrics, path map.\n"
            "Run Track Path first to enable."
        )
        self._analysis_btn.clicked.connect(self._show_analysis_panel)
        toolbar.addWidget(self._analysis_btn)
        self._config_btn = QPushButton("Aa")
        self._config_btn.setFixedWidth(34)
        self._config_btn.setToolTip("Font size, color theme, recognition window settings")
        self._config_btn.clicked.connect(self._open_config_dialog)
        toolbar.addWidget(self._config_btn)
        self._cancel_btn = QPushButton("✕  Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.setStyleSheet(
            "QPushButton { color: #ff6666; border-color: #884444; }"
            "QPushButton:hover { background: #4e2222; }"
            "QPushButton:disabled { color: #664444; border-color: #442222; }"
        )
        self._cancel_btn.setToolTip(
            "Cancel the running analysis.\n"
            "The current frame batch will finish before stopping."
        )
        self._cancel_btn.clicked.connect(self._cancel_worker)

        toolbar.addWidget(self._phase_label)
        toolbar.addWidget(self._progress)
        toolbar.addWidget(self._cancel_btn)
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

        self._trail_vis_chk = QCheckBox("Trail")
        self._trail_vis_chk.setChecked(True)
        self._trail_vis_chk.setStyleSheet("font-size:11px;")
        self._trail_vis_chk.setToolTip(
            "Show/hide the movement trail overlay on the video.\n"
            "Trail length: last 5 seconds (blue→red gradient)."
        )
        self._trail_vis_chk.stateChanged.connect(
            lambda s: self._video.set_trail_visible(bool(s))
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
        parambar.addWidget(self._trail_vis_chk)
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
        modelbar.addSpacing(24)

        self._action_merge_chk = QCheckBox("Merge clips")
        self._action_merge_chk.setChecked(False)
        self._action_merge_chk.setStyleSheet("font-size:11px;")
        self._action_merge_chk.setToolTip(
            "Merge consecutive action clips with the same label\n"
            "if the gap between them is within the threshold below.\n"
            "Useful to collapse repeated same-action windows into one segment."
        )
        self._action_merge_chk.stateChanged.connect(self._schedule_refilter)

        self._action_merge_gap_spin = QSpinBox()
        self._action_merge_gap_spin.setRange(0, 5000)
        self._action_merge_gap_spin.setSingleStep(100)
        self._action_merge_gap_spin.setValue(500)
        self._action_merge_gap_spin.setFixedWidth(62)
        self._action_merge_gap_spin.setToolTip(
            "Maximum gap (ms) between two same-label clips\n"
            "for them to be merged into one.  0 = only merge overlapping clips."
        )
        self._action_merge_gap_spin.valueChanged.connect(self._schedule_refilter)

        modelbar.addWidget(self._action_merge_chk)
        modelbar.addWidget(_model_label("gap ≤"))
        modelbar.addWidget(self._action_merge_gap_spin)
        modelbar.addWidget(_model_label("ms"))
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

        subj_lbl = QLabel("Subject classes:")
        subj_lbl.setStyleSheet("font-size:11px; color:#bbcc88;")

        self._subject_classes_edit = QLineEdit()
        self._subject_classes_edit.setPlaceholderText(
            "e.g.  hand, foot  (always includes person)"
        )
        self._subject_classes_edit.setToolTip(
            "Comma-separated YOLO-World labels to treat as SUBJECTS.\n"
            "Subjects interact WITH objects — subject↔subject pairs are ignored.\n"
            "'person' is always a subject.  Add 'hand', 'foot' here if detected.\n"
            "These labels are also automatically added to the detection class list."
        )
        self._subject_classes_edit.setFixedWidth(260)
        self._subject_classes_edit.textChanged.connect(self._on_subject_classes_changed)

        world_layout.addWidget(world_lbl)
        world_layout.addWidget(self._world_classes_edit, stretch=1)
        world_layout.addSpacing(16)
        world_layout.addWidget(subj_lbl)
        world_layout.addWidget(self._subject_classes_edit)
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
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(6)

        self._video = VideoPlayerWidget()
        self._video.position_changed.connect(self._on_position_changed)
        self._video.video_loaded.connect(self._on_video_loaded)
        self._video.region_drawn.connect(self._on_region_drawn)
        self._video.calibration_click.connect(self._on_calibration_click)

        # ── Three-tab results panel ──
        self._tabs = QTabWidget()

        # Tab 0 — YOLO Object Interactions (existing schema)
        self._results = ResultsTableWidget()
        self._results.filter_btn_clicked.connect(self._open_label_filter)
        self._results.hide_label_requested.connect(self._hide_label)
        self._results._table.clicked.connect(
            lambda idx: self._on_table_row_clicked(idx, 0,
                self._results._proxy, self._result_time_ranges)
        )
        self._tabs.addTab(self._results, "Object Interactions")

        # Tab 1 — Region Presence
        self._region_table = GenericTableWidget(
            ["Start", "End", "Duration", "Region", "Frames"],
            stretch_col=3,
        )
        self._region_table._table.clicked.connect(
            lambda idx: self._on_table_row_clicked(idx, 1,
                self._region_table._proxy, self._region_time_ranges)
        )
        self._tabs.addTab(self._region_table, "Regions")

        # Tab 2 — MMAction2 Action Clips
        self._action_table = GenericTableWidget(
            ["Start", "End", "Duration", "Action", "Confidence", "Model"],
            stretch_col=3,
        )
        self._action_table._table.clicked.connect(
            lambda idx: self._on_table_row_clicked(idx, 2,
                self._action_table._proxy, self._action_time_ranges)
        )
        self._tabs.addTab(self._action_table, "Actions (MMAction2)")

        # Tab 3 — Path & Heatmap
        from ui.path_map_widget import PathMapWidget
        _path_tab_container = QWidget()
        _path_tab_layout    = QVBoxLayout(_path_tab_container)
        _path_tab_layout.setContentsMargins(4, 4, 4, 4)
        _path_tab_layout.setSpacing(4)

        # Toggle row for path tab layers
        _path_toggle_row = QHBoxLayout()
        _path_toggle_row.setSpacing(8)

        self._path_show_path_chk = QCheckBox("Path line")
        self._path_show_path_chk.setChecked(True)
        self._path_show_path_chk.setStyleSheet("font-size:11px;")
        self._path_show_path_chk.setToolTip(
            "Show/hide the movement path polyline.\n"
            "Blue = early, Red = late.  Dotted = interpolated."
        )

        self._path_show_heatmap_chk = QCheckBox("Heatmap")
        self._path_show_heatmap_chk.setChecked(True)
        self._path_show_heatmap_chk.setStyleSheet("font-size:11px;")
        self._path_show_heatmap_chk.setToolTip(
            "Show/hide the density heatmap overlay.\n"
            "Blue = low density, Red = high density."
        )

        self._path_cal_status_lbl = QLabel("No calibration")
        self._path_cal_status_lbl.setStyleSheet("font-size:10px; color:#8899aa;")

        self._save_cal_btn = QPushButton("Save Cal…")
        self._save_cal_btn.setFixedWidth(90)
        self._save_cal_btn.setToolTip(
            "Save the current homography calibration to a chosen file.\n"
            "Useful when you have multiple cameras or room setups."
        )
        self._save_cal_btn.clicked.connect(self._save_cal_dialog)

        self._load_cal_btn = QPushButton("Load Cal…")
        self._load_cal_btn.setFixedWidth(90)
        self._load_cal_btn.setToolTip(
            "Load a homography calibration file from any location."
        )
        self._load_cal_btn.clicked.connect(self._load_cal_dialog)

        self._path_live_chk = QCheckBox("Live path")
        self._path_live_chk.setChecked(False)
        self._path_live_chk.setStyleSheet("font-size:11px;")
        self._path_live_chk.setToolTip(
            "Live: path grows as the video plays.\n"
            "Seek anywhere — the map shows only the path up to that moment.\n"
            "Off: always show the complete session path."
        )

        _path_toggle_row.addWidget(self._path_show_path_chk)
        _path_toggle_row.addWidget(self._path_show_heatmap_chk)
        _path_toggle_row.addWidget(self._path_live_chk)
        _path_toggle_row.addStretch()
        _path_toggle_row.addWidget(self._path_cal_status_lbl)
        _path_toggle_row.addSpacing(8)
        _path_toggle_row.addWidget(self._save_cal_btn)
        _path_toggle_row.addWidget(self._load_cal_btn)
        _path_tab_layout.addLayout(_path_toggle_row)

        # Stats + export row
        _path_stats_row = QHBoxLayout()
        self._path_stats_lbl = QLabel("No path data — run Track Path first.")
        self._path_stats_lbl.setStyleSheet("font-size:10px; color:#8899aa;")
        self._export_path_btn = QPushButton("Export Path CSV…")
        self._export_path_btn.setEnabled(False)
        self._export_path_btn.clicked.connect(self._export_path_csv)
        _path_stats_row.addWidget(self._path_stats_lbl, stretch=1)
        _path_stats_row.addWidget(self._export_path_btn)
        _path_tab_layout.addLayout(_path_stats_row)

        self._path_map = PathMapWidget(room_size_cm=self._room_size_cm)
        _path_tab_layout.addWidget(self._path_map, stretch=1)

        self._path_show_path_chk.stateChanged.connect(
            lambda s: self._path_map.set_show_path(bool(s))
        )
        self._path_show_heatmap_chk.stateChanged.connect(
            lambda s: self._path_map.set_show_heatmap(bool(s))
        )
        self._path_live_chk.stateChanged.connect(self._on_path_live_toggled)

        self._tabs.addTab(_path_tab_container, "Path & Heatmap")

        self._splitter.addWidget(self._video)
        self._splitter.addWidget(self._tabs)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        root.addWidget(self._splitter, stretch=1)

        # ── Analysis panel (initially hidden, takes over full content area) ──
        self._build_analysis_panel(root)

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
        self._track_path_btn.setEnabled(True)
        self._calibrate_btn.setEnabled(True)
        self._edit_regions_btn.setEnabled(True)
        self._load_regions_btn.setEnabled(True)

        # Auto-load homography if it exists beside the video
        from backend.homography_config import (
            default_homography_path, load_homography,
        )
        hom_path = default_homography_path(path)
        if hom_path.exists():
            try:
                hom_data = load_homography(hom_path)
                self._homography_matrix = hom_data["matrix"]
                self._room_size_cm      = tuple(hom_data["room_size_cm"])
                self._cal_pixel_pts     = hom_data.get("pixel_points", [])
                self._cal_world_pts     = hom_data.get("world_points_cm", [])
                self._path_map.set_room_size(*self._room_size_cm)
                self._path_cal_status_lbl.setText(
                    f"Calibration loaded  ({self._room_size_cm[0]/100:.1f} m × "
                    f"{self._room_size_cm[1]/100:.1f} m)"
                )
                self._path_cal_status_lbl.setStyleSheet(
                    "font-size:10px; color:#88cc88;"
                )
            except Exception as exc:
                self._status.showMessage(
                    f"Warning: could not load homography: {exc}"
                )

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

    def _on_subject_classes_changed(self, text: str):
        """Parse the subject classes field; always include 'person'."""
        extra = {c.strip().lower() for c in text.split(",") if c.strip()}
        self._subject_labels = {"person"} | extra

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
            # Save subject classes too (exclude "person" — always implicit — but
            # include everything else the user typed in the Subject classes field)
            extra_subjects = sorted(self._subject_labels - {"person"})
            save_class_list(
                path, config_name.strip(), self._world_classes,
                subject_classes=extra_subjects if extra_subjects else None,
            )
            self._status.showMessage(
                f"Class list saved → {Path(path).name}  "
                f"({len(self._world_classes)} classes, "
                f"{len(extra_subjects)} extra subject(s))"
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
            config_name, classes, subject_classes = load_class_list(path)
            self._world_classes = classes
            self._world_classes_edit.setText(", ".join(classes))
            # Restore subject classes if the file has them
            if subject_classes:
                self._subject_classes_edit.setText(", ".join(subject_classes))
                # _on_subject_classes_changed fires via textChanged signal
            msg = (
                f"Class list loaded: '{config_name}'  "
                f"({len(classes)} class{'es' if len(classes) != 1 else ''}"
            )
            if subject_classes:
                msg += f", subjects: {', '.join(subject_classes)}"
            msg += ")"
            self._status.showMessage(msg)
        except Exception as exc:
            self._status.showMessage(f"Failed to load class list: {exc}")

    # ------------------------------------------------------------------ region editing

    # ------------------------------------------------------------------ label filter

    def _get_all_object_labels(self) -> list[str]:
        """All unique non-subject detected labels from the current YOLO run."""
        if not self._raw_frames:
            return []
        return sorted({
            d.label
            for fd in self._raw_frames
            for d in fd.detections
            if d.label not in self._subject_labels
        })

    def _get_detected_subjects(self) -> list[str]:
        """Subject labels that were actually detected in the current run."""
        if not self._raw_frames:
            return ["person"]
        detected = {
            d.label
            for fd in self._raw_frames
            for d in fd.detections
            if d.label in self._subject_labels
        }
        # Always include "person" even if not detected (so it shows in dialog)
        detected.add("person")
        return sorted(detected)

    def _open_label_filter(self):
        """Open (or bring to front) the non-modal interaction filter dialog."""
        all_labels   = self._get_all_object_labels()
        all_subjects = self._get_detected_subjects()
        if not all_labels and not all_subjects:
            self._status.showMessage(
                "No YOLO data loaded yet — run YOLO first."
            )
            return
        if self._label_filter_dlg and self._label_filter_dlg.isVisible():
            self._label_filter_dlg.populate(
                all_subjects, all_labels,
                self._subject_filter_excluded, self._label_filter_excluded,
            )
            self._label_filter_dlg.raise_()
            self._label_filter_dlg.activateWindow()
        else:
            self._label_filter_dlg = _InteractionFilterDialog(
                all_subjects=all_subjects,
                all_labels=all_labels,
                excl_subjects=self._subject_filter_excluded,
                excl_labels=self._label_filter_excluded,
                on_change=self._on_interaction_filter_changed,
                parent=self,
            )
            self._label_filter_dlg.show()

    def _on_interaction_filter_changed(
        self, excl_subjects: set[str], excl_labels: set[str]
    ):
        """Called by the dialog whenever any checkbox changes."""
        self._subject_filter_excluded = excl_subjects
        self._label_filter_excluded   = excl_labels
        n_hidden = len(excl_subjects) + len(excl_labels)
        self._results.set_filter_status(n_hidden)
        self._schedule_refilter()

    def _hide_label(self, label: str):
        """Called by the right-click 'Hide' action on a table row."""
        self._label_filter_excluded.add(label)
        if self._label_filter_dlg and self._label_filter_dlg.isVisible():
            self._label_filter_dlg.set_excluded_labels(self._label_filter_excluded)
            self._label_filter_dlg._obj_list.itemChanged.connect(
                self._label_filter_dlg._changed
            )
        n_hidden = len(self._label_filter_excluded) + len(self._subject_filter_excluded)
        self._results.set_filter_status(n_hidden)
        self._schedule_refilter()

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
            self._settings().setValue("last_regions_path", path)
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
            self._settings().setValue("last_regions_path", path)
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

            # Restore path tracking data if present
            path_section = data.get("path_tracking", {})
            if path_section and path_section.get("track_points"):
                try:
                    from backend.results_format import load_path_section
                    from backend.path_analyzer import (
                        compute_heatmap, heatmap_to_rgba,
                    )
                    path_pts = load_path_section(path_section)
                    self._path_points = path_pts
                    room_size = tuple(path_section.get("room_size_cm",
                                                       self._room_size_cm))
                    self._room_size_cm = room_size
                    self._path_map.set_room_size(*room_size)
                    trail = [(p.frame_index, p.cx_px, p.cy_px) for p in path_pts]
                    self._video.set_trail_points(trail)
                    self._path_map.set_path(path_pts)
                    heatmap_grid = compute_heatmap(path_pts, room_size)
                    if heatmap_grid.max() > 0:
                        self._path_map.set_heatmap(heatmap_to_rgba(heatmap_grid))
                    stats_d = path_section.get("stats", {})
                    if stats_d.get("calibrated"):
                        stats_text = (
                            f"Distance: {stats_d.get('total_distance_m', 0):.1f} m  ·  "
                            f"Avg speed: {stats_d.get('avg_speed_m_s', 0):.2f} m/s  ·  "
                            f"Duration: {_fmt_dur(stats_d.get('duration_s', 0) * 1000)}  ·  "
                            f"{stats_d.get('n_points', len(path_pts))} pts"
                        )
                    else:
                        stats_text = (
                            f"{stats_d.get('n_points', len(path_pts))} track pts  ·  "
                            "Calibrate camera for real-world distances"
                        )
                    self._path_stats_lbl.setText(stats_text)
                    self._path_stats_lbl.setStyleSheet("font-size:10px; color:#aabbcc;")
                    self._export_path_btn.setEnabled(True)
                    self._analysis_btn.setEnabled(True)
                    self._tabs.setTabText(3, f"Path & Heatmap ({len(path_pts)} pts)")
                except Exception as path_exc:
                    print(f"[LoadJSON] Could not restore path data: {path_exc}")

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
        if self._action_merge_chk.isChecked() and filtered_clips:
            gap_ms = self._action_merge_gap_spin.value()
            filtered_clips = _merge_action_clips(filtered_clips, gap_ms)

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
        mapped = InteractionMapper(
            proximity_px=prox,
            subject_labels=self._subject_labels,
        ).map(filtered)
        # Object Interactions tab shows pure YOLO spatial labels only.
        # MMAction2 clips are displayed separately in the Actions tab — passing
        # them here would cause apply_action_labels() to replace "near ball" with
        # "Running ball", mixing two unrelated data streams.
        segs, ui_boxes = SegmentBuilder(fw, fh, min_duration_ms=min_dur).build(mapped)
        self._video.set_frame_detections(_convert_boxes(ui_boxes))

        # Apply subject + label exclusion filters before building table rows
        visible_segs = segs
        if self._subject_filter_excluded:
            visible_segs = [
                s for s in visible_segs
                if s.subject_label not in self._subject_filter_excluded
            ]
        table_rows = [s.as_table_row() for s in visible_segs]
        if self._label_filter_excluded:
            table_rows = [
                r for r in table_rows
                if r[3] not in self._label_filter_excluded
            ]
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
        n      = len(table_rows)   # post-filter count
        n_reg  = len(region_segs)
        n_clip = len(filtered_clips)
        n_hidden = len(self._label_filter_excluded) + len(self._subject_filter_excluded)
        self._results.set_filter_status(n_hidden)
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
        if not self._check_same_settings("yolo"):
            return
        self._launch_worker(_YoloWorker)

    def _run_full(self):
        if not self._check_same_settings("yolo"):
            return
        self._launch_worker(_FullAnalysisWorker)

    def _run_mmaction2(self):
        """Launch MMAction2 only, without re-running YOLO."""
        if not self._video_path:
            self._status.showMessage("No video loaded.")
            return
        if self._worker_thread and self._worker_thread.isRunning():
            return
        if not self._check_same_settings("action"):
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
        # Merge subject classes (hand, foot…) into the detection list so they
        # get detected by YOLO-World even if the user forgot to add them above.
        if is_world:
            # YOLO-World needs every label listed explicitly — auto-add all
            # subject classes (including "person") if not already present.
            merged_cls = list(dict.fromkeys(
                self._world_classes + [s for s in self._subject_labels
                                       if s not in self._world_classes]
            ))
            world_cls = merged_cls
        else:
            world_cls = None
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
            yolo_run_settings=self._get_current_yolo_run_settings(),
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
        if diagnostic == "Cancelled":
            self._status.showMessage("Analysis cancelled.")
            return
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
        # Refresh the filter dialog if open (new labels/subjects may have appeared)
        if self._label_filter_dlg and self._label_filter_dlg.isVisible():
            self._label_filter_dlg.populate(
                self._get_detected_subjects(),
                self._get_all_object_labels(),
                self._subject_filter_excluded,
                self._label_filter_excluded,
            )
        self._refilter()

    def _on_mmaction2_finished(self, action_clips, diagnostic: str):
        """Slot for standalone MMAction2 run (no YOLO data involved)."""
        self._set_running(False)
        if diagnostic == "Cancelled":
            self._status.showMessage("Analysis cancelled.")
            return
        self._action_clips = action_clips

        # Upsert action section in JSON (preserves YOLO + path sections)
        if self._video_path:
            from backend.results_format import (
                default_output_path, upsert_action_section,
            )
            json_path = default_output_path(self._video_path)
            try:
                upsert_action_section(
                    output_path  = json_path,
                    video_path   = self._video_path,
                    fps          = self._video.fps,
                    fw           = self._video.frame_width,
                    fh           = self._video.frame_height,
                    total_frames = self._video.total_frames,
                    action_clips = action_clips,
                    run_settings = self._get_current_action_run_settings(),
                )
            except Exception as exc:
                print(f"[UI] WARNING: could not save action JSON: {exc}")

        n = len(action_clips)
        self._source_label.setText(
            f"MMAction2 complete  ·  {n} clip{'s' if n != 1 else ''} captured  ·  "
            f"use 'Action conf ≥' slider to filter"
        )
        self._status.showMessage(f"MMAction2 complete — {diagnostic}")
        # refilter updates the banner and source label (YOLO part skipped if no raw_frames)
        self._refilter()

    # ------------------------------------------------------------------ path tracking

    def _run_path_tracking(self):
        """Launch the ByteTrack path-tracking worker."""
        if not self._video_path:
            self._status.showMessage("No video loaded.")
            return
        if self._worker_thread and self._worker_thread.isRunning():
            return
        if not self._check_same_settings("path"):
            return

        yolo_model = self._get_yolo_model_name()
        self._set_running(True)
        self._worker = _PathTrackingWorker(
            self._video_path,
            yolo_model=yolo_model,
            homography_matrix=self._homography_matrix,
            room_size_cm=self._room_size_cm,
        )
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.status.connect(self._on_worker_status)
        self._worker.finished.connect(self._on_path_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(lambda *_: self._worker_thread.quit())
        self._worker.error.connect(   lambda *_: self._worker_thread.quit())

        self._worker_thread.start()

    def _on_path_finished(self, path_pts: list, diagnostic: str):
        """Receive completed path data, build heatmap, update map widget."""
        self._set_running(False)
        if not path_pts and diagnostic == "Cancelled":
            self._status.showMessage("Path tracking cancelled.")
            return

        self._path_points = path_pts

        # Push trail overlay to video player — (frame_index, cx_px, cy_px)
        trail = [(p.frame_index, p.cx_px, p.cy_px) for p in path_pts]
        self._video.set_trail_points(trail)

        # Update top-down map
        self._path_map.set_path(path_pts)
        if self._path_live_mode:
            self._path_map.set_time_cutoff(self._last_position_ms)

        # Build heatmap if calibrated
        from backend.path_analyzer import (
            compute_heatmap, heatmap_to_rgba, compute_stats,
        )
        heatmap_grid = compute_heatmap(path_pts, self._room_size_cm)
        if heatmap_grid.max() > 0:
            rgba = heatmap_to_rgba(heatmap_grid)
            self._path_map.set_heatmap(rgba)
        else:
            self._path_map.set_heatmap(None)

        # Compute and display stats
        self._path_stats = compute_stats(path_pts)
        st = self._path_stats
        if st.calibrated:
            stats_text = (
                f"Distance: {st.total_distance_m:.1f} m  ·  "
                f"Avg speed: {st.avg_speed_m_s:.2f} m/s  ·  "
                f"Duration: {_fmt_dur(st.duration_s * 1000)}  ·  "
                f"{st.n_points} pts"
                + (f"  ({st.n_interpolated} interp.)" if st.n_interpolated else "")
            )
        else:
            stats_text = (
                f"{st.n_points} track pts  ·  "
                f"Duration: {_fmt_dur(st.duration_s * 1000)}  ·  "
                "Calibrate camera for real-world distances"
            )
        self._path_stats_lbl.setText(stats_text)
        self._path_stats_lbl.setStyleSheet("font-size:10px; color:#aabbcc;")
        self._export_path_btn.setEnabled(len(path_pts) > 0)

        n = len(path_pts)
        self._tabs.setTabText(3, f"Path & Heatmap ({n} pts)")
        self._analysis_btn.setEnabled(n > 0)
        self._status.showMessage(f"Path tracking complete — {diagnostic}")

        # Upsert path section in JSON (preserves YOLO + action sections)
        if self._video_path and path_pts:
            from backend.results_format import (
                default_output_path, upsert_path_section,
            )
            json_path = default_output_path(self._video_path)
            try:
                upsert_path_section(
                    output_path  = json_path,
                    video_path   = self._video_path,
                    fps          = self._video.fps,
                    fw           = self._video.frame_width,
                    fh           = self._video.frame_height,
                    total_frames = self._video.total_frames,
                    track_points = path_pts,
                    room_size_cm = self._room_size_cm,
                    run_settings = self._get_current_path_run_settings(),
                )
            except Exception as exc:
                print(f"[UI] WARNING: could not save path JSON: {exc}")

    # ------------------------------------------------------------------ camera calibration

    def _start_calibration(self):
        """
        Begin 4-point homography calibration.
        The user must click 4 known floor points in the video frame
        and type in their real-world X, Y positions in cm.
        """
        if not self._video_path:
            self._status.showMessage("Load a video before calibrating.")
            return

        # Ask for room dimensions before starting click sequence
        w_txt, ok = QInputDialog.getText(
            self, "Room Width",
            "Room width in cm (across the camera view):",
            text=str(int(self._room_size_cm[0])),
        )
        if not ok or not w_txt.strip():
            return
        d_txt, ok = QInputDialog.getText(
            self, "Room Depth",
            "Room depth in cm (front-to-far wall, away from camera):",
            text=str(int(self._room_size_cm[1])),
        )
        if not ok or not d_txt.strip():
            return

        try:
            new_w = float(w_txt.strip())
            new_d = float(d_txt.strip())
        except ValueError:
            self._status.showMessage("Invalid room dimensions — enter numbers.")
            return

        self._room_size_cm = (new_w, new_d)
        self._path_map.set_room_size(new_w, new_d)

        # Reset calibration state
        self._cal_step      = 1
        self._cal_pixel_pts = []
        self._cal_world_pts = []
        self._video.set_calibration_mode(True)
        self._video.set_cal_markers([])

        self._status.showMessage(
            "Calibration: click point 1 / 4 on the video frame."
        )

    def _on_calibration_click(self, nx: float, ny: float):
        """
        Handle each of the 4 calibration clicks.
        nx, ny are normalised (0–1) video-frame coordinates.
        A re-entry guard prevents double-firing while the dialog is open.
        """
        if self._cal_step == 0 or self._cal_dialog_active:
            return

        self._cal_dialog_active = True
        try:
            self._handle_calibration_click(nx, ny)
        finally:
            self._cal_dialog_active = False

    def _handle_calibration_click(self, nx: float, ny: float):
        """Inner implementation — called from the guarded wrapper above."""
        step = self._cal_step
        fw   = self._video.frame_width  or 1
        fh   = self._video.frame_height or 1
        px   = nx * fw
        py   = ny * fh

        dlg = _CalibPointDialog(step, px, py, self._room_size_cm, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return   # user cancelled — keep the same step

        vals = dlg.result_values()
        if vals is None:
            self._status.showMessage(
                "Invalid coordinates — enter numbers and try again."
            )
            return

        x_cm, y_cm = vals
        self._cal_pixel_pts.append([px, py])
        self._cal_world_pts.append([x_cm, y_cm])

        # Update crosshair markers on the video
        self._video.set_cal_markers([
            (p[0] / fw, p[1] / fh)
            for p in self._cal_pixel_pts
        ])

        if step < 4:
            self._cal_step += 1
            self._status.showMessage(
                f"Calibration: click point {self._cal_step} / 4 on the video frame."
            )
        else:
            # All 4 points collected — compute and save
            self._cal_step = 0
            self._video.set_calibration_mode(False)
            self._video.set_cal_markers([])
            self._finish_calibration(
                self._cal_pixel_pts,
                self._cal_world_pts,
                self._room_size_cm,
            )

    def _finish_calibration(
        self,
        pixel_pts: list,
        world_pts: list,
        room_size_cm: tuple,
        save_path: str | None = None,
    ):
        """
        Compute the homography from 4 point pairs and persist it.
        save_path: if None, auto-derives the path from the video filename.
        """
        try:
            from backend.homography_config import (
                save_homography, default_homography_path,
            )
            path = (Path(save_path)
                    if save_path
                    else default_homography_path(self._video_path))
            matrix = save_homography(
                path, pixel_pts, world_pts, list(room_size_cm),
            )
            self._homography_matrix = matrix
            self._room_size_cm      = tuple(room_size_cm)
            self._path_map.set_room_size(*room_size_cm)
            self._path_cal_status_lbl.setText(
                f"Calibrated  ({room_size_cm[0]/100:.1f} m × "
                f"{room_size_cm[1]/100:.1f} m)"
            )
            self._path_cal_status_lbl.setStyleSheet(
                "font-size:10px; color:#88cc88;"
            )
            self._status.showMessage(
                f"Calibration saved → {path.name}  —  "
                f"Run 'Track Path' to see the floor map."
            )
        except Exception as exc:
            self._status.showMessage(f"Calibration failed: {exc}")
            print(f"[Calibration] Error: {exc}")

    # ------------------------------------------------------------------ save/load calibration

    def _save_cal_dialog(self):
        """Save the current homography to a user-chosen file."""
        if self._homography_matrix is None:
            self._status.showMessage(
                "No calibration to save — run 'Calibrate Camera…' first."
            )
            return
        default_dir = (str(Path(self._video_path).parent)
                       if self._video_path else "")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Calibration", default_dir,
            "Homography (*.homography.json);;JSON (*.json);;All files (*)"
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".homography.json"
        try:
            from backend.homography_config import save_homography
            save_homography(
                path,
                self._cal_pixel_pts,
                self._cal_world_pts,
                list(self._room_size_cm),
            )
            self._status.showMessage(
                f"Calibration saved → {Path(path).name}"
            )
        except Exception as exc:
            self._status.showMessage(f"Failed to save calibration: {exc}")

    def _load_cal_dialog(self):
        """Load a homography file from any location."""
        default_dir = (str(Path(self._video_path).parent)
                       if self._video_path else "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration", default_dir,
            "Homography (*.homography.json *.json);;All files (*)"
        )
        if not path:
            return
        try:
            from backend.homography_config import load_homography
            data = load_homography(path)
            self._homography_matrix = data["matrix"]
            self._room_size_cm      = tuple(data["room_size_cm"])
            self._cal_pixel_pts     = data["pixel_points"]
            self._cal_world_pts     = data["world_points_cm"]
            self._path_map.set_room_size(*self._room_size_cm)
            self._path_cal_status_lbl.setText(
                f"Calibration loaded  ({self._room_size_cm[0]/100:.1f} m × "
                f"{self._room_size_cm[1]/100:.1f} m)"
            )
            self._path_cal_status_lbl.setStyleSheet(
                "font-size:10px; color:#88cc88;"
            )
            self._status.showMessage(
                f"Calibration loaded: {Path(path).name}"
            )
        except Exception as exc:
            self._status.showMessage(f"Failed to load calibration: {exc}")

    def _on_error(self, message: str):
        self._set_running(False)
        self._status.showMessage(f"Error: {message}")
        print(f"[Worker] Error: {message}")

    def _set_running(self, running: bool):
        has_video = self._video_path is not None
        self._yolo_btn.setEnabled(not running and has_video)
        self._full_btn.setEnabled(not running and has_video)
        self._mmaction2_btn.setEnabled(not running and has_video)
        self._track_path_btn.setEnabled(not running and has_video)
        self._calibrate_btn.setEnabled(not running and has_video)
        self._load_json_btn.setEnabled(not running)
        self._mock_toggle.setEnabled(not running)
        self._sample_spin.setEnabled(not running)
        self._all_classes_chk.setEnabled(not running)
        self._progress.setValue(0)
        self._progress.setVisible(running)
        self._phase_label.setVisible(running)
        self._cancel_btn.setVisible(running)
        self._cancel_btn.setEnabled(running)
        if not running:
            self._phase_label.setText("")

    # ------------------------------------------------------------------ seek sync

    def _on_position_changed(self, position_ms: int):
        self._last_position_ms = position_ms
        if self._path_live_mode and self._path_points:
            self._path_map.set_time_cutoff(position_ms)
        if self._result_time_ranges:
            self._results.highlight_row_at(position_ms, self._result_time_ranges)
        if self._region_time_ranges:
            self._region_table.highlight_row_at(position_ms, self._region_time_ranges)
        if self._action_time_ranges:
            self._action_table.highlight_row_at(position_ms, self._action_time_ranges)

    # ------------------------------------------------------------------ seekbar highlight

    def _on_table_row_clicked(self, proxy_idx, tab: int, proxy, time_ranges: list):
        """
        Toggle a seek-bar highlight for the clicked row's time range.
        Clicking the same row again clears the highlight.
        """
        src_row = proxy.mapToSource(proxy_idx).row()
        if self._hl_tab == tab and self._hl_src_row == src_row:
            # Same row clicked again → clear
            self._hl_tab     = -1
            self._hl_src_row = -1
            self._video.clear_seekbar_highlight()
        else:
            if 0 <= src_row < len(time_ranges):
                start_ms, end_ms = time_ranges[src_row]
                self._video.set_seekbar_highlight(start_ms, end_ms)
                self._hl_tab     = tab
                self._hl_src_row = src_row

    # ------------------------------------------------------------------ analysis panel

    def _build_analysis_panel(self, root: QVBoxLayout):
        """Build the full-window analysis panel (initially hidden)."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

        self._analysis_panel = QWidget()
        self._analysis_panel.setVisible(False)
        panel_layout = QVBoxLayout(self._analysis_panel)
        panel_layout.setContentsMargins(8, 4, 8, 4)
        panel_layout.setSpacing(4)

        # ── Navigation bar ──
        nav = QHBoxLayout()
        nav.setSpacing(6)

        back_btn = QPushButton("← Back")
        back_btn.setFixedWidth(80)
        back_btn.setToolTip("Return to video + tables view")
        back_btn.clicked.connect(self._hide_analysis_panel)
        nav.addWidget(back_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #3a3a5a;")
        nav.addWidget(sep)

        # Sub-view toggle buttons
        self._analysis_view_btns: list[QPushButton] = []
        for i, (icon, name) in enumerate([
            ("≡", "Summary"),
            ("📈", "Speed"),
            ("▶▐", "Episodes"),
            ("⬡", "Regions"),
            ("🗺", "Path Map"),
        ]):
            btn = QPushButton(f"{icon}  {name}")
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.setFixedWidth(105)
            btn.clicked.connect(lambda checked, idx=i: self._set_analysis_view(idx))
            nav.addWidget(btn)
            self._analysis_view_btns.append(btn)

        nav.addStretch()

        self._analysis_export_btn = QPushButton("Export CSV…")
        self._analysis_export_btn.setEnabled(False)
        self._analysis_export_btn.setToolTip("Export current view to CSV")
        self._analysis_export_btn.clicked.connect(self._export_analysis_csv)
        nav.addWidget(self._analysis_export_btn)

        panel_layout.addLayout(nav)

        # ── Parameter bar ──
        param_frame = QFrame()
        param_frame.setObjectName("parambar")
        param_bar = QHBoxLayout(param_frame)
        param_bar.setContentsMargins(8, 4, 8, 4)
        param_bar.setSpacing(6)

        def _albl(txt):
            l = QLabel(txt)
            l.setStyleSheet("font-size:11px; color:#99aacc;")
            return l

        param_bar.addWidget(_albl("Smooth:"))
        self._analysis_smooth_spin = QSpinBox()
        self._analysis_smooth_spin.setRange(1, 31)
        self._analysis_smooth_spin.setSingleStep(2)
        self._analysis_smooth_spin.setValue(self._analysis_smooth_window)
        self._analysis_smooth_spin.setFixedWidth(52)
        self._analysis_smooth_spin.setToolTip(
            "Rolling-average window (frames) applied before computing speed and episodes.\n"
            "Higher = smoother graph lines, less jitter in distance calculations.\n"
            "1 = no smoothing (raw tracker output)."
        )
        self._analysis_smooth_spin.valueChanged.connect(self._on_analysis_param_changed)
        param_bar.addWidget(self._analysis_smooth_spin)
        param_bar.addWidget(_albl("frames"))

        param_bar.addSpacing(14)
        param_bar.addWidget(_albl("Stationary < "))
        self._analysis_thresh_spin = QDoubleSpinBox()
        self._analysis_thresh_spin.setRange(0.01, 2.0)
        self._analysis_thresh_spin.setSingleStep(0.05)
        self._analysis_thresh_spin.setValue(self._analysis_stat_thresh)
        self._analysis_thresh_spin.setDecimals(2)
        self._analysis_thresh_spin.setFixedWidth(70)
        self._analysis_thresh_spin.setToolTip(
            "Speed threshold for classifying movement as 'stationary' (m/s).\n"
            "Points below this speed → stationary episode.\n"
            "Default 0.10 m/s (10 cm/s).  Adjust for the child's typical pace."
        )
        self._analysis_thresh_spin.valueChanged.connect(self._on_analysis_param_changed)
        param_bar.addWidget(self._analysis_thresh_spin)
        param_bar.addWidget(_albl("m/s"))

        param_bar.addSpacing(14)
        param_bar.addWidget(_albl("Min episode:"))
        self._analysis_minep_spin = QSpinBox()
        self._analysis_minep_spin.setRange(100, 10000)
        self._analysis_minep_spin.setSingleStep(100)
        self._analysis_minep_spin.setValue(int(self._analysis_min_ep_ms))
        self._analysis_minep_spin.setFixedWidth(68)
        self._analysis_minep_spin.setToolTip(
            "Episodes shorter than this (ms) are merged into their neighbours.\n"
            "Prevents a single noisy frame from fragmenting the episode sequence.\n"
            "Default 500 ms."
        )
        self._analysis_minep_spin.valueChanged.connect(self._on_analysis_param_changed)
        param_bar.addWidget(self._analysis_minep_spin)
        param_bar.addWidget(_albl("ms"))

        param_bar.addStretch()
        panel_layout.addWidget(param_frame)

        # ── Content stack ──
        self._analysis_stack = QStackedWidget()
        panel_layout.addWidget(self._analysis_stack, stretch=1)

        # Page 0 — Summary
        self._build_analysis_summary_page()

        # Page 1 — Speed graph (matplotlib)
        self._analysis_speed_fig = Figure(tight_layout=True)
        self._analysis_speed_fig.patch.set_facecolor("#12122a")
        self._analysis_speed_ax = self._analysis_speed_fig.add_subplot(111)
        self._analysis_speed_canvas = FigureCanvas(self._analysis_speed_fig)
        self._analysis_speed_canvas.mpl_connect(
            "button_press_event", self._on_speed_graph_click
        )
        self._analysis_stack.addWidget(self._analysis_speed_canvas)

        # Page 2 — Episodes table
        self._episodes_table = QTableWidget(0, 7)
        self._episodes_table.setHorizontalHeaderLabels(
            ["Start", "End", "Duration", "Type", "Distance", "Avg Speed", "Max Speed"]
        )
        hdr = self._episodes_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setStretchLastSection(True)
        self._episodes_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._episodes_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._episodes_table.setAlternatingRowColors(True)
        self._episodes_table.clicked.connect(self._on_episode_row_clicked)
        self._analysis_stack.addWidget(self._episodes_table)

        # Page 3 — Regions table
        self._regions_analysis_table = QTableWidget(0, 5)
        self._regions_analysis_table.setHorizontalHeaderLabels(
            ["Region", "Dwell Time", "% Session", "Visits", "First Visit"]
        )
        rhdr = self._regions_analysis_table.horizontalHeader()
        rhdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        rhdr.setStretchLastSection(True)
        self._regions_analysis_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._regions_analysis_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._regions_analysis_table.setAlternatingRowColors(True)
        self._analysis_stack.addWidget(self._regions_analysis_table)

        # Page 4 — Full-screen path map
        from ui.path_map_widget import PathMapWidget
        path_page = QWidget()
        path_page_layout = QVBoxLayout(path_page)
        path_page_layout.setContentsMargins(0, 2, 0, 0)
        path_page_layout.setSpacing(4)

        path_ctrl = QHBoxLayout()
        path_ctrl.setSpacing(8)
        self._analysis_show_path_chk = QCheckBox("Path line")
        self._analysis_show_path_chk.setChecked(True)
        self._analysis_show_path_chk.stateChanged.connect(
            lambda s: self._analysis_path_map.set_show_path(bool(s))
        )
        self._analysis_show_hm_chk = QCheckBox("Heatmap")
        self._analysis_show_hm_chk.setChecked(True)
        self._analysis_show_hm_chk.stateChanged.connect(
            lambda s: self._analysis_path_map.set_show_heatmap(bool(s))
        )
        path_ctrl.addWidget(self._analysis_show_path_chk)
        path_ctrl.addWidget(self._analysis_show_hm_chk)
        path_ctrl.addStretch()
        path_page_layout.addLayout(path_ctrl)

        self._analysis_path_map = PathMapWidget(room_size_cm=self._room_size_cm)
        path_page_layout.addWidget(self._analysis_path_map, stretch=1)
        self._analysis_stack.addWidget(path_page)

        root.addWidget(self._analysis_panel, stretch=1)

    def _build_analysis_summary_page(self):
        """Build the scrollable summary statistics page (page 0 of _analysis_stack)."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        grid  = QGridLayout(inner)
        grid.setSpacing(10)
        grid.setContentsMargins(20, 20, 20, 20)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setColumnStretch(5, 1)

        self._summary_labels: dict[str, QLabel] = {}

        stat_defs = [
            # (key, display_name)
            ("total_distance",  "Total Distance"),
            ("avg_speed",       "Avg Speed"),
            ("max_speed",       "Peak Speed"),
            ("duration",        "Session Duration"),
            ("coverage",        "Floor Coverage"),
            ("active_ratio",    "Active / Stationary"),
            ("n_active_ep",     "Active Episodes"),
            ("n_stationary_ep", "Stationary Episodes"),
            ("avg_active_dur",  "Avg Active Duration"),
            ("avg_stat_dur",    "Avg Stationary Dur."),
            ("n_points",        "Track Points"),
            ("n_interp",        "Interpolated Points"),
        ]

        for i, (key, label) in enumerate(stat_defs):
            row = i // 3
            col = (i % 3) * 2

            name_lbl = QLabel(label)
            name_lbl.setStyleSheet(
                "font-size:10px; color:#7799bb; font-weight:bold;"
                " padding-right:6px;"
            )
            name_lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )

            val_lbl = QLabel("—")
            val_lbl.setStyleSheet("font-size:17px; color:#ddeeff;")
            val_lbl.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            val_lbl.setMinimumWidth(160)

            grid.addWidget(name_lbl, row, col)
            grid.addWidget(val_lbl,  row, col + 1)
            self._summary_labels[key] = val_lbl

        grid.setRowStretch(len(stat_defs) // 3 + 1, 1)
        scroll.setWidget(inner)
        self._analysis_stack.addWidget(scroll)

    # ------------------------------------------------------------------ analysis control

    def _show_analysis_panel(self):
        """Hide video+tabs splitter and show the full-window analysis panel."""
        self._splitter.hide()
        self._analysis_panel.show()
        self._refresh_analysis()

    def _hide_analysis_panel(self):
        """Return to normal video + tables view."""
        self._analysis_panel.hide()
        self._splitter.show()

    def _set_analysis_view(self, index: int):
        """Switch the analysis sub-view and update toggle-button states."""
        for i, btn in enumerate(self._analysis_view_btns):
            btn.setChecked(i == index)
        self._analysis_stack.setCurrentIndex(index)

    def _on_analysis_param_changed(self):
        """Called when smooth / threshold / min-episode spinboxes change."""
        self._analysis_smooth_window = self._analysis_smooth_spin.value()
        self._analysis_stat_thresh   = self._analysis_thresh_spin.value()
        self._analysis_min_ep_ms     = float(self._analysis_minep_spin.value())
        if self._analysis_panel.isVisible():
            self._refresh_analysis()

    def _refresh_analysis(self):
        """Smooth the path, recompute all metrics, update every sub-view."""
        if not self._path_points:
            return

        from backend.path_analyzer import (
            smooth_path, compute_stats, compute_speed_series,
            compute_episodes, episode_summary,
            compute_region_stats, compute_coverage_pct,
            compute_heatmap, heatmap_to_rgba,
        )

        pts = smooth_path(self._path_points, window=self._analysis_smooth_window)

        stats    = compute_stats(pts)
        speed_s  = compute_speed_series(pts)
        episodes = compute_episodes(
            pts,
            stationary_threshold_m_s = self._analysis_stat_thresh,
            min_episode_ms           = self._analysis_min_ep_ms,
        )
        ep_sum   = episode_summary(episodes)
        fw, fh   = self._analysis_fw, self._analysis_fh
        regions  = getattr(self, "_regions", [])
        reg_stats = (
            compute_region_stats(pts, regions, fw, fh)
            if regions and fw > 0 else []
        )
        coverage = (
            compute_coverage_pct(pts, self._room_size_cm)
            if stats.calibrated else 0.0
        )

        # Cache for export
        self._analysis_episodes  = episodes
        self._analysis_reg_stats = reg_stats
        self._analysis_speed_s   = speed_s
        self._analysis_stats     = stats

        self._update_analysis_summary(stats, ep_sum, coverage)
        self._update_speed_graph(speed_s, episodes)
        self._update_episodes_table(episodes)
        self._update_regions_table(reg_stats)

        # Full-screen path map
        self._analysis_path_map.set_room_size(*self._room_size_cm)
        self._analysis_path_map.set_path(pts)
        grid = compute_heatmap(pts, self._room_size_cm)
        if grid.max() > 0:
            self._analysis_path_map.set_heatmap(heatmap_to_rgba(grid))
        else:
            self._analysis_path_map.set_heatmap(None)

        self._analysis_export_btn.setEnabled(True)

    def _update_analysis_summary(self, stats, ep_sum: dict, coverage: float):
        lbl = self._summary_labels
        if stats.calibrated:
            lbl["total_distance"].setText(f"{stats.total_distance_m:.2f} m")
            lbl["avg_speed"].setText(f"{stats.avg_speed_m_s:.3f} m/s")
            lbl["max_speed"].setText(f"{stats.max_speed_m_s:.3f} m/s")
            lbl["coverage"].setText(f"{coverage:.1f} %")
        else:
            lbl["total_distance"].setText("(no calibration)")
            lbl["avg_speed"].setText("(no calibration)")
            lbl["max_speed"].setText("(no calibration)")
            lbl["coverage"].setText("(no calibration)")
        lbl["duration"].setText(_fmt_dur(stats.duration_s * 1000))
        lbl["n_points"].setText(str(stats.n_points))
        lbl["n_interp"].setText(str(stats.n_interpolated))
        active_pct = ep_sum.get("active_ratio", 0.0) * 100.0
        stat_pct   = 100.0 - active_pct
        lbl["active_ratio"].setText(f"{active_pct:.1f} % / {stat_pct:.1f} %")
        lbl["n_active_ep"].setText(str(ep_sum.get("n_active_episodes", 0)))
        lbl["n_stationary_ep"].setText(str(ep_sum.get("n_stationary_episodes", 0)))
        lbl["avg_active_dur"].setText(
            f"{ep_sum.get('avg_active_dur_s', 0.0):.1f} s"
        )
        lbl["avg_stat_dur"].setText(
            f"{ep_sum.get('avg_stationary_dur_s', 0.0):.1f} s"
        )

    def _update_speed_graph(self, speed_series: list, episodes: list):
        ax = self._analysis_speed_ax
        ax.clear()

        if not speed_series:
            self._analysis_speed_canvas.draw()
            return

        ts_s = [s.timestamp_ms / 1000.0 for s in speed_series]
        spds = [s.speed_m_s for s in speed_series]

        # Shade episode regions
        for ep in episodes:
            color = "#1a3a1a" if ep.kind == "active" else "#3a1a1a"
            ax.axvspan(
                ep.start_ms / 1000.0, ep.end_ms / 1000.0,
                alpha=0.35, color=color, linewidth=0,
            )

        ax.plot(ts_s, spds, color="#5588ff", linewidth=1.0, alpha=0.9)
        ax.axhline(
            self._analysis_stat_thresh,
            color="#ff8844", linewidth=0.9,
            linestyle="--", alpha=0.8,
            label=f"Threshold  {self._analysis_stat_thresh:.2f} m/s",
        )

        ax.set_xlabel("Time (s)", color="#aabbcc", fontsize=9)
        ax.set_ylabel("Speed (m/s)", color="#aabbcc", fontsize=9)
        ax.set_title(
            "Speed over Time  —  click to seek video",
            color="#ddeeff", fontsize=10,
        )
        ax.tick_params(colors="#aabbcc", labelsize=8)
        ax.set_facecolor("#0e0e22")
        self._analysis_speed_fig.patch.set_facecolor("#12122a")
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color("#3a3a5a")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            fontsize=8, labelcolor="#aabbcc",
            facecolor="#1e1e3e", edgecolor="#3a3a5a",
        )
        ax.grid(True, color="#1e1e3e", linewidth=0.5, alpha=0.6)
        self._analysis_speed_canvas.draw()

    def _on_speed_graph_click(self, event):
        """Seek video to the clicked time on the speed graph."""
        if event.xdata is None:
            return
        seek_ms = max(0, int(event.xdata * 1000))
        self._video.seek_to_ms(seek_ms)

    def _update_episodes_table(self, episodes: list):
        tbl = self._episodes_table
        tbl.setRowCount(len(episodes))
        for row, ep in enumerate(episodes):
            vals = ep.as_table_row()
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                if ep.kind == "active":
                    item.setForeground(QColor("#88dd88"))
                else:
                    item.setForeground(QColor("#dd8888"))
                tbl.setItem(row, col, item)
        tbl.resizeColumnsToContents()

    def _on_episode_row_clicked(self, index):
        """Seek video to the start of the clicked episode."""
        row = index.row()
        if 0 <= row < len(self._analysis_episodes):
            seek_ms = int(self._analysis_episodes[row].start_ms)
            self._video.seek_to_ms(seek_ms)

    def _update_regions_table(self, reg_stats: list):
        tbl = self._regions_analysis_table
        tbl.setRowCount(len(reg_stats))
        for row, rs in enumerate(reg_stats):
            for col, val in enumerate(rs.as_summary_row()):
                item = QTableWidgetItem(val)
                item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                tbl.setItem(row, col, item)
        tbl.resizeColumnsToContents()

    def _export_analysis_csv(self):
        """Export the currently visible analysis sub-view to a CSV file."""
        import csv
        view = self._analysis_stack.currentIndex()
        view_names = ["summary", "speed", "episodes", "regions", "path_map"]
        view_name  = view_names[view] if view < len(view_names) else "data"

        default_stem = Path(self._video_path).stem if self._video_path else "analysis"
        default_dir  = str(Path(self._video_path).parent) if self._video_path else ""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Analysis CSV",
            str(Path(default_dir) / f"{default_stem}_{view_name}.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return

        try:
            if view == 0:  # Summary
                rows = []
                st = getattr(self, "_analysis_stats", None)
                for key, lbl in self._summary_labels.items():
                    rows.append([key, lbl.text()])
                with open(path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows([["Metric", "Value"]] + rows)

            elif view == 1:  # Speed
                header = ["timestamp_ms", "speed_m_s", "speed_cm_s"]
                rows   = [
                    [f"{s.timestamp_ms:.1f}", f"{s.speed_m_s:.4f}", f"{s.speed_cm_s:.2f}"]
                    for s in self._analysis_speed_s
                ]
                with open(path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows([header] + rows)

            elif view == 2:  # Episodes
                header = ["Start", "End", "Duration_s", "Type",
                          "Distance_m", "Avg_Speed_m_s", "Max_Speed_m_s"]
                rows = []
                for ep in self._analysis_episodes:
                    rows.append([
                        f"{ep.start_ms:.0f}", f"{ep.end_ms:.0f}",
                        f"{ep.duration_s:.3f}", ep.kind,
                        f"{ep.distance_m:.4f}", f"{ep.avg_speed_m_s:.4f}",
                        f"{ep.max_speed_m_s:.4f}",
                    ])
                with open(path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows([header] + rows)

            elif view == 3:  # Regions summary
                header = ["Region", "Dwell_s", "Dwell_pct", "Visit_count",
                          "First_visit_ms"]
                rows = []
                for rs in self._analysis_reg_stats:
                    rows.append([
                        rs.region_name,
                        f"{rs.total_dwell_s:.3f}",
                        f"{rs.dwell_pct:.2f}",
                        str(rs.visit_count),
                        f"{rs.first_visit_ms:.0f}" if rs.first_visit_ms is not None else "",
                    ])
                with open(path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows([header] + rows)

            elif view == 4:  # Path map — export raw smoothed path points
                header = ["frame_index", "timestamp_ms", "cx_px", "cy_px",
                          "x_cm", "y_cm", "interpolated"]
                from backend.path_analyzer import smooth_path
                pts = smooth_path(self._path_points, self._analysis_smooth_window)
                rows = [
                    [p.frame_index, f"{p.timestamp_ms:.1f}",
                     f"{p.cx_px:.1f}", f"{p.cy_px:.1f}",
                     f"{p.x_cm:.2f}" if p.x_cm is not None else "",
                     f"{p.y_cm:.2f}" if p.y_cm is not None else "",
                     int(p.interpolated)]
                    for p in pts
                ]
                with open(path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows([header] + rows)

            self._status.showMessage(f"Exported: {Path(path).name}")

        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))

    # ------------------------------------------------------------------ workers

    def _cancel_worker(self):
        """Request cancellation of the currently running worker."""
        if self._worker is not None and hasattr(self._worker, "cancel"):
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)
            self._status.showMessage(
                "Cancelling — finishing current frame batch…"
            )

    def _export_path_csv(self):
        """Export the tracked path points to a CSV file."""
        default_dir = (str(Path(self._video_path).parent)
                       if self._video_path else "")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Path CSV", default_dir,
            "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            import csv as _csv
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = _csv.writer(f)
                writer.writerow([
                    "frame_index", "timestamp_ms",
                    "cx_px", "cy_px",
                    "x_cm", "y_cm",
                    "interpolated",
                ])
                for p in self._path_points:
                    writer.writerow([
                        p.frame_index,
                        f"{p.timestamp_ms:.1f}",
                        f"{p.cx_px:.1f}",
                        f"{p.cy_px:.1f}",
                        f"{p.x_cm:.1f}" if p.x_cm is not None else "",
                        f"{p.y_cm:.1f}" if p.y_cm is not None else "",
                        "1" if p.interpolated else "0",
                    ])
            n = len(self._path_points)
            self._status.showMessage(
                f"Path exported → {Path(path).name}  ({n} points)"
            )
        except Exception as exc:
            self._status.showMessage(f"Failed to export path: {exc}")

    def _on_path_live_toggled(self, state: int):
        """Switch path map between live (up-to-playhead) and full-session view."""
        self._path_live_mode = bool(state)
        if self._path_live_mode:
            self._path_map.set_time_cutoff(self._last_position_ms)
        else:
            self._path_map.set_time_cutoff(None)

    def _clear_seekbar_highlight(self):
        self._hl_tab     = -1
        self._hl_src_row = -1
        self._video.clear_seekbar_highlight()

    # ------------------------------------------------------------------ config / settings

    def _open_config_dialog(self):
        dlg = _ConfigDialog(
            current_size        = self._font_size,
            current_theme       = self._theme_name,
            current_clip_len    = self._clip_len_frames,
            current_clip_stride = self._clip_stride_frames,
            parent              = self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._font_size          = dlg.selected_size()
            self._theme_name         = dlg.selected_theme()
            self._clip_len_frames    = dlg.clip_len()
            self._clip_stride_frames = dlg.clip_stride()
            self.setStyleSheet(_make_qss(self._font_size, self._theme_name))

    # ------------------------------------------------------------------ run-settings helpers

    def _get_current_yolo_run_settings(self) -> dict:
        yolo_model = self._get_yolo_model_name()
        is_world   = "world" in yolo_model.lower()
        return {
            "yolo_model":         yolo_model,
            "frame_stride":       self._sample_spin.value(),
            "detect_all_classes": self._all_classes_chk.isChecked(),
            "world_classes":      sorted(self._world_classes) if is_world else [],
        }

    def _get_current_action_run_settings(self) -> dict:
        return {
            "enabled_models": self._get_enabled_mmaction2_models(),
            "clip_length":    self._clip_len_frames,
            "clip_stride":    self._clip_stride_frames,
        }

    def _get_current_path_run_settings(self) -> dict:
        return {
            "yolo_model": self._get_yolo_model_name(),
        }

    def _check_same_settings(self, run_type: str) -> bool:
        """
        Compare current run settings with what's stored in the JSON.
        Returns True  = proceed with run (settings differ, or user confirmed).
        Returns False = user cancelled.
        """
        if not self._video_path:
            return True
        from backend.results_format import (
            default_output_path,
            get_yolo_run_settings, get_action_run_settings, get_path_run_settings,
        )
        json_path = default_output_path(self._video_path)
        if not json_path.exists():
            return True

        if run_type == "yolo":
            stored  = get_yolo_run_settings(json_path)
            current = self._get_current_yolo_run_settings()
        elif run_type == "action":
            stored  = get_action_run_settings(json_path)
            current = self._get_current_action_run_settings()
        elif run_type == "path":
            stored  = get_path_run_settings(json_path)
            current = self._get_current_path_run_settings()
        else:
            return True

        if not stored:
            return True

        # Strip the stored date before comparing
        stored_cmp = {k: v for k, v in stored.items() if k != "date"}
        if stored_cmp != current:
            return True  # Settings differ — proceed without warning

        # Identical settings — warn the user
        msg = QMessageBox(self)
        msg.setWindowTitle("Same settings as last run")
        msg.setText(
            "The current settings match the last saved run for this video.\n"
            "Running again will overwrite the existing results."
        )
        msg.setInformativeText("Run anyway?")
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        msg.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return msg.exec() == QMessageBox.StandardButton.Yes

    # ------------------------------------------------------------------ settings persistence

    def _settings(self) -> QSettings:
        return QSettings("PlayroomAnnotator", "PlayroomAnnotator")

    def _load_settings(self):
        s = self._settings()
        # Spinboxes / numeric params
        self._conf_spin.setValue(           float(s.value("yolo_conf",   0.25)))
        self._mmaction2_conf_spin.setValue( float(s.value("action_conf", 0.50)))
        self._prox_spin.setValue(           int(  s.value("proximity",   150)))
        self._min_dur_spin.setValue(        float(s.value("min_dur",     0.2)))
        self._sample_spin.setValue(         int(  s.value("sample_rate", 1)))
        # Checkboxes
        self._all_classes_chk.setChecked( s.value("all_classes",   True,  type=bool))
        self._yolo_vis_chk.setChecked(    s.value("yolo_vis",      True,  type=bool))
        self._action_vis_chk.setChecked(  s.value("action_vis",    True,  type=bool))
        self._trail_vis_chk.setChecked(   s.value("trail_vis",     True,  type=bool))
        self._debug_toggle.setChecked(    s.value("debug_mode",    False, type=bool))
        # YOLO model
        idx = int(s.value("yolo_model_idx", 0))
        if 0 <= idx < self._yolo_model_combo.count():
            self._yolo_model_combo.setCurrentIndex(idx)
        # MMAction2 model checkboxes
        for act in self._mm_model_actions:
            key = f"mm_model_{act.data()}"
            act.setChecked(s.value(key, True, type=bool))
        # YOLO-World classes
        world_text = s.value("world_classes_text", "")
        if world_text:
            self._world_classes_edit.setText(world_text)
        # Subject classes
        subj_text = s.value("subject_classes_text", "")
        if subj_text:
            self._subject_classes_edit.setText(subj_text)
        # Appearance / recognition-window
        self._font_size          = int(  s.value("font_size",         12))
        self._theme_name         = str(  s.value("theme_name",        "Dark Blue"))
        self._clip_len_frames    = int(  s.value("clip_len_frames",   32))
        self._clip_stride_frames = int(  s.value("clip_stride_frames",16))
        # Merge controls
        self._action_merge_chk.setChecked(  s.value("action_merge",      False, type=bool))
        self._action_merge_gap_spin.setValue(int(s.value("action_merge_gap", 500)))
        # Auto-reload last-used regions file
        last_regions = s.value("last_regions_path", "")
        if last_regions and Path(last_regions).exists():
            try:
                from backend.region_config import load_region_config
                config_name, regions = load_region_config(last_regions)
                self._regions = regions
                self._video.set_regions(self._regions)
                self._update_region_ui()
            except Exception:
                pass   # silently skip — file may be corrupt or moved

    def _save_settings(self):
        s = self._settings()
        s.setValue("yolo_conf",           self._conf_spin.value())
        s.setValue("action_conf",         self._mmaction2_conf_spin.value())
        s.setValue("proximity",           self._prox_spin.value())
        s.setValue("min_dur",             self._min_dur_spin.value())
        s.setValue("sample_rate",         self._sample_spin.value())
        s.setValue("all_classes",         self._all_classes_chk.isChecked())
        s.setValue("yolo_vis",            self._yolo_vis_chk.isChecked())
        s.setValue("action_vis",          self._action_vis_chk.isChecked())
        s.setValue("trail_vis",           self._trail_vis_chk.isChecked())
        s.setValue("debug_mode",          self._debug_toggle.isChecked())
        s.setValue("yolo_model_idx",      self._yolo_model_combo.currentIndex())
        for act in self._mm_model_actions:
            s.setValue(f"mm_model_{act.data()}", act.isChecked())
        s.setValue("world_classes_text",   self._world_classes_edit.text())
        s.setValue("subject_classes_text", self._subject_classes_edit.text())
        s.setValue("font_size",            self._font_size)
        s.setValue("theme_name",           self._theme_name)
        s.setValue("clip_len_frames",      self._clip_len_frames)
        s.setValue("clip_stride_frames",   self._clip_stride_frames)
        s.setValue("action_merge",         self._action_merge_chk.isChecked())
        s.setValue("action_merge_gap",     self._action_merge_gap_spin.value())

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_action_clips(clips: list, gap_ms: float) -> list:
    """
    Merge consecutive clips that share the same action_label and whose gap
    is within gap_ms, keeping the max confidence and spanning start→end.
    Input clips do not need to be sorted (sorted internally).
    """
    if not clips:
        return clips
    from backend.action_recognizer import ActionClip
    sorted_clips = sorted(clips, key=lambda c: c.start_ms)
    merged = []
    cur = sorted_clips[0]
    for nxt in sorted_clips[1:]:
        if (nxt.action_label == cur.action_label
                and nxt.start_ms - cur.end_ms <= gap_ms):
            cur = ActionClip(
                start_ms     = cur.start_ms,
                end_ms       = max(cur.end_ms, nxt.end_ms),
                action_label = cur.action_label,
                raw_label    = cur.raw_label,
                confidence   = max(cur.confidence, nxt.confidence),
                model_name   = cur.model_name,
            )
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)
    return merged


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
    yolo_run_settings=None,
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
            yolo_run_settings=yolo_run_settings,
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
