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
#   _YoloWorker         — YOLO-only pipeline

from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, QSettings, pyqtSignal, QObject
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSplitter, QStatusBar,
    QProgressBar, QCheckBox, QFrame, QSpinBox, QDoubleSpinBox,
    QFileDialog, QInputDialog, QTabWidget, QComboBox, QLineEdit,
    QDialog, QDialogButtonBox, QFormLayout, QMessageBox,
    QScrollArea,
    QButtonGroup, QRadioButton,
    QStackedWidget, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QGridLayout, QSizePolicy, QListWidget, QListWidgetItem,
)

from ui.video_player import VideoPlayerWidget
from ui.results_table import GenericTableWidget

# DEPRECATED — regions are now persisted via QSettings recent-files lists and
# explicit .regions.json saves; the silent default-file autosave was removed
# because it clobbered the last-used config on startup.
_DEFAULT_REGIONS_PATH = Path(__file__).parent.parent / "default.regions.json"


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

class _UserGuideDialog(QDialog):
    """Scrollable in-app user guide."""

    _GUIDE_HTML = """
<h2>Getting Started</h2>
<ul>
<li>Open any video file (MP4, AVI, MOV, etc.).</li>
<li>If results from a previous session exist (a <code>_results.json</code> file next to
the video), they load automatically.</li>
<li>The last-used regions file is auto-loaded on startup.</li>
</ul>

<h2>First-Time Camera Setup</h2>
<p><b>Before Track Path gives meaningful results</b>, you need to calibrate the camera.</p>
<ol>
<li>Click <b>Calibrate Camera</b> &mdash; it asks for the room width and depth in
centimetres.</li>
<li>You then click 4 known floor points in the video frame and type their
real-world X, Y positions (in cm).</li>
<li>The coordinate system: origin = front-left corner (camera side, left),
+X = across the room width, +Y = toward the far wall.</li>
</ol>
<p>Calibration is saved automatically as a <code>.homography.json</code> file next to the
video &mdash; it auto-loads next time you open that video.</p>
<p>The <b>calibration dropdown</b> (Path &amp; Heatmap tab) lists recently used
calibrations &mdash; pick one to reuse it for a video from the same camera.
<b>Load Cal&hellip;</b> adds a calibration file to that list (useful after a
re-install or on a new PC). <b>Save Cal&hellip;</b> exports the current one.</p>
<p><b>Changing the calibration instantly reprojects the loaded path</b> &mdash; no
re-run of Track Path is needed. If a video has no calibration file beside it,
no calibration is applied (one from a previous video is never silently reused).</p>
<p><b>Without calibration:</b> Track Path still works (you get a path and point count),
but distances, speeds, episodes, and coverage will show &ldquo;no calibration&rdquo;.</p>

<h2>Running Analysis</h2>
<ul>
<li><b>Run YOLO</b> &mdash; Detects objects (child, toys, etc.) in every frame. Required
before anything else.</li>
<li><b>Track Path</b> &mdash; Uses ByteTrack to follow the child across frames, then maps
pixel positions to floor coordinates using the calibration. Requires YOLO data.</li>
<li><b>Run All</b> &mdash; Runs YOLO then Track Path automatically in sequence. The
recommended one-click option. Note it takes roughly twice as long as a single
run: detection and tracking are separate passes over the video.</li>
</ul>
<p>Results are auto-saved to <code>&lt;videoname&gt;_results.json</code> next to the video file.
If you run with the same settings as last time, the app warns you before
overwriting.</p>

<h2>Toolbar Parameters</h2>
<ul>
<li><b>All objects</b> &mdash; When checked, YOLO detects all 80 COCO classes. When
unchecked, uses a curated playroom subset. Changes require a re-run.</li>
<li><b>Sample every N frames</b> &mdash; Process every Nth frame (1 = every frame). Lower
= faster but less data. Changes require a re-run.</li>
<li><b>YOLO conf</b> &mdash; Confidence threshold for displaying detections. This is a
<b>live filter</b> &mdash; drag it and results update instantly, no re-run needed.</li>
<li><b>Min duration</b> &mdash; Minimum duration for region presence segments. Also a
<b>live filter</b>.</li>
<li><b>YOLO boxes / Trail</b> &mdash; Toggle visibility overlays on the video. Cosmetic
only.</li>
<li><b>Person only</b> &mdash; Draw only &ldquo;person&rdquo; boxes (default). Uncheck to
also draw every detected object.</li>
</ul>

<h2>YOLO Models</h2>
<p>Multiple YOLO model sizes are available (nano through extra-large). Larger =
more accurate but slower.</p>
<p><b>YOLO-World</b> is an open-vocabulary model. When selected, a text field appears
where you type comma-separated class names (e.g. &ldquo;toy, ball, slinky,
book&rdquo;). Plain English nouns work best.</p>
<p>Class lists can be saved/loaded as <code>.classes.json</code> files.</p>

<h2>Regions</h2>
<p>Regions are named rectangular zones drawn on the video frame (e.g.
&ldquo;Toy shelf&rdquo;, &ldquo;Play mat&rdquo;).</p>
<ul>
<li>Toggle <b>Edit Regions</b>, then drag a rectangle on the video. You will be
prompted to name it.</li>
<li><b>Manage&hellip;</b> opens a list of the current regions where you can rename
or delete individual ones without redrawing the rest. Remember to
<b>Save Regions&hellip;</b> afterwards to persist the changes.</li>
<li>Regions are stored in normalised coordinates (resolution-independent) &mdash; the
same config works for any video from that camera.</li>
<li>The <b>regions dropdown</b> lists recently used configs &mdash; pick one to apply
it instantly. <b>Load Regions&hellip;</b> adds a config file to the list;
<b>Save Regions&hellip;</b> writes the current regions to a
<code>.regions.json</code> file. The last-used config auto-reloads on launch.</li>
<li>The <b>Regions tab</b> shows time segments where the child was inside each
region. Once Track Path has run, these segments come from the tracked path
(same definition as the Analysis panel, so the numbers match); before that,
a YOLO bounding-box fallback is used and labelled as such.</li>
</ul>

<h2>Path &amp; Heatmap Tab</h2>
<ul>
<li>Shows the child's movement path on a top-down floor map.</li>
<li><b>Path line</b> &mdash; Blue-to-red gradient showing early-to-late trajectory.
Dotted segments = interpolated frames.</li>
<li><b>Heatmap</b> &mdash; Density overlay (blue = low, red = high dwell time).</li>
<li><b>Regions overlay</b> &mdash; Shows named regions projected onto the floor map
(requires calibration).</li>
<li><b>Live path</b> &mdash; When enabled, the map shows the path up to the current
video playhead position. Scrub the video to watch the path grow.</li>
<li><b>Blueprint&hellip;</b> &mdash; Load a top-down room image drawn under the path
and heatmap (orient it with the camera side at the bottom). It is remembered
across sessions and embedded in the Excel report. The &#10005; button removes it.</li>
<li><b>Export Path CSV</b> &mdash; Exports raw path points (frame, timestamp, pixel
coords, world coords).</li>
</ul>

<h2>Analysis Panel</h2>
<p>Click <b>Analysis</b> (enabled after Track Path completes) to open the full-window
analysis view. Press <b>Back</b> to return to the normal video + tables view.</p>

<h3>Summary</h3>
<p>Key metrics in a grid: total distance, average speed, session duration,
floor coverage %, active/stationary ratio, episode counts, average durations,
point counts. Distances, speeds, and episode metrics require calibration.</p>
<p><i>Floor coverage</i> is the percentage of a fine floor grid (50&times;120
cells) that the child's path passed through &mdash; small values are normal;
use it to compare sessions, not as an absolute measure.</p>

<h3>Speed Graph</h3>
<p>Speed over time with episode shading (green = active, red = stationary). The
orange dashed line is the stationary threshold. <b>Click anywhere on the graph to
seek the video to that moment.</b></p>

<h3>Episodes Table</h3>
<p>Lists each active and stationary period with start/end times, duration,
distance, and speed. Active rows are green, stationary are red. <b>Click a row to
seek the video to that episode's start.</b></p>

<h3>Regions Table</h3>
<p>Per-region dwell time (seconds and %), visit count, and first-visit latency.</p>

<h3>Path Map</h3>
<p>Same path map as the tab, but full-screen &mdash; useful for presentations and
screenshots.</p>

<h3>Analysis Parameters</h3>
<ul>
<li><b>Smooth</b> &mdash; Rolling-average window (frames) applied before computing
speed. Higher = smoother, less tracker noise. 1 = raw. This is the canonical
treatment: it applies to the Path tab stats, the saved JSON stats, and the
Analysis panel alike, so every view reports the same numbers.</li>
<li><b>Stationary threshold</b> &mdash; Speed below this (m/s) counts as stationary.
Default 0.10 m/s (10 cm/s). Adjust for the child's typical pace.</li>
<li><b>Min episode</b> &mdash; Episodes shorter than this (ms) get merged into neighbours.
Prevents single noisy frames from creating false micro-episodes. Default
500 ms.</li>
</ul>
<p>All parameters update the analysis in real time &mdash; experiment freely.</p>
<p><b>Export CSV</b> exports the currently visible sub-view to CSV.</p>
<p><b>Export Excel</b> writes the complete session report to one styled
<code>.xlsx</code> workbook: summary metrics, rendered path-map images (plain,
with region footprints, and heatmap &mdash; over the blueprint if one is set),
episodes, region metrics, region segments, the full speed series, and all path
points. Use these workbooks to compare sessions.</p>

<h2>Appearance</h2>
<p>Click the <b>Aa</b> button for font size (10 / 12 / 14 pt) and colour theme
(Dark Blue, Dark Neutral, Warm Dark).</p>

<h2>Tips &amp; Caveats</h2>
<ul>
<li>The calibration is <b>camera-specific</b> &mdash; if you move the camera,
recalibrate. Changing the calibration reprojects the loaded path instantly.</li>
<li>Track Path follows the person tracked for the most frames. If multiple
children are visible, results may be unreliable.</li>
<li>When loading a new video, all old data (including calibration) is cleared
automatically. If the new video has partial results (e.g. YOLO but no path
tracking), only the available data is shown.</li>
<li>All results are saved per-video. Changing filter settings (YOLO conf, min
duration) does <b>not</b> require re-running &mdash; these update live.</li>
<li>&ldquo;Sample every N frames&rdquo; applies to both Run YOLO and Track Path
&mdash; raise it for a faster preview pass on long videos.</li>
</ul>
"""

    def __init__(self, parent=None, theme: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("User Guide")
        self.setModal(True)
        self.setMinimumSize(640, 520)
        self.resize(720, 600)

        t = theme or _THEMES["Dark Blue"]
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)

        from PyQt6.QtWidgets import QTextBrowser
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(self._GUIDE_HTML)
        browser.setStyleSheet(
            f"QTextBrowser {{"
            f"  background: {t['bg_w']}; color: {t['text']};"
            f"  font-size: 12pt; padding: 16px;"
            f"  border: none;"
            f"}}"
        )
        layout.addWidget(browser, stretch=1)

        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(self.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)


class _RegionManagerDialog(QDialog):
    """
    Rename or delete individual regions without redrawing everything.
    Works on a copy; the caller reads .result_regions() after accept.
    """

    def __init__(self, regions: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Regions")
        self.setModal(True)
        self.setMinimumSize(340, 320)
        self._regions = [dict(r) for r in regions]   # working copy

        vbox = QVBoxLayout(self)
        vbox.setSpacing(8)

        self._list = QListWidget()
        self._rebuild_list()
        self._list.itemDoubleClicked.connect(lambda _: self._rename())
        vbox.addWidget(self._list, stretch=1)

        btn_row = QHBoxLayout()
        rename_btn = QPushButton("Rename…")
        rename_btn.clicked.connect(self._rename)
        delete_btn = QPushButton("Delete")
        delete_btn.setToolTip("Remove the selected region")
        delete_btn.clicked.connect(self._delete)
        btn_row.addWidget(rename_btn)
        btn_row.addWidget(delete_btn)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        vbox.addWidget(buttons)

    def _rebuild_list(self):
        self._list.clear()
        for r in self._regions:
            QListWidgetItem(r["name"], self._list)

    def _selected_index(self) -> int:
        return self._list.currentRow()

    def _rename(self):
        idx = self._selected_index()
        if idx < 0:
            return
        current = self._regions[idx]["name"]
        name, ok = QInputDialog.getText(
            self, "Rename Region", "New name:", text=current,
        )
        if ok and name.strip():
            self._regions[idx]["name"] = name.strip()
            self._rebuild_list()
            self._list.setCurrentRow(idx)

    def _delete(self):
        idx = self._selected_index()
        if idx < 0:
            return
        del self._regions[idx]
        self._rebuild_list()
        self._list.setCurrentRow(min(idx, len(self._regions) - 1))

    def result_regions(self) -> list[dict]:
        return self._regions


class _ConfigDialog(QDialog):
    """App-wide appearance settings (font size and color theme)."""

    def __init__(
        self,
        current_size: int,
        current_theme: str,
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
        self._yolo_run_settings = yolo_run_settings
        self._cancelled         = False

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
            # DEPRECATED — object-interaction pipeline removed from the UI.
            # Kept for future reference; re-enable the imports and the mapper
            # block below to restore interaction segments.
            # from backend.interaction_mapper import InteractionMapper
            # from backend.segment_builder    import SegmentBuilder

            is_world = "world" in self._yolo_model.lower()
            allowed  = None if (self._all_classes or is_world) else DEFAULT_CLASSES

            # ── Phase 1: YOLO at capture floor ──
            self.status.emit("Phase 1/2 — YOLO detection…")
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
                f"Phase 2/2 — Saving results…  "
                f"({n_person}/{n_frames} frames with person, "
                f"{n_objects} frames with objects)"
            )

            # DEPRECATED — interaction mapping (see note at the imports above).
            # filtered = _filter_frames(raw_frames, self._display_conf)
            # mapped   = InteractionMapper().map(filtered)
            # n_inter  = sum(1 for mf in mapped if mf.interacting_objects())
            # segs, _  = SegmentBuilder(self._fw, self._fh).build(mapped)
            segs: list = []

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
                yolo_run_settings=self._yolo_run_settings,
            )

            diag = (
                f"YOLO: {n_frames} frames  |  "
                f"Person: {n_person}  |  "
                f"Objects: {n_objects}"
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
        frame_stride: int = 1,
    ):
        super().__init__()
        self._path     = video_path
        self._model    = yolo_model
        self._matrix   = homography_matrix
        self._room     = room_size_cm
        self._stride   = frame_stride
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
                frame_stride=self._stride,
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
        self._worker: QObject | None = None
        self._worker_thread: QThread | None = None
        self._region_time_ranges: list = []

        # Raw FrameDetections cached for live refilter
        self._raw_frames: list | None = None
        # Named spatial regions
        self._regions: list[dict] = []
        # YOLO-World custom class list
        from backend.yolo_classes_config import DEFAULT_PLAYROOM_CLASSES
        self._world_classes: list[str] = list(DEFAULT_PLAYROOM_CLASSES)

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

        # Appearance settings (persisted via QSettings)
        self._font_size: int  = 12
        self._theme_name: str = "Dark Blue"

        # Blueprint image drawn under the path map (per-room, persisted)
        self._blueprint_path: str | None = None

        # Analysis panel state
        self._analysis_smooth_window: int   = 5     # smoothing window (frames)
        self._analysis_stat_thresh:   float = 0.10  # stationary threshold (m/s)
        self._analysis_min_ep_ms:     float = 500.0 # min episode duration (ms)
        self._analysis_episodes:      list  = []    # cached list[MovementEpisode]
        self._analysis_reg_stats:     list  = []    # cached list[RegionStats]
        self._analysis_speed_s:       list  = []    # cached list[SpeedSample]

        # "Run All" chaining flag — when True, YOLO completion auto-launches Track Path
        self._run_all_mode: bool = False

        # Debounce timer — refilter fires 250 ms after the last slider change
        self._refilter_timer = QTimer(self)
        self._refilter_timer.setSingleShot(True)
        self._refilter_timer.setInterval(250)
        self._refilter_timer.timeout.connect(self._refilter)

        self._build_ui()
        self._refresh_cal_combo()
        self._refresh_regions_combo()
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

        # ── Toolbar row 1: title + action buttons ──
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 4, 0, 4)
        toolbar.setSpacing(6)

        app_title = QLabel("Playroom Action Annotator")
        app_title.setStyleSheet("font-size:14px; font-weight:bold; color:#aad;")

        self._yolo_btn = QPushButton("Run YOLO")
        self._yolo_btn.setEnabled(False)
        self._yolo_btn.setToolTip("YOLO detection only")
        self._yolo_btn.clicked.connect(self._run_yolo)

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
        self._run_all_btn = QPushButton("▶▶  Run All")
        self._run_all_btn.setEnabled(False)
        self._run_all_btn.setToolTip(
            "Run YOLO detection, then immediately run Track Path.\n"
            "Equivalent to clicking Run YOLO → Track Path in sequence."
        )
        self._run_all_btn.clicked.connect(self._run_all)

        toolbar.addSpacing(8)
        toolbar.addWidget(self._run_all_btn)
        toolbar.addWidget(self._yolo_btn)
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
        self._help_btn = QPushButton("?")
        self._help_btn.setFixedWidth(34)
        self._help_btn.setToolTip("Open user guide")
        self._help_btn.clicked.connect(self._show_user_guide)
        toolbar.addWidget(self._help_btn)
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

        self._yolo_vis_chk = QCheckBox("YOLO boxes")
        self._yolo_vis_chk.setChecked(True)
        self._yolo_vis_chk.setStyleSheet("font-size:11px;")
        self._yolo_vis_chk.stateChanged.connect(
            lambda s: self._video.set_yolo_visible(bool(s))
        )

        self._person_only_chk = QCheckBox("Person only")
        self._person_only_chk.setChecked(True)
        self._person_only_chk.setStyleSheet("font-size:11px;")
        self._person_only_chk.setToolTip(
            "Show only 'person' bounding boxes on the video.\n"
            "Uncheck to also draw every detected object."
        )
        self._person_only_chk.stateChanged.connect(self._schedule_refilter)

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
        parambar.addWidget(_param_label("Min duration ≥"))
        parambar.addWidget(self._min_dur_spin)
        parambar.addWidget(_param_label("s"))
        parambar.addSpacing(12)
        parambar.addWidget(self._yolo_vis_chk)
        parambar.addWidget(self._person_only_chk)
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

        modelbar.addWidget(_model_label("YOLO model:"))
        modelbar.addWidget(self._yolo_model_combo)
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

        self._regions_combo = QComboBox()
        self._regions_combo.setFixedWidth(150)
        self._regions_combo.setToolTip(
            "Quick-pick a recently used region config.\n"
            "Use 'Load Regions…' to add a config file to this list."
        )
        self._regions_combo.currentIndexChanged.connect(
            self._on_regions_combo_changed
        )

        self._regions_vis_chk = QCheckBox("Show")
        self._regions_vis_chk.setChecked(True)
        self._regions_vis_chk.setStyleSheet("font-size:11px;")
        self._regions_vis_chk.setToolTip("Toggle region overlay visibility")
        self._regions_vis_chk.stateChanged.connect(
            lambda s: self._video.set_regions_visible(bool(s))
        )

        self._manage_regions_btn = QPushButton("Manage…")
        self._manage_regions_btn.setEnabled(False)
        self._manage_regions_btn.setToolTip(
            "Rename or delete individual regions."
        )
        self._manage_regions_btn.clicked.connect(self._open_region_manager)

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
        regionbar.addWidget(self._regions_combo)
        regionbar.addWidget(self._edit_regions_btn)
        regionbar.addWidget(self._region_count_lbl)
        regionbar.addWidget(self._regions_vis_chk)
        regionbar.addSpacing(8)
        regionbar.addWidget(self._manage_regions_btn)
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

        # ── Two-tab results panel ──
        self._tabs = QTabWidget()

        # Tab 0 — Region Presence
        self._region_table = GenericTableWidget(
            ["Start", "End", "Duration", "Region", "Frames"],
            stretch_col=3,
        )
        self._region_table._table.clicked.connect(
            lambda idx: self._on_table_row_clicked(idx, 0,
                self._region_table._proxy, self._region_time_ranges)
        )
        self._tabs.addTab(self._region_table, "Regions")

        # Tab 1 — Path & Heatmap
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

        self._cal_combo = QComboBox()
        self._cal_combo.setFixedWidth(150)
        self._cal_combo.setToolTip(
            "Quick-pick a recently used calibration.\n"
            "Applying one instantly reprojects the loaded path — no re-run needed.\n"
            "Use 'Load Cal…' to add a calibration file to this list."
        )
        self._cal_combo.currentIndexChanged.connect(self._on_cal_combo_changed)

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

        self._blueprint_btn = QPushButton("Blueprint…")
        self._blueprint_btn.setFixedWidth(90)
        self._blueprint_btn.setToolTip(
            "Load a top-down room image drawn under the path and heatmap.\n"
            "Orient it like this map: camera side at the bottom.\n"
            "Also embedded in the Excel report."
        )
        self._blueprint_btn.clicked.connect(self._load_blueprint_dialog)

        self._clear_blueprint_btn = QPushButton("✕")
        self._clear_blueprint_btn.setFixedWidth(24)
        self._clear_blueprint_btn.setToolTip("Remove the blueprint image")
        self._clear_blueprint_btn.setVisible(False)
        self._clear_blueprint_btn.clicked.connect(
            lambda: self._set_blueprint(None)
        )

        self._path_show_regions_chk = QCheckBox("Regions")
        self._path_show_regions_chk.setChecked(True)
        self._path_show_regions_chk.setStyleSheet("font-size:11px;")
        self._path_show_regions_chk.setToolTip(
            "Show/hide the named region overlays on the floor map.\n"
            "Requires camera calibration to reproject regions into world coordinates."
        )

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
        _path_toggle_row.addWidget(self._path_show_regions_chk)
        _path_toggle_row.addWidget(self._path_live_chk)
        _path_toggle_row.addStretch()
        _path_toggle_row.addWidget(self._path_cal_status_lbl)
        _path_toggle_row.addSpacing(8)
        _path_toggle_row.addWidget(self._cal_combo)
        _path_toggle_row.addWidget(self._save_cal_btn)
        _path_toggle_row.addWidget(self._load_cal_btn)
        _path_toggle_row.addWidget(self._blueprint_btn)
        _path_toggle_row.addWidget(self._clear_blueprint_btn)
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
        self._path_show_regions_chk.stateChanged.connect(
            lambda s: self._path_map.set_show_regions(bool(s))
        )
        self._path_live_chk.stateChanged.connect(self._on_path_live_toggled)

        self._tabs.addTab(_path_tab_container, "Path & Heatmap")  # Tab 1

        self._splitter.addWidget(self._video)
        self._splitter.addWidget(self._tabs)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        # Explicit minimums override the layouts' size hints — without these,
        # the widest toolbar row inside each pane (dropdowns + buttons) pins
        # the divider into a narrow drag range.
        self._video.setMinimumWidth(320)
        self._tabs.setMinimumWidth(240)
        root.addWidget(self._splitter, stretch=1)

        # ── Analysis panel (initially hidden, takes over full content area) ──
        self._build_analysis_panel(root)

        self._status = QStatusBar()
        self._status.setStyleSheet("color:#888; font-size:11px;")
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — open a video file to begin.")

    # ------------------------------------------------------------------ video loaded

    def _clear_all_results(self):
        """
        Wipe all analysis data from a previous video so stale results
        are never displayed against a newly loaded file.
        Called when a video loads with no existing _results.json.
        """
        # State
        self._raw_frames         = None
        self._path_points        = []
        self._path_stats         = None
        self._analysis_episodes  = []
        self._analysis_reg_stats = []
        self._analysis_speed_s   = []
        self._region_time_ranges = []
        self._capture_conf_floor = 0.05

        # Video overlays
        self._video.set_frame_detections({})
        self._video.set_trail_points([])

        # Path map
        self._path_map.set_path([])
        self._path_map.set_heatmap(None)
        self._path_stats_lbl.setText("No path data — run Track Path first.")
        self._path_stats_lbl.setStyleSheet("font-size:10px; color:#8899aa;")
        self._export_path_btn.setEnabled(False)
        self._analysis_btn.setEnabled(False)

        # Tables
        self._region_table.set_rows([])

        # Tab labels
        self._tabs.setTabText(0, "Regions")
        self._tabs.setTabText(1, "Path & Heatmap")

        # Seekbar highlight
        self._clear_seekbar_highlight()

        # Labels
        self._source_label.setText("")

    def _on_video_loaded(self, path: str):
        self._video_path = path
        self._run_all_btn.setEnabled(True)
        self._yolo_btn.setEnabled(True)
        self._track_path_btn.setEnabled(True)
        self._calibrate_btn.setEnabled(True)
        self._edit_regions_btn.setEnabled(True)
        self._load_regions_btn.setEnabled(True)

        # Clear any data from the previous video FIRST — the calibration step
        # below triggers a path reprojection, which must never see (or save)
        # the previous video's points.
        self._clear_all_results()

        # Auto-load homography if it exists beside the video; otherwise clear
        # any calibration left over from the previous video so a wrong matrix
        # is never silently applied.  (Reuse across videos = pick from the
        # calibration dropdown.)
        from backend.homography_config import default_homography_path
        hom_path = default_homography_path(path)
        if hom_path.exists():
            self._apply_calibration_file(str(hom_path), push_recent=False)
        else:
            self._clear_calibration()

        # Auto-load results JSON if it exists beside the video
        from backend.results_format import default_output_path
        json_path = default_output_path(path)
        if json_path.exists():
            self._load_json(str(json_path), auto=True)
        else:
            self._status.showMessage(f"Loaded: {path}")

    # ------------------------------------------------------------------ model selection

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
            save_class_list(
                path, config_name.strip(), self._world_classes,
            )
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
            config_name, classes, _subject_classes = load_class_list(path)
            self._world_classes = classes
            self._world_classes_edit.setText(", ".join(classes))
            msg = (
                f"Class list loaded: '{config_name}'  "
                f"({len(classes)} class{'es' if len(classes) != 1 else ''})"
            )
            self._status.showMessage(msg)
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
        self._regions_combo.blockSignals(True)
        self._regions_combo.setCurrentIndex(0)   # "(no regions)"
        self._regions_combo.blockSignals(False)
        self._schedule_refilter()
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
            self._push_recent_file("recent_region_files", path)
            self._refresh_regions_combo(select_path=path)
            self._status.showMessage(
                f"Regions saved → {Path(path).name}  ({len(self._regions)} regions)"
            )
        except Exception as exc:
            self._status.showMessage(f"Failed to save regions: {exc}")

    def _load_regions_dialog(self):
        """Browse for a region config — also adds it to the recents dropdown."""
        default_dir = str(Path(self._video_path).parent) if self._video_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Region Config", default_dir,
            "Region config (*.regions.json *.json);;All files (*)"
        )
        if path:
            self._apply_regions_file(path)

    def _apply_regions_file(self, path: str, push_recent: bool = True):
        """Load a region config file and apply it."""
        try:
            from backend.region_config import load_region_config
            config_name, regions = load_region_config(path)
        except Exception as exc:
            self._status.showMessage(f"Failed to load regions: {exc}")
            return
        self._regions = regions
        self._video.set_regions(self._regions)
        self._update_region_ui()
        self._settings().setValue("last_regions_path", path)
        if push_recent:
            self._push_recent_file("recent_region_files", path)
        self._refresh_regions_combo(select_path=path)
        self._status.showMessage(
            f"Regions loaded: '{config_name}'  "
            f"({len(regions)} region{'s' if len(regions) != 1 else ''})"
        )
        self._schedule_refilter()

    def _refresh_regions_combo(self, select_path: str | None = None):
        """Rebuild the regions dropdown from the recent-files list."""
        combo = self._regions_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("(no regions)", None)
        for p in self._recent_files("recent_region_files"):
            combo.addItem(Path(p).stem.replace(".regions", ""), p)
        idx = combo.findData(select_path) if select_path else 0
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _on_regions_combo_changed(self, index: int):
        path = self._regions_combo.itemData(index)
        if path is None:
            self._clear_regions()
        else:
            self._apply_regions_file(path)

    def _update_region_ui(self):
        n = len(self._regions)
        self._region_count_lbl.setText(f"{n} region{'s' if n != 1 else ''}")
        has = n > 0
        self._manage_regions_btn.setEnabled(has)
        self._clear_regions_btn.setEnabled(has)
        self._save_regions_btn.setEnabled(has)
        self._path_map.set_regions(self._regions)
        self._analysis_path_map.set_regions(self._regions)

    def _open_region_manager(self):
        """Rename / delete individual regions via a small list dialog."""
        dlg = _RegionManagerDialog(self._regions, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._regions = dlg.result_regions()
        self._video.set_regions(self._regions)
        self._update_region_ui()
        self._schedule_refilter()
        self._status.showMessage(
            f"Regions updated  ({len(self._regions)} total). "
            "Use 'Save Regions…' to persist the changes to the config file."
        )

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

            raw_frames = raw_frames_to_frame_detections(
                data["frame_detections"]["frames"], fps
            )

            self._raw_frames         = raw_frames
            self._analysis_fw        = fw
            self._analysis_fh        = fh
            self._capture_conf_floor = float(
                data["frame_detections"].get("capture_conf_floor", 0.05)
            )

            # Restore path tracking data if present
            path_section = data.get("path_tracking", {})
            if path_section and path_section.get("track_points"):
                try:
                    from backend.results_format import load_path_section
                    self._path_points  = load_path_section(path_section)
                    self._room_size_cm = tuple(path_section.get(
                        "room_size_cm", self._room_size_cm))
                    # If a calibration is active but the stored points are
                    # pixel-only (or from a different calibration), reproject.
                    if self._homography_matrix is not None:
                        from backend.path_analyzer import build_path_points
                        self._path_points = build_path_points(
                            self._path_points, self._homography_matrix
                        )
                    self._update_path_tab_display()
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
        Apply current YOLO conf and min-duration thresholds.
        Redraws YOLO boxes on the video and recomputes the Region Presence tab.

        Region presence uses the tracked path when available (the official
        definition — matches the Analysis panel exactly); otherwise it falls
        back to raw YOLO bounding boxes.
        """
        conf    = self._conf_spin.value()
        min_dur = int(self._min_dur_spin.value() * 1000)
        fw      = self._analysis_fw
        fh      = self._analysis_fh

        if fw == 0 or fh == 0:
            fw = self._video.frame_width
            fh = self._video.frame_height
            if fw == 0 or fh == 0:
                return
            self._analysis_fw = fw
            self._analysis_fh = fh

        if not self._raw_frames and not self._path_points:
            return

        # ── YOLO bounding-box overlay ──
        if self._raw_frames:
            self._video.set_frame_detections(
                _raw_detections_to_ui_boxes(
                    self._raw_frames, conf, fw, fh,
                    person_only=self._person_only_chk.isChecked(),
                )
            )

        # ── Tab 0: Region Presence ──
        if self._path_points:
            region_segs = _compute_region_segments_from_path(
                self._smoothed_points(), self._regions, fw, fh, min_dur
            )
            source = "path-based"
        elif self._raw_frames:
            region_segs = _compute_region_segments(
                self._raw_frames, conf, self._regions, fw, fh, min_dur
            )
            source = "bbox-based — run Track Path for path-based segments"
        else:
            region_segs = []
            source = ""

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
        n_reg = len(region_segs)
        self._tabs.setTabText(0, f"Regions ({n_reg})")

        floor     = self._capture_conf_floor
        floor_warn = (
            f"  ⚠ YOLO conf below capture floor ({floor:.2f})"
            if conf < floor else ""
        )
        self._source_label.setText(
            f"YOLO conf ≥ {conf:.2f}  ·  min {self._min_dur_spin.value():.1f} s  —  "
            f"{n_reg} region segment{'s' if n_reg != 1 else ''}"
            f" ({source}){floor_warn}"
        )
        self._status.showMessage(
            f"{n_reg} region segment{'s' if n_reg != 1 else ''}  "
            f"(YOLO ≥ {conf:.2f}, min {self._min_dur_spin.value():.1f} s)"
        )

    # ------------------------------------------------------------------ pipeline launchers

    def _run_all(self):
        """Run YOLO then automatically chain into Track Path on completion."""
        if not self._check_same_settings("yolo"):
            return
        self._run_all_mode = True
        self._launch_worker(_YoloWorker)

    def _run_yolo(self):
        self._run_all_mode = False   # standalone YOLO — no chaining
        if not self._check_same_settings("yolo"):
            return
        self._launch_worker(_YoloWorker)

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

    def _on_finished(self, raw_frames, _action_clips, diagnostic: str):
        self._set_running(False)
        if diagnostic == "Cancelled":
            self._run_all_mode = False
            self._status.showMessage("Analysis cancelled.")
            return

        self._raw_frames         = raw_frames
        self._analysis_fw        = self._video.frame_width
        self._analysis_fh        = self._video.frame_height
        self._capture_conf_floor = 0.05

        from backend.results_format import default_output_path
        json_path = default_output_path(self._video_path)
        self._source_label.setText(
            f"YOLO complete  ·  auto-saved → {json_path.name}  ·  refiltering…"
        )
        self._status.showMessage(f"YOLO complete — {diagnostic}")
        self._refilter()

        # Chain into Track Path if "Run All" was requested.
        # Deferred via QTimer: this slot runs before the worker thread's quit()
        # slot, so a direct call would see the thread still running and abort.
        if self._run_all_mode:
            self._run_all_mode = False
            self._status.showMessage("YOLO done — starting Track Path…")
            QTimer.singleShot(
                100, lambda: self._run_path_tracking(skip_settings_check=True)
            )

    # ------------------------------------------------------------------ path tracking

    def _run_path_tracking(self, skip_settings_check: bool = False):
        """Launch the ByteTrack path-tracking worker.

        skip_settings_check is True when chained from Run All — no dialog
        mid-chain, and the previous worker thread may still be winding down,
        in which case we retry shortly instead of silently aborting.
        """
        if not self._video_path:
            self._status.showMessage("No video loaded.")
            return
        if self._worker_thread and self._worker_thread.isRunning():
            if skip_settings_check:
                QTimer.singleShot(
                    100, lambda: self._run_path_tracking(skip_settings_check=True)
                )
            return
        if not skip_settings_check and not self._check_same_settings("path"):
            return

        yolo_model = self._get_yolo_model_name()
        self._set_running(True)
        self._worker = _PathTrackingWorker(
            self._video_path,
            yolo_model=yolo_model,
            homography_matrix=self._homography_matrix,
            room_size_cm=self._room_size_cm,
            frame_stride=self._sample_spin.value(),
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
        """Receive completed path data, refresh all path views, save JSON."""
        self._set_running(False)
        if not path_pts and diagnostic == "Cancelled":
            self._status.showMessage("Path tracking cancelled.")
            return

        # Raw (unsmoothed) points are the stored source of truth; the
        # smoothing window is applied consistently at display time.
        self._path_points = path_pts
        self._update_path_tab_display()
        self._status.showMessage(f"Path tracking complete — {diagnostic}")
        self._save_path_json()
        # Region Presence tab switches to path-based segments once path exists
        self._refilter()

    def _smoothed_points(self) -> list:
        """The canonical smoothed path used by every display and metric."""
        from backend.path_analyzer import smooth_path
        return smooth_path(self._path_points, self._analysis_smooth_window)

    def _update_path_tab_display(self):
        """Refresh trail, map, heatmap, and the stats label from _path_points."""
        from backend.path_analyzer import (
            compute_heatmap, heatmap_to_rgba, compute_stats,
        )
        pts = self._smoothed_points()

        # Trail overlay on the video (raw pixel positions of the detections)
        trail = [(p.frame_index, p.cx_px, p.cy_px) for p in self._path_points]
        self._video.set_trail_points(trail)

        self._path_map.set_room_size(*self._room_size_cm)
        self._path_map.set_path(pts)
        if self._path_live_mode:
            self._path_map.set_time_cutoff(self._last_position_ms)

        heatmap_grid = compute_heatmap(pts, self._room_size_cm)
        self._path_map.set_heatmap(
            heatmap_to_rgba(heatmap_grid) if heatmap_grid.max() > 0 else None
        )

        self._path_stats = compute_stats(pts)
        st = self._path_stats
        if st.n_points == 0:
            stats_text = "No path data — run Track Path first."
        elif st.calibrated:
            stats_text = (
                f"Distance: {st.total_distance_m:.1f} m  ·  "
                f"Avg speed: {st.avg_speed_m_s:.2f} m/s  ·  "
                f"Duration: {_fmt_dur(st.duration_s * 1000)}  ·  "
                f"{st.n_points} pts"
                + (f"  ({st.n_interpolated} interp.)" if st.n_interpolated else "")
                + f"  ·  smooth {self._analysis_smooth_window}"
            )
        else:
            stats_text = (
                f"{st.n_points} track pts  ·  "
                f"Duration: {_fmt_dur(st.duration_s * 1000)}  ·  "
                "Calibrate camera for real-world distances"
            )
        self._path_stats_lbl.setText(stats_text)
        self._path_stats_lbl.setStyleSheet("font-size:10px; color:#aabbcc;")

        n = len(self._path_points)
        self._export_path_btn.setEnabled(n > 0)
        self._tabs.setTabText(1, f"Path & Heatmap ({n} pts)" if n else "Path & Heatmap")
        self._analysis_btn.setEnabled(n > 0)

    def _save_path_json(self):
        """Upsert the path section (raw points + smoothed stats) into the JSON."""
        if not (self._video_path and self._path_points):
            return
        from backend.results_format import (
            default_output_path, upsert_path_section,
        )
        json_path = default_output_path(self._video_path)
        try:
            upsert_path_section(
                output_path   = json_path,
                video_path    = self._video_path,
                fps           = self._video.fps,
                fw            = self._video.frame_width,
                fh            = self._video.frame_height,
                total_frames  = self._video.total_frames,
                track_points  = self._path_points,
                room_size_cm  = self._room_size_cm,
                run_settings  = self._get_current_path_run_settings(),
                smooth_window = self._analysis_smooth_window,
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
            self._path_map.set_homography(matrix, self._video.frame_width, self._video.frame_height)
            self._analysis_path_map.set_homography(matrix, self._video.frame_width, self._video.frame_height)
            self._path_cal_status_lbl.setText(
                f"Calibrated  ({room_size_cm[0]/100:.1f} m × "
                f"{room_size_cm[1]/100:.1f} m)"
            )
            self._path_cal_status_lbl.setStyleSheet(
                "font-size:10px; color:#88cc88;"
            )
            self._push_recent_file("recent_cal_files", str(path))
            self._refresh_cal_combo(select_path=str(path))
            self._reproject_path()
            self._status.showMessage(
                f"Calibration saved → {path.name}"
                + ("" if self._path_points
                   else "  —  Run 'Track Path' to see the floor map.")
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
            self._push_recent_file("recent_cal_files", path)
            self._refresh_cal_combo(select_path=path)
            self._status.showMessage(
                f"Calibration saved → {Path(path).name}"
            )
        except Exception as exc:
            self._status.showMessage(f"Failed to save calibration: {exc}")

    def _load_cal_dialog(self):
        """Browse for a homography file — also adds it to the recents dropdown."""
        default_dir = (str(Path(self._video_path).parent)
                       if self._video_path else "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration", default_dir,
            "Homography (*.homography.json *.json);;All files (*)"
        )
        if path:
            self._apply_calibration_file(path)

    def _apply_calibration_file(self, path: str, push_recent: bool = True):
        """Load a homography file, apply it, and reproject any loaded path."""
        try:
            from backend.homography_config import load_homography
            data = load_homography(path)
        except Exception as exc:
            self._status.showMessage(f"Failed to load calibration: {exc}")
            return

        self._homography_matrix = data["matrix"]
        self._room_size_cm      = tuple(data["room_size_cm"])
        self._cal_pixel_pts     = data["pixel_points"]
        self._cal_world_pts     = data["world_points_cm"]
        self._path_map.set_room_size(*self._room_size_cm)
        self._path_map.set_homography(
            data["matrix"], self._video.frame_width, self._video.frame_height
        )
        self._analysis_path_map.set_homography(
            data["matrix"], self._video.frame_width, self._video.frame_height
        )
        self._path_cal_status_lbl.setText(
            f"Calibration: {Path(path).stem.replace('.homography', '')}  "
            f"({self._room_size_cm[0]/100:.1f} m × "
            f"{self._room_size_cm[1]/100:.1f} m)"
        )
        self._path_cal_status_lbl.setStyleSheet(
            "font-size:10px; color:#88cc88;"
        )
        self._status.showMessage(f"Calibration loaded: {Path(path).name}")
        if push_recent:
            self._push_recent_file("recent_cal_files", path)
        self._refresh_cal_combo(select_path=path)
        self._reproject_path()

    def _clear_calibration(self):
        """Remove the active calibration (no homography for this video)."""
        self._homography_matrix = None
        self._cal_pixel_pts     = []
        self._cal_world_pts     = []
        self._path_map.set_homography(None, 0, 0)
        self._analysis_path_map.set_homography(None, 0, 0)
        self._path_cal_status_lbl.setText("No calibration")
        self._path_cal_status_lbl.setStyleSheet("font-size:10px; color:#8899aa;")
        self._refresh_cal_combo(select_path=None)

    def _reproject_path(self):
        """
        Re-apply the current homography to the loaded path points and refresh
        every path-derived view.  Called whenever the calibration changes —
        pixel coordinates are the source of truth, so no re-run is needed.
        """
        if not self._path_points:
            return
        from backend.path_analyzer import build_path_points
        self._path_points = build_path_points(
            self._path_points, self._homography_matrix
        )
        self._update_path_tab_display()
        if self._analysis_panel.isVisible():
            self._refresh_analysis()
        self._save_path_json()
        self._status.showMessage(
            "Calibration applied — path reprojected to floor coordinates."
        )

    # ------------------------------------------------------------------ recent files

    def _recent_files(self, key: str) -> list[str]:
        """Return the QSettings recent-file list for *key* (existing files only)."""
        raw = self._settings().value(key, [])
        if isinstance(raw, str):
            raw = [raw]
        return [p for p in (raw or []) if p and Path(p).exists()]

    def _push_recent_file(self, key: str, path: str, limit: int = 10):
        """Prepend *path* to the QSettings recent-file list for *key*."""
        paths = [p for p in self._recent_files(key) if p != path]
        paths.insert(0, path)
        self._settings().setValue(key, paths[:limit])

    def _refresh_cal_combo(self, select_path: str | None = None):
        """Rebuild the calibration dropdown from the recent-files list."""
        combo = self._cal_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("(no calibration)", None)
        for p in self._recent_files("recent_cal_files"):
            combo.addItem(Path(p).stem.replace(".homography", ""), p)
        idx = combo.findData(select_path) if select_path else 0
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _on_cal_combo_changed(self, index: int):
        path = self._cal_combo.itemData(index)
        if path is None:
            self._clear_calibration()
            self._reproject_path()
        else:
            self._apply_calibration_file(path)

    # ------------------------------------------------------------------ blueprint

    def _load_blueprint_dialog(self):
        default_dir = (str(Path(self._video_path).parent)
                       if self._video_path else "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Blueprint Image", default_dir,
            "Images (*.png *.jpg *.jpeg *.bmp);;All files (*)"
        )
        if path:
            self._set_blueprint(path)

    def _set_blueprint(self, path: str | None):
        """Apply (or clear) the blueprint on both path maps and persist it."""
        self._blueprint_path = path
        self._path_map.set_blueprint(path)
        self._analysis_path_map.set_blueprint(path)
        self._settings().setValue("blueprint_path", path or "")
        self._clear_blueprint_btn.setVisible(bool(path))
        self._blueprint_btn.setText("Blueprint ✓" if path else "Blueprint…")
        self._status.showMessage(
            f"Blueprint: {Path(path).name}" if path else "Blueprint removed."
        )

    def _on_error(self, message: str):
        self._run_all_mode = False   # abort any pending chain
        self._set_running(False)
        self._status.showMessage(f"Error: {message}")
        print(f"[Worker] Error: {message}")

    def _set_running(self, running: bool):
        has_video = self._video_path is not None
        self._yolo_btn.setEnabled(not running and has_video)
        self._track_path_btn.setEnabled(not running and has_video)
        self._calibrate_btn.setEnabled(not running and has_video)
        self._load_json_btn.setEnabled(not running)
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
        if self._region_time_ranges:
            self._region_table.highlight_row_at(position_ms, self._region_time_ranges)

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

        self._analysis_xlsx_btn = QPushButton("Export Excel…")
        self._analysis_xlsx_btn.setEnabled(False)
        self._analysis_xlsx_btn.setToolTip(
            "Export the full session report to one styled .xlsx workbook:\n"
            "summary, path map image, episodes, regions, speed, path points."
        )
        self._analysis_xlsx_btn.clicked.connect(self._export_analysis_xlsx)
        nav.addWidget(self._analysis_xlsx_btn)

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
        self._analysis_speed_fig.patch.set_facecolor(self._theme()["bg"])
        self._analysis_speed_ax = self._analysis_speed_fig.add_subplot(111)
        self._analysis_speed_canvas = FigureCanvas(self._analysis_speed_fig)
        self._analysis_speed_canvas.mpl_connect(
            "button_press_event", self._on_speed_graph_click
        )
        self._analysis_stack.addWidget(self._analysis_speed_canvas)

        # Page 2 — Episodes table
        self._episodes_table = QTableWidget(0, 6)
        self._episodes_table.setHorizontalHeaderLabels(
            ["Start", "End", "Duration", "Type", "Distance", "Avg Speed"]
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
        self._analysis_show_reg_chk = QCheckBox("Regions")
        self._analysis_show_reg_chk.setChecked(True)
        self._analysis_show_reg_chk.setToolTip(
            "Show region floor footprints (requires camera calibration)."
        )
        self._analysis_show_reg_chk.stateChanged.connect(
            lambda s: self._analysis_path_map.set_show_regions(bool(s))
        )
        path_ctrl.addWidget(self._analysis_show_path_chk)
        path_ctrl.addWidget(self._analysis_show_hm_chk)
        path_ctrl.addWidget(self._analysis_show_reg_chk)
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
        smooth_changed = (
            self._analysis_smooth_window != self._analysis_smooth_spin.value()
        )
        self._analysis_smooth_window = self._analysis_smooth_spin.value()
        self._analysis_stat_thresh   = self._analysis_thresh_spin.value()
        self._analysis_min_ep_ms     = float(self._analysis_minep_spin.value())
        # Smoothing is the canonical treatment — it applies to the Path tab,
        # the stored JSON stats, and the analysis panel alike.
        if smooth_changed and self._path_points:
            self._update_path_tab_display()
            self._save_path_json()
        if self._analysis_panel.isVisible():
            self._refresh_analysis()

    def _refresh_analysis(self):
        """Smooth the path, recompute all metrics, update every sub-view."""
        if not self._path_points:
            return

        from backend.path_analyzer import (
            compute_stats, compute_speed_series,
            compute_episodes, episode_summary,
            compute_region_stats, compute_coverage_pct,
            compute_heatmap, heatmap_to_rgba,
        )

        pts = self._smoothed_points()

        stats    = compute_stats(pts)
        # Speeds and episodes are meaningless without real-world scale —
        # show "requires calibration" instead of a fake 100%-stationary session.
        if stats.calibrated:
            speed_s  = compute_speed_series(pts)
            episodes = compute_episodes(
                pts,
                stationary_threshold_m_s = self._analysis_stat_thresh,
                min_episode_ms           = self._analysis_min_ep_ms,
            )
        else:
            speed_s  = []
            episodes = []
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
        self._analysis_xlsx_btn.setEnabled(True)
        self._analysis_coverage = coverage
        self._analysis_ep_sum   = ep_sum

    def _update_analysis_summary(self, stats, ep_sum: dict, coverage: float):
        lbl = self._summary_labels
        lbl["duration"].setText(_fmt_dur(stats.duration_s * 1000))
        lbl["n_points"].setText(str(stats.n_points))
        lbl["n_interp"].setText(str(stats.n_interpolated))

        if not stats.calibrated:
            # No real-world scale: distance, speed, and episode metrics
            # would be fabricated numbers — say so instead.
            for key in ("total_distance", "avg_speed", "coverage",
                        "active_ratio", "n_active_ep", "n_stationary_ep",
                        "avg_active_dur", "avg_stat_dur"):
                lbl[key].setText("(no calibration)")
            return

        lbl["total_distance"].setText(f"{stats.total_distance_m:.2f} m")
        lbl["avg_speed"].setText(f"{stats.avg_speed_m_s:.3f} m/s")
        lbl["coverage"].setText(f"{coverage:.1f} %")
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

    def _theme(self) -> dict:
        """The active theme's colour dict."""
        return _THEMES.get(self._theme_name, _THEMES["Dark Blue"])

    def _update_speed_graph(self, speed_series: list, episodes: list):
        t  = self._theme()
        ax = self._analysis_speed_ax
        ax.clear()
        ax.set_facecolor(t["parambar"])
        self._analysis_speed_fig.patch.set_facecolor(t["bg"])

        if not speed_series:
            msg = ("Calibrate the camera to compute speeds"
                   if self._path_points else "No path data")
            ax.text(
                0.5, 0.5, msg,
                transform=ax.transAxes,
                ha="center", va="center",
                color=t["text2"], fontsize=11,
            )
            ax.set_xticks([]); ax.set_yticks([])
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

        ax.plot(ts_s, spds, color=t["accent2"], linewidth=1.0, alpha=0.9)
        ax.axhline(
            self._analysis_stat_thresh,
            color="#ff8844", linewidth=0.9,
            linestyle="--", alpha=0.8,
            label=f"Threshold  {self._analysis_stat_thresh:.2f} m/s",
        )

        ax.set_xlabel("Time (s)", color=t["text2"], fontsize=9)
        ax.set_ylabel("Speed (m/s)", color=t["text2"], fontsize=9)
        ax.set_title(
            "Speed over Time  —  click to seek video",
            color=t["text"], fontsize=10,
        )
        ax.tick_params(colors=t["text2"], labelsize=8)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color(t["border2"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            fontsize=8, labelcolor=t["text2"],
            facecolor=t["bg_alt"], edgecolor=t["border2"],
        )
        ax.grid(True, color=t["bg_alt"], linewidth=0.5, alpha=0.6)
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
                          "Distance_m", "Avg_Speed_m_s"]
                rows = []
                for ep in self._analysis_episodes:
                    rows.append([
                        f"{ep.start_ms:.0f}", f"{ep.end_ms:.0f}",
                        f"{ep.duration_s:.3f}", ep.kind,
                        f"{ep.distance_m:.4f}", f"{ep.avg_speed_m_s:.4f}",
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

    def _export_analysis_xlsx(self):
        """Export the complete session report to one styled .xlsx workbook."""
        if not self._path_points:
            return
        # Make sure the cached analysis data reflects the current parameters
        self._refresh_analysis()

        default_stem = (Path(self._video_path).stem
                        if self._video_path else "session")
        default_dir  = (str(Path(self._video_path).parent)
                        if self._video_path else "")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Session Report (Excel)",
            str(Path(default_dir) / f"{default_stem}_report.xlsx"),
            "Excel workbook (*.xlsx)",
        )
        if not path:
            return
        if not path.endswith(".xlsx"):
            path += ".xlsx"

        try:
            from backend.excel_export import export_session_excel

            pts     = self._smoothed_points()
            stats   = self._analysis_stats
            ep_sum  = getattr(self, "_analysis_ep_sum", {})
            fw, fh  = self._analysis_fw, self._analysis_fh
            min_dur = int(self._min_dur_spin.value() * 1000)
            region_segments = _compute_region_segments_from_path(
                pts, self._regions, fw, fh, min_dur
            ) if fw > 0 else []

            from backend.path_analyzer import project_region_shapes
            region_shapes = project_region_shapes(
                self._regions, self._homography_matrix,
                fw, fh, self._room_size_cm,
            )

            export_session_excel(
                xlsx_path       = path,
                video_name      = (Path(self._video_path).name
                                   if self._video_path else "unknown"),
                stats           = stats,
                ep_sum          = ep_sum,
                coverage_pct    = getattr(self, "_analysis_coverage", 0.0),
                episodes        = self._analysis_episodes,
                reg_stats       = self._analysis_reg_stats,
                region_segments = region_segments,
                speed_series    = self._analysis_speed_s,
                path_points     = pts,
                room_size_cm    = self._room_size_cm,
                analysis_params = {
                    "Smoothing window (frames)":     self._analysis_smooth_window,
                    "Stationary threshold (m/s)":    self._analysis_stat_thresh,
                    "Min episode duration (ms)":     int(self._analysis_min_ep_ms),
                    "Min region segment (s)":        self._min_dur_spin.value(),
                },
                run_info        = {
                    "YOLO model":     self._get_yolo_model_name(),
                    "Frame stride":   self._sample_spin.value(),
                },
                blueprint_path  = self._blueprint_path,
                region_shapes   = region_shapes,
            )
            self._status.showMessage(f"Excel report exported: {Path(path).name}")
        except Exception as exc:
            import traceback; traceback.print_exc()
            QMessageBox.warning(self, "Excel export failed", str(exc))

    # ------------------------------------------------------------------ workers

    def _cancel_worker(self):
        """Request cancellation of the currently running worker."""
        self._run_all_mode = False   # abort any pending chain
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
            current_size  = self._font_size,
            current_theme = self._theme_name,
            parent        = self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._font_size  = dlg.selected_size()
            self._theme_name = dlg.selected_theme()
            self.setStyleSheet(_make_qss(self._font_size, self._theme_name))

    def _show_user_guide(self):
        _UserGuideDialog(parent=self, theme=self._theme()).exec()

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

    def _get_current_path_run_settings(self) -> dict:
        return {
            "yolo_model":   self._get_yolo_model_name(),
            "frame_stride": self._sample_spin.value(),
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
            get_yolo_run_settings, get_path_run_settings,
        )
        json_path = default_output_path(self._video_path)
        if not json_path.exists():
            return True

        if run_type == "yolo":
            stored  = get_yolo_run_settings(json_path)
            current = self._get_current_yolo_run_settings()
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
        self._conf_spin.setValue(  float(s.value("yolo_conf",   0.25)))
        self._min_dur_spin.setValue(float(s.value("min_dur",    0.2)))
        self._sample_spin.setValue( int(  s.value("sample_rate", 1)))
        # Checkboxes
        self._all_classes_chk.setChecked(s.value("all_classes", True, type=bool))
        self._yolo_vis_chk.setChecked(   s.value("yolo_vis",    True, type=bool))
        self._person_only_chk.setChecked(s.value("person_only", True, type=bool))
        self._trail_vis_chk.setChecked(  s.value("trail_vis",   True, type=bool))
        # YOLO model
        idx = int(s.value("yolo_model_idx", 0))
        if 0 <= idx < self._yolo_model_combo.count():
            self._yolo_model_combo.setCurrentIndex(idx)
        # YOLO-World classes
        world_text = s.value("world_classes_text", "")
        if world_text:
            self._world_classes_edit.setText(world_text)
        # Appearance
        self._font_size  = int(s.value("font_size",  12))
        self._theme_name = str(s.value("theme_name", "Dark Blue"))
        # Analysis smoothing window (canonical treatment — affects all metrics)
        self._analysis_smooth_window = int(s.value("analysis_smooth", 5))
        self._analysis_smooth_spin.setValue(self._analysis_smooth_window)
        # Auto-reload last-used regions file
        last_regions = s.value("last_regions_path", "")
        if last_regions and Path(last_regions).exists():
            self._apply_regions_file(last_regions)
        # Blueprint image (room-specific, survives restarts)
        bp = s.value("blueprint_path", "")
        if bp and Path(bp).exists():
            self._set_blueprint(bp)

    def _save_settings(self):
        s = self._settings()
        s.setValue("yolo_conf",          self._conf_spin.value())
        s.setValue("min_dur",            self._min_dur_spin.value())
        s.setValue("sample_rate",        self._sample_spin.value())
        s.setValue("all_classes",        self._all_classes_chk.isChecked())
        s.setValue("yolo_vis",           self._yolo_vis_chk.isChecked())
        s.setValue("person_only",        self._person_only_chk.isChecked())
        s.setValue("trail_vis",          self._trail_vis_chk.isChecked())
        s.setValue("yolo_model_idx",     self._yolo_model_combo.currentIndex())
        s.setValue("world_classes_text", self._world_classes_edit.text())
        s.setValue("font_size",          self._font_size)
        s.setValue("theme_name",         self._theme_name)
        s.setValue("analysis_smooth",    self._analysis_smooth_window)

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)


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
    stride, display_conf, all_classes,
    yolo_run_settings=None,
) -> "Path | None":
    """Save _results.json next to the video.  Never raises — logs on failure."""
    try:
        from backend.results_format import (
            save_results, default_output_path, CAPTURE_CONF_FLOOR,
        )
        output_path = default_output_path(video_path)
        settings = {
            "frame_sample_rate":      stride,
            "capture_conf_floor":     CAPTURE_CONF_FLOOR,
            "display_conf_threshold": display_conf,
            "all_classes_mode":       all_classes,
            "min_segment_duration_sec": 0.2,
        }
        modules_used = {
            "yolo":            True,
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
            action_clips=[],
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


def _raw_detections_to_ui_boxes(
    raw_frames: list, conf: float, fw: int, fh: int,
    person_only: bool = False,
) -> dict:
    """Convert raw FrameDetection list directly to video-overlay box format."""
    result: dict[int, list] = {}
    for fd in raw_frames:
        boxes = []
        for d in fd.detections:
            if d.confidence < conf:
                continue
            if person_only and d.label != "person":
                continue
            color = (255, 130, 50) if d.label == "person" else (80, 160, 255)
            boxes.append((d.label, color,
                          d.x1 / fw, d.y1 / fh,
                          (d.x2 - d.x1) / fw, (d.y2 - d.y1) / fh))
        if boxes:
            result[fd.frame_index] = boxes
    return result


# Single shared timestamp formatter (H:MM:SS.d) — one definition project-wide.
from backend.path_analyzer import _fmt_ms  # noqa: E402


def _fmt_dur(dur_ms: float) -> str:
    """Format a duration in ms as a readable string (e.g. '4.2 s')."""
    s = dur_ms / 1000.0
    if s < 60:
        return f"{s:.1f} s"
    m = int(s) // 60
    s = s % 60
    return f"{m}m {s:.0f}s"


def _foot_region_overlap(person, rx1: float, ry1: float, rx2: float, ry2: float) -> float:
    """
    Horizontal overlap length between the person's foot-line and a region.
    Returns 0 if the foot y (person.y2) is outside the region's y-range.
    """
    if not (ry1 <= person.y2 <= ry2):
        return 0.0
    return max(0.0, min(person.x2, rx2) - max(person.x1, rx1))


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
    For each frame, assign the person to the region whose x-span overlaps
    the most with the person's foot-line (bottom bbox edge).  When multiple
    regions qualify, the one with the greatest horizontal overlap wins —
    so the assignment shifts naturally as the person crosses a boundary.

    Returns a list of dicts sorted by start_ms:
        {"start_ms", "end_ms", "region", "frames"}
    """
    if not regions or not raw_frames:
        return []

    # Pre-compute pixel coords for each region once
    reg_boxes = [
        (r["name"],
         r["nx"] * fw,
         r["ny"] * fh,
         (r["nx"] + r["nw"]) * fw,
         (r["ny"] + r["nh"]) * fh)
        for r in regions
    ]

    # Per-region open-segment state
    state = {name: {"start": None, "end": None, "frames": 0}
             for name, *_ in reg_boxes}

    segments: list[dict] = []

    def _close(rname: str) -> None:
        s = state[rname]
        if s["start"] is None:
            return
        dur = s["end"] - s["start"]
        if dur >= min_dur_ms:
            segments.append({
                "start_ms": s["start"],
                "end_ms":   s["end"],
                "region":   rname,
                "frames":   s["frames"],
            })
        s["start"] = s["end"] = None
        s["frames"] = 0

    for fd in raw_frames:
        persons = [
            d for d in fd.detections
            if d.label == "person" and d.confidence >= conf
        ]

        # Use the largest person bbox when multiple are detected
        person = max(persons, key=lambda d: d.area) if persons else None

        # Winner = region with greatest foot-line horizontal overlap
        best_region: str | None = None
        if person is not None:
            best_overlap = 0.0
            for rname, rx1, ry1, rx2, ry2 in reg_boxes:
                overlap = _foot_region_overlap(person, rx1, ry1, rx2, ry2)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_region = rname

        for rname, *_ in reg_boxes:
            s = state[rname]
            if rname == best_region:
                if s["start"] is None:
                    s["start"]  = fd.timestamp_ms
                    s["frames"] = 0
                s["end"]     = fd.timestamp_ms
                s["frames"] += 1
            elif s["start"] is not None:
                gap = fd.timestamp_ms - s["end"]
                if gap > gap_tolerance_ms:
                    _close(rname)

    # Close any segments still open at end of video
    for rname, *_ in reg_boxes:
        _close(rname)

    return sorted(segments, key=lambda s: s["start_ms"])


def _compute_region_segments_from_path(
    path_points: list,
    regions: list[dict],
    fw: int,
    fh: int,
    min_dur_ms: int,
) -> list[dict]:
    """
    Region presence segments from the tracked path (the official definition —
    identical containment test to path_analyzer.compute_region_stats, so the
    Regions tab matches the Analysis panel's region metrics exactly).

    A point is inside a region when its pixel centroid (bottom-center of the
    person bbox = feet) falls within the region rectangle.

    Returns a list of dicts sorted by start_ms:
        {"start_ms", "end_ms", "region", "frames"}
    """
    if not regions or not path_points:
        return []

    segments: list[dict] = []

    for region in regions:
        rx0 = region["nx"]                  * fw
        ry0 = region["ny"]                  * fh
        rx1 = (region["nx"] + region["nw"]) * fw
        ry1 = (region["ny"] + region["nh"]) * fh

        in_region  = False
        entry_ms   = 0.0
        n_points   = 0
        last_ms    = 0.0

        def _close():
            if entry_ms is not None and last_ms - entry_ms >= min_dur_ms:
                segments.append({
                    "start_ms": entry_ms,
                    "end_ms":   last_ms,
                    "region":   region["name"],
                    "frames":   n_points,
                })

        for p in path_points:
            inside = (rx0 <= p.cx_px <= rx1 and ry0 <= p.cy_px <= ry1)
            if inside:
                if not in_region:
                    in_region = True
                    entry_ms  = p.timestamp_ms
                    n_points  = 0
                last_ms   = p.timestamp_ms
                n_points += 1
            elif in_region:
                in_region = False
                _close()

        if in_region:
            _close()

    return sorted(segments, key=lambda s: s["start_ms"])
