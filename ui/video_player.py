# ui/video_player.py
# VideoPlayerWidget: OpenCV frame reading, playback, seek bar, bbox overlay,
# and interactive region-capture (rubber-band drawing of named ROIs).
#
# Overlay modes (most-recently-set wins):
#   • Time-range mode  — set_bounding_boxes()    — mock data
#   • Frame-index mode — set_frame_detections()  — real YOLO output
#
# Region editing:
#   Call set_region_edit_mode(True) to enter drawing mode.
#   The user rubber-bands a rectangle on the video frame.
#   On mouse release the widget emits region_drawn(nx, ny, nw, nh)
#   with normalised coordinates (0–1), accounting for KeepAspectRatio
#   letterboxing.  The caller (MainWindow) prompts for a name and calls
#   set_regions() to display the overlay.
#
# Flicker fix: between sampled frames, boxes show the nearest preceding
# sample rather than going blank (sorted index + bisect).

import bisect
import time
import cv2
from PyQt6.QtCore import Qt, QTimer, QRect, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QSizePolicy, QFileDialog, QRubberBand,
)

# ---------------------------------------------------------------------------
# Region colour palette (cycles if there are more than 8 regions)
# ---------------------------------------------------------------------------

_REGION_COLORS = [
    (255, 100, 100),   # red
    (100, 220, 100),   # green
    (100, 140, 255),   # blue
    (255, 220,  50),   # yellow
    (255, 100, 220),   # pink
    ( 80, 220, 220),   # cyan
    (255, 160,  40),   # orange
    (180, 100, 255),   # purple
]


# ---------------------------------------------------------------------------
# _HighlightSlider — QSlider with an optional highlighted time-range band
# ---------------------------------------------------------------------------

class _HighlightSlider(QSlider):
    """
    Horizontal seek slider that can display a coloured band over a time range.
    The band is rendered via a QSS gradient so it plays nicely with the app
    stylesheet — no custom paintEvent needed.
    """

    _BASE_STYLE = (
        "QSlider::groove:horizontal {"
        "  height: 4px; background: #2e2e5e; border-radius: 2px; }"
        "QSlider::handle:horizontal {"
        "  background: #7070cc; border-radius: 6px;"
        "  width: 12px; height: 12px; margin: -4px 0; }"
    )

    def set_highlight(self, start_frac: float, end_frac: float):
        """
        Colour the groove between *start_frac* and *end_frac* (both 0.0–1.0).
        """
        p1 = round(max(0.0, min(0.9998, start_frac)), 4)
        p2 = round(max(p1 + 0.0002, min(1.0, end_frac)), 4)
        # Use tiny offsets so QSS gradient stops don't coincide
        a = f"{p1:.4f}"
        b = f"{min(p1 + 0.0001, p2):.4f}"
        c = f"{max(p2 - 0.0001, p1):.4f}"
        d = f"{p2:.4f}"
        self.setStyleSheet(
            f"QSlider::groove:horizontal {{"
            f"  height: 4px; border-radius: 2px;"
            f"  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f"    stop:0 #2e2e5e, stop:{a} #2e2e5e,"
            f"    stop:{b} #40b8ff, stop:{c} #40b8ff,"
            f"    stop:{d} #2e2e5e, stop:1 #2e2e5e); }}"
            f"QSlider::handle:horizontal {{"
            f"  background: #7070cc; border-radius: 6px;"
            f"  width: 12px; height: 12px; margin: -4px 0; }}"
        )

    def clear_highlight(self):
        """Remove any highlight and restore the default groove colour."""
        self.setStyleSheet("")   # inherits from parent app stylesheet


# ---------------------------------------------------------------------------
# _FrameLabel — QLabel subclass with rubber-band region drawing
# ---------------------------------------------------------------------------

class _FrameLabel(QLabel):
    """
    QLabel that intercepts mouse events for two exclusive modes:

    Region edit mode  — rubber-band drag → region_drawn(nx, ny, nw, nh)
    Calibration mode  — single click    → calibration_click(nx, ny)
    """
    region_drawn      = pyqtSignal(float, float, float, float)
    calibration_click = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._edit_mode        = False
        self._calibration_mode = False
        self._press_px: tuple | None = None
        self._video_w     = 0
        self._video_h     = 0
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)

    # ---------------------------------------------------------------- public

    def set_edit_mode(self, enabled: bool):
        self._edit_mode = enabled
        if not enabled:
            self._rubber_band.hide()
            self._press_px = None
        self.setCursor(
            Qt.CursorShape.CrossCursor
            if enabled else Qt.CursorShape.ArrowCursor
        )

    def set_calibration_mode(self, enabled: bool):
        self._calibration_mode = enabled
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif not self._edit_mode:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def set_video_size(self, w: int, h: int):
        self._video_w = w
        self._video_h = h

    # ---------------------------------------------------------------- mouse events

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_px = (event.position().x(), event.position().y())
            if self._edit_mode:
                self._rubber_band.setGeometry(
                    QRect(int(self._press_px[0]), int(self._press_px[1]), 0, 0)
                )
                self._rubber_band.show()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._edit_mode and self._press_px is not None:
            x0, y0 = int(self._press_px[0]), int(self._press_px[1])
            x1 = int(event.position().x())
            y1 = int(event.position().y())
            rect = QRect(
                min(x0, x1), min(y0, y1),
                abs(x1 - x0), abs(y1 - y0)
            )
            self._rubber_band.setGeometry(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if (self._press_px is not None
                and event.button() == Qt.MouseButton.LeftButton):
            x0, y0 = self._press_px
            x1 = event.position().x()
            y1 = event.position().y()
            self._press_px = None
            is_click = abs(x1 - x0) < 8 and abs(y1 - y0) < 8

            ox, oy, sw, sh = self._video_rect()

            if self._calibration_mode and is_click and sw > 0:
                # Single click in calibration mode → emit normalised position
                nx = max(0.0, min(1.0, (x1 - ox) / sw))
                ny = max(0.0, min(1.0, (y1 - oy) / sh))
                self.calibration_click.emit(nx, ny)

            elif self._edit_mode:
                self._rubber_band.hide()
                if not is_click and sw > 0:
                    nx = max(0.0, min(1.0, (min(x0, x1) - ox) / sw))
                    ny = max(0.0, min(1.0, (min(y0, y1) - oy) / sh))
                    nw = max(0.0, min(1.0 - nx, abs(x1 - x0) / sw))
                    nh = max(0.0, min(1.0 - ny, abs(y1 - y0) / sh))
                    if nw > 0.01 and nh > 0.01:
                        self.region_drawn.emit(nx, ny, nw, nh)

        super().mouseReleaseEvent(event)

    # ---------------------------------------------------------------- helpers

    def _video_rect(self) -> tuple[int, int, int, int]:
        """
        Return (ox, oy, scaled_w, scaled_h) of the video image within
        this label, accounting for KeepAspectRatio letterboxing.
        """
        lw, lh = self.width(),  self.height()
        vw, vh = self._video_w, self._video_h
        if vw == 0 or vh == 0 or lw == 0 or lh == 0:
            return 0, 0, lw, lh
        scale = min(lw / vw, lh / vh)
        sw    = int(vw * scale)
        sh    = int(vh * scale)
        ox    = (lw - sw) // 2
        oy    = (lh - sh) // 2
        return ox, oy, sw, sh


# ---------------------------------------------------------------------------
# VideoPlayerWidget
# ---------------------------------------------------------------------------

class VideoPlayerWidget(QWidget):
    position_changed   = pyqtSignal(int)   # current position in ms
    video_loaded       = pyqtSignal(str)   # path of newly loaded video
    region_drawn       = pyqtSignal(float, float, float, float)   # nx, ny, nw, nh
    calibration_click  = pyqtSignal(float, float)                  # nx, ny

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap = None
        self._total_frames = 0
        self._fps = 30.0
        self._current_frame = 0
        self._playing = False
        self._play_start_time: float = 0.0
        self._play_start_frame: int  = 0

        # Time-range overlay (mock data)
        # Format: [(start_ms, end_ms, label, color_rgb, nx, ny, nw, nh), ...]
        self._bounding_boxes: list = []

        # Frame-indexed overlay (real YOLO / loaded JSON)
        # Format: {frame_index: [(label, color_rgb, nx, ny, nw, nh), ...]}
        self._frame_detections: dict[int, list] = {}
        self._sorted_sample_frames: list[int] = []
        self._use_frame_detections = False

        # MMAction2 temporal clips for action-label banner overlay
        # Each entry: {"start_ms": float, "end_ms": float,
        #               "action_label": str, "confidence": float}
        self._action_clips: list[dict] = []

        # Named spatial regions (normalised coords)
        self._regions: list[dict] = []
        self._region_edit_mode: bool = False

        # Path trail overlay — pixel-space track history
        # Format: [(frame_index, cx_px, cy_px), ...]  sorted by frame_index
        self._trail_points: list[tuple[int, float, float]] = []
        self._trail_length_s: float = 5.0    # seconds of history to show

        # Calibration-point overlays (drawn while user is clicking points)
        # Format: [(nx, ny), ...]  up to 4 points
        self._cal_markers: list[tuple[float, float]] = []

        # Visibility toggles
        self._show_yolo:      bool = True
        self._show_mmaction2: bool = True
        self._show_regions:   bool = True
        self._show_trail:     bool = True

        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)

    # ------------------------------------------------------------------ layout

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self._frame_label = _FrameLabel("No video loaded")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setStyleSheet("background:#1a1a2e; color:#888;")
        self._frame_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._frame_label.setMinimumSize(480, 320)
        self._frame_label.region_drawn.connect(self.region_drawn)
        self._frame_label.calibration_click.connect(self.calibration_click)
        root.addWidget(self._frame_label, stretch=1)

        self._seek = _HighlightSlider(Qt.Orientation.Horizontal)
        self._seek.setRange(0, 0)
        self._seek.sliderMoved.connect(self._seek_to)
        root.addWidget(self._seek)

        controls = QHBoxLayout()
        controls.setSpacing(6)

        self._open_btn = QPushButton("Open Video…")
        self._open_btn.clicked.connect(self.open_file_dialog)

        self._play_btn = QPushButton("Play")
        self._play_btn.setFixedWidth(64)
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._toggle_play)

        self._time_label = QLabel("0:00.000 / 0:00.000")

        controls.addWidget(self._open_btn)
        controls.addWidget(self._play_btn)
        controls.addStretch()
        controls.addWidget(self._time_label)
        root.addLayout(controls)

    # ------------------------------------------------------------------ public API

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.wmv);;All files (*)"
        )
        if path:
            self.load_video(path)

    def load_video(self, path: str):
        self._stop()
        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            self._frame_label.setText(f"Could not open:\n{path}")
            return
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_frame = 0
        self._seek.setRange(0, max(0, self._total_frames - 1))
        self._play_btn.setEnabled(True)
        self._render_frame(0)
        self.video_loaded.emit(path)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def frame_height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def set_bounding_boxes(self, boxes: list):
        """Time-range mock overlay."""
        self._bounding_boxes = boxes
        self._use_frame_detections = False
        self._render_frame(self._current_frame)

    def set_frame_detections(self, detections: dict[int, list]):
        """
        Real YOLO / JSON overlay.
        detections: {frame_index: [(label, color_rgb, nx, ny, nw, nh), ...]}

        Frames between sampled indices show the nearest preceding sample's
        boxes — no flickering even at high sample strides.
        """
        self._frame_detections = detections
        self._sorted_sample_frames = sorted(detections.keys())
        self._use_frame_detections = True
        self._render_frame(self._current_frame)

    def set_action_clips(self, clips: list[dict]):
        """
        Set MMAction2 temporal action clips for banner overlay.
        Each dict: {"start_ms", "end_ms", "action_label", "confidence"}
        """
        self._action_clips = clips
        self._render_frame(self._current_frame)

    def set_yolo_visible(self, visible: bool):
        self._show_yolo = visible
        self._render_frame(self._current_frame)

    def set_mmaction2_visible(self, visible: bool):
        self._show_mmaction2 = visible
        self._render_frame(self._current_frame)

    def set_regions_visible(self, visible: bool):
        self._show_regions = visible
        self._render_frame(self._current_frame)

    # ------------------------------------------------------------------ trail overlay API

    def set_trail_points(self, points: list[tuple[int, float, float]]):
        """
        Set path trail data.  points: [(frame_index, cx_px, cy_px), ...]
        Must be sorted by frame_index.
        """
        self._trail_points = points
        self._render_frame(self._current_frame)

    def set_trail_visible(self, visible: bool):
        self._show_trail = visible
        self._render_frame(self._current_frame)

    def set_trail_length(self, seconds: float):
        self._trail_length_s = max(0.5, seconds)
        self._render_frame(self._current_frame)

    # ------------------------------------------------------------------ seekbar highlight API

    def set_seekbar_highlight(self, start_ms: float, end_ms: float):
        """Colour the seek-bar groove between *start_ms* and *end_ms*."""
        if self._total_frames <= 0 or self._fps <= 0:
            return
        total_ms = self._total_frames / self._fps * 1000.0
        if total_ms <= 0:
            return
        self._seek.set_highlight(start_ms / total_ms, end_ms / total_ms)

    def clear_seekbar_highlight(self):
        """Remove any seek-bar highlight."""
        self._seek.clear_highlight()

    def seek_to_ms(self, ms: float):
        """Seek the video to the given millisecond position."""
        if self._fps <= 0 or self._total_frames <= 0:
            return
        frame_index = int(ms / 1000.0 * self._fps)
        frame_index = max(0, min(frame_index, self._total_frames - 1))
        self._seek_to(frame_index)

    # ------------------------------------------------------------------ calibration overlay API

    def set_calibration_mode(self, enabled: bool):
        """Enter/leave calibration click mode (crosshair cursor, no rubber band)."""
        self._frame_label.set_calibration_mode(enabled)

    def set_cal_markers(self, markers: list[tuple[float, float]]):
        """
        Display numbered calibration point markers on the video.
        markers: [(nx, ny), ...]  up to 4 points in normalised coords.
        """
        self._cal_markers = list(markers)
        self._render_frame(self._current_frame)

    def clear_detections(self):
        self._frame_detections = {}
        self._sorted_sample_frames = []
        self._bounding_boxes = []
        self._use_frame_detections = False
        self._action_clips = []
        self._render_frame(self._current_frame)

    # ------------------------------------------------------------------ region API

    def set_region_edit_mode(self, enabled: bool):
        """
        Enter / leave rubber-band region drawing mode.
        In edit mode the cursor changes to a crosshair and mouse drags
        draw a region rectangle.  A region_drawn signal is emitted when
        the mouse is released; the caller should then prompt for a name
        and call set_regions().
        """
        self._region_edit_mode = enabled
        self._frame_label.set_edit_mode(enabled)

    def set_regions(self, regions: list[dict]):
        """
        Replace the displayed region list and refresh the frame.
        Each dict must have keys: name, nx, ny, nw, nh (normalised 0–1).
        """
        self._regions = list(regions)
        self._render_frame(self._current_frame)

    def get_regions(self) -> list[dict]:
        """Return a copy of the current region list."""
        return list(self._regions)

    def clear_regions(self):
        """Remove all regions and refresh."""
        self._regions = []
        self._render_frame(self._current_frame)

    # ------------------------------------------------------------------ playback

    def _toggle_play(self):
        self._stop() if self._playing else self._start()

    def _start(self):
        if not self._cap:
            return
        self._play_start_time  = time.monotonic()
        self._play_start_frame = self._current_frame
        self._timer.start(16)  # poll at ~60 Hz; wall clock decides actual frame
        self._playing = True
        self._play_btn.setText("Pause")

    def _stop(self):
        self._timer.stop()
        self._playing = False
        self._play_btn.setText("Play")

    def _advance_frame(self):
        elapsed = time.monotonic() - self._play_start_time
        target  = self._play_start_frame + int(elapsed * self._fps)
        target  = min(target, self._total_frames - 1)

        if target <= self._current_frame:
            return  # not time for the next frame yet

        if target >= self._total_frames - 1:
            self._current_frame = self._total_frames - 1
            self._render_frame(self._current_frame)
            self._stop()
            return

        sequential      = (target - self._current_frame) == 1
        self._current_frame = target
        self._render_frame(self._current_frame, sequential=sequential)

    def _seek_to(self, frame_index: int):
        was_playing = self._playing
        self._stop()
        self._current_frame = frame_index
        self._render_frame(frame_index)
        if was_playing:
            self._start()  # resets _play_start_time/_play_start_frame

    # ------------------------------------------------------------------ rendering

    def _render_frame(self, frame_index: int, sequential: bool = False):
        if not self._cap:
            return
        if not sequential:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = self._cap.read()
        if not ok:
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        # Keep _FrameLabel up to date so rubber-band coords are correct
        self._frame_label.set_video_size(w, h)

        qimg   = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        painter = QPainter(pixmap)

        # ── Path trail overlay (pixel-space, no homography needed) ──
        if self._show_trail and self._trail_points:
            current_ms  = frame_index / self._fps * 1000.0
            cutoff_ms   = current_ms - self._trail_length_s * 1000.0
            trail_now   = [
                (fi, cx, cy) for fi, cx, cy in self._trail_points
                if fi / self._fps * 1000.0 >= cutoff_ms
                and fi / self._fps * 1000.0 <= current_ms
            ]
            if len(trail_now) >= 2:
                n = len(trail_now)
                for i in range(1, n):
                    t     = i / max(n - 1, 1)
                    alpha = int(80 + t * 175)   # fade older segments
                    r     = int(t * 220)
                    b_    = int((1 - t) * 220)
                    pen   = QPen(QColor(r, 60, b_, alpha), 3)
                    painter.setPen(pen)
                    x0 = int(trail_now[i - 1][1])
                    y0 = int(trail_now[i - 1][2])
                    x1_ = int(trail_now[i][1])
                    y1_ = int(trail_now[i][2])
                    painter.drawLine(x0, y0, x1_, y1_)
            # Latest position dot
            if trail_now:
                cx_now = int(trail_now[-1][1])
                cy_now = int(trail_now[-1][2])
                painter.setPen(QPen(QColor(255, 220, 80), 1))
                painter.setBrush(QColor(255, 220, 80, 200))
                painter.drawEllipse(cx_now - 5, cy_now - 5, 10, 10)
                painter.setBrush(Qt.BrushStyle.NoBrush)  # reset — don't tint later draws

        # ── Calibration point markers ──
        if self._cal_markers:
            for idx, (mnx, mny) in enumerate(self._cal_markers):
                mx = int(mnx * w)
                my = int(mny * h)
                # Crosshair
                painter.setPen(QPen(QColor(255, 200, 0), 2))
                painter.drawLine(mx - 12, my, mx + 12, my)
                painter.drawLine(mx, my - 12, mx, my + 12)
                # Numbered circle
                painter.setBrush(QColor(255, 200, 0, 200))
                painter.setPen(QPen(Qt.GlobalColor.black, 1))
                painter.drawEllipse(mx - 9, my - 9, 18, 18)
                painter.setPen(QPen(Qt.GlobalColor.black))
                painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
                painter.drawText(mx - 4, my + 4, str(idx + 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)  # reset after markers

        # ── Named region overlays ──
        if self._show_regions and self._regions:
            for idx, region in enumerate(self._regions):
                color = _REGION_COLORS[idx % len(_REGION_COLORS)]
                rx = int(region["nx"] * w)
                ry = int(region["ny"] * h)
                rw = int(region["nw"] * w)
                rh = int(region["nh"] * h)
                # Dashed border — thicker so it reads clearly on any background
                pen = QPen(QColor(*color), 3, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawRect(rx, ry, rw, rh)
                # Name tag — font scales with video width
                name      = region["name"]
                font_pt   = max(14, w // 45)
                tag_h     = font_pt + 10
                tag_w     = len(name) * (font_pt - 2) + 14
                painter.fillRect(rx, ry, tag_w, tag_h, QColor(*color, 220))
                painter.setPen(QPen(Qt.GlobalColor.white))
                painter.setFont(QFont("Arial", font_pt, QFont.Weight.Bold))
                painter.drawText(rx + 6, ry + tag_h - 5, name)

        # ── YOLO bounding boxes ──
        if self._show_yolo:
            active = self._get_active_boxes(frame_index, w, h)
            font = QFont("Arial", max(8, w // 60))
            painter.setFont(font)
            for label, color, x, y, bw, bh in active:
                pen = QPen(QColor(*color), 2)
                painter.setPen(pen)
                painter.drawRect(x, y, bw, bh)
                painter.fillRect(x, y - 18, len(label) * 8 + 6, 18,
                                 QColor(*color, 180))
                painter.setPen(QPen(Qt.GlobalColor.white))
                painter.drawText(x + 3, y - 4, label)

        # ── MMAction2 action-label banner ──
        if self._show_mmaction2:
            action = self._get_active_action(frame_index)
            if action:
                banner_h = max(22, h // 20)
                painter.fillRect(0, h - banner_h, w, banner_h,
                                 QColor(0, 0, 0, 180))
                painter.setPen(QPen(QColor(120, 210, 255)))
                font_sz = max(8, h // 36)
                painter.setFont(QFont("Arial", font_sz, QFont.Weight.Bold))
                text = (f"▶ {action['action_label']}"
                        f"  ({action['confidence']:.2f})")
                painter.drawText(6, h - banner_h + font_sz + 2, text)

        painter.end()

        transform = (
            Qt.TransformationMode.FastTransformation
            if self._playing
            else Qt.TransformationMode.SmoothTransformation
        )
        scaled = pixmap.scaled(
            self._frame_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            transform,
        )
        self._frame_label.setPixmap(scaled)

        self._seek.blockSignals(True)
        self._seek.setValue(frame_index)
        self._seek.blockSignals(False)

        cur_s = frame_index / self._fps
        tot_s = self._total_frames / self._fps
        self._time_label.setText(f"{self._fmt(cur_s)} / {self._fmt(tot_s)}")
        self.position_changed.emit(int(cur_s * 1000))

    def _get_active_boxes(
        self, frame_index: int, w: int, h: int
    ) -> list[tuple]:
        """Return (label, color, x_px, y_px, w_px, h_px) for this frame."""
        if self._use_frame_detections:
            boxes = self._lookup_nearest_sample(frame_index)
            return [
                (label, color,
                 int(nx * w), int(ny * h), int(nw * w), int(nh * h))
                for label, color, nx, ny, nw, nh in boxes
            ]

        # Time-range mock boxes
        current_ms = int(frame_index / self._fps * 1000)
        result = []
        for start_ms, end_ms, label, color, nx, ny, nw, nh in self._bounding_boxes:
            if start_ms <= current_ms < end_ms:
                result.append(
                    (label, color,
                     int(nx * w), int(ny * h), int(nw * w), int(nh * h))
                )
        return result

    def _get_active_action(self, frame_index: int) -> dict | None:
        """Return the highest-confidence action clip covering this frame, or None."""
        if not self._action_clips:
            return None
        current_ms = frame_index / self._fps * 1000.0
        best: dict | None = None
        for clip in self._action_clips:
            if clip["start_ms"] <= current_ms < clip["end_ms"]:
                if best is None or clip["confidence"] > best["confidence"]:
                    best = clip
        return best

    def _lookup_nearest_sample(self, frame_index: int) -> list:
        """
        Return the detection list for frame_index if it was sampled, otherwise
        fall back to the most-recently-sampled frame before frame_index.
        This prevents boxes from disappearing between sample points.
        """
        if not self._sorted_sample_frames:
            return []
        boxes = self._frame_detections.get(frame_index)
        if boxes is not None:
            return boxes
        pos = bisect.bisect_right(self._sorted_sample_frames, frame_index) - 1
        if pos >= 0:
            return self._frame_detections.get(self._sorted_sample_frames[pos], [])
        return []

    @staticmethod
    def _fmt(seconds: float) -> str:
        m = int(seconds) // 60
        s = seconds % 60
        return f"{m}:{s:06.3f}"
