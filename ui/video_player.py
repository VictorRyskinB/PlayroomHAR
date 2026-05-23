# ui/video_player.py
# VideoPlayerWidget: OpenCV frame reading, playback, seek bar, bbox overlay.
#
# Two overlay modes (whichever was set most recently takes priority):
#   • Time-range mode  — set_bounding_boxes()    — mock data
#   • Frame-index mode — set_frame_detections()  — real YOLO output
#
# Flicker fix: when using frame-index mode with a sample stride > 1, frames
# between sample points show the most-recently-seen detections rather than
# going blank.  This uses a sorted list of sampled frame indices + bisect.

import bisect
import cv2
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QSizePolicy, QFileDialog,
)


class VideoPlayerWidget(QWidget):
    position_changed = pyqtSignal(int)   # current position in ms
    video_loaded     = pyqtSignal(str)   # path of newly loaded video

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap = None
        self._total_frames = 0
        self._fps = 30.0
        self._current_frame = 0
        self._playing = False

        # Time-range overlay (mock data)
        # Format: [(start_ms, end_ms, label, color_rgb, nx, ny, nw, nh), ...]
        self._bounding_boxes: list = []

        # Frame-indexed overlay (real YOLO / loaded JSON)
        # Format: {frame_index: [(label, color_rgb, nx, ny, nw, nh), ...]}
        self._frame_detections: dict[int, list] = {}
        self._sorted_sample_frames: list[int] = []   # sorted keys of _frame_detections
        self._use_frame_detections = False

        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)

    # ------------------------------------------------------------------ layout

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self._frame_label = QLabel("No video loaded")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setStyleSheet("background:#1a1a2e; color:#888;")
        self._frame_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._frame_label.setMinimumSize(480, 320)
        root.addWidget(self._frame_label, stretch=1)

        self._seek = QSlider(Qt.Orientation.Horizontal)
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

    def clear_detections(self):
        self._frame_detections = {}
        self._sorted_sample_frames = []
        self._bounding_boxes = []
        self._use_frame_detections = False
        self._render_frame(self._current_frame)

    # ------------------------------------------------------------------ playback

    def _toggle_play(self):
        self._stop() if self._playing else self._start()

    def _start(self):
        if not self._cap:
            return
        self._timer.start(max(1, int(1000 / self._fps)))
        self._playing = True
        self._play_btn.setText("Pause")

    def _stop(self):
        self._timer.stop()
        self._playing = False
        self._play_btn.setText("Play")

    def _advance_frame(self):
        if self._current_frame >= self._total_frames - 1:
            self._stop()
            return
        self._current_frame += 1
        self._render_frame(self._current_frame)

    def _seek_to(self, frame_index: int):
        was_playing = self._playing
        self._stop()
        self._current_frame = frame_index
        self._render_frame(frame_index)
        if was_playing:
            self._start()

    # ------------------------------------------------------------------ rendering

    def _render_frame(self, frame_index: int):
        if not self._cap:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = self._cap.read()
        if not ok:
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg  = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        active = self._get_active_boxes(frame_index, w, h)
        if active:
            painter = QPainter(pixmap)
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
            painter.end()

        scaled = pixmap.scaled(
            self._frame_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
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

    def _lookup_nearest_sample(self, frame_index: int) -> list:
        """
        Return the detection list for frame_index if it was sampled, otherwise
        fall back to the most-recently-sampled frame before frame_index.
        This prevents boxes from disappearing between sample points.
        """
        if not self._sorted_sample_frames:
            return []
        # Exact hit
        boxes = self._frame_detections.get(frame_index)
        if boxes is not None:
            return boxes
        # Nearest preceding sample
        pos = bisect.bisect_right(self._sorted_sample_frames, frame_index) - 1
        if pos >= 0:
            return self._frame_detections.get(self._sorted_sample_frames[pos], [])
        return []

    @staticmethod
    def _fmt(seconds: float) -> str:
        m = int(seconds) // 60
        s = seconds % 60
        return f"{m}:{s:06.3f}"
