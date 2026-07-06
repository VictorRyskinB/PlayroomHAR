# ui/path_map_widget.py
# Top-down room map: draws the child's path on a plain rectangle.
#
# Coordinate convention (matches path_analyzer.py):
#   Origin (0,0) = front-left (camera side, left)
#   +X = across the room width
#   +Y = toward the far wall
#   Room default: 250 cm × 600 cm
#
# Display orientation:
#   Camera side at the BOTTOM, far wall at the TOP — natural top-down view.
#
# Layers (toggle independently):
#   1. Room rectangle + labels
#   2. Path polyline (time-gradient: blue early → red late)
#   3. Heatmap overlay (semi-transparent RGBA)
#   4. Named region overlays (re-projected to world coords if homography set)

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont, QImage, QPixmap,
    QLinearGradient,
)
from PyQt6.QtWidgets import QWidget, QSizePolicy

from backend.path_analyzer import PathPoint


# Padding around the room rectangle (widget pixels)
_MARGIN = 30


class PathMapWidget(QWidget):
    """
    Renders a top-down room map with path and optional heatmap.

    Call set_path() after a tracking run.
    Call set_heatmap() to overlay the density map.
    Toggle show_path / show_heatmap / show_regions at any time.
    """

    def __init__(
        self,
        room_size_cm: tuple[float, float] = (250.0, 600.0),
        parent=None,
    ):
        super().__init__(parent)
        self._room_w, self._room_d = room_size_cm

        self._path_points:  list[PathPoint] = []
        self._heatmap_rgba: np.ndarray | None = None   # shape (H, W, 4) uint8
        self._regions:      list[dict] = []            # {name, nx, ny, nw, nh}
        self._homography:   np.ndarray | None = None   # 3×3 pixel→world matrix
        self._video_fw:     int = 0
        self._video_fh:     int = 0
        self._blueprint:    QPixmap | None = None      # top-down room image

        self._show_path    = True
        self._show_heatmap = True
        self._show_regions = True
        self._time_cutoff_ms: float | None = None  # None = show all; set in live-path mode

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(200, 300)

    # ------------------------------------------------------------------ public

    def set_room_size(self, width_cm: float, depth_cm: float):
        self._room_w = width_cm
        self._room_d = depth_cm
        self.update()

    def set_blueprint(self, image_path: str | None):
        """
        Load a top-down room image drawn under the path/heatmap.
        Orient the image like this map: camera side at the BOTTOM.
        Pass None to remove the blueprint.
        """
        if image_path:
            pm = QPixmap(image_path)
            self._blueprint = pm if not pm.isNull() else None
        else:
            self._blueprint = None
        self.update()

    def set_path(self, points: list[PathPoint]):
        self._path_points = points
        self.update()

    def set_heatmap(self, rgba: np.ndarray | None):
        """rgba: (H, W, 4) uint8 array from path_analyzer.heatmap_to_rgba()."""
        self._heatmap_rgba = rgba
        self.update()

    def set_regions(self, regions: list[dict]):
        """Regions in normalised video coords. Drawn on the map if homography is set."""
        self._regions = regions
        self.update()

    def set_homography(self, matrix: np.ndarray | None, fw: int, fh: int):
        """Pass the pixel→world homography so regions can be reprojected onto the map."""
        self._homography = matrix
        self._video_fw   = fw
        self._video_fh   = fh
        self.update()

    def set_show_path(self, v: bool):
        self._show_path = v
        self.update()

    def set_show_heatmap(self, v: bool):
        self._show_heatmap = v
        self.update()

    def set_show_regions(self, v: bool):
        self._show_regions = v
        self.update()

    def set_time_cutoff(self, ms: float | None):
        """
        Restrict the path line to points with timestamp_ms <= ms.
        Pass None to show the full path (non-live mode).
        """
        self._time_cutoff_ms = ms
        self.update()

    def clear(self):
        self._path_points  = []
        self._heatmap_rgba = None
        self.update()

    # ------------------------------------------------------------------ paint

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor("#12122a"))

        # Compute the room rectangle in widget space
        rx, ry, rw, rh = self._room_rect(w, h)

        # ── Blueprint layer (under everything else) ──
        if self._blueprint is not None:
            painter.drawPixmap(
                QRectF(rx, ry, rw, rh),
                self._blueprint,
                QRectF(self._blueprint.rect()),
            )

        # ── Heatmap layer ──
        if self._show_heatmap and self._heatmap_rgba is not None:
            self._draw_heatmap(painter, rx, ry, rw, rh)

        # ── Room outline ──
        overlay_present = self._blueprint is not None or (
            self._heatmap_rgba is not None and self._show_heatmap
        )
        painter.setPen(QPen(QColor("#556677"), 2))
        painter.setBrush(QBrush(QColor(20, 30, 50, 0 if overlay_present else 40)))
        painter.drawRect(int(rx), int(ry), int(rw), int(rh))

        # Axis labels
        painter.setPen(QPen(QColor("#667788")))
        font = QFont("Arial", 8)
        painter.setFont(font)
        # "Camera" label at bottom
        painter.drawText(
            QRectF(rx, ry + rh + 4, rw, 16),
            Qt.AlignmentFlag.AlignCenter,
            "▲ Camera"
        )
        # "Far wall" at top
        painter.drawText(
            QRectF(rx, ry - 18, rw, 16),
            Qt.AlignmentFlag.AlignCenter,
            "Far wall"
        )
        # Width label
        painter.drawText(
            QRectF(0, ry + rh / 2 - 8, _MARGIN - 2, 16),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{self._room_w/100:.1f}m"
        )
        # Depth label (rotated)
        painter.save()
        painter.translate(w - 4, ry + rh / 2)
        painter.rotate(-90)
        painter.drawText(
            QRectF(-40, -8, 80, 16),
            Qt.AlignmentFlag.AlignCenter,
            f"{self._room_d/100:.1f}m"
        )
        painter.restore()

        # ── Path layer ──
        if self._show_path and self._path_points:
            self._draw_path(painter, rx, ry, rw, rh)

        # ── Region overlays ──
        if self._show_regions and self._regions and self._homography is not None:
            self._draw_regions(painter, rx, ry, rw, rh)

        # ── No-data message ──
        if not self._path_points:
            painter.setPen(QPen(QColor("#445566")))
            painter.setFont(QFont("Arial", 11))
            painter.drawText(
                QRectF(0, 0, w, h),
                Qt.AlignmentFlag.AlignCenter,
                "No path data\nRun 'Track Path' to populate this view",
            )

        painter.end()

    # ------------------------------------------------------------------ helpers

    def _room_rect(self, w: int, h: int) -> tuple[float, float, float, float]:
        """
        Compute (rx, ry, rw, rh) of the room rectangle in widget pixels,
        filling as much space as possible while keeping aspect ratio.
        """
        avail_w = w - 2 * _MARGIN
        avail_h = h - 2 * _MARGIN
        scale   = min(avail_w / self._room_w, avail_h / self._room_d)
        rw      = self._room_w * scale
        rh      = self._room_d * scale
        rx      = (w - rw) / 2
        ry      = (h - rh) / 2
        return rx, ry, rw, rh

    def _world_to_widget(
        self,
        x_cm: float, y_cm: float,
        rx: float, ry: float, rw: float, rh: float,
    ) -> QPointF:
        """
        Map floor (x_cm, y_cm) → widget pixel QPointF.
        y_cm=0 (camera side) → bottom of rectangle.
        y_cm=room_d (far wall) → top of rectangle.
        """
        px = rx + (x_cm / self._room_w) * rw
        py = ry + rh - (y_cm / self._room_d) * rh   # flip y
        return QPointF(px, py)

    def _draw_path(
        self,
        painter: QPainter,
        rx: float, ry: float, rw: float, rh: float,
    ):
        cutoff = self._time_cutoff_ms
        pts = [p for p in self._path_points
               if p.x_cm is not None and p.y_cm is not None
               and (cutoff is None or p.timestamp_ms <= cutoff)]
        if not pts:
            if cutoff is None:
                # Full-path mode, no calibrated points → show calibration hint
                painter.setPen(QPen(QColor("#445566")))
                painter.setFont(QFont("Arial", 9))
                painter.drawText(
                    QRectF(rx, ry, rw, rh),
                    Qt.AlignmentFlag.AlignCenter,
                    "Calibrate camera\nto see floor path",
                )
            # Live mode with no points in range yet: show empty room silently
            return

        n = len(pts)
        # Draw path as line segments, coloured blue→red along time axis
        for i in range(1, n):
            t    = i / max(n - 1, 1)
            r    = int(t * 220)
            b    = int((1 - t) * 220)
            pen  = QPen(QColor(r, 60, b), 2)
            prev = self._world_to_widget(
                pts[i - 1].x_cm, pts[i - 1].y_cm, rx, ry, rw, rh
            )
            curr = self._world_to_widget(
                pts[i].x_cm, pts[i].y_cm, rx, ry, rw, rh
            )
            # Draw dashed for interpolated segments
            if pts[i].interpolated:
                pen.setStyle(Qt.PenStyle.DotLine)
            painter.setPen(pen)
            painter.drawLine(prev, curr)

        # Start and end markers
        if n > 0:
            start = self._world_to_widget(
                pts[0].x_cm, pts[0].y_cm, rx, ry, rw, rh
            )
            end   = self._world_to_widget(
                pts[-1].x_cm, pts[-1].y_cm, rx, ry, rw, rh
            )
            # Start: green circle
            painter.setPen(QPen(QColor("#44ff88"), 1))
            painter.setBrush(QBrush(QColor("#44ff88")))
            painter.drawEllipse(start, 5, 5)
            # End: red circle
            painter.setPen(QPen(QColor("#ff4444"), 1))
            painter.setBrush(QBrush(QColor("#ff4444")))
            painter.drawEllipse(end, 5, 5)

    def _draw_regions(
        self,
        painter: QPainter,
        rx: float, ry: float, rw: float, rh: float,
    ):
        """
        Draw each region's floor footprint: all 4 corners of the video-space
        rectangle projected through the homography → a quadrilateral on the
        map, clipped to the room.

        Regions with vertical extent (walls, shelves) project their top edge
        far toward/behind the horizon; when the projection blows up beyond a
        sanity bound, fall back to a dot at the bottom-center floor point.
        """
        import cv2
        from PyQt6.QtGui import QPolygonF

        _COLORS = [
            (255, 100, 100), (100, 220, 100), (100, 140, 255), (255, 220, 50),
            (255, 100, 220), ( 80, 220, 220), (255, 160,  40), (180, 100, 255),
        ]
        mat = self._homography
        fw, fh = self._video_fw, self._video_fh
        if fw == 0 or fh == 0:
            return

        dot_r     = max(6, int(rw / 30))
        font_size = max(9, int(rw / 28))
        painter.setFont(QFont("Arial", font_size, QFont.Weight.Bold))

        # Projections landing further out than this are treated as degenerate
        # (region extends toward the image horizon).
        bound_x = (-0.5 * self._room_w, 1.5 * self._room_w)
        bound_y = (-0.5 * self._room_d, 1.5 * self._room_d)

        for idx, region in enumerate(self._regions):
            color = QColor(*_COLORS[idx % len(_COLORS)])

            x0 = region["nx"] * fw
            y0 = region["ny"] * fh
            x1 = (region["nx"] + region["nw"]) * fw
            y1 = (region["ny"] + region["nh"]) * fh
            corners_px = np.float32([[
                [x0, y0], [x1, y0], [x1, y1], [x0, y1],
            ]])
            corners_w = cv2.perspectiveTransform(corners_px, mat)[0]

            sane = all(
                bound_x[0] <= float(cx) <= bound_x[1]
                and bound_y[0] <= float(cy) <= bound_y[1]
                for cx, cy in corners_w
            )

            if sane:
                poly = QPolygonF([
                    self._world_to_widget(float(cx), float(cy), rx, ry, rw, rh)
                    for cx, cy in corners_w
                ])
                # Clip to the room rectangle so partial overshoot stays tidy
                painter.save()
                painter.setClipRect(QRectF(rx, ry, rw, rh))
                fill = QColor(color); fill.setAlpha(60)
                painter.setBrush(QBrush(fill))
                painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
                painter.drawPolygon(poly)
                painter.restore()

                # Label at the polygon centroid (clamped into the room)
                cx = sum(p.x() for p in poly) / 4
                cy = sum(p.y() for p in poly) / 4
                cx = min(max(cx, rx + 4), rx + rw - 4)
                cy = min(max(cy, ry + font_size), ry + rh - 4)
                painter.setPen(QPen(Qt.GlobalColor.white))
                painter.drawText(QPointF(cx, cy), region["name"])
                continue

            # ── Fallback: bottom-center floor dot (degenerate projection) ──
            cx_px = (region["nx"] + region["nw"] / 2) * fw
            cy_px = (region["ny"] + region["nh"]) * fh
            pt_w  = cv2.perspectiveTransform(
                np.float32([[[cx_px, cy_px]]]), mat
            )
            x_cm, y_cm = float(pt_w[0][0][0]), float(pt_w[0][0][1])
            wpt = self._world_to_widget(x_cm, y_cm, rx, ry, rw, rh)
            if not (rx <= wpt.x() <= rx + rw and ry <= wpt.y() <= ry + rh):
                continue
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawEllipse(wpt, dot_r, dot_r)
            painter.setPen(QPen(Qt.GlobalColor.white))
            painter.drawText(
                QPointF(wpt.x() + dot_r + 3, wpt.y() + font_size // 2),
                region["name"]
            )

    def _draw_heatmap(
        self,
        painter: QPainter,
        rx: float, ry: float, rw: float, rh: float,
    ):
        rgba = self._heatmap_rgba
        if rgba is None or rgba.sum() == 0:
            return
        h_grid, w_grid = rgba.shape[:2]
        # Build QImage from RGBA bytes
        img = QImage(
            rgba.tobytes(),
            w_grid, h_grid,
            w_grid * 4,
            QImage.Format.Format_RGBA8888,
        )
        pixmap = QPixmap.fromImage(img)
        # Draw scaled to the room rectangle
        painter.drawPixmap(
            QRectF(rx, ry, rw, rh),
            pixmap,
            QRectF(0, 0, w_grid, h_grid),
        )
