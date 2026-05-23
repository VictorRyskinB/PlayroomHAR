# ui/results_table.py
# Two table widgets used in the three-tab results panel:
#
# ResultsTableWidget  — YOLO Object Interactions tab
#   Fixed 6-column schema with optional debug columns.
#   Columns: Start, End, Action, Object, YOLO Conf*, Action Conf*
#   (* hidden until debug mode enabled)
#
# GenericTableWidget  — Regions tab and MMAction2 Actions tab
#   Flexible column schema passed at construction.
#   Full-width stretch on the last column by default.

import csv
from PyQt6.QtCore import Qt, QSortFilterProxyModel
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableView, QLabel, QFileDialog, QHeaderView,
)

ALL_COLUMNS = ["Start Time", "End Time", "Action", "Object",
               "YOLO Conf", "Action Conf"]
DEBUG_COLUMNS = {4, 5}   # column indices that are hidden in normal mode


class ResultsTableWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list = []
        self._debug_mode = False
        self._build_ui()

    # ------------------------------------------------------------------ layout

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # Header bar
        header_row = QHBoxLayout()
        title = QLabel("Annotation Results")
        title.setStyleSheet("font-weight:bold; font-size:13px;")
        self._count_label = QLabel("0 rows")
        self._count_label.setStyleSheet("color:#888; font-size:11px;")
        header_row.addWidget(title)
        header_row.addStretch()
        header_row.addWidget(self._count_label)
        root.addLayout(header_row)

        # Model + proxy
        self._model = QStandardItemModel(0, len(ALL_COLUMNS))
        self._model.setHorizontalHeaderLabels(ALL_COLUMNS)

        self._proxy = QSortFilterProxyModel()
        self._proxy.setSourceModel(self._model)

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            "QTableView { gridline-color: #2e2e4e; }"
            "QTableView::item:selected { background: #3a3a6e; }"
        )
        root.addWidget(self._table, stretch=1)

        # Initially hide debug columns
        for col in DEBUG_COLUMNS:
            self._table.setColumnHidden(col, True)

        # Bottom bar
        btn_row = QHBoxLayout()
        self._action_source_label = QLabel("")
        self._action_source_label.setStyleSheet("color:#888; font-size:10px;")

        self._export_btn = QPushButton("Export CSV…")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_csv)

        btn_row.addWidget(self._action_source_label)
        btn_row.addStretch()
        btn_row.addWidget(self._export_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------ public

    def set_results(self, rows: list):
        """
        Accepts 4-tuple (mock) or 6-tuple (real backend) rows.
        4-tuple rows have confidence columns padded with "—".
        """
        self._rows = [_normalize_row(r) for r in rows]
        self._model.removeRows(0, self._model.rowCount())

        mmaction2_count = 0
        for row in self._rows:
            items = [QStandardItem(cell) for cell in row]
            for item in items:
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
                )
            self._model.appendRow(items)
            if row[5] != "—":   # action_conf present → MMAction2 was used
                mmaction2_count += 1

        n = len(self._rows)
        self._count_label.setText(f"{n} row{'s' if n != 1 else ''}")
        self._export_btn.setEnabled(n > 0)

        if mmaction2_count:
            self._action_source_label.setText(
                f"MMAction2 labels: {mmaction2_count}/{n}"
            )
        else:
            self._action_source_label.setText("")

    def set_debug_mode(self, enabled: bool):
        self._debug_mode = enabled
        for col in DEBUG_COLUMNS:
            self._table.setColumnHidden(col, not enabled)

    def highlight_row_at(self, position_ms: int, time_rows: list):
        for i, (start_ms, end_ms) in enumerate(time_rows):
            if start_ms <= position_ms < end_ms:
                proxy_idx = self._proxy.mapFromSource(self._model.index(i, 0))
                self._table.selectRow(proxy_idx.row())
                self._table.scrollTo(proxy_idx)
                return
        self._table.clearSelection()

    # ------------------------------------------------------------------ export

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "annotations.csv",
            "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return

        if self._debug_mode:
            header = ALL_COLUMNS
            data   = self._rows
        else:
            header = ALL_COLUMNS[:4]
            data   = [r[:4] for r in self._rows]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_row(row: tuple | list) -> tuple[str, str, str, str, str, str]:
    """Pad 4-tuple rows to 6 values; leave 6-tuple rows unchanged."""
    r = tuple(str(v) for v in row)
    if len(r) == 4:
        return r + ("—", "—")
    if len(r) >= 6:
        return r[:6]
    return r + ("—",) * (6 - len(r))


# ---------------------------------------------------------------------------
# GenericTableWidget — flexible column schema for Regions and Actions tabs
# ---------------------------------------------------------------------------

class GenericTableWidget(QWidget):
    """
    A sortable, exportable table with a caller-defined column schema.

    Usage:
        w = GenericTableWidget(["Start", "End", "Duration", "Region", "Frames"])
        w.set_rows([ ("0:00:01.000", "0:00:15.000", "14.0 s", "Play mat", "42"), ... ])
    """

    def __init__(self, columns: list[str], stretch_col: int = -1, parent=None):
        """
        columns    : list of column header strings
        stretch_col: column index to fill remaining width (-1 = last column)
        """
        super().__init__(parent)
        self._columns    = columns
        self._stretch_col = len(columns) - 1 if stretch_col == -1 else stretch_col
        self._rows: list[tuple] = []
        self._build_ui()

    # ------------------------------------------------------------------ layout

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        header_row = QHBoxLayout()
        self._count_label = QLabel("0 rows")
        self._count_label.setStyleSheet("color:#888; font-size:11px;")
        header_row.addStretch()
        header_row.addWidget(self._count_label)
        root.addLayout(header_row)

        self._model = QStandardItemModel(0, len(self._columns))
        self._model.setHorizontalHeaderLabels(self._columns)

        self._proxy = QSortFilterProxyModel()
        self._proxy.setSourceModel(self._model)

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            self._stretch_col, QHeaderView.ResizeMode.Stretch
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            "QTableView { gridline-color: #2e2e4e; }"
            "QTableView::item:selected { background: #3a3a6e; }"
        )
        root.addWidget(self._table, stretch=1)

        btn_row = QHBoxLayout()
        self._export_btn = QPushButton("Export CSV…")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_csv)
        btn_row.addStretch()
        btn_row.addWidget(self._export_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------ public

    def set_rows(self, rows: list[tuple]):
        """Replace all table content with *rows* (each a tuple of strings)."""
        self._rows = [tuple(str(v) for v in r) for r in rows]
        self._model.removeRows(0, self._model.rowCount())
        for row in self._rows:
            items = [QStandardItem(cell) for cell in row]
            for item in items:
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
                )
            self._model.appendRow(items)
        n = len(self._rows)
        self._count_label.setText(f"{n} row{'s' if n != 1 else ''}")
        self._export_btn.setEnabled(n > 0)

    def highlight_row_at(self, position_ms: int, time_rows: list):
        """
        Select the first row whose (start_ms, end_ms) covers position_ms.
        time_rows must be pre-computed as [(start_ms, end_ms), ...].
        """
        for i, (start_ms, end_ms) in enumerate(time_rows):
            if start_ms <= position_ms < end_ms:
                proxy_idx = self._proxy.mapFromSource(self._model.index(i, 0))
                self._table.selectRow(proxy_idx.row())
                self._table.scrollTo(proxy_idx)
                return
        self._table.clearSelection()

    # ------------------------------------------------------------------ export

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "export.csv",
            "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self._columns)
            writer.writerows(self._rows)
