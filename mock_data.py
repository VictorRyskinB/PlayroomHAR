# mock_data.py
# DEPRECATED (2026-07) — the mock-data injection path was removed from the UI.
# Not imported by any live code path; kept for future reference.
#
# Placeholder data used during UI development.
# BACKEND INJECTION POINT: Replace MOCK_BOUNDING_BOXES with real YOLO detections
# and MOCK_RESULTS with merged YOLO + MMAction2 output once the backend is ready.

# Each bounding box entry: (start_ms, end_ms, label, color_rgb, x, y, w, h)
# Coordinates are normalized [0.0, 1.0] relative to frame size.
MOCK_BOUNDING_BOXES = [
    (0,     3000,  "child",  (255,  80,  80), 0.10, 0.15, 0.30, 0.60),
    (0,     3000,  "toy car",(80,  180, 255), 0.55, 0.60, 0.15, 0.12),
    (3000,  7000,  "child",  (255,  80,  80), 0.20, 0.10, 0.30, 0.65),
    (3000,  7000,  "ball",   (80,  220, 120), 0.65, 0.55, 0.12, 0.12),
    (7000,  12000, "child",  (255,  80,  80), 0.35, 0.20, 0.28, 0.58),
    (7000,  12000, "block",  (240, 180,  50), 0.50, 0.65, 0.10, 0.10),
    (12000, 18000, "child",  (255,  80,  80), 0.15, 0.25, 0.30, 0.60),
    (18000, 25000, "child",  (255,  80,  80), 0.40, 0.15, 0.30, 0.60),
    (18000, 25000, "toy car",(80,  180, 255), 0.60, 0.58, 0.15, 0.12),
]

# Each result row: (start_time_str, end_time_str, action, object_name)
# BACKEND INJECTION POINT: Replace with rows parsed from the merged CSV output
# produced by the YOLO + MMAction2 pipeline.
MOCK_RESULTS = [
    ("00:00:00", "00:00:03", "Reaches for object",  "toy car"),
    ("00:00:03", "00:00:07", "Rolls object",         "ball"),
    ("00:00:07", "00:00:12", "Stacks objects",       "block"),
    ("00:00:12", "00:00:18", "Idle / no interaction","—"),
    ("00:00:18", "00:00:25", "Pushes object",        "toy car"),
]
