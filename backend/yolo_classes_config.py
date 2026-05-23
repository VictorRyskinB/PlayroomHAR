# backend/yolo_classes_config.py
# Save / load YOLO-World custom class lists.
#
# A class list config is a small JSON file:
#   {
#     "config_name": "Playroom – Room A",
#     "classes": ["toy", "ball", "slinky", ...]
#   }
#
# Classes are free-form text strings passed directly to YOLO-World's
# set_classes() API, so they should be plain English nouns or short
# noun phrases (e.g. "building block", not "block_toy").

from __future__ import annotations
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Default class list — a reasonable starting point for a child playroom.
# Edit freely in the UI and save to a per-room .classes.json file.
# ---------------------------------------------------------------------------

DEFAULT_PLAYROOM_CLASSES: list[str] = [
    # Iconic toys
    "slinky",
    "teddy bear",
    "stuffed animal",
    "doll",
    "action figure",
    "toy car",
    "toy truck",
    "toy train",
    "building block",
    "lego brick",
    "puzzle piece",
    # Sports / active play
    "ball",
    "sports ball",
    "frisbee",
    "jump rope",
    # Art / learning
    "book",
    "crayon",
    "marker",
    "pencil",
    "drawing pad",
    # Containers / props
    "toy",          # generic fallback YOLO-World understands well
    "box",
    "basket",
    "bucket",
    "bag",
    # Furniture / surfaces likely in frame
    "mat",
    "blanket",
    "pillow",
    "chair",
    "table",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_class_list(
    path: str | Path,
    config_name: str,
    classes: list[str],
) -> None:
    """Write a class-list config to *path* (overwrites if exists)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"config_name": config_name, "classes": classes},
                  f, indent=2)


def load_class_list(path: str | Path) -> tuple[str, list[str]]:
    """
    Load a class-list config.
    Returns (config_name, classes).
    Raises ValueError if the file is malformed.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "classes" not in data:
        raise ValueError(f"No 'classes' key in {path}")
    classes = [str(c).strip() for c in data["classes"] if str(c).strip()]
    return data.get("config_name", Path(path).stem), classes
