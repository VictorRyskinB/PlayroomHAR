# backend/region_config.py
# Save / load named region configs (camera-specific rectangular ROIs).
#
# A region config is a small JSON file stored in a user-chosen location.
# Each region is stored in normalised coordinates (0–1) so the config is
# resolution-independent and works for any video from the same camera.
#
# Schema:
#   {
#     "config_name": "Room A – Camera 1",
#     "regions": [
#       { "name": "Toy shelf",  "nx": 0.05, "ny": 0.10, "nw": 0.30, "nh": 0.25 },
#       { "name": "Play mat",   "nx": 0.05, "ny": 0.55, "nw": 0.80, "nh": 0.40 }
#     ]
#   }

from __future__ import annotations
import json
from pathlib import Path


def save_region_config(
    path: str | Path,
    config_name: str,
    regions: list[dict],
) -> None:
    """Write a region config to *path* (overwrites if exists)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"config_name": config_name, "regions": regions},
                  f, indent=2)


def load_region_config(path: str | Path) -> tuple[str, list[dict]]:
    """
    Load a region config.
    Returns (config_name, regions).
    Raises ValueError if the file is malformed.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "regions" not in data:
        raise ValueError(f"No 'regions' key in {path}")
    return data.get("config_name", Path(path).stem), data["regions"]
