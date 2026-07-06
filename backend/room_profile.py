# backend/room_profile.py
# Room profiles: one named bundle referencing a calibration file, a regions
# file, and a blueprint image.  A profile is a LOOSE bundle — each component
# can still be changed individually in the app at any time; the profile then
# shows as "modified" until re-saved.
#
# Schema (.room.json):
#   {
#     "name":             "Room A – Camera 1",
#     "calibration_path": "C:/…/roomA.homography.json"  | null,
#     "regions_path":     "C:/…/roomA.regions.json"     | null,
#     "blueprint_path":   "C:/…/roomA_floorplan.png"    | null
#   }
#
# Paths are stored absolute.  Missing/None components are simply skipped on
# apply, so a profile with only a calibration is valid.

from __future__ import annotations
import json
from pathlib import Path


def save_room_profile(
    path: str | Path,
    name: str,
    calibration_path: str | None,
    regions_path: str | None,
    blueprint_path: str | None,
) -> None:
    """Write a room profile to *path* (overwrites if exists)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "name":             name,
            "calibration_path": calibration_path,
            "regions_path":     regions_path,
            "blueprint_path":   blueprint_path,
        }, f, indent=2)


def load_room_profile(path: str | Path) -> dict:
    """
    Load a room profile.  Returns a dict with keys
    name / calibration_path / regions_path / blueprint_path.
    Raises ValueError if the file is malformed.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "name" not in data:
        raise ValueError(f"Not a room profile file: {path}")
    return {
        "name":             data.get("name", Path(path).stem),
        "calibration_path": data.get("calibration_path"),
        "regions_path":     data.get("regions_path"),
        "blueprint_path":   data.get("blueprint_path"),
    }
