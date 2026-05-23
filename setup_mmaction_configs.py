"""
Download MMAction2 config files needed by backend/action_recognizer.py.

Run once after installing mmaction2 via pip:
    python setup_mmaction_configs.py

The configs are stored in backend/mmaction_configs/ and referenced at runtime.
Requires internet access (~5 KB total, no model weights downloaded here).
"""

import sys
import urllib.request
from pathlib import Path

BRANCH   = "main"
BASE_URL = f"https://raw.githubusercontent.com/open-mmlab/mmaction2/{BRANCH}"
OUT_ROOT = Path(__file__).parent / "backend" / "mmaction_configs"

# ── files to download (relative to repo root) ──────────────────────────────
# Configs are flat files in configs/recognition/tsn/ (no subdirectories).
# Base deps verified against mmaction2 main branch 2025-05.
FILES = [
    # TSN Kinetics-400  (3 clips × 1 frame)
    "configs/recognition/tsn/"
    "tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py",

    # TSN Something-Something v2  (8 clips × 1 frame)
    "configs/recognition/tsn/"
    "tsn_imagenet-pretrained-r50_8xb32-1x1x8-50e_sthv2-rgb.py",

    # Shared base configs
    "configs/_base_/models/tsn_r50.py",
    "configs/_base_/schedules/sgd_100e.py",
    "configs/_base_/schedules/sgd_50e.py",
    "configs/_base_/default_runtime.py",

    # Label maps (used as fallback when checkpoint meta lacks class names)
    "tools/data/kinetics/label_map_k400.txt",
    "tools/data/sthv2/label_map.txt",
]


def download(rel_path: str) -> bool:
    url  = f"{BASE_URL}/{rel_path}"
    dest = OUT_ROOT / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"  {rel_path} … ", end="", flush=True)
        urllib.request.urlretrieve(url, dest)
        print("OK")
        return True
    except Exception as exc:
        print(f"FAILED ({exc})")
        return False


def main():
    print(f"Downloading MMAction2 configs (branch: {BRANCH}) → {OUT_ROOT}\n")
    ok = all(download(f) for f in FILES)
    if ok:
        print("\nAll configs downloaded.  You can now run python main.py.")
    else:
        print("\nSome downloads failed — check your internet connection and retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
