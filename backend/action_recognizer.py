# backend/action_recognizer.py
# MMAction2-based action recognition using a sliding-window approach.
#
# AVAILABILITY: MMAction2 requires mmcv + mmaction2, which as of 2026-05 have
# no prebuilt wheels for Python 3.13 or CPU-only PyTorch on Windows.
# This module detects availability at import time and sets MMACTION2_AVAILABLE.
# All public methods work regardless — they return empty results with a warning
# when the stack is missing, so the rest of the pipeline degrades gracefully.
#
# To enable:
#   1. Use Python 3.10–3.11 with CUDA-enabled PyTorch
#   2. pip install -U openmim
#   3. mim install mmengine mmcv
#   4. mim install mmaction2
#
# REGION COUNTING INJECTION POINT: Once spatial region counts are added
# in interaction_mapper.py, the region where an action occurs (e.g. "near
# toy shelf") can be used here to filter which action clips are relevant
# to each spatial region, improving label accuracy.

from __future__ import annotations
import os
import tempfile
from dataclasses import dataclass
from typing import Callable

import cv2

# ---------------------------------------------------------------------------
# Availability check — performed once at import time
# ---------------------------------------------------------------------------

MMACTION2_AVAILABLE = False
_UNAVAILABLE_REASON = ""

try:
    import mmcv                                          # noqa: F401
    from mmaction.apis import init_recognizer, inference_recognizer  # noqa: F401
    MMACTION2_AVAILABLE = True
except ImportError as _e:
    _UNAVAILABLE_REASON = str(_e)

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------

ACTION_CONF_THRESHOLD = 0.50   # minimum confidence to accept an MMAction2 label
CLIP_LENGTH_FRAMES    = 32     # frames extracted per sliding window
CLIP_STRIDE_FRAMES    = 16     # step between windows (50 % overlap)
CLIP_TEMP_FPS         = 8      # fps when writing temp clips (lower = faster)

# Label map files (relative to project/backend/mmaction_configs/).
# Used as fallback when the checkpoint's meta dict lacks class names.
# Downloaded by setup_mmaction_configs.py.
_LABEL_MAPS: dict[str, str] = {
    "kinetics400": "tools/data/kinetics/label_map_k400.txt",
    "kinetics700": "tools/data/kinetics/label_map_k700.txt",
    "sthv2":       "tools/data/sthv2/label_map.txt",
}

# All supported models.  The waterfall iterates only the entries whose
# "dataset" key appears in ActionRecognizer's enabled_models list.
# Keys used in the UI: "kinetics400", "sthv2".
_CANDIDATE_MODELS: list[dict] = [
    {
        "name":    "TSN Kinetics-400",
        "dataset": "kinetics400",
        "desc":    "General actions (running, jumping, sports, play)",
        "config":  "tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/"
            "tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/"
            "tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_"
            "20220906-cd10898e.pth"
        ),
    },
    {
        "name":    "TSN SomethingV2",
        "dataset": "sthv2",
        "desc":    "Hand-object interactions (picking up, pushing, throwing)",
        "config":  "tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-50e_sthv2-rgb.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/"
            "tsn_imagenet-pretrained-r50_8xb32-1x1x8-50e_sthv2-rgb/"
            "tsn_imagenet-pretrained-r50_8xb32-1x1x8-50e_sthv2-rgb_"
            "20230313-06ad7d03.pth"
        ),
    },
    {
        "name":    "TSN Kinetics-700",
        "dataset": "kinetics700",
        "desc":    "700-category general actions — better child-play coverage than K400",
        "config":  "tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics700-rgb.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/"
            "tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics700-rgb/"
            "tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics700-rgb_"
            "20220906-797d6b77.pth"
        ),
    },
]

# Convenience: dataset key → display name (used by the UI button label)
MODEL_DISPLAY_NAMES: dict[str, str] = {
    m["dataset"]: m["name"] for m in _CANDIDATE_MODELS
}

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ActionClip:
    """One recognized action window from the video."""
    start_ms: float
    end_ms: float
    action_label: str      # simplified, display-ready
    raw_label: str         # original model output
    confidence: float
    model_name: str        # which candidate model produced this


# ---------------------------------------------------------------------------
# Recognizer
# ---------------------------------------------------------------------------

class ActionRecognizer:
    """
    Runs MMAction2 inference on a video using a sliding window.

    Usage:
        recognizer = ActionRecognizer()
        if not recognizer.is_available():
            print(recognizer.unavailable_reason())
        clips = recognizer.recognize("session.mp4", progress_cb=lambda p: ...)
        # clips: list[ActionClip], empty when MMAction2 is not available
    """

    def __init__(
        self,
        conf_threshold: float = ACTION_CONF_THRESHOLD,
        clip_length: int = CLIP_LENGTH_FRAMES,
        clip_stride: int = CLIP_STRIDE_FRAMES,
        enabled_models: list[str] | None = None,
    ):
        """
        enabled_models: list of dataset keys to try, in priority order.
            e.g. ["kinetics400"]  or  ["sthv2"]  or  ["kinetics400", "sthv2"]
            None → try all models in _CANDIDATE_MODELS order (original behaviour).
        """
        self._conf           = conf_threshold
        self._clip_len       = clip_length
        self._stride         = clip_stride
        self._enabled_models = enabled_models   # None = all
        self._model          = None
        self._model_meta: dict = {}

    # ------------------------------------------------------------------ public

    @staticmethod
    def is_available() -> bool:
        return MMACTION2_AVAILABLE

    @staticmethod
    def unavailable_reason() -> str:
        if MMACTION2_AVAILABLE:
            return ""
        return (
            f"MMAction2 is not available on this environment "
            f"(Python 3.13, CPU-only PyTorch, Windows). "
            f"No mmcv wheels exist for cp313. "
            f"To enable: use Python 3.10–3.11 with a CUDA PyTorch build, "
            f"then: pip install openmim && mim install mmengine mmcv mmaction2. "
            f"Import error: {_UNAVAILABLE_REASON}"
        )

    def recognize(
        self,
        video_path: str,
        progress_cb: Callable[[int], None] | None = None,
    ) -> list[ActionClip]:
        """
        Run sliding-window action recognition on the video.
        Returns empty list (with a console warning) when MMAction2 is unavailable.
        """
        if not MMACTION2_AVAILABLE:
            print(f"[ActionRecognizer] WARNING: {self.unavailable_reason()}")
            return []

        if self._model is None:
            self._load_model()
            if self._model is None:
                return []

        return self._run_sliding_window(video_path, progress_cb)

    # ------------------------------------------------------------------ model loading

    def _load_model(self):
        from mmaction.apis import init_recognizer
        import mmaction

        mmaction_root = os.path.dirname(mmaction.__file__)

        # Search order for config files:
        #   1. <package>/.mim/configs/recognition/  (mim install)
        #   2. <package>/configs/recognition/        (some pip builds)
        #   3. <project>/backend/mmaction_configs/configs/recognition/  (our download)
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _cfg_search = [
            os.path.join(mmaction_root, ".mim",    "configs", "recognition"),
            os.path.join(mmaction_root,            "configs", "recognition"),
            os.path.join(_project_root, "backend", "mmaction_configs",
                         "configs", "recognition"),
        ]

        # Filter candidate list to only enabled models, preserving order
        candidates = (
            _CANDIDATE_MODELS if self._enabled_models is None
            else [m for m in _CANDIDATE_MODELS
                  if m["dataset"] in self._enabled_models]
        )
        if not candidates:
            print("[ActionRecognizer] ERROR: No models enabled — "
                  "check your model selection.")
            return

        for candidate in candidates:
            cfg_path = None
            for root in _cfg_search:
                candidate_path = os.path.join(root, candidate["config"])
                if os.path.exists(candidate_path):
                    cfg_path = candidate_path
                    break

            if cfg_path is None:
                print(f"[ActionRecognizer] Config not found in any search path: "
                      f"{candidate['config']}")
                print(f"[ActionRecognizer]   Run:  python setup_mmaction_configs.py")
                continue
            try:
                print(f"[ActionRecognizer] Loading {candidate['name']} from {cfg_path}…")
                self._model = init_recognizer(
                    cfg_path,
                    candidate["checkpoint"],
                    device="cuda:0",
                )
                self._model_meta = candidate

                # mmaction2 v1.2.0 init_recognizer does NOT set dataset_meta.
                # Try three sources in order:
                #   1. Cached checkpoint meta dict  (most accurate)
                #   2. Project label map file       (downloaded by setup script)
                if not getattr(self._model, "dataset_meta", None):
                    self._model.dataset_meta = {}
                if not self._model.dataset_meta.get("classes"):
                    classes = (
                        self._classes_from_checkpoint(candidate["checkpoint"])
                        or self._classes_from_label_map(candidate["dataset"])
                    )
                    if classes:
                        self._model.dataset_meta["classes"] = classes
                        print(f"[ActionRecognizer]   {len(classes)} class names loaded")

                print(f"[ActionRecognizer] Loaded: {candidate['name']}")
                return
            except Exception as exc:
                print(f"[ActionRecognizer] Failed to load {candidate['name']}: {exc}")

        print("[ActionRecognizer] ERROR: All candidate models failed to load.")

    # ------------------------------------------------------------------ class name helpers

    @staticmethod
    def _classes_from_checkpoint(checkpoint_url: str) -> list[str]:
        """Try to read CLASSES from the cached .pth file's meta dict."""
        import torch as _torch
        ckpt_name = os.path.basename(checkpoint_url)
        ckpt_path = os.path.join(
            os.path.expanduser("~"),
            ".cache", "torch", "hub", "checkpoints", ckpt_name,
        )
        if not os.path.exists(ckpt_path):
            return []
        try:
            meta    = _torch.load(ckpt_path, map_location="cpu").get("meta", {})
            classes = meta.get("CLASSES") or meta.get("classes", [])
            return list(classes) if classes else []
        except Exception:
            return []

    @staticmethod
    def _classes_from_label_map(dataset: str) -> list[str]:
        """Read class names from the project-local label map file."""
        rel = _LABEL_MAPS.get(dataset, "")
        if not rel:
            return []
        label_map_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "backend", "mmaction_configs", rel,
        )
        if not os.path.exists(label_map_path):
            return []
        try:
            with open(label_map_path, encoding="utf-8") as f:
                classes = [ln.strip() for ln in f if ln.strip()]
            return classes
        except Exception:
            return []

    # ------------------------------------------------------------------ inference

    def _run_sliding_window(
        self, video_path: str, progress_cb: Callable[[int], None] | None
    ) -> list[ActionClip]:
        from mmaction.apis import inference_recognizer

        cap = cv2.VideoCapture(video_path)
        fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        clips: list[ActionClip] = []
        window_count = max(1, (total - self._clip_len) // self._stride + 1)
        done = 0

        start_frame = 0
        while start_frame + self._clip_len <= total:
            frames = self._extract_frames(cap, start_frame, self._clip_len)
            if not frames:
                break

            clip = self._infer_clip(
                frames,
                start_ms=start_frame / fps * 1000,
                end_ms=(start_frame + self._clip_len) / fps * 1000,
                inference_fn=inference_recognizer,
            )
            # Store ALL clips — UI filters by threshold live (same as YOLO)
            if clip is not None:
                clips.append(clip)
                if clip.confidence >= self._conf:
                    print(
                        f"[ActionRecognizer] {clip.start_ms/1000:.1f}s–"
                        f"{clip.end_ms/1000:.1f}s  {clip.action_label!r}  "
                        f"conf={clip.confidence:.3f}"
                    )

            start_frame += self._stride
            done += 1
            if progress_cb:
                progress_cb(int(done / window_count * 100))

        cap.release()
        n_above = sum(1 for c in clips if c.confidence >= self._conf)
        print(f"[ActionRecognizer] {len(clips)} total clips captured "
              f"({n_above} above display threshold {self._conf}) from {done} windows.")
        return clips

    def _extract_frames(
        self, cap: cv2.VideoCapture, start: int, length: int
    ) -> list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(length):
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(bgr)
        return frames

    def _infer_clip(
        self,
        frames: list,
        start_ms: float,
        end_ms: float,
        inference_fn,
    ) -> ActionClip | None:
        # Write frames to a temp video file so inference_recognizer can read it
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tmp_path = tf.name

        try:
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                tmp_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                CLIP_TEMP_FPS,
                (w, h),
            )
            for f in frames:
                writer.write(f)
            writer.release()

            result = inference_fn(self._model, tmp_path)
            # mmaction2 v1.x: per-class scores live in pred_score (tensor)
            scores = result.pred_score.cpu().numpy()
            top_idx = int(scores.argmax())
            confidence = float(scores[top_idx])

            # Retrieve label name (dataset_meta may be absent in older mmaction2 builds)
            label_map = []
            if hasattr(self._model, "dataset_meta") and self._model.dataset_meta:
                label_map = self._model.dataset_meta.get("classes", [])
            raw_label = (label_map[top_idx]
                         if top_idx < len(label_map)
                         else f"action_{top_idx}")
            action_label = _simplify_label(raw_label)

            return ActionClip(
                start_ms=start_ms,
                end_ms=end_ms,
                action_label=action_label,
                raw_label=raw_label,
                confidence=confidence,
                model_name=self._model_meta.get("name", "unknown"),
            )
        except Exception as exc:
            print(f"[ActionRecognizer] Inference error: {exc}")
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _simplify_label(raw: str) -> str:
    """
    Convert verbose model labels to short display strings.
    Examples:
        "Moving something up"           → "Moving"
        "Throwing something"            → "Throwing"
        "playing with (not on) computer"→ "Playing"
    """
    label = raw.strip()
    # SSv2-style: remove "something/somewhere" and trailing qualifiers
    for filler in (" something", " somewhere", " from left to right",
                   " from right to left", " toward the camera",
                   " away from the camera"):
        label = label.replace(filler, "")
    # Capitalize first word only
    words = label.split()
    if words:
        label = words[0].capitalize() + (" " + " ".join(words[1:]) if len(words) > 1 else "")
    return label or raw


def apply_action_labels(
    segments: list,
    action_clips: list[ActionClip],
    conf_threshold: float = ACTION_CONF_THRESHOLD,
) -> list:
    """
    Overlay MMAction2 action clips onto YOLO-derived segments.

    For each segment, find the action clip with the highest confidence that
    overlaps the segment's time range.  If its confidence exceeds the threshold,
    replace segment.interaction_type with the action verb and record the
    confidence in segment.action_confidence.

    Returns the same segment list (mutated in-place for efficiency).
    """
    if not action_clips:
        return segments

    for seg in segments:
        best_clip: ActionClip | None = None
        best_conf = 0.0
        # Treat zero-duration segments (single detected frame) as 1 ms wide
        seg_end = seg.end_ms if seg.end_ms > seg.start_ms else seg.start_ms + 1
        for clip in action_clips:
            # Overlap check: [seg.start_ms, seg_end) ∩ [clip.start_ms, clip.end_ms)
            if clip.end_ms <= seg.start_ms or clip.start_ms >= seg_end:
                continue
            if clip.confidence > best_conf:
                best_conf = clip.confidence
                best_clip = clip

        if best_clip and best_conf >= conf_threshold:
            seg.interaction_type  = best_clip.action_label
            seg.action_confidence = best_conf
            seg.action_source     = "mmaction2"

    return segments
