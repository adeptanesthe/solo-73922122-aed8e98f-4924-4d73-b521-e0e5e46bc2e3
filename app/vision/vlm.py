"""Vision-language backends.

Two implementations behind a common protocol:

* ``SmolVLMBackend``    - loads a local image-text model (default
  ``HuggingFaceTB/SmolVLM-256M-Instruct``) via transformers and runs real
  multimodal inference on CPU/GPU. Weights come from a local cache dir
  (``MODEL_CACHE_DIR``); set ``VLM_LOCAL_FILES_ONLY=true`` to forbid downloads.
* ``HeuristicVisionBackend`` - deterministic fallback built on classical CV
  region features only. Used when the learned model is unavailable so the API
  stays functional; every response honestly reports which backend ran.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

from PIL import Image

from app.config import Settings
from app.vision.regions import Region


def _cpu_quota() -> int | None:
    """Detect cgroup CPU quota (v2 then v1); returns core count or None."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:  # cgroup v2
            parts = fh.read().split()
            if parts[0] != "max":
                return max(1, int(int(parts[0]) // int(parts[1])))
    except (OSError, ValueError, IndexError):
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            quota = int(fh.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            period = int(fh.read().strip())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return None


def _tune_torch_threads() -> None:
    """Match torch threads to the cgroup CPU quota.

    Default torch uses all visible cores; under a container CPU quota that
    causes heavy thread thrashing (observed 20x slowdown).
    """
    quota = _cpu_quota()
    if quota is None:
        return
    try:
        import torch

        torch.set_num_threads(quota)
        os.environ.setdefault("OMP_NUM_THREADS", str(quota))
    except ImportError:  # pragma: no cover
        pass


class VLMUnavailable(RuntimeError):
    """Raised when the learned vision model cannot be loaded."""


@dataclass
class VisionAnalysis:
    backend_name: str
    model_id: str | None
    model_loaded: bool
    fallback_reason: str | None
    image_summary: str
    region_captions: list[str] = field(default_factory=list)


class SmolVLMBackend:
    name = "smolvlm"

    def __init__(self, settings: Settings) -> None:
        _tune_torch_threads()
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:  # pragma: no cover - depends on env
            raise VLMUnavailable(f"transformers/torch not installed: {exc}") from exc

        self._settings = settings
        self.model_id = settings.vlm_model_id
        self._torch = torch
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=settings.model_cache_dir,
                local_files_only=settings.vlm_local_files_only,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                cache_dir=settings.model_cache_dir,
                local_files_only=settings.vlm_local_files_only,
                dtype=torch.float32,
            )
            self._model.eval()
        except Exception as exc:  # noqa: BLE001 - surface any load failure as unavailable
            raise VLMUnavailable(f"failed to load {self.model_id}: {exc}") from exc
        self._lock = threading.Lock()

    def _generate(self, image: Image.Image, prompt_text: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=[image], return_tensors="pt")
        with self._lock, self._torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self._settings.vlm_max_new_tokens,
                do_sample=False,
            )
        new_tokens = out[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    def analyze(
        self, image: Image.Image, question: str, regions: list[Region]
    ) -> VisionAnalysis:
        # Downscale for the model: fewer vision tokens -> much faster CPU
        # inference. Region detection already ran on the full-resolution image.
        vlm_image = image.copy()
        side = self._settings.vlm_max_image_side
        vlm_image.thumbnail((side, side))
        scale = vlm_image.size[0] / image.size[0] if image.size[0] else 1.0

        summary = self._generate(
            vlm_image,
            "This is a screenshot from a software troubleshooting case. "
            f"Troubleshooting question: {question!r}. "
            "Describe any error messages, dialogs, status codes or warning "
            "indicators visible in the screenshot, in up to 3 short sentences.",
        )
        captions: list[str] = []
        w, h = vlm_image.size
        for region in regions[: self._settings.vlm_region_caption_limit]:
            pad = 12
            box = (
                max(int(region.x * scale) - pad, 0),
                max(int(region.y * scale) - pad, 0),
                min(int((region.x + region.width) * scale) + pad, w),
                min(int((region.y + region.height) * scale) + pad, h),
            )
            crop = vlm_image.crop(box)
            caption = self._generate(
                crop,
                "This is a cropped region of a troubleshooting screenshot. "
                "In one short sentence, what UI element or message does it show?",
            )
            captions.append(caption)
        return VisionAnalysis(
            backend_name=self.name,
            model_id=self.model_id,
            model_loaded=True,
            fallback_reason=None,
            image_summary=summary,
            region_captions=captions,
        )


class HeuristicVisionBackend:
    """Fallback: derives observations from CV region features only."""

    name = "heuristic"

    def __init__(self, reason: str | None = None) -> None:
        self.model_id = None
        self._reason = reason

    def analyze(
        self, image: Image.Image, question: str, regions: list[Region]
    ) -> VisionAnalysis:
        parts: list[str] = []
        red = [r for r in regions if r.kind == "red_alert"]
        amber = [r for r in regions if r.kind == "amber_warning"]
        blocks = [r for r in regions if r.kind == "contrast_block"]
        if red:
            parts.append(
                f"{len(red)} red alert-style region(s) detected, typically error banners or failure buttons"
            )
        if amber:
            parts.append(f"{len(amber)} amber warning-style region(s) detected")
        if blocks:
            parts.append(
                f"{len(blocks)} high-contrast panel/dialog-like region(s) detected"
            )
        if not parts:
            parts.append("no visually salient alert regions detected")
        summary = (
            "[heuristic vision backend - no learned model] " + "; ".join(parts) + "."
        )
        captions = [
            f"{r.kind} region ({r.width}x{r.height}px), score {r.score:.2f}"
            for r in regions
        ]
        return VisionAnalysis(
            backend_name=self.name,
            model_id=None,
            model_loaded=False,
            fallback_reason=self._reason,
            image_summary=summary,
            region_captions=captions,
        )


def build_backend(settings: Settings):
    """Select the vision backend according to configuration."""
    choice = settings.vlm_backend.lower()
    if choice == "heuristic":
        return HeuristicVisionBackend(reason="disabled by configuration (VLM_BACKEND=heuristic)")
    if choice in ("smolvlm", "auto"):
        try:
            return SmolVLMBackend(settings)
        except VLMUnavailable as exc:
            if choice == "smolvlm":
                raise
            return HeuristicVisionBackend(reason=str(exc))
    raise ValueError(f"unknown VLM_BACKEND: {settings.vlm_backend!r}")
