"""Runtime configuration, sourced from environment variables."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    # Vision-language backend: "smolvlm" | "heuristic" | "auto"
    # auto = try smolvlm, fall back to heuristic when it cannot be loaded.
    vlm_backend: str = "auto"
    vlm_model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    model_cache_dir: str = ".cache/models"
    # Only use files already present in the cache (no network download at runtime).
    vlm_local_files_only: bool = False
    vlm_max_new_tokens: int = 160
    # Max number of screenshot regions that get an individual VLM caption.
    vlm_region_caption_limit: int = 2
    # Longest image side fed to the VLM (downscaled for CPU-friendly latency).
    vlm_max_image_side: int = 512

    max_image_bytes: int = 15 * 1024 * 1024
    max_log_chars: int = 200_000
    max_regions: int = 8

    # A hypothesis must reach this confidence to be reported as reliable.
    confidence_threshold: float = 0.45


@lru_cache
def get_settings() -> Settings:
    return Settings()
