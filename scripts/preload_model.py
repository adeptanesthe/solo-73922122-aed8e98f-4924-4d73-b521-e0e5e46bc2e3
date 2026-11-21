"""Pre-download the VLM weights into the local cache (used at image build time)."""
import os

from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: F401

model_id = os.environ.get("VLM_MODEL_ID", "HuggingFaceTB/SmolVLM-256M-Instruct")
cache = os.environ.get("MODEL_CACHE_DIR", "/models")

print(f"downloading {model_id} into {cache} ...")
AutoProcessor.from_pretrained(model_id, cache_dir=cache)
AutoModelForImageTextToText.from_pretrained(model_id, cache_dir=cache)
print("done")
