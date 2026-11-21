import base64
import io
import os

# Force the deterministic heuristic backend for tests (no model download).
os.environ["VLM_BACKEND"] = "heuristic"

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def make_error_screenshot() -> bytes:
    img = Image.new("RGB", (800, 450), (245, 246, 248))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 40, 800, 90], fill=(198, 40, 40))  # red error banner
    d.text((20, 58), "ERROR 502 - upstream unavailable", fill=(255, 255, 255))
    d.rectangle([0, 100, 800, 124], fill=(255, 171, 0))  # amber warning
    d.rectangle([200, 160, 600, 320], fill=(255, 255, 255), outline=(60, 60, 60), width=2)
    d.rectangle([200, 160, 600, 190], fill=(58, 58, 60))  # dialog header
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_blank_screenshot() -> bytes:
    img = Image.new("RGB", (640, 360), (245, 246, 248))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


ERROR_LOGS = "\n".join(
    [
        "2026-11-18T10:15:01Z INFO  checkout-api request received",
        "2026-11-18T10:15:07Z ERROR payment-gateway client timeout after 5000ms req=req-7f3a2c",
        "2026-11-18T10:15:07Z ERROR checkout-api charge failed: upstream connect timeout req=req-7f3a2c",
        "2026-11-18T10:15:07Z INFO  checkout-api responded 502 to client req=req-7f3a2c",
    ]
)

CLEAN_LOGS = "\n".join(
    [
        "2026-11-18T10:15:01Z INFO  checkout-api request received",
        "2026-11-18T10:15:02Z INFO  checkout-api responded 200 ok",
    ]
)
