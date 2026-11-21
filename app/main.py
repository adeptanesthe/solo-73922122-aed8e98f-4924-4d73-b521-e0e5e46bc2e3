"""FastAPI application: multimodal troubleshooting evidence correlation."""
from __future__ import annotations

import base64
import binascii
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.correlate.engine import correlate
from app.loganalysis.analyzer import analyze_logs
from app.schemas import (
    BackendInfo,
    DiagnoseJsonRequest,
    DiagnoseResponse,
    HealthResponse,
    ModelInfoResponse,
)
from app.vision.regions import detect_regions
from app.vision.vlm import HeuristicVisionBackend, build_backend


def _backend_info(backend) -> BackendInfo:
    return BackendInfo(
        vision_backend=backend.name,
        vision_model=getattr(backend, "model_id", None),
        vision_model_loaded=not isinstance(backend, HeuristicVisionBackend),
        fallback_reason=getattr(backend, "_reason", None),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.vision_backend = build_backend(settings)
    yield


app = FastAPI(
    title="Multimodal Troubleshooting Evidence Correlation API",
    description=(
        "Accepts a fault screenshot, a troubleshooting question and related log "
        "snippets; locates key screenshot regions and log anomalies with a "
        "locally loaded vision-language model (heuristic fallback), then emits "
        "numbered evidence, ranked hypotheses, support/contradiction relations "
        "and next-step suggestions. Returns `status=uncertain` with reasons "
        "when no reliable judgment can be formed."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse, tags=["meta"])
def healthz() -> HealthResponse:
    return HealthResponse()


@app.get("/v1/model/info", response_model=ModelInfoResponse, tags=["meta"])
def model_info(request: Request) -> ModelInfoResponse:
    settings = request.app.state.settings
    return ModelInfoResponse(
        backend=_backend_info(request.app.state.vision_backend),
        settings={
            "vlm_backend": settings.vlm_backend,
            "vlm_model_id": settings.vlm_model_id,
            "vlm_local_files_only": settings.vlm_local_files_only,
            "confidence_threshold": settings.confidence_threshold,
            "max_regions": settings.max_regions,
        },
    )


def _decode_image(data: bytes, settings) -> Image.Image:
    if len(data) > settings.max_image_bytes:
        raise HTTPException(413, f"screenshot too large (>{settings.max_image_bytes} bytes)")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(422, f"cannot decode screenshot as an image: {exc}") from exc
    return img.convert("RGB")


def _run_diagnosis(request: Request, image: Image.Image, question: str, logs: str) -> DiagnoseResponse:
    settings = request.app.state.settings
    backend = request.app.state.vision_backend
    if len(logs) > settings.max_log_chars:
        logs = logs[: settings.max_log_chars]
    regions = detect_regions(image, max_regions=settings.max_regions)
    vision = backend.analyze(image, question, regions)
    log_analysis = analyze_logs(logs)
    return correlate(
        question=question,
        regions=regions,
        vision=vision,
        logs=log_analysis,
        settings=settings,
        backend=_backend_info(backend),
    )


@app.post("/v1/diagnose", response_model=DiagnoseResponse, tags=["diagnosis"])
async def diagnose(
    request: Request,
    screenshot: UploadFile = File(..., description="fault screenshot (PNG/JPEG)"),
    question: str = Form(..., description="troubleshooting question"),
    logs: str = Form("", description="related log snippet (plain text)"),
) -> DiagnoseResponse:
    if not question.strip():
        raise HTTPException(422, "question must not be empty")
    data = await screenshot.read()
    image = _decode_image(data, request.app.state.settings)
    # CPU-bound work (CV + VLM inference) must not block the event loop
    return await run_in_threadpool(_run_diagnosis, request, image, question.strip(), logs)


@app.post("/v1/diagnose/json", response_model=DiagnoseResponse, tags=["diagnosis"])
def diagnose_json(request: Request, body: DiagnoseJsonRequest) -> DiagnoseResponse:
    if not body.question.strip():
        raise HTTPException(422, "question must not be empty")
    try:
        data = base64.b64decode(body.screenshot_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(422, f"screenshot_base64 is not valid base64: {exc}") from exc
    image = _decode_image(data, request.app.state.settings)
    return _run_diagnosis(request, image, body.question.strip(), body.logs)
