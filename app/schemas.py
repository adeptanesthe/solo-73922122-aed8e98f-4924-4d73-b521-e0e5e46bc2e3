"""Pydantic schemas for the diagnosis API."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class DiagnosisStatus(str, Enum):
    OK = "ok"
    UNCERTAIN = "uncertain"


class EvidenceSource(str, Enum):
    SCREENSHOT = "screenshot"
    LOG = "log"


class BoundingBox(BaseModel):
    x: int = Field(..., description="left, pixels")
    y: int = Field(..., description="top, pixels")
    width: int
    height: int


class Evidence(BaseModel):
    id: str = Field(..., description="evidence id, e.g. E1")
    source: EvidenceSource
    kind: str = Field(..., description="e.g. red_alert_region, log_anomaly, vlm_observation")
    description: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    region: BoundingBox | None = None
    log_lines: list[int] = Field(default_factory=list, description="1-based line numbers")
    matched_keywords: list[str] = Field(default_factory=list)


class RelationType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INSUFFICIENT = "insufficient"  # evidence gap: needed to confirm, currently missing


class EvidenceRelation(BaseModel):
    evidence_id: str
    hypothesis_id: str
    relation: RelationType
    rationale: str


class Hypothesis(BaseModel):
    id: str = Field(..., description="hypothesis id, e.g. H1")
    statement: str
    category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(
        default_factory=list, description="what evidence would be needed to confirm"
    )


class BackendInfo(BaseModel):
    vision_backend: str
    vision_model: str | None = None
    vision_model_loaded: bool = False
    fallback_reason: str | None = None


class DiagnoseResponse(BaseModel):
    status: DiagnosisStatus
    uncertainty_reasons: list[str] = Field(default_factory=list)
    question: str
    evidence: list[Evidence]
    hypotheses: list[Hypothesis]
    relations: list[EvidenceRelation]
    next_steps: list[str]
    backend: BackendInfo


class DiagnoseJsonRequest(BaseModel):
    question: str = Field(..., min_length=1)
    logs: str = ""
    screenshot_base64: str = Field(..., description="base64-encoded PNG/JPEG image")
    filename: str | None = None


class ModelInfoResponse(BaseModel):
    backend: BackendInfo
    settings: dict[str, str | int | float | bool]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
