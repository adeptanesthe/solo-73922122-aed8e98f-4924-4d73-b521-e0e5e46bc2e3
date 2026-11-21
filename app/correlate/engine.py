"""Evidence-correlation engine.

Turns vision observations + log clues into:
  * numbered evidence items (E1..En)
  * ranked fault hypotheses (H1..Hn) with confidence
  * typed relations: supports / contradicts / insufficient
  * next-step suggestions
  * an overall status: "ok" or "uncertain" (with explicit reasons)

Everything is deterministic and explainable: every relation carries a
rationale string, and every hypothesis lists the evidence behind it.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.loganalysis.analyzer import LogAnalysis
from app.schemas import (
    BackendInfo,
    BoundingBox,
    DiagnoseResponse,
    DiagnosisStatus,
    Evidence,
    EvidenceRelation,
    EvidenceSource,
    Hypothesis,
    RelationType,
)
from app.vision.regions import Region
from app.vision.vlm import VisionAnalysis

# fault categories with keywords used to match question / VLM text
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "dependency": [
        "upstream", "downstream", "timeout", "timed out", "gateway", "circuit",
        "third-party", "third party", "latency", "unavailable", "502", "503", "504",
        "payment", "slow", "external",
    ],
    "network": [
        "network", "dns", "connection refused", "connection reset", "unreachable",
        "no route", "socket", "tls", "ssl", "handshake",
    ],
    "database": [
        "database", "db", "sql", "deadlock", "connection pool", "too many connections",
        "migration", "query",
    ],
    "auth": [
        "auth", "login", "token", "401", "403", "permission", "unauthorized",
        "forbidden", "credential", "sso",
    ],
    "resource": [
        "memory", "oom", "disk", "cpu", "space", "throttl", "quota", "limit",
        "no space",
    ],
    "application": [
        "500", "exception", "error", "crash", "bug", "panic", "traceback",
        "failed", "failure", "blank", "broken",
    ],
    "config": [
        "config", "configuration", "env var", "environment", "missing key",
        "misconfig", "flag", "setting",
    ],
}

CATEGORY_STATEMENTS: dict[str, str] = {
    "dependency": "An upstream/dependency service is failing or too slow, causing the observed errors.",
    "network": "A network connectivity problem is preventing the service from reaching its peers.",
    "database": "The database layer is failing (errors, pool exhaustion or deadlock).",
    "auth": "Authentication/authorization is rejecting otherwise valid requests.",
    "resource": "The service is hitting a resource limit (memory, disk or CPU throttling).",
    "application": "A defect or crash in the application itself is producing the errors.",
    "config": "A configuration problem (missing/incorrect setting) is breaking the service.",
}

CATEGORY_NEXT_STEPS: dict[str, list[str]] = {
    "dependency": [
        "Check health/status pages and recent deploys of the upstream dependency named in the logs.",
        "Inspect upstream latency/error metrics around the failing timestamps; verify timeout and circuit-breaker settings.",
        "Capture one failing request end-to-end (trace id) across caller and dependency.",
    ],
    "network": [
        "From the affected host, test DNS resolution and TCP connectivity to the failing destination.",
        "Check recent changes to security groups, firewall rules, proxies and service-mesh policies.",
    ],
    "database": [
        "Check database availability, connection-pool usage and slow-query/deadlock logs.",
        "Verify recent migrations and credential/secret rotations for the database user.",
    ],
    "auth": [
        "Verify token/credential validity and expiry for the failing identity.",
        "Check recent changes to IAM policies, scopes and ACLs; compare a working vs failing request's auth headers.",
    ],
    "resource": [
        "Check memory/disk/CPU usage and OOM-kill events for the service around the incident window.",
        "Review configured limits/requests and recent traffic growth; consider temporary scale-up.",
    ],
    "application": [
        "Pull the full stack trace for the failing request id and correlate with the latest deploy/commit.",
        "Roll back or feature-flag the most recent change if errors started right after a release.",
    ],
    "config": [
        "Diff current configuration/secrets against the last known-good version.",
        "Validate required environment variables and config files are present in the failing environment.",
    ],
}

GENERIC_NEXT_STEPS = [
    "Collect a longer log window (at least 15 minutes around the incident) with higher verbosity if possible.",
    "Attach infrastructure metrics (CPU/memory/network) for the affected instance around the incident.",
    "Note exact reproduction steps and whether the issue is continuous or intermittent.",
]

_REGION_KIND_CATEGORY = {
    "red_alert": {"application", "dependency"},
    "amber_warning": {"dependency", "resource"},
    "contrast_block": {"application"},
}


@dataclass
class _Ctx:
    question: str
    regions: list[Region]
    vision: VisionAnalysis
    logs: LogAnalysis
    settings: Settings


def _keyword_hits(text: str, category: str) -> list[str]:
    low = text.lower()
    return [kw for kw in CATEGORY_KEYWORDS[category] if kw in low]


def _build_evidence(ctx: _Ctx) -> list[Evidence]:
    evidence: list[Evidence] = []

    def next_id() -> str:
        return f"E{len(evidence) + 1}"

    # 1) screenshot regions (+ optional VLM caption per region)
    for i, region in enumerate(ctx.regions):
        caption = ""
        if i < len(ctx.vision.region_captions):
            caption = ctx.vision.region_captions[i]
        desc = f"Screenshot region '{region.kind}' at ({region.x},{region.y}) {region.width}x{region.height}px"
        if caption:
            desc += f" — {ctx.vision.backend_name} caption: {caption}"
        evidence.append(
            Evidence(
                id=next_id(),
                source=EvidenceSource.SCREENSHOT,
                kind=region.kind,
                description=desc,
                confidence=region.score,
                region=BoundingBox(x=region.x, y=region.y, width=region.width, height=region.height),
                matched_keywords=[],
            )
        )

    # 2) VLM overall observation
    if ctx.vision.image_summary:
        conf = 0.65 if ctx.vision.model_loaded else 0.4
        evidence.append(
            Evidence(
                id=next_id(),
                source=EvidenceSource.SCREENSHOT,
                kind="vlm_observation",
                description=f"Vision analysis of full screenshot: {ctx.vision.image_summary}",
                confidence=conf,
            )
        )

    # 3) log clues (highest severity first, already capped by analyzer)
    for clue in ctx.logs.clues:
        evidence.append(
            Evidence(
                id=next_id(),
                source=EvidenceSource.LOG,
                kind="log_anomaly",
                description=f"Log line {clue.line_no} [{clue.label}]: {clue.text}",
                confidence=round(clue.severity, 3),
                log_lines=[clue.line_no],
                matched_keywords=clue.matched,
            )
        )

    # 4) recovery / healthy signals (used for contradictions)
    if ctx.logs.recovery_after_error:
        lines = [p for p in ctx.logs.positive_lines if p > 0]
        evidence.append(
            Evidence(
                id=next_id(),
                source=EvidenceSource.LOG,
                kind="log_recovery_signal",
                description=(
                    f"Healthy/recovery markers in logs (lines {lines}) appear AFTER "
                    "the last error line — the failure may be transient or already mitigated."
                ),
                confidence=0.55,
                log_lines=lines,
            )
        )
    return evidence


def _evidence_category(ev: Evidence, ctx: _Ctx) -> set[str]:
    """Which fault categories this evidence points to."""
    cats: set[str] = set()
    if ev.kind in _REGION_KIND_CATEGORY:
        cats |= _REGION_KIND_CATEGORY[ev.kind]
    if ev.kind == "log_anomaly":
        # recover the analyzer category from the matching clue
        for clue in ctx.logs.clues:
            if clue.line_no in ev.log_lines:
                cats.add(clue.category)
                break
    if ev.kind == "vlm_observation":
        for cat in CATEGORY_KEYWORDS:
            if _keyword_hits(ev.description, cat):
                cats.add(cat)
    return cats


def correlate(
    question: str,
    regions: list[Region],
    vision: VisionAnalysis,
    logs: LogAnalysis,
    settings: Settings,
    backend: BackendInfo,
) -> DiagnoseResponse:
    ctx = _Ctx(question=question, regions=regions, vision=vision, logs=logs, settings=settings)
    evidence = _build_evidence(ctx)
    ev_cats = {ev.id: _evidence_category(ev, ctx) for ev in evidence}

    vision_text = " ".join([vision.image_summary, *vision.region_captions])

    # ---- score each category ---------------------------------------------
    scores: dict[str, float] = {}
    parts: dict[str, tuple[float, float, float]] = {}
    for cat in CATEGORY_KEYWORDS:
        log_sev = sum(c.severity for c in logs.clues if c.category == cat)
        log_part = min(log_sev / 1.6, 1.0)  # two strong clues saturate
        vision_hits = _keyword_hits(vision_text, cat)
        region_boost = any(
            ev.kind in _REGION_KIND_CATEGORY and cat in _REGION_KIND_CATEGORY[ev.kind]
            for ev in evidence
        )
        vision_part = min(0.34 * len(vision_hits) + (0.32 if region_boost else 0.0), 1.0)
        question_part = min(0.5 * len(_keyword_hits(question, cat)), 1.0)
        score = 0.55 * log_part + 0.30 * vision_part + 0.15 * question_part
        if score > 0.05:
            scores[cat] = score
            parts[cat] = (log_part, vision_part, question_part)

    # keep categories with a meaningful signal only (below ~0.15 is noise)
    ranked = [kv for kv in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if kv[1] >= 0.15][:3]

    # ---- build hypotheses + relations ------------------------------------
    hypotheses: list[Hypothesis] = []
    relations: list[EvidenceRelation] = []
    for idx, (cat, score) in enumerate(ranked, start=1):
        hid = f"H{idx}"
        confidence = round(min(score, 0.97), 3)
        supporting: list[str] = []
        contradicting: list[str] = []
        weak_support: list[str] = []

        for ev in evidence:
            if cat not in ev_cats.get(ev.id, set()):
                continue
            if ev.confidence >= 0.5:
                supporting.append(ev.id)
                relations.append(
                    EvidenceRelation(
                        evidence_id=ev.id,
                        hypothesis_id=hid,
                        relation=RelationType.SUPPORTS,
                        rationale=f"Evidence category signal matches '{cat}'.",
                    )
                )
            else:
                weak_support.append(ev.id)
                relations.append(
                    EvidenceRelation(
                        evidence_id=ev.id,
                        hypothesis_id=hid,
                        relation=RelationType.INSUFFICIENT,
                        rationale=(
                            f"Weak signal only (confidence {ev.confidence:.2f} < 0.5); "
                            "insufficient on its own to confirm this hypothesis."
                        ),
                    )
                )

        # contradiction: recovery signals in logs vs ongoing-failure hypotheses
        if cat in ("application", "dependency", "network"):
            for ev in evidence:
                if ev.kind == "log_recovery_signal":
                    contradicting.append(ev.id)
                    relations.append(
                        EvidenceRelation(
                            evidence_id=ev.id,
                            hypothesis_id=hid,
                            relation=RelationType.CONTRADICTS,
                            rationale=(
                                "Logs show healthy/recovery markers after the errors, "
                                "which argues against an ongoing outage of this kind."
                            ),
                        )
                    )

        log_part, vision_part, question_part = parts[cat]
        missing: list[str] = []
        if log_part == 0:
            missing.append("log entries naming this failure category")
        if vision_part == 0:
            missing.append("screenshot evidence (visible error dialog/banner) for this category")
        if not supporting:
            missing.append("any direct evidence; hypothesis comes only from question wording")

        hypotheses.append(
            Hypothesis(
                id=hid,
                statement=CATEGORY_STATEMENTS[cat],
                category=cat,
                confidence=confidence,
                supporting_evidence=supporting,
                contradicting_evidence=contradicting,
                missing_evidence=missing,
            )
        )

    # ---- status + uncertainty --------------------------------------------
    uncertainty: list[str] = []
    if not evidence:
        uncertainty.append("No usable evidence could be extracted from the screenshot or logs.")
    if logs.total_lines == 0:
        uncertainty.append("No log snippet was provided; analysis relies on the screenshot only.")
    if not regions:
        uncertainty.append("No salient regions were detected in the screenshot.")
    if not hypotheses:
        uncertainty.append("Evidence does not point to any known failure category.")
    top_conf = hypotheses[0].confidence if hypotheses else 0.0
    if hypotheses and top_conf < settings.confidence_threshold:
        uncertainty.append(
            f"Top hypothesis confidence {top_conf:.2f} is below the reliability "
            f"threshold {settings.confidence_threshold:.2f}."
        )
    strong_contra = any(h.contradicting_evidence for h in hypotheses[:1])
    if strong_contra and top_conf < 0.7:
        uncertainty.append("Contradicting evidence (recovery signals) is present while confidence is moderate.")

    status = DiagnosisStatus.UNCERTAIN if uncertainty else DiagnosisStatus.OK

    # ---- next steps --------------------------------------------------------
    next_steps: list[str] = []
    for h in hypotheses[:2]:
        next_steps.extend(CATEGORY_NEXT_STEPS.get(h.category, [])[:2])
    if status == DiagnosisStatus.UNCERTAIN or not hypotheses:
        next_steps.extend(GENERIC_NEXT_STEPS[: 3 if not hypotheses else 2])
    # de-duplicate, keep order
    seen: set[str] = set()
    next_steps = [s for s in next_steps if not (s in seen or seen.add(s))][:6]

    return DiagnoseResponse(
        status=status,
        uncertainty_reasons=uncertainty,
        question=question,
        evidence=evidence,
        hypotheses=hypotheses,
        relations=relations,
        next_steps=next_steps,
        backend=backend,
    )
