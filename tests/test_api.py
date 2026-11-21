"""End-to-end API tests using the deterministic heuristic vision backend."""
from __future__ import annotations

import base64

from tests.conftest import (
    CLEAN_LOGS,
    ERROR_LOGS,
    make_blank_screenshot,
    make_error_screenshot,
)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_info_reports_heuristic_backend(client):
    r = client.get("/v1/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"]["vision_backend"] == "heuristic"
    assert body["backend"]["vision_model_loaded"] is False
    assert body["backend"]["fallback_reason"]


def test_docs_available(client):
    r = client.get("/docs")
    assert r.status_code == 200
    assert "swagger" in r.text.lower()


def test_diagnose_multipart_with_error_evidence(client):
    r = client.post(
        "/v1/diagnose",
        files={"screenshot": ("fault.png", make_error_screenshot(), "image/png")},
        data={
            "question": "Checkout shows 502 and payments time out. Likely cause?",
            "logs": ERROR_LOGS,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()

    # basic shape
    assert body["status"] in ("ok", "uncertain")
    assert body["evidence"], "expected evidence items"
    assert body["hypotheses"], "expected at least one hypothesis"
    assert body["next_steps"], "expected next steps"

    # evidence ids are unique and well-formed
    ev_ids = [e["id"] for e in body["evidence"]]
    assert len(ev_ids) == len(set(ev_ids))
    assert all(e.startswith("E") for e in ev_ids)

    # screenshot evidence carries bounding boxes; log evidence carries line numbers
    shot_ev = [e for e in body["evidence"] if e["source"] == "screenshot"]
    log_ev = [e for e in body["evidence"] if e["source"] == "log"]
    assert shot_ev and log_ev
    assert any(e["region"] for e in shot_ev if e["kind"] != "vlm_observation")
    assert any(e["log_lines"] for e in log_ev)

    # every relation references existing evidence and hypothesis ids
    hyp_ids = {h["id"] for h in body["hypotheses"]}
    for rel in body["relations"]:
        assert rel["evidence_id"] in ev_ids
        assert rel["hypothesis_id"] in hyp_ids
        assert rel["relation"] in ("supports", "contradicts", "insufficient")
        assert rel["rationale"]

    # hypotheses reference existing evidence and are ranked by confidence
    for h in body["hypotheses"]:
        for ref in h["supporting_evidence"] + h["contradicting_evidence"]:
            assert ref in ev_ids
    confs = [h["confidence"] for h in body["hypotheses"]]
    assert confs == sorted(confs, reverse=True)

    # the dependency/upstream hypothesis should win for this scenario
    top = body["hypotheses"][0]
    assert top["category"] == "dependency"
    assert top["confidence"] >= 0.45
    assert body["status"] == "ok"
    assert body["backend"]["vision_backend"] == "heuristic"


def test_diagnose_uncertain_when_no_evidence(client):
    r = client.post(
        "/v1/diagnose",
        files={"screenshot": ("blank.png", make_blank_screenshot(), "image/png")},
        data={"question": "Something is wrong somewhere.", "logs": CLEAN_LOGS},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "uncertain"
    assert body["uncertainty_reasons"], "uncertain status must carry reasons"


def test_diagnose_json_base64(client):
    payload = {
        "question": "502 errors on checkout, payments time out",
        "logs": ERROR_LOGS,
        "screenshot_base64": base64.b64encode(make_error_screenshot()).decode(),
    }
    r = client.post("/v1/diagnose/json", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["hypotheses"]
    assert body["evidence"]


def test_diagnose_rejects_invalid_image(client):
    r = client.post(
        "/v1/diagnose",
        files={"screenshot": ("x.png", b"not an image", "image/png")},
        data={"question": "anything", "logs": ""},
    )
    assert r.status_code == 422


def test_diagnose_rejects_empty_question(client):
    r = client.post(
        "/v1/diagnose",
        files={"screenshot": ("fault.png", make_error_screenshot(), "image/png")},
        data={"question": "   ", "logs": ""},
    )
    assert r.status_code == 422


def test_diagnose_json_rejects_bad_base64(client):
    r = client.post(
        "/v1/diagnose/json",
        json={"question": "q", "logs": "", "screenshot_base64": "!!!not-base64!!!"},
    )
    assert r.status_code == 422
