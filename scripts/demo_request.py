"""End-to-end demo against a running service (no model weights needed).

Starts nothing itself: point it at a service started with
``docker compose -f docker-compose.yml -f docker-compose.demo.yml up``
(heuristic backend), or any other deployment.

Usage:
    python scripts/demo_request.py --base-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


def _post_multipart(base_url: str) -> dict:
    boundary = "----demo-boundary-7f3a2c"
    parts: list[bytes] = []
    fields = {
        "question": (SAMPLES / "question.txt").read_text().strip(),
        "logs": (SAMPLES / "app_logs.txt").read_text(),
    }
    for name, value in fields.items():
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode()
        )
    image = (SAMPLES / "fault_screenshot.png").read_bytes()
    parts.append(
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="screenshot"; filename="fault_screenshot.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode()
        + image
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        f"{base_url}/v1/diagnose",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def _post_json(base_url: str) -> dict:
    import base64

    payload = {
        "question": (SAMPLES / "question.txt").read_text().strip(),
        "logs": (SAMPLES / "app_logs.txt").read_text(),
        "screenshot_base64": base64.b64encode((SAMPLES / "fault_screenshot.png").read_bytes()).decode(),
        "filename": "fault_screenshot.png",
    }
    req = urllib.request.Request(
        f"{base_url}/v1/diagnose/json",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def _summary(body: dict) -> str:
    lines = [f"status: {body['status']}  backend: {body['backend']['vision_backend']}"]
    for ev in body["evidence"]:
        region = ev.get("region")
        where = f" region={region}" if region else ""
        lines.append(f"  {ev['id']:<4} {ev['kind']:<22} confidence={ev['confidence']:.3f}{where}")
    for hyp in body["hypotheses"]:
        lines.append(
            f"  {hyp['id']} {hyp['category']:<12} confidence={hyp['confidence']:.3f} "
            f"supports={hyp['supporting_evidence']}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    try:
        with urllib.request.urlopen(f"{base}/healthz", timeout=10) as resp:
            print("healthz:", resp.read().decode())
        print("\n== POST /v1/diagnose (multipart) ==")
        multipart = _post_multipart(base)
        print(_summary(multipart))
        print("\n== POST /v1/diagnose/json ==")
        json_resp = _post_json(base)
        print(_summary(json_resp))

        def evidence_shape(body: dict) -> list:
            return [
                (e["id"], e["kind"], e.get("region"), e.get("log_lines"), e["confidence"])
                for e in body["evidence"]
            ]

        same = evidence_shape(multipart) == evidence_shape(json_resp)
        print(f"\nevidence identical across both endpoints: {same}")
        return 0 if same else 1
    except urllib.error.URLError as exc:
        print(f"could not reach {base}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
