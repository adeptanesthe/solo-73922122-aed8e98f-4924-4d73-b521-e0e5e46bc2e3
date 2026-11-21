# Multimodal Troubleshooting Evidence Correlation API

A locally runnable FastAPI service that correlates **fault screenshots**, a
**troubleshooting question** and **log snippets** into numbered evidence,
ranked fault hypotheses, support/contradiction relations and next-step
suggestions. When the evidence is too weak or contradictory, it answers with
`status=uncertain` and explicit reasons instead of guessing.

## How it works

```
screenshot ──┬─ CV region detection (red alerts / amber warnings / dialog blocks)
             └─ local VLM (SmolVLM-256M) ── full-image observation + region captions
logs ────────┴─ rule-based anomaly extraction (severity + category, line numbers)
                              │
                    correlation engine
                              │
   evidence E1..En ─ hypotheses H1..Hn ─ relations (supports/contradicts/insufficient)
                              │
              status ok|uncertain + reasons + next steps
```

* **Vision-language model** — `HuggingFaceTB/SmolVLM-256M-Instruct` loaded
  locally via transformers (CPU-friendly, ~500 MB weights). If the model cannot
  be loaded (no weights, no network), the service degrades to a deterministic
  heuristic backend and **says so in every response** (`backend` field).
* **Region localization** — classical CV (numpy) finds error-colored and
  high-contrast regions; the VLM describes the full screenshot and the top
  regions. Bounding boxes are returned in pixel coordinates.
* **Log analysis** — regex rules map lines to severity + fault category
  (network / database / auth / resource / dependency / application / config)
  and detect recovery signals used for contradiction checks.
* **Correlation** — deterministic scoring per fault category; every hypothesis
  lists supporting/contradicting/missing evidence, every relation carries a
  rationale. No hidden state, fully explainable output.

## Quick start (Docker Compose)

```bash
docker compose up --build
# open http://localhost:8000/docs
```

The compose build pre-downloads the SmolVLM weights into the image
(`PRELOAD_MODEL=true`), so the container runs fully offline at runtime. If
huggingface.co is unreachable, set the `HF_ENDPOINT` build arg in
`docker-compose.yml` to a mirror. To skip the baked-in model, set
`PRELOAD_MODEL=false` — weights are then downloaded on first start into the
`models` volume.

## Quick start (local, no Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt

# generate a demo screenshot + logs
python scripts/make_sample.py --out samples

# run (heuristic backend — no model download needed)
VLM_BACKEND=heuristic uvicorn app.main:app --port 8000

# or run with the real local VLM (downloads ~500 MB once into ./.cache/models)
MODEL_CACHE_DIR=.cache/models VLM_BACKEND=auto uvicorn app.main:app --port 8000
```

> CPU note: on a 2-core container, one diagnosis with the VLM takes roughly
> 1–3 minutes (one full-image pass + up to `VLM_REGION_CAPTION_LIMIT` region
> captions). The heuristic backend answers in milliseconds. The app
> auto-detects the cgroup CPU quota and matches torch threads to it.

## API

| Method | Path             | Purpose                                             |
|--------|------------------|-----------------------------------------------------|
| GET    | `/healthz`       | liveness                                            |
| GET    | `/v1/model/info` | active vision backend + settings                    |
| POST   | `/v1/diagnose`   | multipart: `screenshot` (file), `question`, `logs`  |
| POST   | `/v1/diagnose/json` | JSON: `screenshot_base64`, `question`, `logs`    |
| GET    | `/docs`          | Swagger UI (try the endpoints interactively)        |

### Example

```bash
curl -s http://localhost:8000/v1/diagnose \
  -F "screenshot=@samples/fault_screenshot.png" \
  -F "question=Checkout page shows 502 and payments time out. Likely cause?" \
  -F "logs=$(cat samples/app_logs.txt)" | jq
```

Response (abridged):

```json
{
  "status": "ok",
  "uncertainty_reasons": [],
  "evidence": [
    {"id": "E1", "source": "screenshot", "kind": "red_alert",
     "region": {"x": 0, "y": 60, "width": 960, "height": 60}, "confidence": 0.98, ...},
    {"id": "E6", "source": "log", "kind": "log_anomaly",
     "log_lines": [3], "confidence": 0.8, ...}
  ],
  "hypotheses": [
    {"id": "H1", "category": "dependency", "confidence": 0.8,
     "supporting_evidence": ["E1", "E2", "E6", ...],
     "contradicting_evidence": ["E10"], "missing_evidence": []}
  ],
  "relations": [
    {"evidence_id": "E6", "hypothesis_id": "H1", "relation": "supports", "rationale": "..."},
    {"evidence_id": "E10", "hypothesis_id": "H1", "relation": "contradicts", "rationale": "..."}
  ],
  "next_steps": ["Check health/status pages of the upstream dependency ...", ...],
  "backend": {"vision_backend": "smolvlm", "vision_model_loaded": true, ...}
}
```

When the evidence is insufficient, the response becomes:

```json
{"status": "uncertain",
 "uncertainty_reasons": ["Top hypothesis confidence 0.31 is below the reliability threshold 0.45.", ...],
 "hypotheses": [...], "next_steps": [...]}
```

## Configuration (environment variables)

| Variable                   | Default                                  | Meaning                                   |
|----------------------------|------------------------------------------|-------------------------------------------|
| `VLM_BACKEND`              | `auto`                                   | `auto` / `smolvlm` / `heuristic`          |
| `VLM_MODEL_ID`             | `HuggingFaceTB/SmolVLM-256M-Instruct`    | any transformers image-text-to-text model |
| `MODEL_CACHE_DIR`          | `/models` (image) / `.cache/models` (default) | local weights cache                  |
| `VLM_LOCAL_FILES_ONLY`     | `true` (image) / `false` (defaults)      | forbid runtime downloads                  |
| `VLM_MAX_NEW_TOKENS`       | `160`                                    | generation budget per VLM call            |
| `VLM_REGION_CAPTION_LIMIT` | `2`                                      | region crops captioned per request        |
| `VLM_MAX_IMAGE_SIDE`       | `512`                                    | downscale bound for VLM input             |
| `CONFIDENCE_THRESHOLD`     | `0.45`                                   | min confidence for `status=ok`            |
| `MAX_IMAGE_BYTES`          | `15728640`                               | upload size limit                         |
| `MAX_LOG_CHARS`            | `200000`                                 | log snippet size limit                    |

## Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v        # uses the deterministic heuristic backend
```

## Known limitations

* SmolVLM-256M is a small model: it reads prominent dialog/banner text well
  but can miss fine print; treat its observations as hints, the CV/log signals
  as the backbone.
* CPU inference is slow (see note above); use a GPU or the heuristic backend
  for interactive latency.
* Log rules are English-keyword oriented.
