# BrandFit Creator Persona Core

BrandFit Core converts stored, consented Instagram-style evidence into immutable, auditable
creator personas. It does **not** accept Instagram passwords, operate a scraper, match campaigns,
send outreach, or provide a UI in v1.

```text
upstream collector -> MinIO media + ingestion manifest
                   -> post extraction and observation
                   -> creator planner and deterministic aggregates
                   -> grounded synthesis and claim verification
                   -> reviewed persona version, embeddings, relationships, exports
```

## Runtime architecture

- FastAPI exposes the ingestion, jobs, creator, review, relationship, and export APIs.
- PostgreSQL with pgvector is canonical. Every product record is workspace-scoped.
- MinIO owns source media, derived keyframes/audio, and versioned datasets.
- Celery and Redis run ingestion, rebuild, export, and weekly-new-evidence jobs.
- LangGraph executes planner -> post analyst -> synthesizer -> verifier -> publisher and uses
  PostgreSQL checkpoints.
- The fixed multilingual embedding model is `BAAI/bge-m3`; generative providers are swappable.

Comments are retained in `stored_comments` but no analysis or persona query reads them. Source
media remains until an explicit future deletion workflow is introduced.

## Agent contracts

| Component | Typed input | Typed output | Authority |
|---|---|---|---|
| Creator Analysis Planner | creator inventory, prior observations, confidence, six-call budget | `CreatorAnalysisPlan` | chooses priority/sample; cannot remove valid selected evidence silently |
| Post Analyst | caption, frames, transcript, OCR, metrics, public profile | `PostObservation` | reads evidence; bounded ReAct only below threshold |
| Persona Synthesizer | deterministic all-history/recent aggregates and evidence examples | `PersonaProposal` | writes narrative claims only; cannot publish |
| Persona Verifier | proposal plus exact observations | `PersonaVerificationReport` | rejects invented evidence and prohibited inference |
| Deterministic Publisher | accepted typed proposal/report | immutable persona version | sole writer of persona versions/latest pointers |

Working memory is LangGraph state/checkpoints. Plans, steps, attempts, errors, and verification are
episodic rows. Approved observations/personas/taxonomy/embeddings are semantic memory. Versioned
prompts/contracts/configuration are procedural memory. Human reviews are workspace-private
feedback memory; agents never promote their own free text into shared memory.

## Start locally

Requirements: Docker with Compose. Copy the environment file and start the stack:

```powershell
Copy-Item .env.example .env
docker compose up --build -d
Invoke-RestMethod http://localhost:8000/health
```

The default `fake:fixture-v1` provider is deterministic and offline. For a real chain, set for
example:

```text
BRANDFIT_PROVIDER_CHAIN=gemini:gemini-2.5-flash,openai:gpt-5-mini,groq:your-model,ollama:qwen2.5vl
```

Only configured operational failures trigger fallback; a safety refusal stops immediately.
Provider attempts, latency, token counts, and nullable cost estimates are persisted.
Configure current cloud-model rates through `BRANDFIT_PROVIDER_PRICING_PER_MILLION`; exact
`provider:model` entries take precedence over provider defaults.

Generate and load the CC0 synthetic multilingual benchmark:

```powershell
docker compose exec api python fixtures/generate_dataset.py
docker compose exec api python fixtures/load_dataset.py
```

For a host-side load, install the project first and use `python fixtures/load_dataset.py`.

## Public API

- `POST /v1/ingestion/uploads` creates a presigned MinIO upload URL.
- `POST /v1/ingestion/manifests` validates and atomically stores valid posts while quarantining
  corrupt/invalid posts.
- `GET /v1/ingestion/batches/{id}` and `GET /v1/jobs/{id}` report asynchronous progress.
- `GET /v1/creators/{id}/persona`, `/persona/versions`, `/evidence`, and `/relationships` read data.
- `POST /v1/creators/{id}/rebuild` runs a versioned rebuild.
- `POST /v1/creators/{id}/reviews` approves/rejects the newest pending version.
- `POST /v1/exports` and `GET /v1/exports/{id}` create reproducible JSONL/Parquet snapshots.

The single v1 admin workspace uses `X-Workspace-Id`; omitted means the seeded default workspace.

## Quality checks

```powershell
uv sync --extra dev --extra media
uv run ruff check .
uv run pytest
docker compose config --quiet
```

With the stack and all ten fixtures processed, run:

```powershell
uv run python scripts/run_acceptance.py
```

First and verifier-flagged personas are stored as pending versions. A human must approve them
before they become current or participate in nearest-creator discovery. Campaign scoring remains
intentionally outside this repository until persona acceptance gates pass.
