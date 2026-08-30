from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from brandfit_core import __version__
from brandfit_core.api.routers import creators, exports, ingestion, jobs

app = FastAPI(
    title="BrandFit Creator Persona Core",
    version=__version__,
    description=(
        "Evidence-grounded creator persona pipeline. No Instagram credentials are accepted."
    ),
)
app.include_router(ingestion.router)
app.include_router(jobs.router)
app.include_router(creators.router)
app.include_router(exports.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


def run() -> None:
    uvicorn.run("brandfit_core.api.main:app", host="0.0.0.0", port=8000, reload=False)
