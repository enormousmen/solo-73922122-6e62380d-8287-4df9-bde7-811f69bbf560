"""FastAPI surface for the single-job resumable training service.

Endpoints (interactive docs at /docs):

    GET  /                     service info
    GET  /config               effective default config + runtime info
    POST /train/start          start a fresh run (optional config overrides, force)
    POST /train/pause          stop training; 200 only after a full checkpoint is persisted
    POST /train/resume         continue from the latest checkpoint / paused state
    GET  /train/status         progress and checkpoint health
    GET  /train/losses         per-batch loss history
    GET  /train/parameters     download model parameters (.pt) from latest checkpoint
    GET  /train/checkpoint     download the full checkpoint (.pt)
    POST /train/drift          PSI distribution-drift score for an unlabeled batch
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, field_validator

from .config import load_default_config
from .trainer import (
    CheckpointSaveError,
    ConflictError,
    DriftUnavailableError,
    JobManager,
    NotFoundError,
    PauseTimeoutError,
)

manager: JobManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    runs_dir = os.environ.get("RUNS_DIR", "runs")
    manager = JobManager(runs_dir=runs_dir, default_config=load_default_config())
    yield


app = FastAPI(
    title="Resumable Training API",
    version="1.0.0",
    description=(
        "Single-job CPU training service (two-layer FC net + Dropout + Adam). "
        "Pause/resume and kill/restart recovery reproduce uninterrupted runs "
        "bit-exactly under the same environment, seed and config."
    ),
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def _json_safe_validation_handler(request: Request,
                                        exc: RequestValidationError) -> Response:
    """422 handler whose payload is always JSON-encodable.

    Pydantic echoes the offending input inside validation errors; for NaN /
    Infinity values that echo is not representable in strict JSON and makes
    the default handler crash with a 500.  Strip non-finite floats (and any
    other non-encodable objects) before serializing.
    """
    def sanitize(obj: Any) -> Any:
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        return obj

    errors = []
    for err in exc.errors():
        clean = {"type": err.get("type"), "loc": list(err.get("loc", []))}
        if err.get("msg"):
            clean["msg"] = str(err["msg"])
        if "input" in err:
            try:
                clean["input"] = sanitize(err["input"])
            except (TypeError, ValueError):
                clean["input"] = None
        errors.append(clean)
    return Response(
        content=json.dumps({"detail": errors}),
        status_code=422,
        media_type="application/json",
    )


def _conflict(exc: Exception) -> HTTPException:
    return HTTPException(status_code=409, detail=str(exc))


# Incoming unlabeled batch for POST /train/drift: 20..200 finite numeric
# vectors, each ordered exactly like the training CSV feature columns.
MIN_DRIFT_SAMPLES = 20
MAX_DRIFT_SAMPLES = 200


class DriftRequest(BaseModel):
    samples: list[list[float]] = Field(
        ...,
        description=(
            "20-200 unlabeled numeric rows, each a vector ordered exactly "
            "like the training CSV feature columns"
        ),
    )

    @field_validator("samples")
    @classmethod
    def _validate_samples(cls, samples: list[list[float]]) -> list[list[float]]:
        if not (MIN_DRIFT_SAMPLES <= len(samples) <= MAX_DRIFT_SAMPLES):
            raise ValueError(
                f"samples must contain between {MIN_DRIFT_SAMPLES} and "
                f"{MAX_DRIFT_SAMPLES} rows, got {len(samples)}")
        width = len(samples[0])
        if width == 0:
            raise ValueError("each sample must contain at least one feature value")
        for i, row in enumerate(samples):
            if len(row) != width:
                raise ValueError(
                    f"every sample must have the same length; row 0 has {width} "
                    f"values but row {i} has {len(row)}")
            for j, value in enumerate(row):
                # NB: do not interpolate the offending value into the message:
                # NaN/Inf are not representable in JSON error responses.
                if not math.isfinite(value):
                    raise ValueError(
                        f"all values must be finite; sample {i} column {j} "
                        "is NaN or infinite")
        return samples


# TEST-ONLY endpoint (enabled via ENABLE_TEST_CRASH_ENDPOINT=1): hard-exit the
# process with os._exit, simulating SIGKILL / preemption / power loss.  Used by
# scripts/verify_repro.py; never enable in production.
if os.environ.get("ENABLE_TEST_CRASH_ENDPOINT") == "1":
    @app.post("/test/crash", include_in_schema=False)
    def _test_crash() -> None:  # pragma: no cover - kills the process
        os._exit(1)


@app.get("/")
def root() -> dict:
    return {
        "service": "resumable-training-api",
        "docs": "/docs",
        "endpoints": [
            "GET /config",
            "POST /train/start",
            "POST /train/pause",
            "POST /train/resume",
            "GET /train/status",
            "GET /train/losses",
            "GET /train/parameters",
            "GET /train/checkpoint",
            "POST /train/drift",
        ],
    }


@app.get("/config")
def get_config() -> dict:
    return {
        "default_config": manager.default_config.to_dict(),
        "runs_dir": str(manager.runs_dir),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
    }


@app.post("/train/start")
def start(body: dict[str, Any] | None = Body(default=None)) -> dict:
    """Start a fresh run. Body: {"force": bool, "config": {overrides}}."""
    body = body or {}
    force = bool(body.get("force", False))
    overrides = body.get("config") or {}
    try:
        return manager.start(overrides=overrides, force=force)
    except ConflictError as exc:
        raise _conflict(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/train/pause")
async def pause() -> dict:
    """Stop training; responds only after the trainer stopped AND a complete
    checkpoint is persisted.  Runs in a thread so other queries stay live."""
    try:
        return await asyncio.to_thread(manager.pause)
    except PauseTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except CheckpointSaveError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(exc) from exc


@app.post("/train/resume")
def resume() -> dict:
    """Resume from the paused state / latest checkpoint. Only one trainer can
    run at a time: concurrent resumes yield exactly one 200 and 409s."""
    try:
        return manager.resume()
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(exc) from exc


@app.get("/train/status")
def status() -> dict:
    return manager.status()


@app.get("/train/losses")
def losses(
    since_step: int | None = Query(default=None, description="only losses with step > this"),
    limit: int | None = Query(default=None, description="max number of entries"),
) -> dict:
    return manager.get_losses(since_step=since_step, limit=limit)


@app.get("/train/parameters")
def parameters() -> Response:
    """Download current model parameters (.pt) from the latest checkpoint."""
    try:
        data = manager.read_parameters()
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="model_parameters.pt"'},
    )


@app.get("/train/checkpoint")
def checkpoint() -> FileResponse:
    """Download the full checkpoint (model + optimizer + RNG + history)."""
    try:
        path = manager.checkpoint_file()
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, filename="checkpoint.pt",
                        media_type="application/octet-stream")


@app.post(
    "/train/drift",
    responses={
        200: {"description": "Per-column PSI scores, alerts and baseline metadata"},
        400: {"description": "Sample width does not match the training feature columns"},
        404: {"description": "No published drift baseline (no checkpoint, or a "
                             "forced new run before its first successful save)"},
        422: {"description": "Body is not 20-200 finite, rectangular numeric rows"},
    },
)
def drift(request: DriftRequest) -> dict:
    """Population Stability Index (PSI) of an unlabeled online batch.

    Accepts 20-200 finite numeric vectors in training-CSV feature-column
    order.  For every feature column the baseline (the CSV actually loaded
    for the run) is split into five quantile buckets with coincident edges
    merged; out-of-range values fall into the end buckets.  Both sides add
    1e-6 to bucket proportions and renormalize, then PSI is
    ``sum (p_new - p_base) * ln(p_new / p_base)``.

    The baseline is stored with the first successful checkpoint of each run
    and never recomputed from a rewritten CSV after restart.  Returns 404
    with an explicit reason when no published baseline exists yet (no
    checkpoint at all, or a forced new run before its first save).
    """
    # Sync handler: FastAPI runs it in its worker threadpool, so the numeric
    # PSI work never blocks the event loop (training and status queries stay
    # responsive regardless).  The manager itself only takes short locks.
    try:
        return manager.compute_drift(request.samples)
    except DriftUnavailableError as exc:
        raise HTTPException(status_code=404, detail={
            "error": "drift_baseline_unavailable",
            "message": str(exc),
        }) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
