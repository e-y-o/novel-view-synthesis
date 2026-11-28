"""Routes that use a session's depth: coverage, limit, render and result download.

Poses are validated here (torch-free, numpy only) and again by the worker: R must be a finite 3x3 proper
rotation, t finite with |t| < 1000. The client never chooses the resolution or the intrinsics: the worker
always renders with the session's own size and focal length.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, field_validator

from nvs.api.errors import ApiError
from nvs.api.sessions import SessionStore
from nvs.jobs.scheduler import JobHandle, JobOutcome, Priority, Scheduler

MAX_T = 1000.0
ROT_TOL = 1e-4  # same tolerance as nvs.core.camera.make_camera
POLL_S = 0.25
RESULT_TYPES = {".png": "image/png", ".jpg": "image/jpeg"}
STATUS_FOR_CODE = {
    "worker_unavailable": 503,
    "worker_died": 503,
    "gpu_oom": 507,
    "timeout": 504,
    "depth_missing": 409,
    "not_ready": 409,
    "invalid_request": 422,
}


def check_pose(R: list[list[float]], t: list[float]) -> None:
    """Raise ``ValueError`` unless (R, t) is a finite 3x3 proper rotation and a translation under 1 km."""
    try:
        r = np.asarray(R, dtype=np.float64)
        tv = np.asarray(t, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError("R and t must be numeric") from e
    if r.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got shape {list(r.shape)}")
    if tv.shape != (3,):
        raise ValueError(f"t must have 3 numbers, got shape {list(tv.shape)}")
    if not (np.isfinite(r).all() and np.isfinite(tv).all()):
        raise ValueError("R and t must be finite")
    if np.abs(r.T @ r - np.eye(3)).max() > ROT_TOL:
        raise ValueError("R is not orthonormal")
    if np.linalg.det(r) < 0:
        raise ValueError("R is a reflection (det < 0), not a rotation")
    if np.linalg.norm(tv) >= MAX_T:
        raise ValueError(f"|t| must be below {MAX_T:g} m")


def _finite(v: float, name: str, limit: float) -> float:
    if not math.isfinite(v) or abs(v) > limit:
        raise ValueError(f"{name} must be a finite number with magnitude <= {limit:g}")
    return v


class Orbit(BaseModel):
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    dolly: float = 0.0
    pan: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    pivot_z: float

    @field_validator("yaw_deg")
    @classmethod
    def _yaw(cls, v: float) -> float:
        return _finite(v, "yaw_deg", 180.0)

    @field_validator("pitch_deg")
    @classmethod
    def _pitch(cls, v: float) -> float:
        return _finite(v, "pitch_deg", 89.0)

    @field_validator("dolly")
    @classmethod
    def _dolly(cls, v: float) -> float:
        return _finite(v, "dolly", MAX_T)

    @field_validator("pan")
    @classmethod
    def _pan(cls, v: list[float]) -> list[float]:
        if len(v) != 2:
            raise ValueError("pan must have 2 numbers")
        return [_finite(x, "pan", MAX_T) for x in v]

    @field_validator("pivot_z")
    @classmethod
    def _pivot(cls, v: float) -> float:
        if not (math.isfinite(v) and 0.0 < v <= MAX_T):
            raise ValueError(f"pivot_z must be in (0, {MAX_T:g}] metres")
        return v


class PoseBody(BaseModel):
    R: list[list[float]]
    t: list[float]
    orbit: Orbit | None = None  # informational (UI display, EXIF note); R and t are authoritative

    @field_validator("t")
    @classmethod
    def _check(cls, t: list[float], info) -> list[float]:
        R = info.data.get("R")
        if R is not None:
            check_pose(R, t)
        return t


class RenderBody(PoseBody):
    size: Literal["source", "preview"] = "source"
    hole: Literal["alpha", "magenta", "black"] = "alpha"
    fill_specks: bool = True
    format: Literal["png", "jpg"] = "png"


class LimitBody(BaseModel):
    orbit: Orbit
    max_hole_fraction: float | None = Field(default=None, gt=0.0, le=1.0)


def result_name(body: RenderBody, hole: str) -> str:
    """Content-addressed file name: identical requests map to the same file."""

    def q(x: float) -> float:
        return round(x, 4) + 0.0  # "+ 0.0" turns -0.0 into 0.0

    key = json.dumps(
        [
            [[q(x) for x in row] for row in body.R],
            [q(x) for x in body.t],
            body.size,
            hole,
            body.fill_specks,
            body.format,
        ]
    )
    return f"{hashlib.sha1(key.encode()).hexdigest()[:12]}.{body.format}"


def make_render_router(store: SessionStore, scheduler: Scheduler, cache_dir) -> APIRouter:
    router = APIRouter(prefix="/api/sessions")

    def require_ready(session_id: str):
        session = store.get(session_id)
        if session is None:
            raise ApiError(404, "not_found", "no such session")
        if session.state != "ready":
            raise ApiError(409, "not_ready", f"the session is {session.state}, not ready")
        return session

    def common(session_id: str) -> dict[str, Any]:
        return {
            "session": session_id,
            "session_dir": str(store.root / session_id),
            "cache_dir": str(cache_dir),
        }

    async def run(request: Request, handle: JobHandle) -> dict[str, Any]:
        """Wait for the job; cancel it if the client goes away; map every outcome to a result or an error."""
        waiter = asyncio.ensure_future(handle.result())
        try:
            while not waiter.done():
                await asyncio.wait({waiter}, timeout=POLL_S)
                if not waiter.done() and await request.is_disconnected():
                    handle.cancel()
                    break
            outcome: JobOutcome = await waiter
        except asyncio.CancelledError:  # the server is cancelling this handler (client gone)
            handle.cancel()
            raise
        if outcome.ok:
            return outcome.data
        if outcome.status == "superseded":
            raise ApiError(409, "superseded", "a newer request replaced this one")
        if outcome.status == "cancelled":
            raise ApiError(409, "cancelled", "the request was cancelled")
        err = outcome.error or {"code": "internal", "message": "job failed"}
        code = err.get("code", "internal")
        raise ApiError(STATUS_FOR_CODE.get(code, 500), code, err.get("message", "job failed"))

    @router.post("/{session_id}/coverage")
    async def coverage(session_id: str, body: PoseBody, request: Request) -> dict:
        require_ready(session_id)
        handle = scheduler.submit(
            "coverage",
            {**common(session_id), "R": body.R, "t": body.t},
            priority=Priority.INTERACTIVE,
            session=session_id,
            coalesce_key=f"cov:{session_id}",
        )
        data = await run(request, handle)
        return {"coverage": data["coverage"], "ms": data["ms"]}

    @router.post("/{session_id}/limit")
    async def limit(session_id: str, body: LimitBody, request: Request) -> dict:
        require_ready(session_id)
        handle = scheduler.submit(
            "limit",
            {
                **common(session_id),
                "orbit": body.orbit.model_dump(),
                "max_hole_fraction": body.max_hole_fraction,
            },
            priority=Priority.INTERACTIVE,
            session=session_id,
            coalesce_key=f"lim:{session_id}",
        )
        data = await run(request, handle)
        return {k: v for k, v in data.items() if k != "ms"}

    @router.post("/{session_id}/render")
    async def render(session_id: str, body: RenderBody, request: Request) -> dict:
        require_ready(session_id)
        hole = "magenta" if body.format == "jpg" and body.hole == "alpha" else body.hole  # JPEG: no alpha
        name = result_name(body, hole)
        url = f"/api/sessions/{session_id}/results/{name}"
        existing = store.result_path(session_id, name)
        sidecar = store.result_path(session_id, name.rsplit(".", 1)[0] + ".json")
        if existing is not None and sidecar is not None:  # identical request: no GPU job
            try:
                info = json.loads(sidecar.read_text(encoding="utf-8"))
                return {**info, "url": url}
            except (OSError, ValueError):
                pass
        original = store.original_path(session_id)
        handle = scheduler.submit(
            "render",
            {
                **common(session_id),
                "original": str(original),
                "R": body.R,
                "t": body.t,
                "orbit": body.orbit.model_dump() if body.orbit else None,
                "size": body.size,
                "hole": hole,
                "fill_specks": body.fill_specks,
                "name": name,
            },
            priority=Priority.RENDER,
            session=session_id,
        )
        data = await run(request, handle)
        return {**data, "url": url}

    @router.get("/{session_id}/results/{name}")
    async def get_result(session_id: str, name: str, request: Request, download: int = 0) -> Response:
        session = store.get(session_id)
        path = store.result_path(session_id, name) if session else None
        if session is None or path is None or path.suffix not in RESULT_TYPES:
            raise ApiError(404, "not_found", "no such result")
        stat = await run_in_threadpool(path.stat)
        headers = {
            "Cache-Control": "private, max-age=31536000, immutable",
            "ETag": f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"',
        }
        if request.headers.get("if-none-match") == headers["ETag"]:
            return Response(status_code=304, headers=headers)
        if download:
            stem = session.filename.rsplit(".", 1)[0] or "image"
            safe = "".join(c if c.isalnum() or c in " ._-()" else "_" for c in stem)[:80]
            headers["Content-Disposition"] = f'attachment; filename="{safe}-view{path.suffix}"'
        return FileResponse(path, media_type=RESULT_TYPES[path.suffix], headers=headers)

    return router
