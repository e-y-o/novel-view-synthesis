"""FastAPI application factory. Routes for sessions, jobs and events are added by later tickets."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from nvs.api.analysis import AnalysisService
from nvs.api.config import ServerConfig
from nvs.api.errors import ApiError
from nvs.api.routes_events import make_events_router
from nvs.api.routes_render import make_render_router
from nvs.api.routes_sessions import make_sessions_router
from nvs.api.security import SecurityMiddleware
from nvs.api.sessions import Session, SessionStore
from nvs.jobs.events import EventHub
from nvs.jobs.scheduler import Scheduler
from nvs.worker.client import WorkerClient
from nvs.worker.protocol import WorkerConfig, WorkerStatus

FALLBACK_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>nvs</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 40rem; margin: 4rem auto">
<h1>nvs server is running</h1>
<p>The frontend is not built. Run <code>npm run build</code> in <code>web/</code> and reload.</p>
<p>API health: <a href="/api/health"><code>/api/health</code></a></p>
</body></html>
"""


def _version() -> str:
    try:
        return version("nvs")
    except PackageNotFoundError:
        return "unknown"


def _default_web_dist() -> Path:
    return Path(__file__).resolve().parents[3] / "web" / "dist"


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def status_message(status: WorkerStatus) -> dict:
    return {"type": "status", "worker": status.state, "detail": status.detail}


def create_app(
    config: ServerConfig | None = None,
    on_created: Callable[[Session], None] | None = None,
    worker: WorkerClient | None = None,
) -> FastAPI:
    config = config or ServerConfig()
    store = SessionStore(config.data_dir / "sessions")
    hub = EventHub()
    worker = worker or WorkerClient(WorkerConfig(stub=config.stub, preload_depth=config.preload_depth))
    scheduler = Scheduler(worker)
    analysis = AnalysisService(store, scheduler, hub, config.data_dir / "cache")
    worker.add_status_listener(lambda s: hub.publish(status_message(s)))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if config.start_worker:
            await worker.start()
        scheduler.start()
        try:
            yield
        finally:
            hub.close()  # WebSocket handlers finish and close their sockets
            await analysis.close()
            await scheduler.close()
            if config.start_worker:
                await worker.stop()

    app = FastAPI(title="nvs", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.config = config
    app.state.sessions = store
    app.state.hub = hub
    app.state.worker = worker
    app.state.scheduler = scheduler
    app.state.analysis = analysis

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return _error(exc.status_code, f"http_{exc.status_code}", str(exc.detail))

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        message = str(first.get("msg", "invalid request")).removeprefix("Value error, ")
        return _error(422, "invalid_request", f"{where}: {message}" if where else message)

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "version": _version(), "stub": config.stub}

    app.include_router(
        make_sessions_router(
            store, config.max_upload_bytes, on_created or analysis.on_created, analysis.on_deleted
        )
    )
    app.include_router(make_events_router(hub, store, scheduler))
    app.include_router(make_render_router(store, scheduler, config.data_dir / "cache"))

    @app.api_route(
        "/api/{rest:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    async def api_not_found(rest: str) -> JSONResponse:
        return _error(404, "not_found", "no such API route")

    dist = config.web_dist or _default_web_dist()
    if (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    else:

        @app.get("/", include_in_schema=False)
        async def frontend_missing() -> HTMLResponse:
            return HTMLResponse(FALLBACK_PAGE)

    # Added last so it is outermost: every request, including 404s and static files, is checked first.
    app.add_middleware(SecurityMiddleware, config=config)
    return app


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
