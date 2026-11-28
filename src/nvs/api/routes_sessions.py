"""REST routes for sessions. Analysis is started through the injected ``on_created`` hook (T05)."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response

from nvs.api.sessions import ImageTooLarge, Session, SessionStore, UnsupportedImage
from nvs.api.uploads import read_upload_limited

LIST_LIMIT = 50
IMMUTABLE = "private, max-age=31536000, immutable"
CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".json": "application/json",
}


def _asset_url(session_id: str, name: str) -> str:
    return f"/api/sessions/{session_id}/assets/{name}"


def _summary(s: Session) -> dict:
    out = s.to_json()
    out["thumbnail"] = _asset_url(s.id, "color.jpg") if s.state == "ready" else None
    return out


def make_sessions_router(
    store: SessionStore,
    max_upload_bytes: int,
    on_created: Callable[[Session], None] | None = None,
    on_deleted: Callable[[str], None] | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/sessions")

    def require(session_id: str) -> Session:
        session = store.get(session_id)  # invalid ids simply return None
        if session is None:
            raise HTTPException(404, "no such session")
        return session

    @router.post("", status_code=201)
    async def upload(file: UploadFile) -> dict:
        stored = await read_upload_limited(file, max_upload_bytes, dest_dir=store.tmp_dir)
        try:
            session, existed = await run_in_threadpool(
                store.create, stored.path, file.filename, stored.sha256
            )
        except UnsupportedImage as e:
            raise HTTPException(415, str(e)) from e
        except ImageTooLarge as e:
            raise HTTPException(413, str(e)) from e
        if on_created is not None:
            on_created(session)
            session = store.get(session.id) or session  # the hook may have changed the state
        return {"id": session.id, "state": session.state, "filename": session.filename, "existing": existed}

    @router.get("")
    async def list_sessions() -> dict:
        sessions = await run_in_threadpool(store.list)
        return {"sessions": [_summary(s) for s in sessions[:LIST_LIMIT]]}

    @router.get("/{session_id}")
    async def get_session(session_id: str) -> dict:
        session = require(session_id)
        meta = await run_in_threadpool(store.meta, session_id) if session.state == "ready" else None
        return {**_summary(session), "meta": meta}

    @router.get("/{session_id}/assets/{name}")
    async def get_asset(session_id: str, name: str, request: Request) -> Response:
        require(session_id)
        path = store.asset_path(session_id, name)
        if path is None:
            raise HTTPException(404, "no such asset")
        stat = path.stat()
        headers = {"Cache-Control": IMMUTABLE, "ETag": f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'}
        if request.headers.get("if-none-match") == headers["ETag"]:
            return Response(status_code=304, headers=headers)
        return FileResponse(
            path,
            media_type=CONTENT_TYPES.get(path.suffix, "application/octet-stream"),
            headers=headers,  # content-addressed, so immutable
        )

    @router.delete("/{session_id}", status_code=204)
    async def delete_session(session_id: str) -> Response:
        require(session_id)
        if on_deleted is not None:
            on_deleted(session_id)
        await run_in_threadpool(store.delete, session_id)
        return Response(status_code=204)

    return router
