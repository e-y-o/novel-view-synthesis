"""Glue between uploads, the scheduler and the event hub: runs the ``analyze`` job for a new session.

State machine: ``uploaded -> analyzing -> ready | error`` (``analyzing`` is reset to ``uploaded`` by the store
after a restart). A session whose ``preview/`` already holds a complete, current-schema asset set becomes
``ready`` at once without touching the worker, so re-uploading a photo is instant even when the model is cold.
Torch-free.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from nvs.api.sessions import Session, SessionStore
from nvs.jobs.events import EventHub
from nvs.jobs.scheduler import JobHandle, Priority, Scheduler
from nvs.preview.layout import validate_dir
from nvs.worker.protocol import Event

log = logging.getLogger("nvs.api.analysis")

FRIENDLY = {
    "worker_died": "The GPU worker crashed while analysing the photo. Try again.",
    "worker_unavailable": "The depth model is not ready yet. Wait a moment and upload again.",
    "timeout": "Analysis took too long and was stopped. Try a smaller photo.",
}


class AnalysisService:
    def __init__(self, store: SessionStore, scheduler: Scheduler, hub: EventHub, cache_dir: Path):
        self.store = store
        self.scheduler = scheduler
        self.hub = hub
        self.cache_dir = cache_dir
        self._jobs: dict[str, JobHandle] = {}
        self._tasks: set[asyncio.Task] = set()

    # -------------------------------------------------------------------------------------------- hooks
    def on_created(self, session: Session) -> None:
        """Called after every upload (new or repeated). Must run on the event loop."""
        sid = session.id
        current = self.store.get(sid) or session
        preview = self.store.root / sid / "preview"
        if validate_dir(preview) is not None:  # idempotent: nothing to compute
            if current.state != "ready":
                self.store.update(sid, state="ready", error=None)
            self.hub.publish({"type": "ready", "session": sid}, sid)
            return
        job = self._jobs.get(sid)
        if job is not None and not job.done:
            return  # already being analysed
        original = self.store.original_path(sid)
        if original is None:
            return
        self.store.update(sid, state="analyzing", error=None)
        self._progress(sid, "upload", 0.0)

        def forward(event: Event) -> None:
            if event.type == "progress":
                self._progress(sid, event.data.get("phase", "depth"), float(event.data.get("pct", 0.0)))

        handle = self.scheduler.submit(
            "analyze",
            {
                "path": str(original),
                "out_dir": str(preview),
                "filename": current.filename,
                "cache_dir": str(self.cache_dir),
            },
            priority=Priority.ANALYZE,
            session=sid,
            coalesce_key=f"analyze:{sid}",
            on_event=forward,
        )
        self._jobs[sid] = handle
        task = asyncio.get_running_loop().create_task(self._watch(sid, handle))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def on_deleted(self, session_id: str) -> None:
        job = self._jobs.pop(session_id, None)
        if job is not None:
            job.cancel()
        self.hub.forget(session_id)

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # ------------------------------------------------------------------------------------------ internals
    def _progress(self, sid: str, phase: str, pct: float) -> None:
        self.hub.publish({"type": "analyze.progress", "session": sid, "phase": phase, "pct": pct}, sid)

    async def _watch(self, sid: str, handle: JobHandle) -> None:
        outcome = await handle.result()
        if self._jobs.get(sid) is handle:
            del self._jobs[sid]
        if self.store.get(sid) is None:  # deleted meanwhile
            return
        if outcome.ok:
            fields = {"state": "ready", "error": None}
            if outcome.data.get("width"):
                fields.update(width=int(outcome.data["width"]), height=int(outcome.data["height"]))
            self.store.update(sid, **fields)
            self.hub.publish({"type": "ready", "session": sid}, sid)
        elif outcome.status == "cancelled":
            self.store.update(sid, state="uploaded", error=None)
        else:
            err = outcome.error or {"code": "internal", "message": "analysis failed"}
            code = err.get("code", "internal")
            message = FRIENDLY.get(code, err.get("message", "analysis failed"))
            log.warning("analysis of %s failed: %s (%s)", sid[:12], code, err.get("message"))
            self.store.update(sid, state="error", error={"code": code, "message": message})
            self.hub.publish({"type": "error", "session": sid, "code": code, "message": message}, sid)
