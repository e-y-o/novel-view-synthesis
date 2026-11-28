"""Messages exchanged between the server process and the GPU worker.

Everything here is a frozen dataclass of plain, picklable types (no tensors, no callables): on Windows the
worker is started with ``spawn`` and the messages cross a pipe. Torch-free: the web process imports this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Event types. result / error / cancelled are terminal: exactly one of them ends every request.
STARTED = "started"
PROGRESS = "progress"
RESULT = "result"
ERROR = "error"
CANCELLED = "cancelled"
TERMINAL = frozenset({RESULT, ERROR, CANCELLED})

# Worker states
STARTING = "starting"
READY = "ready"
FAILED = "error"


@dataclass(frozen=True)
class WorkerConfig:
    """Everything the worker needs at start-up."""

    stub: bool = False  # fake depth, no CUDA, no weights
    preload_depth: bool = True  # load the model before reporting ``ready``
    depth_model: str = "depthpro2-h"
    precision: str = "bf16"
    max_mp: float = 4.0
    vram_fraction: float = 0.92


@dataclass(frozen=True)
class Request:
    id: str
    kind: str  # key into ``nvs.worker.main.HANDLERS``
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # lower runs first; ordering is the scheduler's job (T04), the worker is FIFO
    coalesce_key: str | None = None  # a newer request with the same key supersedes older queued ones (T04)


@dataclass(frozen=True)
class Event:
    request_id: str
    type: str  # started | progress | result | error | cancelled
    data: dict[str, Any] = field(default_factory=dict)  # error: {"code", "message"}; progress: {"pct", ...}

    @property
    def terminal(self) -> bool:
        return self.type in TERMINAL


@dataclass(frozen=True)
class WorkerStatus:
    state: str  # starting | ready | error
    detail: str = ""


# Control messages (server -> worker)
@dataclass(frozen=True)
class Cancel:
    request_id: str


@dataclass(frozen=True)
class Shutdown:
    pass
