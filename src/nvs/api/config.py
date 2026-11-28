"""Server configuration: a dataclass filled from ``NVS_*`` environment variables and CLI flags."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
DEV_ORIGIN = "http://localhost:5173"  # Vite dev server, allowed only with ``dev=True``


def _flag(text: str) -> bool:
    return text.strip().lower() in ("1", "true", "yes", "on")


def _default_data_dir() -> Path:
    from nvs.licence import data_dir

    return data_dir()


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    data_dir: Path = field(default_factory=_default_data_dir)
    max_upload_mb: int = 200
    stub: bool = False  # no GPU: fake depth (T03)
    preload_depth: bool = True
    open_browser: bool = False
    dev: bool = False  # also accept the Vite dev-server origin
    web_dist: Path | None = None  # built frontend; default <repo>/web/dist

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def allowed_hosts(self) -> frozenset[str]:
        """Exact ``Host`` header values accepted (DNS-rebinding defence)."""
        names = {"127.0.0.1", "localhost", "[::1]"}
        if self.host not in LOOPBACK_HOSTS:  # only reachable with --i-know-this-is-unsafe
            names.add(f"[{self.host}]" if ":" in self.host else self.host)
        return frozenset(f"{n}:{self.port}" for n in names)

    @property
    def allowed_origins(self) -> frozenset[str]:
        origins = {f"http://{h}" for h in self.allowed_hosts}
        if self.dev:
            origins.add(DEV_ORIGIN)
        return frozenset(origins)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> ServerConfig:
        env = os.environ if environ is None else environ
        updates: dict[str, object] = {}
        if "NVS_PORT" in env:
            updates["port"] = int(env["NVS_PORT"])
        if "NVS_DATA_DIR" in env:
            updates["data_dir"] = Path(env["NVS_DATA_DIR"])
        if "NVS_MAX_UPLOAD_MB" in env:
            updates["max_upload_mb"] = int(env["NVS_MAX_UPLOAD_MB"])
        for name, key in [
            ("NVS_STUB", "stub"),
            ("NVS_PRELOAD_DEPTH", "preload_depth"),
            ("NVS_OPEN_BROWSER", "open_browser"),
            ("NVS_DEV", "dev"),
        ]:
            if name in env:
                updates[key] = _flag(env[name])
        return replace(cls(), **updates)

    def with_overrides(self, **overrides: object) -> ServerConfig:
        """Return a copy with every override whose value is not ``None`` applied (CLI flags win over env)."""
        known = {f.name for f in fields(self)}
        bad = set(overrides) - known
        if bad:
            raise TypeError(f"unknown config fields: {sorted(bad)}")
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})
