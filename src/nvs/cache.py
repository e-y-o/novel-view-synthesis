"""On-disk depth cache: depth depends on the image (and model/focal length), not on the camera.

Layout: ``<root>/depth/<key[:2]>/<key>/{depth.npy, sky.npy, meta.json}`` with float32 depth (float16 is not
precise enough for the sky clamp), bool sky mask and a small JSON header. Writes go to a temporary sibling
directory and are published with ``os.replace``, so a crash never leaves a half-written visible entry.
Entries are validated on read (schema version, shapes, dtypes, finite values); corrupt ones are deleted and
reported as a miss. Total size is capped (default 5 GiB) with least-recently-used eviction by directory mtime
(``get`` touches it). ``max_bytes=0`` disables caching.

Default location: ``data/cache`` under the repository root (git-ignored), ``NVS_CACHE_DIR`` overrides it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from nvs.core.types import DepthResult

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_MAX_BYTES = 5 * 2**30
_TMP_PREFIX = ".tmp-"
_STALE_TMP_SECONDS = 3600


def cache_key(image_sha256: str, backend_name: str, precision: str, f_px_used: float) -> str:
    """Hex key over everything that changes the depth: image, model, precision, f_px (0.1 px), schema."""
    parts = [f"schema={SCHEMA_VERSION}", image_sha256, backend_name, precision, f"f={round(f_px_used * 10)}"]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def default_cache_root() -> Path:
    env = os.environ.get("NVS_CACHE_DIR")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[2]
    base = repo if (repo / "pyproject.toml").exists() else Path.cwd()
    return base / "data" / "cache"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


class DepthCache:
    def __init__(self, root: str | Path | None = None, max_bytes: int = DEFAULT_MAX_BYTES):
        self.root = Path(root) if root is not None else default_cache_root()
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    @property
    def _depth_dir(self) -> Path:
        return self.root / "depth"

    def _entry(self, key: str) -> Path:
        return self._depth_dir / key[:2] / key

    def _entries(self) -> list[Path]:
        if not self._depth_dir.exists():
            return []
        return [
            e
            for shard in self._depth_dir.iterdir()
            if shard.is_dir()
            for e in shard.iterdir()
            if e.is_dir() and not e.name.startswith((_TMP_PREFIX, ".trash-"))
        ]

    # ---- read ----
    def get(self, key: str, device: str | torch.device = "cpu") -> DepthResult | None:
        if self.max_bytes == 0:
            return None
        entry = self._entry(key)
        if not entry.is_dir():
            return None
        try:
            meta = json.loads((entry / "meta.json").read_text())
            if meta.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"schema_version {meta.get('schema_version')!r}")
            shape = tuple(meta["shape"])
            depth = np.load(entry / "depth.npy", allow_pickle=False)
            sky = np.load(entry / "sky.npy", allow_pickle=False)
            if depth.dtype != np.float32 or sky.dtype != np.bool_:
                raise ValueError("unexpected array dtype")
            if depth.shape != shape or sky.shape != shape:
                raise ValueError(f"shape mismatch: {depth.shape}/{sky.shape} vs {shape}")
            if not np.isfinite(depth).all():
                raise ValueError("non-finite depth")
            result = DepthResult(
                depth=torch.from_numpy(depth).to(device),
                f_px=float(meta["f_px"]),
                sky_mask=torch.from_numpy(sky).to(device),
                model=str(meta["model"]),
                infer_size=tuple(meta["infer_size"]),
                notes=list(meta["notes"]),
            )
        except Exception as e:  # corrupt or foreign entry: drop it and report a miss
            log.warning("depth cache entry %s is corrupt (%s); removing it", key[:12], e)
            shutil.rmtree(entry, ignore_errors=True)
            return None
        try:
            os.utime(entry)  # LRU touch
        except OSError:
            pass
        return result

    # ---- write ----
    def put(self, key: str, result: DepthResult, image_sha256: str | None = None) -> None:
        if self.max_bytes == 0:
            return
        final = self._entry(key)
        parent = final.parent
        parent.mkdir(parents=True, exist_ok=True)
        self._remove_stale_tmp(parent)
        tmp = parent / f"{_TMP_PREFIX}{key[:12]}-{uuid.uuid4().hex}"
        tmp.mkdir()
        try:
            depth = result.depth.detach().to("cpu", torch.float32).contiguous().numpy()
            sky = result.sky_mask.detach().to("cpu", torch.bool).contiguous().numpy()
            np.save(tmp / "depth.npy", depth, allow_pickle=False)
            np.save(tmp / "sky.npy", sky, allow_pickle=False)
            meta = {
                "schema_version": SCHEMA_VERSION,
                "f_px": float(result.f_px),
                "model": result.model,
                "infer_size": list(result.infer_size),
                "notes": list(result.notes),
                "shape": list(depth.shape),
                "created": datetime.now(UTC).isoformat(),
                "image_sha256": image_sha256,
            }
            (tmp / "meta.json").write_text(json.dumps(meta))
            size = _dir_size(tmp)
            if size > self.max_bytes:
                log.warning("depth entry (%d bytes) exceeds the cache cap; not stored", size)
                shutil.rmtree(tmp, ignore_errors=True)
                return
            with self._lock:
                self._publish(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        self._evict(keep=final)

    @staticmethod
    def _publish(tmp: Path, final: Path) -> None:
        last: OSError | None = None
        for _ in range(5):
            try:
                os.replace(tmp, final)
                return
            except OSError as e:  # target exists (directory replace is not allowed over a non-empty dir)
                last = e
                trash = final.with_name(f".trash-{uuid.uuid4().hex}")
                try:
                    os.replace(final, trash)
                    shutil.rmtree(trash, ignore_errors=True)
                except OSError:
                    time.sleep(0.01)  # someone else is replacing it right now; retry
        assert last is not None
        raise last

    @staticmethod
    def _remove_stale_tmp(parent: Path) -> None:
        now = time.time()
        for p in parent.glob(f"{_TMP_PREFIX}*"):
            try:
                if now - p.stat().st_mtime > _STALE_TMP_SECONDS:
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass

    def _evict(self, keep: Path) -> None:
        sized = []
        for e in self._entries():
            try:
                sized.append((e.stat().st_mtime, e, _dir_size(e)))
            except OSError:
                continue  # removed concurrently
        total = sum(s for _, _, s in sized)
        for _, e, s in sorted(sized, key=lambda t: t[0]):
            if total <= self.max_bytes:
                break
            if e == keep:
                continue
            shutil.rmtree(e, ignore_errors=True)
            total -= s

    # ---- housekeeping ----
    def size_bytes(self) -> int:
        total = 0
        for e in self._entries():
            try:
                total += _dir_size(e)
            except OSError:
                pass
        return total

    def clear(self) -> None:
        shutil.rmtree(self._depth_dir, ignore_errors=True)
