"""Upload handling: stream a multipart file to disk with a hard size limit."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

CHUNK = 1024 * 1024


@dataclass(frozen=True)
class StoredUpload:
    path: Path
    size: int
    sha256: str


def _write(out, digest, chunk: bytes) -> None:
    digest.update(chunk)
    out.write(chunk)


async def read_upload_limited(
    upload: UploadFile, max_bytes: int, dest_dir: Path | None = None
) -> StoredUpload:
    """Stream ``upload`` to a temp file (in ``dest_dir`` or the system temp dir), hashing as it goes.

    Aborts with HTTP 413 as soon as more than ``max_bytes`` have been read; the partial file is removed, so
    nothing oversized is ever kept. The caller owns the returned file (move or delete it).
    """
    if dest_dir is not None:
        dest_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="upload-", suffix=".part", dir=dest_dir)
    path = Path(name)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await upload.read(CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"file larger than the {max_bytes // (1024 * 1024)} MB limit",
                    )
                await run_in_threadpool(_write, out, digest, chunk)  # keep disk/hash work off the event loop
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return StoredUpload(path=path, size=size, sha256=digest.hexdigest())
