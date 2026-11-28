"""On-disk session store. One session = one uploaded photo (id = SHA-256 of its bytes) and its derived files.

Layout under ``root`` (git-ignored, ``data/sessions``)::

    <id>/original.<ext>   session.json   preview/{color.jpg,depth.f16,sky.u8,meta.json}   results/<name>.png

Torch-free on purpose: the web process must never import CUDA code. All methods are blocking; async handlers
call them through ``run_in_threadpool``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from nvs.core import _pillow
from nvs.core._pillow import MAX_PIXELS

ID_RE = re.compile(r"^[0-9a-f]{64}$")
RESULT_NAME_RE = re.compile(r"^[a-z0-9._-]{1,64}$")
PREVIEW_ASSETS = ("color.jpg", "depth.f16", "sky.u8", "meta.json")
FORMAT_EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "HEIF": "heic", "MPO": "jpg"}
STATES = ("uploaded", "analyzing", "ready", "error")
MAX_FILENAME = 120
DEFAULT_MAX_BYTES = 10 * 1024**3

assert _pillow  # registers HEIF support


class UnsupportedImage(ValueError):
    """Content is not an image Pillow can open in a supported format (HTTP 415)."""


class ImageTooLarge(ValueError):
    """More than ``MAX_PIXELS`` pixels (HTTP 413)."""


def sanitize_filename(name: str | None) -> str:
    """Display-only name: no directories, no control characters, at most 120 characters."""
    text = (name or "").replace("\\", "/").rsplit("/", 1)[-1]
    text = "".join(c for c in text if unicodedata.category(c) not in ("Cc", "Cf", "Cs", "Co", "Cn"))
    text = text.strip(" .")
    if not text:
        return "image"
    return text[:MAX_FILENAME]


def sniff_image(path: Path) -> tuple[str, int, int]:
    """Return ``(ext, width, height)`` from the content (header only, no pixel decode)."""
    try:
        with Image.open(path) as im:
            ext = FORMAT_EXT.get(im.format or "")
            width, height = im.size
    except Image.DecompressionBombError as e:
        raise ImageTooLarge(f"image exceeds the {MAX_PIXELS // 1_000_000} MP limit") from e
    except Exception as e:  # UnidentifiedImageError, OSError, SyntaxError from broken headers, ...
        raise UnsupportedImage("not a supported image (JPEG, PNG, WebP or HEIC)") from e
    if ext is None:
        raise UnsupportedImage(f"unsupported image format {im.format!r}: use JPEG, PNG, WebP or HEIC")
    if width * height > MAX_PIXELS:
        raise ImageTooLarge(
            f"image is {width * height / 1e6:.0f} MP; the limit is {MAX_PIXELS // 1_000_000} MP"
        )
    return ext, width, height


def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON so readers see the old or the new file, never a partial one."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Session:
    id: str
    filename: str
    created: float
    last_used: float
    state: str = "uploaded"
    error: dict | None = None  # {"code", "message"}
    width: int = 0
    height: int = 0
    bytes: int = 0

    def to_json(self) -> dict:
        return asdict(self)


class SessionStore:
    def __init__(self, root: Path, max_bytes: int = DEFAULT_MAX_BYTES):
        self.root = Path(root)
        self.max_bytes = max_bytes
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(exist_ok=True)
        for leftover in self.tmp_dir.iterdir():  # aborted uploads
            leftover.unlink(missing_ok=True)
        for s in self.list():  # the server died mid-job: analysis must be requested again
            if s.state == "analyzing":
                self.update(s.id, state="uploaded", error=None)

    # ------------------------------------------------------------------------------------------ paths
    @property
    def tmp_dir(self) -> Path:
        return self.root / ".tmp"

    def session_dir(self, session_id: str) -> Path | None:
        if not ID_RE.match(session_id):
            return None
        return self.root / session_id

    def _json_path(self, session_id: str) -> Path | None:
        d = self.session_dir(session_id)
        return d / "session.json" if d else None

    def _inside(self, base: Path, candidate: Path) -> Path | None:
        """``candidate`` resolved, only if it stays inside ``base`` and is an existing file."""
        try:
            resolved = candidate.resolve(strict=True)
            if resolved.is_file() and resolved.is_relative_to(base.resolve()):
                return resolved
        except (OSError, ValueError):
            pass
        return None

    def asset_path(self, session_id: str, name: str) -> Path | None:
        """A preview asset (allow-listed names only), or ``None``."""
        d = self.session_dir(session_id)
        if d is None or name not in PREVIEW_ASSETS:
            return None
        return self._inside(d, d / "preview" / name)

    def result_path(self, session_id: str, name: str) -> Path | None:
        d = self.session_dir(session_id)
        if d is None or not RESULT_NAME_RE.match(name) or name.startswith("."):
            return None
        return self._inside(d, d / "results" / name)

    def original_path(self, session_id: str) -> Path | None:
        d = self.session_dir(session_id)
        if d is None or not d.is_dir():
            return None
        for ext in ("jpg", "png", "webp", "heic"):
            if (d / f"original.{ext}").is_file():
                return d / f"original.{ext}"
        return None

    # ----------------------------------------------------------------------------------------- reads
    def get(self, session_id: str) -> Session | None:
        path = self._json_path(session_id)
        if path is None:
            return None
        try:
            return Session(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None

    def list(self) -> list[Session]:
        """All sessions, newest first."""
        out = []
        for d in self.root.iterdir():
            if d.is_dir() and ID_RE.match(d.name) and (s := self.get(d.name)) is not None:
                out.append(s)
        return sorted(out, key=lambda s: s.created, reverse=True)

    def meta(self, session_id: str) -> dict | None:
        path = self.asset_path(session_id, "meta.json")
        if path is None:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def total_bytes(self) -> int:
        total = 0
        for d in self.root.iterdir():
            if d.is_dir() and ID_RE.match(d.name):
                total += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        return total

    # ---------------------------------------------------------------------------------------- writes
    def create(
        self, upload_tmp_path: Path, original_filename: str | None, sha256: str | None = None
    ) -> tuple[Session, bool]:
        """Adopt an uploaded temp file. Returns ``(session, existed)``; the temp file is always consumed.

        Raises ``UnsupportedImage`` (415) or ``ImageTooLarge`` (413); the temp file is removed first.
        """
        tmp = Path(upload_tmp_path)
        try:
            ext, width, height = sniff_image(tmp)
            session_id = sha256 or sha256_file(tmp)
            size = tmp.stat().st_size
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        with self._lock:
            existing = self.get(session_id)
            if existing is not None and self.original_path(session_id) is not None:
                tmp.unlink(missing_ok=True)
                self.touch(session_id)
                return self.get(session_id) or existing, True
            d = self.root / session_id
            d.mkdir(parents=True, exist_ok=True)
            os.replace(tmp, d / f"original.{ext}")
            now = time.time()
            session = Session(
                id=session_id,
                filename=sanitize_filename(original_filename),
                created=now,
                last_used=now,
                width=width,
                height=height,
                bytes=size,
            )
            atomic_write_json(d / "session.json", session.to_json())
            self.evict(keep=session_id)
            return session, False

    def update(self, session_id: str, **fields) -> Session | None:
        with self._lock:
            session = self.get(session_id)
            if session is None:
                return None
            if "state" in fields and fields["state"] not in STATES:
                raise ValueError(f"bad state {fields['state']!r}")
            for k, v in fields.items():
                if not hasattr(session, k) or k == "id":
                    raise ValueError(f"cannot update field {k!r}")
                setattr(session, k, v)
            atomic_write_json(self._json_path(session_id), session.to_json())  # type: ignore[arg-type]
            return session

    def touch(self, session_id: str) -> None:
        self.update(session_id, last_used=time.time())

    def delete(self, session_id: str) -> bool:
        with self._lock:
            d = self.session_dir(session_id)
            if d is None or not d.is_dir():
                return False
            shutil.rmtree(d, ignore_errors=True)
            return True

    def evict(self, keep: str | None = None) -> list[str]:
        """Delete least-recently-used sessions until under ``max_bytes`` (skips analyzing and ``keep``)."""
        evicted = []
        with self._lock:
            total = self.total_bytes()
            if total <= self.max_bytes:
                return evicted
            for s in sorted(self.list(), key=lambda s: s.last_used):
                if total <= self.max_bytes:
                    break
                if s.state == "analyzing" or s.id == keep:
                    continue
                d = self.root / s.id
                total -= sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                shutil.rmtree(d, ignore_errors=True)
                evicted.append(s.id)
        return evicted
