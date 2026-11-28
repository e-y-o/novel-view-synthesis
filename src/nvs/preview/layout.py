"""Preview asset set: names, schema version and validation. Torch-free (the web process uses it)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

SCHEMA = 1
ASSET_NAMES = ("color.jpg", "depth.f16", "sky.u8", "meta.json")


def validate_dir(directory: Path) -> dict | None:
    """The parsed ``meta.json`` if ``directory`` holds a complete, consistent, current-schema asset set."""
    try:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if meta.get("schema") != SCHEMA:
            return None
        pw, ph = int(meta["preview_width"]), int(meta["preview_height"])
        if (directory / "depth.f16").stat().st_size != pw * ph * 2:
            return None
        if (directory / "sky.u8").stat().st_size != pw * ph:
            return None
        with Image.open(directory / "color.jpg") as im:
            if im.size != (pw, ph):
                return None
        return meta
    except (OSError, ValueError, KeyError, TypeError):
        return None
