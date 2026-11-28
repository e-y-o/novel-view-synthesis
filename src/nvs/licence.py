"""Acceptance gate for the Depth Pro 2 model licence (research-only weights).

Shared by ``nvs render`` and ``nvs serve``: call ``ensure_license`` before weights are downloaded or used.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

LICENSE_NOTICE = """\
Depth Pro 2 model weights are released by Apple under a research-only licence (see LICENSE_MODEL in
https://github.com/apple-aiml-research/ml-depth-pro-2). They are downloaded to your machine on first use
and are not part of this repository. By continuing you confirm that your use complies with that licence.
This project is not endorsed by Apple."""

ACCEPT_ENV = "NVS_ACCEPT_DEPTHPRO2_LICENSE"
_MARKER = "depthpro2-license-accepted"


def data_dir() -> Path:
    from nvs.cache import default_cache_root

    return default_cache_root().parent


def license_accepted() -> bool:
    return (data_dir() / _MARKER).exists()


def record_license() -> None:
    d = data_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / _MARKER).write_text("accepted\n")


def ensure_license(accept: bool = False) -> None:
    """Make sure the user has accepted the model licence; raise ``CliError`` otherwise."""
    from nvs.cli.errors import CliError

    if license_accepted():
        return
    if accept or os.environ.get(ACCEPT_ENV) == "1":
        record_license()
        return
    if sys.stdin is not None and sys.stdin.isatty():
        print(LICENSE_NOTICE)
        if input("Accept and continue? [y/N] ").strip().lower() in ("y", "yes"):
            record_license()
            return
        raise CliError("licence not accepted")
    raise CliError(
        f"the Depth Pro 2 licence must be accepted: pass --accept-license or set {ACCEPT_ENV}=1\n"
        + LICENSE_NOTICE
    )
