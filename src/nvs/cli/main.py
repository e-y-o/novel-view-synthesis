"""`nvs` console entry point."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError, version

# Must be set before torch initialises CUDA (see nvs.depth.base.configure_cuda_env); harmless otherwise.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

Handler = Callable[[argparse.Namespace], int]


def _version() -> str:
    try:
        return version("nvs")
    except PackageNotFoundError:
        return "unknown"


def _commands() -> list[Callable[[argparse._SubParsersAction], None]]:
    from nvs.cli import export_preview, render, serve

    return [render.register, serve.register, export_preview.register]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nvs", description="Fast single-image novel view synthesis.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for register in _commands():
        register(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Handler | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    from nvs.cli.errors import CliError

    try:
        return handler(args)
    except CliError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 1
    except Exception as e:
        if getattr(args, "verbose", False):
            raise
        print(f"error: {type(e).__name__}: {e} (run with -v for a traceback)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
