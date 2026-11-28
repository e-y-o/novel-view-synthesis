"""`nvs` console entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError, version

Handler = Callable[[argparse.Namespace], int]


def _version() -> str:
    try:
        return version("nvs")
    except PackageNotFoundError:
        return "unknown"


def _register_render(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("render", help="render a novel view from a single image (not implemented yet)")
    p.set_defaults(handler=_render)


def _render(args: argparse.Namespace) -> int:
    print("render: not implemented")
    return 1


COMMANDS: list[Callable[[argparse._SubParsersAction], None]] = [_register_render]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nvs", description="Fast single-image novel view synthesis.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for register in COMMANDS:
        register(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Handler | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
