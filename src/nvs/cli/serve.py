"""``nvs serve``: start the local web server on loopback.

fastapi/uvicorn are imported inside the handler so ``nvs --help`` and ``nvs serve --help`` stay fast.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from nvs.cli.errors import CliError

SAFE_HOSTS = ("127.0.0.1", "localhost", "::1")


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "serve",
        help="start the local web app (loopback only)",
        description="Start the local web server. Environment: NVS_PORT, NVS_DATA_DIR, NVS_MAX_UPLOAD_MB, "
        "NVS_STUB, NVS_PRELOAD_DEPTH, NVS_OPEN_BROWSER, NVS_DEV (flags win).",
    )
    p.add_argument("--host", default="127.0.0.1", help="bind address (loopback only unless unsafe flag)")
    p.add_argument("--port", type=int, default=None, help="port (default 8765)")
    p.add_argument("--data-dir", type=Path, default=None, help="where sessions and caches live")
    p.add_argument("--stub", action="store_true", default=None, help="no GPU: fake depth, no weights")
    p.add_argument(
        "--no-preload",
        dest="preload_depth",
        action="store_false",
        default=None,
        help="do not load the depth model at start",
    )
    p.add_argument("--open", dest="open_browser", action="store_true", default=None, help="open a browser")
    p.add_argument("--dev", action="store_true", default=None, help="also accept the Vite dev origin")
    p.add_argument("--accept-license", action="store_true", help="accept the Depth Pro 2 licence")
    p.add_argument(
        "--i-know-this-is-unsafe",
        action="store_true",
        help="allow binding a non-loopback host (anyone on the network can drive the server)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging and tracebacks")
    p.set_defaults(handler=_serve)


def _serve(args: argparse.Namespace) -> int:
    import logging

    import uvicorn

    from nvs.api.app import configure_logging, create_app
    from nvs.api.config import ServerConfig
    from nvs.licence import ensure_license

    if args.host not in SAFE_HOSTS:
        if not args.i_know_this_is_unsafe:
            raise CliError(
                f"refusing to bind {args.host!r}: the server can run GPU jobs and write files and has no "
                "authentication. Use 127.0.0.1, or pass --i-know-this-is-unsafe."
            )
        bar = "!" * 70
        print(
            f"{bar}\nWARNING: binding {args.host}. Anyone who can reach this port can use your GPU "
            f"and disk.\n{bar}",
            file=sys.stderr,
        )

    config = ServerConfig.from_env().with_overrides(
        host=args.host,
        port=args.port,
        data_dir=args.data_dir,
        stub=args.stub,
        preload_depth=args.preload_depth,
        open_browser=args.open_browser,
        dev=args.dev,
    )
    if not config.stub:
        ensure_license(accept=args.accept_license)  # before the worker may load weights

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    app = create_app(config)
    url = f"http://{'[::1]' if config.host == '::1' else config.host}:{config.port}/"
    print(f"nvs server on {url}" + (" (stub: no GPU)" if config.stub else ""), file=sys.stderr)
    if config.open_browser:
        import webbrowser

        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    # uvicorn's own access log is off: our middleware logs request line, status and time (no query/body).
    uvicorn.run(app, host=config.host, port=config.port, log_level="warning")
    return 0
