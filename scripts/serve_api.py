"""Run the ask API (kaprag.serve.api) on a local port.

Needs the `app` dependency group (`uv sync --group ml --group app --group observability`):

    uv run python scripts/serve_api.py                      # router config, 127.0.0.1:8000
    uv run python scripts/serve_api.py --config rerank --port 8010

The config defaults to $KAPRAG_SERVE_CONFIG, else `router`. The LLM backend is
LLM_BACKEND (from the environment or `.env`): `claude_code` uses the Claude
subscription and is for local use only; a demo opened to others must run with
`LLM_BACKEND=anthropic_api` (API key). The pipeline's models load once at start.

There is no authentication. The server binds to 127.0.0.1; any other address needs
`--allow-public` (and prints a warning), and is refused with the `claude_code` backend.
"""

from __future__ import annotations

import argparse
import ipaddress
import logging
import os
import sys

from dotenv import load_dotenv

from kaprag.corpus import REPO_ROOT

LOOPBACK_NAMES = {"localhost"}


def is_loopback(host: str) -> bool:
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the kaprag ask API.")
    parser.add_argument("--config", default=None, help="configs/<name>.yaml (default: router)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--allow-public",
        action="store_true",
        help="allow a non-loopback --host (no authentication; anthropic_api backend only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(REPO_ROOT / ".env", override=False)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    backend = os.environ.get("LLM_BACKEND", "claude_code")
    if not is_loopback(args.host):
        if not args.allow_public:
            print(
                f"error: refusing to bind to {args.host}: the API has no authentication. "
                "Use the default 127.0.0.1, or pass --allow-public deliberately.",
                file=sys.stderr,
            )
            return 2
        if backend == "claude_code":
            print(
                "error: refusing a public bind with LLM_BACKEND=claude_code: a demo opened "
                "to others must run with LLM_BACKEND=anthropic_api (API key).",
                file=sys.stderr,
            )
            return 2
        print(
            f"WARNING: binding to {args.host}:{args.port} with NO authentication; anyone "
            "who can reach this address can spend LLM credit.",
            file=sys.stderr,
        )

    import uvicorn

    from kaprag.serve import open_service
    from kaprag.serve.api import create_app

    print(f"loading pipeline (config {args.config or 'default'}, LLM backend {backend})...")
    service = open_service(args.config)
    print(f"ready: config {service.config}, LLM backend {service.backend}")
    try:
        # The app closes the service on shutdown: uvicorn may re-raise the stop signal
        # after it returns, so code after `run` is not guaranteed to execute.
        app = create_app(service, close_on_shutdown=True)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        service.close()  # idempotent; covers a failed start
    return 0


if __name__ == "__main__":
    sys.exit(main())
