"""Whether a server without authentication may run on an address with an LLM backend.

The project rule: a demo opened to others must run with an API key
(`anthropic_api`), never the Claude subscription (`claude_code`). A loopback address
is local use and always allowed. Any other address, including an unset or empty one
(Streamlit then listens on all interfaces), counts as public: allowed with a warning
on `anthropic_api`, refused on any other backend.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass

API_KEY_BACKEND = "anthropic_api"


@dataclass(frozen=True)
class Exposure:
    allowed: bool  # may the server answer questions
    public: bool  # reachable beyond this machine: show a warning


def is_loopback(address: str | None) -> bool:
    """`localhost` or a loopback IP (127.0.0.0/8, ::1); unset/empty is not loopback."""
    host = (address or "").strip()
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_exposure(address: str | None, backend: str) -> Exposure:
    if is_loopback(address):
        return Exposure(allowed=True, public=False)
    return Exposure(allowed=backend == API_KEY_BACKEND, public=True)
