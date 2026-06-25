from __future__ import annotations

import pickle
from typing import Any

CONTENT_TYPE = "application/octet-stream"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
DEFAULT_TIMEOUT_S = 600

ROUTE_HEALTH = "/health"
ROUTE_INFO = "/info"
ROUTE_RESET = "/reset"
ROUTE_ACT = "/act"
ROUTE_UPDATE_TIMELINE = "/update_timeline"
ROUTE_SHUTDOWN = "/shutdown"


def dumps(obj: Any) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def loads(data: bytes) -> Any:
    return pickle.loads(data)


def make_url(host: str, port: int, route: str) -> str:
    host = host.strip()
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return f"{host.rstrip('/')}:{int(port)}{route}"
