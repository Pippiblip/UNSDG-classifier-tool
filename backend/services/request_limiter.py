"""Process-local pacing for external API requests."""

from __future__ import annotations

import os
import threading
import time


_lock = threading.Lock()
_last_request = time.monotonic()


def wait_for_request() -> None:
    interval = max(0.0, float(os.getenv("SDG_REQUEST_INTERVAL", "0")))
    if interval == 0:
        return
    global _last_request
    with _lock:
        remaining = interval - (time.monotonic() - _last_request)
        if remaining > 0:
            time.sleep(remaining)
        _last_request = time.monotonic()
