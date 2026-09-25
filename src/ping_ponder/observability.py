"""Machine-readable standard-logging events with monotonic timestamps."""

import json
import logging
import time
from typing import Any


def emit(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, "monotonic_time": time.monotonic(), **fields}, default=str, sort_keys=True))
