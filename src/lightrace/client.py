"""Lightrace client — singleton that manages the exporter and tool WS client."""

from __future__ import annotations

import logging

from .exporter import BatchExporter
from .trace import _set_exporter

logger = logging.getLogger("lightrace")


class Lightrace:
    """Main Lightrace SDK client.

    Usage:
        lt = Lightrace(
            public_key="pk-lt-demo",
            secret_key="sk-lt-demo",
            host="http://localhost:3002",
        )

        @trace()
        def my_function():
            ...

        lt.flush()
        lt.shutdown()
    """

    _instance: Lightrace | None = None

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = "http://localhost:3002",
        flush_at: int = 50,
        flush_interval: float = 5.0,
        timeout: float = 10.0,
        enabled: bool = True,
    ):
        self._public_key = public_key
        self._secret_key = secret_key
        self._host = host.rstrip("/")
        self._enabled = enabled

        if not enabled:
            logger.info("Lightrace disabled — no events will be sent")
            return

        self._exporter = BatchExporter(
            host=self._host,
            public_key=public_key,
            secret_key=secret_key,
            flush_at=flush_at,
            flush_interval=flush_interval,
            timeout=timeout,
        )
        _set_exporter(self._exporter)

        Lightrace._instance = self
        logger.info("Lightrace initialized → %s", self._host)

    @classmethod
    def get_instance(cls) -> Lightrace | None:
        return cls._instance

    def flush(self) -> None:
        """Flush all pending events to the server."""
        if self._enabled and self._exporter:
            self._exporter.flush()

    def shutdown(self) -> None:
        """Flush and shut down the client."""
        if self._enabled and self._exporter:
            self._exporter.shutdown()
            _set_exporter(None)
        Lightrace._instance = None
        logger.info("Lightrace shut down")
