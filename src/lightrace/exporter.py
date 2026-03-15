"""Batch exporter that sends events to the lightrace ingestion endpoint."""

from __future__ import annotations

import base64
import logging
import threading
import time

import httpx

from .types import TraceEvent

logger = logging.getLogger("lightrace")


class BatchExporter:
    """Collects trace events and flushes them in batches to the ingestion API."""

    def __init__(
        self,
        host: str,
        public_key: str,
        secret_key: str,
        flush_at: int = 50,
        flush_interval: float = 5.0,
        timeout: float = 10.0,
        max_retries: int = 2,
    ):
        self._host = host.rstrip("/")
        self._endpoint = f"{self._host}/api/public/ingestion"
        self._auth_header = self._build_auth(public_key, secret_key)
        self._flush_at = flush_at
        self._flush_interval = flush_interval
        self._timeout = timeout
        self._max_retries = max_retries

        self._queue: list[TraceEvent] = []
        self._lock = threading.Lock()
        self._client = httpx.Client(timeout=self._timeout)

        self._running = True
        self._flush_thread = threading.Thread(
            target=self._periodic_flush, daemon=True, name="lightrace-flush"
        )
        self._flush_thread.start()

    @staticmethod
    def _build_auth(public_key: str, secret_key: str) -> str:
        credentials = f"{public_key}:{secret_key}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"

    def enqueue(self, event: TraceEvent) -> None:
        with self._lock:
            self._queue.append(event)
            should_flush = len(self._queue) >= self._flush_at
        if should_flush:
            self._do_flush()

    def flush(self) -> None:
        self._do_flush()

    def shutdown(self) -> None:
        self._running = False
        self._do_flush()
        self._client.close()

    def _periodic_flush(self) -> None:
        while self._running:
            time.sleep(self._flush_interval)
            if self._running:
                self._do_flush()

    def _do_flush(self) -> None:
        with self._lock:
            if not self._queue:
                return
            batch = self._queue[:]
            self._queue.clear()

        payload = {"batch": [event.to_dict() for event in batch]}

        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.post(
                    self._endpoint,
                    json=payload,
                    headers={
                        "Authorization": self._auth_header,
                        "Content-Type": "application/json",
                    },
                )
                if resp.status_code < 400:
                    return
                if resp.status_code >= 500 and attempt < self._max_retries:
                    time.sleep(2**attempt)
                    continue
                logger.warning("Ingestion failed: %d %s", resp.status_code, resp.text[:200])
                return
            except httpx.HTTPError as e:
                if attempt < self._max_retries:
                    time.sleep(2**attempt)
                    continue
                logger.warning("Ingestion request failed: %s", e)
                return
