"""HMAC signing and verification for secure tool invocation."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any


def sign(session_token: str, nonce: str, tool: str, input_data: Any) -> str:
    """Create HMAC-SHA256 signature for a tool invocation."""
    payload = nonce + tool + json.dumps(input_data, sort_keys=True, default=str)
    return hmac.new(
        session_token.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()


def verify(session_token: str, nonce: str, tool: str, input_data: Any, signature: str) -> bool:
    """Verify HMAC-SHA256 signature for a tool invocation."""
    expected = sign(session_token, nonce, tool, input_data)
    return hmac.compare_digest(expected, signature)


class NonceTracker:
    """Tracks single-use nonces with TTL to prevent replay attacks."""

    def __init__(self, ttl_seconds: float = 60.0):
        self._ttl = ttl_seconds
        self._seen: dict[str, float] = {}

    def check_and_mark(self, nonce: str) -> bool:
        """Return True if the nonce is fresh (not seen before). Marks it as used."""
        self._cleanup()
        if nonce in self._seen:
            return False
        self._seen[nonce] = time.monotonic()
        return True

    def _cleanup(self) -> None:
        """Remove expired nonces."""
        cutoff = time.monotonic() - self._ttl
        expired = [n for n, t in self._seen.items() if t < cutoff]
        for n in expired:
            del self._seen[n]
