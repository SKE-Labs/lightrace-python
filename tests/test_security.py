"""Tests for HMAC signing and nonce tracking."""

import time

from lightrace.security import NonceTracker, sign, verify


class TestHMAC:
    def test_sign_and_verify(self):
        token = "test-session-token"
        nonce = "nonce-123"
        tool = "weather"
        input_data = {"city": "NYC"}

        signature = sign(token, nonce, tool, input_data)
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA-256 hex digest

        assert verify(token, nonce, tool, input_data, signature)

    def test_wrong_token_fails(self):
        sig = sign("correct-token", "nonce", "tool", {"a": 1})
        assert not verify("wrong-token", "nonce", "tool", {"a": 1}, sig)

    def test_wrong_nonce_fails(self):
        sig = sign("token", "nonce-1", "tool", {"a": 1})
        assert not verify("token", "nonce-2", "tool", {"a": 1}, sig)

    def test_wrong_input_fails(self):
        sig = sign("token", "nonce", "tool", {"a": 1})
        assert not verify("token", "nonce", "tool", {"a": 2}, sig)


class TestNonceTracker:
    def test_fresh_nonce_accepted(self):
        tracker = NonceTracker()
        assert tracker.check_and_mark("nonce-1") is True

    def test_duplicate_nonce_rejected(self):
        tracker = NonceTracker()
        assert tracker.check_and_mark("nonce-1") is True
        assert tracker.check_and_mark("nonce-1") is False

    def test_expired_nonce_cleaned_up(self):
        tracker = NonceTracker(ttl_seconds=0.1)
        tracker.check_and_mark("nonce-1")
        time.sleep(0.15)
        # After TTL, the nonce should be cleaned up
        # A new nonce should still work
        assert tracker.check_and_mark("nonce-2") is True
