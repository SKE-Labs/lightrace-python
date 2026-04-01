"""Tests for the Lightrace client initialization and lifecycle."""

from unittest.mock import patch

from lightrace.client import Lightrace
from lightrace.trace import _set_otel_exporter


class TestLightraceClient:
    def teardown_method(self):
        instance = Lightrace.get_instance()
        if instance:
            instance.shutdown()
        _set_otel_exporter(None)

    def test_init_disabled(self):
        lt = Lightrace(enabled=False)
        assert lt._otel_exporter is None
        assert lt.dev_server is None

    def test_init_requires_keys(self):
        import pytest

        with pytest.raises(ValueError, match="public_key and secret_key are required"):
            Lightrace(public_key="", secret_key="")

    def test_init_with_dev_server_disabled(self):
        lt = Lightrace(
            public_key="pk-test",
            secret_key="sk-test",
            host="http://localhost:9999",
            dev_server=False,
        )
        assert lt._otel_exporter is not None
        assert lt.dev_server is None

    @patch("lightrace.client.httpx.post")
    def test_init_with_dev_server_enabled(self, mock_post):
        mock_post.return_value = None
        lt = Lightrace(
            public_key="pk-test",
            secret_key="sk-test",
            host="http://localhost:9999",
            dev_server=True,
        )
        assert lt.dev_server is not None
        assert lt.dev_server.port is not None
        assert lt.dev_server.port > 0

    def test_singleton_pattern(self):
        lt1 = Lightrace(
            public_key="pk-test",
            secret_key="sk-test",
            host="http://localhost:9999",
            dev_server=False,
        )
        assert Lightrace.get_instance() is lt1

        lt2 = Lightrace(
            public_key="pk-test2",
            secret_key="sk-test2",
            host="http://localhost:9999",
            dev_server=False,
        )
        assert Lightrace.get_instance() is lt2

    def test_user_id_session_id(self):
        lt = Lightrace(
            public_key="pk-test",
            secret_key="sk-test",
            host="http://localhost:9999",
            dev_server=False,
            user_id="user-1",
            session_id="sess-1",
        )
        assert lt.user_id == "user-1"
        assert lt.session_id == "sess-1"

    @patch("lightrace.client.httpx.post")
    def test_shutdown_stops_dev_server(self, mock_post):
        mock_post.return_value = None
        lt = Lightrace(
            public_key="pk-test",
            secret_key="sk-test",
            host="http://localhost:9999",
            dev_server=True,
        )
        assert lt.dev_server is not None

        lt.shutdown()
        assert lt.dev_server is None
        assert Lightrace.get_instance() is None
