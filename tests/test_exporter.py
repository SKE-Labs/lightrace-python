"""Tests for the batch exporter."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from lightrace.exporter import BatchExporter
from lightrace.types import TraceEvent


class TestBatchExporter:
    def test_enqueue_and_flush(self):
        exporter = BatchExporter(
            host="http://localhost:3002",
            public_key="pk-test",
            secret_key="sk-test",
            flush_at=100,
            flush_interval=9999,  # disable auto-flush
        )

        event = TraceEvent(
            event_id="evt-1",
            event_type="trace-create",
            body={"id": "trace-1", "name": "test"},
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        exporter.enqueue(event)

        with patch.object(exporter, "_client") as mock_client:
            mock_resp = MagicMock()
            mock_resp.status_code = 207
            mock_client.post.return_value = mock_resp

            exporter.flush()

            mock_client.post.assert_called_once()
            call_kwargs = mock_client.post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert len(payload["batch"]) == 1
            assert payload["batch"][0]["type"] == "trace-create"

        exporter.shutdown()

    def test_auto_flush_at_threshold(self):
        exporter = BatchExporter(
            host="http://localhost:3002",
            public_key="pk-test",
            secret_key="sk-test",
            flush_at=2,
            flush_interval=9999,
        )

        with patch.object(exporter, "_do_flush") as mock_flush:
            exporter.enqueue(TraceEvent("1", "trace-create", {"id": "t1"}))
            mock_flush.assert_not_called()

            exporter.enqueue(TraceEvent("2", "trace-create", {"id": "t2"}))
            mock_flush.assert_called_once()

        exporter.shutdown()

    def test_auth_header_format(self):
        exporter = BatchExporter(
            host="http://localhost:3002",
            public_key="pk-test",
            secret_key="sk-test",
        )
        # Basic base64("pk-test:sk-test")
        import base64

        expected = "Basic " + base64.b64encode(b"pk-test:sk-test").decode()
        assert exporter._auth_header == expected
        exporter.shutdown()
