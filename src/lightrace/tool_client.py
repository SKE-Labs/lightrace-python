"""WebSocket client for remote tool invocation."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from typing import Any

from .security import NonceTracker, sign, verify
from .trace import _get_tool_registry
from .utils import generate_id, json_serializable

logger = logging.getLogger("lightrace")


class ToolClient:
    """Connects to lightrace via WebSocket, registers invocable tools,
    and executes tool invocations received from the server."""

    def __init__(
        self,
        host: str,
        public_key: str,
        secret_key: str,
        sdk_instance_id: str | None = None,
        heartbeat_interval: float = 30.0,
        max_reconnect_delay: float = 30.0,
    ):
        self._host = host.rstrip("/")
        self._public_key = public_key
        self._secret_key = secret_key
        self._sdk_instance_id = sdk_instance_id or generate_id()
        self._heartbeat_interval = heartbeat_interval
        self._max_reconnect_delay = max_reconnect_delay

        self._session_token: str | None = None
        self._nonce_tracker = NonceTracker(ttl_seconds=60.0)
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def ws_url(self) -> str:
        base = self._host.replace("http://", "ws://").replace("https://", "wss://")
        return f"{base}/api/public/tools/ws"

    def start(self) -> None:
        """Start the WebSocket client in a background thread."""
        registry = _get_tool_registry()
        if not registry:
            logger.debug("No invocable tools registered — skipping tool client")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="lightrace-tool-client"
        )
        self._thread.start()
        logger.info(
            "Tool client started — %d tool(s) registered: %s",
            len(registry),
            list(registry.keys()),
        )

    def stop(self) -> None:
        self._running = False

    def _run_loop(self) -> None:
        """Run the async WebSocket loop in a new event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._connect_loop())
        finally:
            loop.close()

    async def _connect_loop(self) -> None:
        """Connect with exponential backoff retry."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package required for tool invocation")
            return

        delay = 1.0
        while self._running:
            try:
                auth = base64.b64encode(f"{self._public_key}:{self._secret_key}".encode()).decode()
                async with websockets.connect(
                    self.ws_url,
                    additional_headers={"Authorization": f"Basic {auth}"},
                ) as ws:
                    delay = 1.0  # reset on successful connect
                    await self._handle_connection(ws)
            except Exception as e:  # noqa: BLE001
                if not self._running:
                    break  # type: ignore[unreachable]
                logger.warning("Tool client connection error: %s (retry in %.0fs)", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_reconnect_delay)

    async def _handle_connection(self, ws: Any) -> None:
        """Handle a single WebSocket connection lifecycle."""
        import websockets

        # Wait for connected message with session token
        raw = await asyncio.wait_for(ws.recv(), timeout=10)
        msg = json.loads(raw)
        if msg.get("type") != "connected":
            logger.error("Unexpected initial message: %s", msg.get("type"))
            return

        self._session_token = msg["sessionToken"]
        logger.debug("Tool client connected, session token received")

        # Register tools
        registry = _get_tool_registry()
        register_msg = {
            "type": "register",
            "sdkInstanceId": self._sdk_instance_id,
            "tools": [
                {"name": name, "inputSchema": info.get("input_schema")}
                for name, info in registry.items()
            ],
        }
        await ws.send(json.dumps(register_msg))

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws))

        try:
            async for raw_msg in ws:
                if not self._running:
                    break
                try:
                    msg = json.loads(raw_msg)
                    await self._handle_message(ws, msg)
                except Exception as e:
                    logger.error("Error handling message: %s", e)
        except websockets.ConnectionClosed:
            logger.info("Tool client connection closed")
        finally:
            heartbeat_task.cancel()

    async def _heartbeat_loop(self, ws: Any) -> None:
        while self._running:
            await asyncio.sleep(self._heartbeat_interval)
            try:
                await ws.send(json.dumps({"type": "heartbeat"}))
            except Exception:
                break

    async def _handle_message(self, ws: Any, msg: dict) -> None:
        msg_type = msg.get("type")

        if msg_type == "registered":
            logger.info("Tools registered: %s", msg.get("tools"))

        elif msg_type == "invoke":
            await self._handle_invoke(ws, msg)

        elif msg_type == "heartbeat_ack":
            pass

        elif msg_type == "error":
            logger.error("Server error: %s", msg.get("message"))

    async def _handle_invoke(self, ws: Any, msg: dict) -> None:
        nonce = msg.get("nonce", "")
        tool_name = msg.get("tool", "")
        input_data = msg.get("input")
        signature = msg.get("signature", "")

        # Verify HMAC
        if not self._session_token or not verify(
            self._session_token, nonce, tool_name, input_data, signature
        ):
            logger.warning("Invalid signature for tool invoke: %s", tool_name)
            await ws.send(json.dumps({"type": "error", "message": "Invalid signature"}))
            return

        # Check nonce freshness
        if not self._nonce_tracker.check_and_mark(nonce):
            logger.warning("Replayed nonce for tool invoke: %s", tool_name)
            await ws.send(json.dumps({"type": "error", "message": "Replayed nonce"}))
            return

        # Find and execute tool
        registry = _get_tool_registry()
        tool_info = registry.get(tool_name)
        if not tool_info:
            result_msg = {
                "type": "result",
                "nonce": nonce,
                "output": None,
                "error": f"Tool not found: {tool_name}",
                "durationMs": 0,
                "signature": sign(self._session_token, nonce, tool_name, None),
            }
            await ws.send(json.dumps(result_msg))
            return

        func = tool_info["func"]
        start = time.monotonic()
        try:
            if asyncio.iscoroutinefunction(func):
                if isinstance(input_data, dict):
                    output = await func(**input_data)
                else:
                    output = await func(input_data) if input_data is not None else await func()
            else:
                if isinstance(input_data, dict):
                    output = func(**input_data)
                else:
                    output = func(input_data) if input_data is not None else func()
            duration_ms = (time.monotonic() - start) * 1000
            output = json_serializable(output)
            result_msg = {
                "type": "result",
                "nonce": nonce,
                "output": output,
                "durationMs": round(duration_ms),
                "signature": sign(self._session_token, nonce, tool_name, output),
            }
        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            result_msg = {
                "type": "result",
                "nonce": nonce,
                "output": None,
                "error": str(e),
                "durationMs": round(duration_ms),
                "signature": sign(self._session_token, nonce, tool_name, None),
            }
            logger.error("Tool execution error for %s: %s", tool_name, e)

        await ws.send(json.dumps(result_msg))
