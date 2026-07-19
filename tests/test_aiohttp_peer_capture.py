from __future__ import annotations

import asyncio  # noqa: ANYIO_OK
import importlib
import socket
from types import ModuleType

import pytest


def _module() -> ModuleType:
    return importlib.import_module(
        "astrbot_plugin_image_generation.core.reference_transport"
    )


class _LoopbackResolver:
    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_UNSPEC
    ) -> list[dict[str, str | int]]:
        return [
            {
                "hostname": host,
                "host": "127.0.0.1",
                "port": port,
                "family": socket.AF_INET,
                "proto": 0,
                "flags": 0,
            }
        ]

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_empty_redirect_keeps_connected_peer_for_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    module = _module()
    original_public_ip = module._public_ip

    def public_test_address(value: str) -> str | None:
        if value == "127.0.0.1":
            return "93.184.216.34"
        return original_public_ip(value)

    class LoopbackPublicResolver(module._PublicOnlyResolver):
        def __init__(self) -> None:
            super().__init__(_LoopbackResolver())

    request_count = 0

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        nonlocal request_count
        await reader.readuntil(b"\r\n\r\n")
        request_count += 1
        if request_count == 1:
            reply = (
                b"HTTP/1.1 302 Found\r\nLocation: /image\r\n"
                b"Content-Length: 0\r\nConnection: close\r\n\r\n"
            )
        else:
            reply = (
                b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n"
                b"Connection: close\r\n\r\nimage"
            )
        writer.write(reply)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setattr(module, "_public_ip", public_test_address)
    monkeypatch.setattr(module, "_PublicOnlyResolver", LoopbackPublicResolver)

    # When
    async with server:
        payload = await module.AiohttpRemoteReader().read(
            f"http://loopback.test:{port}/start", 1024
        )

    # Then
    assert payload == b"image"
    assert request_count == 2
