from __future__ import annotations

import asyncio
import socket

import aiohttp
import pytest


class FakeContent:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def iter_chunked(self, _size: int):
        if self._payload:
            yield self._payload


class ErrorContent:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    async def iter_chunked(self, _size: int):
        raise self._error
        yield b""


class BlockingContent:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self._release = asyncio.Event()

    async def iter_chunked(self, _size: int):
        self.entered.set()
        await self._release.wait()
        yield b""


class FakeTransport:
    def __init__(self, peer: tuple[str, int] | None) -> None:
        self._peer = peer

    def get_extra_info(self, name: str):
        return self._peer if name == "peername" else None


class FakeConnection:
    def __init__(self, peer: tuple[str, int] | None) -> None:
        self.transport = FakeTransport(peer)


class FakeResponse:
    def __init__(
        self,
        status: int,
        payload: bytes = b"",
        *,
        location: str | None = None,
        peer: tuple[str, int] | None = ("93.184.216.34", 443),
    ) -> None:
        self.status = status
        self.content_length: int | None = len(payload)
        self.content = FakeContent(payload)
        self.headers = {} if location is None else {"Location": location}
        self.connection = FakeConnection(peer) if peer is not None else None
        self.exited = False

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, *_args) -> None:
        self.exited = True


class FakeSession:
    responses: list[FakeResponse] = []
    instances: list[FakeSession] = []

    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.closed = False
        self.connector = kwargs.get("connector")
        self._responses = list(type(self).responses)
        type(self).instances.append(self)

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *_args) -> None:
        await self.close()

    def get(self, url: str, *, allow_redirects: bool = True) -> FakeResponse:
        self.calls.append((url, allow_redirects))
        return self._responses.pop(0)

    async def close(self) -> None:
        self.closed = True
        if self.connector is not None:
            result = self.connector.close()
            if hasattr(result, "__await__"):
                await result


class FakeResolver:
    def __init__(self, *answer_sets: tuple[str, ...]) -> None:
        self._answer_sets = answer_sets
        self.calls = 0

    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_UNSPEC
    ) -> list[dict[str, str | int]]:
        addresses = self._answer_sets[self.calls]
        self.calls += 1
        return [
            {
                "hostname": host,
                "host": address,
                "port": port,
                "family": socket.AF_INET,
                "proto": 0,
                "flags": 0,
            }
            for address in addresses
        ]

    async def close(self) -> None:
        return None


def install_session(
    monkeypatch: pytest.MonkeyPatch, responses: list[FakeResponse]
) -> None:
    FakeSession.responses = responses
    FakeSession.instances = []
    monkeypatch.setattr(aiohttp, "ClientSession", FakeSession)
