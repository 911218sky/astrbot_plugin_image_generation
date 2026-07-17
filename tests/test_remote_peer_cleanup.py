from __future__ import annotations

import asyncio  # noqa: F401  # noqa: ANYIO_OK
import importlib
import socket
from types import ModuleType
from typing import ClassVar

import pytest

from reference_transport_fakes import (
    BlockingContent,
    ErrorContent,
    FakeResponse,
    FakeSession,
    install_session,
)


def _module() -> ModuleType:
    return importlib.import_module(
        "astrbot_plugin_image_generation.core.reference_collector"
    )


def _transport_module() -> ModuleType:
    return importlib.import_module(
        "astrbot_plugin_image_generation.core.reference_transport"
    )


class TrackingPublicResolver:
    instances: ClassVar[list[TrackingPublicResolver]] = []

    def __init__(self) -> None:
        self.close_calls = 0
        self._addresses = frozenset({"93.184.216.34"})
        type(self).instances.append(self)

    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_UNSPEC
    ) -> list[dict[str, str | int]]:
        return [
            {
                "hostname": host,
                "host": "93.184.216.34",
                "port": port,
                "family": family,
                "proto": 0,
                "flags": 0,
            }
        ]

    def addresses_for(self, _host: str, _port: int) -> frozenset[str]:
        return self._addresses

    async def close(self) -> None:
        self.close_calls += 1


def _install(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[FakeResponse],
    *,
    track_resolver: bool = False,
) -> None:
    install_session(monkeypatch, responses)
    if track_resolver:
        TrackingPublicResolver.instances = []
        monkeypatch.setattr(
            _transport_module(), "_PublicOnlyResolver", TrackingPublicResolver
        )


async def _read_outcome(module: ModuleType, reader, url: str):
    try:
        return await reader.read(url, 1024)
    except module.RemoteReferenceDenied as exc:
        return exc


@pytest.mark.asyncio
async def test_baseline_public_literal_peer_is_accepted_and_resources_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    response = FakeResponse(200, b"image", peer=("93.184.216.34", 443))
    _install(monkeypatch, [response])
    reader = _module().AiohttpRemoteReader()

    # When
    payload = await reader.read("https://93.184.216.34/image.png", 1024)

    # Then
    assert payload == b"image" and response.exited
    assert all(session.closed for session in FakeSession.instances)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "peer",
    (
        pytest.param(None, id="identity_unavailable"),
        pytest.param(("93.184.216.35", 443), id="literal_target_mismatch"),
    ),
)
async def test_unverifiable_or_mismatched_peer_is_rejected(
    peer: tuple[str, int] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    response = FakeResponse(200, b"unsafe", peer=peer)
    _install(monkeypatch, [response])
    module = _module()

    # When
    outcome = await _read_outcome(
        module,
        module.AiohttpRemoteReader(),
        "https://93.184.216.34/image.png",
    )

    # Then
    assert isinstance(outcome, module.RemoteReferenceDenied)
    assert response.exited and all(session.closed for session in FakeSession.instances)


@pytest.mark.asyncio
async def test_redirect_peer_is_verified_before_following(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    redirect = FakeResponse(302, location="/next", peer=("93.184.216.35", 443))
    destination = FakeResponse(200, b"unsafe")
    _install(monkeypatch, [redirect, destination])
    module = _module()

    # When
    outcome = await _read_outcome(
        module,
        module.AiohttpRemoteReader(),
        "https://93.184.216.34/start",
    )

    # Then
    assert isinstance(outcome, module.RemoteReferenceDenied)
    assert redirect.exited and not destination.exited
    assert FakeSession.instances[0].calls == [("https://93.184.216.34/start", False)]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome_kind", ("success", "timeout"))
async def test_custom_resolver_closes_once_on_completion(
    outcome_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    response = FakeResponse(200, b"image")
    if outcome_kind == "timeout":
        response.content_length = None
        response.content = ErrorContent(TimeoutError())
    _install(monkeypatch, [response], track_resolver=True)
    reader = _module().AiohttpRemoteReader()

    # When
    outcome = await reader.read("https://93.184.216.34/image.png", 1024)

    # Then
    assert outcome == (b"image" if outcome_kind == "success" else None)
    assert [item.close_calls for item in TrackingPublicResolver.instances] == [1]


@pytest.mark.asyncio
async def test_custom_resolver_closes_once_when_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    response = FakeResponse(200)
    content = BlockingContent()
    response.content_length = None
    response.content = content
    _install(monkeypatch, [response], track_resolver=True)
    task = asyncio.create_task(
        _module().AiohttpRemoteReader().read("https://93.184.216.34/image.png", 1024)
    )
    await asyncio.wait_for(content.entered.wait(), timeout=1)

    # When
    task.cancel()

    # Then
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [item.close_calls for item in TrackingPublicResolver.instances] == [1]
