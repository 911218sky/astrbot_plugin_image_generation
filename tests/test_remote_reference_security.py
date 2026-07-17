from __future__ import annotations

import asyncio  # noqa: F401  # noqa: ANYIO_OK
import importlib
from types import ModuleType

import pytest

from reference_transport_fakes import (
    BlockingContent,
    ErrorContent,
    FakeResolver,
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


def _install_session(
    monkeypatch: pytest.MonkeyPatch, responses: list[FakeResponse]
) -> None:
    install_session(monkeypatch, responses)


async def _read_outcome(module: ModuleType, reader, url: str):
    denied_type = getattr(module, "RemoteReferenceDenied", None)
    if denied_type is None:
        return await reader.read(url, 1024)
    try:
        return await reader.read(url, 1024)
    except denied_type as exc:
        return exc


@pytest.mark.asyncio
@pytest.mark.parametrize("run", range(2))
async def test_private_literal_is_rejected_before_connect(
    run: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    module = _module()
    _install_session(monkeypatch, [FakeResponse(200, b"secret")])
    reader = module.AiohttpRemoteReader()

    # When
    payload = await _read_outcome(
        module, reader, f"http://127.0.0.1:{8000 + run}/secret"
    )
    await reader.close()

    # Then
    calls = [call for session in FakeSession.instances for call in session.calls]
    denied_type = getattr(module, "RemoteReferenceDenied", ())
    assert denied_type and isinstance(payload, denied_type) and calls == [], (
        "remote references must reject private literal addresses before connecting"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("run", range(2))
async def test_redirect_to_private_address_is_rejected_per_hop(
    run: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    module = _module()
    redirect = f"http://127.0.0.1:{9000 + run}/metadata"
    response = FakeResponse(302, location=redirect)
    _install_session(monkeypatch, [response])
    reader = module.AiohttpRemoteReader()

    # When
    payload = await _read_outcome(module, reader, "https://93.184.216.34/start")
    await reader.close()

    # Then
    calls = [call for session in FakeSession.instances for call in session.calls]
    denied_type = getattr(module, "RemoteReferenceDenied", ())
    assert (
        denied_type
        and isinstance(payload, denied_type)
        and calls == [("https://93.184.216.34/start", False)]
    )
    assert response.exited and all(session.closed for session in FakeSession.instances)


@pytest.mark.asyncio
async def test_dns_rebinding_is_revalidated_and_rejected() -> None:
    # Given
    module = _transport_module()
    delegate = FakeResolver(("93.184.216.34",), ("127.0.0.1",))
    resolver = module._PublicOnlyResolver(delegate)

    # When / Then
    first = await resolver.resolve("example.test", 443)
    assert first[0]["host"] == "93.184.216.34"
    with pytest.raises(OSError, match="public"):
        await resolver.resolve("example.test", 443)
    await resolver.close()


@pytest.mark.asyncio
async def test_mixed_public_and_private_dns_answers_are_denied() -> None:
    # Given
    module = _transport_module()
    resolver = module._PublicOnlyResolver(FakeResolver(("93.184.216.34", "127.0.0.1")))

    # When / Then
    with pytest.raises(OSError, match="public"):
        await resolver.resolve("mixed.example", 443)
    await resolver.close()


@pytest.mark.asyncio
async def test_connected_private_peer_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    module = _module()
    _install_session(
        monkeypatch,
        [FakeResponse(200, b"unsafe", peer=("127.0.0.1", 443))],
    )
    reader = module.AiohttpRemoteReader()

    # When
    payload = await _read_outcome(module, reader, "https://93.184.216.34/image.png")
    await reader.close()

    # Then
    denied_type = getattr(module, "RemoteReferenceDenied", ())
    assert denied_type and isinstance(payload, denied_type)
    assert all(session.closed for session in FakeSession.instances)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    (
        "ftp://93.184.216.34/image.png",
        "https://user:password@93.184.216.34/image.png",
        "https://[::1]/image.png",
        "https://93.184.216.34:70000/image.png",
    ),
)
async def test_malformed_or_unsafe_url_never_opens_session(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    module = _module()
    _install_session(monkeypatch, [FakeResponse(200, b"unsafe")])
    reader = module.AiohttpRemoteReader()

    # When
    payload = await _read_outcome(module, reader, url)
    await reader.close()

    # Then
    denied_type = getattr(module, "RemoteReferenceDenied", ())
    assert denied_type and isinstance(payload, denied_type)
    assert FakeSession.instances == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("redirects", "expected"),
    (
        pytest.param(4, b"image", id="four_allowed"),
        pytest.param(5, None, id="fifth_denied"),
    ),
)
async def test_redirect_limit_closes_every_response(
    redirects: int, expected: bytes | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    responses = [
        FakeResponse(302, location=f"/hop/{index}") for index in range(redirects)
    ]
    if expected is not None:
        responses.append(FakeResponse(200, expected))
    _install_session(monkeypatch, responses)
    module = _module()
    reader = module.AiohttpRemoteReader()

    # When
    outcome = await _read_outcome(module, reader, "https://93.184.216.34/start")

    # Then
    if expected is None:
        assert isinstance(outcome, module.RemoteReferenceDenied)
    else:
        assert outcome == expected
    assert all(response.exited for response in responses)
    assert all(session.closed for session in FakeSession.instances)
    await reader.close()


@pytest.mark.asyncio
async def test_declared_content_length_over_limit_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    response = FakeResponse(200, b"12345678")
    response.content_length = 9
    _install_session(monkeypatch, [response])
    module = _module()
    reader = module.AiohttpRemoteReader()

    # When / Then
    with pytest.raises(module.RemoteLimitExceeded):
        await reader.read("https://93.184.216.34/image.png", 8)
    assert response.exited and all(session.closed for session in FakeSession.instances)
    await reader.close()


@pytest.mark.asyncio
async def test_streamed_body_over_limit_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    response = FakeResponse(200, b"123456789")
    response.content_length = None
    _install_session(monkeypatch, [response])
    module = _module()
    reader = module.AiohttpRemoteReader()

    # When / Then
    with pytest.raises(module.RemoteLimitExceeded):
        await reader.read("https://93.184.216.34/image.png", 8)
    assert response.exited and all(session.closed for session in FakeSession.instances)
    await reader.close()


@pytest.mark.asyncio
async def test_timeout_closes_response_and_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    response = FakeResponse(200)
    response.content_length = None
    response.content = ErrorContent(TimeoutError())
    _install_session(monkeypatch, [response])
    reader = _module().AiohttpRemoteReader()

    # When
    outcome = await reader.read("https://93.184.216.34/image.png", 8)

    # Then
    assert outcome is None and response.exited
    assert all(session.closed for session in FakeSession.instances)
    await reader.close()


@pytest.mark.asyncio
async def test_asyncio_cancellation_propagates_after_resource_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    response = FakeResponse(200)
    content = BlockingContent()
    response.content_length = None
    response.content = content
    _install_session(monkeypatch, [response])
    reader = _module().AiohttpRemoteReader()
    task = asyncio.create_task(reader.read("https://93.184.216.34/image.png", 8))
    await asyncio.wait_for(content.entered.wait(), timeout=1)

    # When
    task.cancel()

    # Then
    with pytest.raises(asyncio.CancelledError):
        await task
    assert response.exited and all(session.closed for session in FakeSession.instances)
    await reader.close()
