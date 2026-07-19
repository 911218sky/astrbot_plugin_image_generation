from __future__ import annotations

import base64
import json

import anyio
import pytest

from astrbot_plugin_image_generation.core.provider_transport import (
    MAX_PROVIDER_IMAGE_BYTES,
    decode_provider_base64,
    provider_image_limit,
    read_provider_json,
)


class _ResponseContent:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def read(self, _limit: int) -> bytes:
        return self._payload


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.content = _ResponseContent(payload)


def test_provider_base64_rejects_oversized_payload_before_decoding() -> None:
    encoded_size = ((MAX_PROVIDER_IMAGE_BYTES + 2) // 3) * 4 + 4
    assert decode_provider_base64("A" * encoded_size) is None


def test_provider_base64_accepts_valid_image_bytes() -> None:
    payload = b"image-bytes"
    assert decode_provider_base64(base64.b64encode(payload).decode()) == payload


@pytest.mark.asyncio
async def test_provider_json_accepts_large_base64_image_response() -> None:
    response = {"data": [{"b64_json": "A" * (3 * 1024 * 1024)}]}

    parsed = await read_provider_json(
        _Response(json.dumps(response, separators=(",", ":")).encode())
    )

    assert parsed == response


@pytest.mark.asyncio
async def test_provider_json_reassembles_chunked_response() -> None:
    response = {"data": [{"b64_json": "A" * (2 * 1024 * 1024)}]}
    payload = json.dumps(response, separators=(",", ":")).encode()

    class ChunkedContent:
        async def read(self, _limit: int) -> bytes:
            return payload[:1024]

        async def iter_chunked(self, size: int):
            for offset in range(0, len(payload), size):
                await anyio.sleep(0)
                yield payload[offset : offset + size]

    class ChunkedResponse:
        def __init__(self) -> None:
            self.content = ChunkedContent()

    parsed = await read_provider_json(ChunkedResponse())

    assert parsed == response


@pytest.mark.asyncio
async def test_provider_json_accepts_64_mib_base64_image_response() -> None:
    response = {"data": [{"b64_json": "A" * (64 * 1024 * 1024)}]}

    parsed = await read_provider_json(
        _Response(json.dumps(response, separators=(",", ":")).encode())
    )

    assert parsed == response


def test_provider_base64_uses_configured_limit_before_decoding() -> None:
    payload = base64.b64encode(b"too-large").decode()

    with provider_image_limit(len(b"too-large") - 1):
        decoded = decode_provider_base64(payload)

    assert decoded is None


@pytest.mark.asyncio
async def test_provider_json_parsing_is_serialized() -> None:
    class TrackingContent:
        active = 0
        peak = 0

        async def read(self, _limit: int) -> bytes:
            type(self).active += 1
            type(self).peak = max(type(self).peak, type(self).active)
            await anyio.sleep(0)
            type(self).active -= 1
            return b'{"data": []}'

    class TrackingResponse:
        def __init__(self) -> None:
            self.content = TrackingContent()

    responses = [TrackingResponse(), TrackingResponse()]

    async with anyio.create_task_group() as task_group:
        for response in responses:
            task_group.start_soon(read_provider_json, response)

    assert TrackingContent.peak == 1
