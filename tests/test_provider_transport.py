from __future__ import annotations

import base64
import json

import pytest

from astrbot_plugin_image_generation.core.provider_transport import (
    MAX_PROVIDER_IMAGE_BYTES,
    decode_provider_base64,
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
