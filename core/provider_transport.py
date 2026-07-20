from __future__ import annotations

import base64
import binascii
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import json
from typing import Any

import anyio

from .reference_transport import (
    AiohttpRemoteReader,
    RemoteLimitExceeded,
    RemoteReferenceDenied,
)

MIB = 1024 * 1024
MAX_PROVIDER_IMAGE_BYTES = 30 * MIB
MAX_PROVIDER_JSON_BYTES = 256 * MIB
PROVIDER_READ_CHUNK_SIZE = 64 * 1024
_PROVIDER_IMAGE_LIMIT = ContextVar(
    "provider_image_limit_bytes", default=MAX_PROVIDER_IMAGE_BYTES
)
_PROVIDER_JSON_LIMITER = anyio.CapacityLimiter(1)


@contextmanager
def provider_image_limit(max_bytes: int) -> Iterator[None]:
    bounded_limit = min(MAX_PROVIDER_IMAGE_BYTES, max(1, max_bytes))
    token = _PROVIDER_IMAGE_LIMIT.set(bounded_limit)
    try:
        yield
    finally:
        _PROVIDER_IMAGE_LIMIT.reset(token)


def _current_provider_image_limit() -> int:
    return _PROVIDER_IMAGE_LIMIT.get()


async def download_provider_image(url: str) -> bytes | None:
    reader = AiohttpRemoteReader()
    try:
        return await reader.read(url, _current_provider_image_limit())
    except (RemoteLimitExceeded, RemoteReferenceDenied):
        return None
    finally:
        await reader.close()


def decode_provider_base64(value: Any) -> bytes | None:
    if not isinstance(value, str):
        return None
    max_bytes = _current_provider_image_limit()
    max_encoded = ((max_bytes + 2) // 3) * 4
    if len(value) > max_encoded:
        return None
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None
    return decoded if len(decoded) <= max_bytes else None


async def read_provider_json(response: Any) -> dict[str, Any] | None:
    async with _PROVIDER_JSON_LIMITER:
        content_length = getattr(response, "content_length", None)
        if isinstance(content_length, int) and content_length > MAX_PROVIDER_JSON_BYTES:
            return None
        content = response.content
        iter_chunked = getattr(content, "iter_chunked", None)
        if callable(iter_chunked):
            payload = bytearray()
            async for chunk in iter_chunked(PROVIDER_READ_CHUNK_SIZE):
                payload.extend(chunk)
                if len(payload) > MAX_PROVIDER_JSON_BYTES:
                    return None
        else:
            payload = await content.read(MAX_PROVIDER_JSON_BYTES + 1)
        if len(payload) > MAX_PROVIDER_JSON_BYTES:
            return None
        try:
            parsed = json.loads(payload.decode("utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None
