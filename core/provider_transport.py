from __future__ import annotations

import base64
import binascii
import json
from typing import Any

from .reference_transport import (
    AiohttpRemoteReader,
    RemoteLimitExceeded,
    RemoteReferenceDenied,
)

MIB = 1024 * 1024
MAX_PROVIDER_IMAGE_BYTES = 30 * MIB
MAX_PROVIDER_JSON_BYTES = 48 * MIB


async def download_provider_image(url: str) -> bytes | None:
    reader = AiohttpRemoteReader()
    try:
        return await reader.read(url, MAX_PROVIDER_IMAGE_BYTES)
    except (RemoteLimitExceeded, RemoteReferenceDenied):
        return None
    finally:
        await reader.close()


def decode_provider_base64(value: Any) -> bytes | None:
    if not isinstance(value, str):
        return None
    max_encoded = ((MAX_PROVIDER_IMAGE_BYTES + 2) // 3) * 4
    if len(value) > max_encoded:
        return None
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None
    return decoded if len(decoded) <= MAX_PROVIDER_IMAGE_BYTES else None


async def read_provider_json(response: Any) -> dict[str, Any] | None:
    payload = await response.content.read(MAX_PROVIDER_JSON_BYTES + 1)
    if len(payload) > MAX_PROVIDER_JSON_BYTES:
        return None
    try:
        parsed = json.loads(payload.decode("utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None
