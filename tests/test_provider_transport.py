from __future__ import annotations

import base64

from astrbot_plugin_image_generation.core.provider_transport import (
    MAX_PROVIDER_IMAGE_BYTES,
    decode_provider_base64,
)


def test_provider_base64_rejects_oversized_payload_before_decoding() -> None:
    encoded_size = ((MAX_PROVIDER_IMAGE_BYTES + 2) // 3) * 4 + 4
    assert decode_provider_base64("A" * encoded_size) is None


def test_provider_base64_accepts_valid_image_bytes() -> None:
    payload = b"image-bytes"
    assert decode_provider_base64(base64.b64encode(payload).decode()) == payload
