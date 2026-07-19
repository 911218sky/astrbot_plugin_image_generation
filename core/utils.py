from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from io import BytesIO
from typing import Any

from PIL import Image

from astrbot.api import logger

from .constants import (
    MASK_MIN_LENGTH,
    MASK_PLACEHOLDER,
    MASK_VISIBLE_CHARS,
    SUPPORTED_ASPECT_RATIOS,
    SUPPORTED_RESOLUTIONS,
)
from .types import ImageData

SUPPORTED_IMAGE_FORMATS = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
}

# Use the constants module values and keep local set lookups for compatibility.
ALLOWED_ASPECT_RATIOS = set(SUPPORTED_ASPECT_RATIOS)
ALLOWED_RESOLUTIONS = set(SUPPORTED_RESOLUTIONS)
SELF_AVATAR_ALIAS_PATTERN = re.compile(r"(?<!\S)@self(?!\S)", re.IGNORECASE)


def detect_mime_type(data: bytes) -> str:
    """Detect MIME type from image magic numbers."""

    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if len(data) > 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"heic", b"heix", b"heim", b"heis"):
            return "image/heic"
        if brand in (b"mif1", b"msf1", b"heif"):
            return "image/heif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def _sync_convert_image_format(image_data: bytes, mime_type: str) -> ImageData:
    """Synchronously convert unsupported image formats to JPEG."""

    try:
        img = Image.open(BytesIO(image_data))

        if img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode in ("P", "LA"):
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[3])
            img = background

        output = BytesIO()
        img.save(output, format="JPEG", quality=95)
        logger.debug("[ImageGen] Converted image to JPEG")
        return ImageData(data=output.getvalue(), mime_type="image/jpeg")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[ImageGen] Failed to convert image format: {exc}")
        return ImageData(data=image_data, mime_type=mime_type)


async def convert_image_format(image_data: bytes, mime_type: str) -> ImageData:
    """Convert an image when the MIME type is unsupported."""

    real_mime = detect_mime_type(image_data)
    if real_mime in SUPPORTED_IMAGE_FORMATS:
        return ImageData(data=image_data, mime_type=real_mime)
    logger.info(f"[ImageGen] Converting image format: {mime_type} -> image/jpeg")
    return await asyncio.to_thread(_sync_convert_image_format, image_data, mime_type)


async def convert_images_batch(images: Iterable[ImageData]) -> list[ImageData]:
    """Convert a batch of images concurrently."""

    tasks = [convert_image_format(img.data, img.mime_type) for img in images]
    return await asyncio.gather(*tasks)


def validate_aspect_ratio(value: str | None) -> str | None:
    """Validate an aspect ratio against the allowed values."""

    if value is None:
        return None
    return value if value in ALLOWED_ASPECT_RATIOS else None


def validate_resolution(value: str | None) -> str | None:
    """Validate a resolution against the allowed values."""

    if value is None:
        return None
    return value if value in ALLOWED_RESOLUTIONS else None


def normalize_batch_count(value: Any, maximum: int) -> int:
    if isinstance(value, bool):
        return 1
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 1
    return min(maximum, max(1, count))


def extract_self_avatar_alias(prompt: str) -> tuple[str, bool]:
    """Strip the ``@self`` alias from a prompt and report whether it was used."""

    if not prompt:
        return "", False

    used_alias = bool(SELF_AVATAR_ALIAS_PATTERN.search(prompt))
    if not used_alias:
        return prompt.strip(), False

    cleaned_prompt = SELF_AVATAR_ALIAS_PATTERN.sub(" ", prompt)
    cleaned_prompt = re.sub(r"\s+", " ", cleaned_prompt).strip()
    return cleaned_prompt, True


def mask_sensitive(
    value: str,
    visible_chars: int = MASK_VISIBLE_CHARS,
    min_length: int = MASK_MIN_LENGTH,
    placeholder: str = MASK_PLACEHOLDER,
) -> str:
    """Mask a sensitive string for logs."""

    if len(value) <= min_length:
        return placeholder
    return f"{value[:visible_chars]}{placeholder}{value[-visible_chars:]}"
