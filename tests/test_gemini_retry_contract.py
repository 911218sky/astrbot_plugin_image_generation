from __future__ import annotations

import pytest

from astrbot_plugin_image_generation.adapter.gemini_adapter import GeminiAdapter
from astrbot_plugin_image_generation.adapter.gemini_openai_adapter import (
    GeminiOpenAIAdapter,
)
from astrbot_plugin_image_generation.core.types import GenerationRequest


@pytest.mark.asyncio
async def test_gemini_preserves_http_error_status_for_retry_policy() -> None:
    adapter = object.__new__(GeminiAdapter)

    async def make_request(_session, _payload, _task_id):
        return None, "API 錯誤 (400)"

    adapter._build_payload = lambda _request: {}
    adapter._get_session = lambda: None
    adapter._make_request = make_request

    result = await adapter._generate_once(GenerationRequest(prompt="cat"))

    assert result == (None, "API 錯誤 (400)")


@pytest.mark.asyncio
async def test_gemini_openai_preserves_http_error_status_for_retry_policy() -> None:
    adapter = object.__new__(GeminiOpenAIAdapter)

    async def make_request(_session, _payload, _task_id):
        return None, "API 錯誤 (400)"

    adapter._build_payload = lambda _request: {}
    adapter._get_session = lambda: None
    adapter._make_request = make_request

    result = await adapter._generate_once(GenerationRequest(prompt="cat"))

    assert result == (None, "API 錯誤 (400)")
