from __future__ import annotations

from astrbot_plugin_image_generation.adapter.openai_adapter import OpenAIAdapter
from astrbot_plugin_image_generation.core.base_adapter import BaseImageAdapter
from astrbot_plugin_image_generation.core.types import AdapterConfig, AdapterType


def test_retry_policy_skips_non_retryable_provider_errors() -> None:
    assert BaseImageAdapter._is_retryable_error("API 錯誤 (400)") is False
    assert BaseImageAdapter._is_retryable_error("API 錯誤 (401)") is False
    assert BaseImageAdapter._is_retryable_error("API 錯誤 (429)") is True
    assert BaseImageAdapter._is_retryable_error("API 錯誤 (503)") is True


def test_retry_policy_keeps_transport_failures_retryable() -> None:
    assert BaseImageAdapter._is_retryable_error("連線逾時") is True
    assert BaseImageAdapter._is_retryable_error("未配置 API Key") is False


def test_openai_error_keeps_structured_provider_code() -> None:
    body = (
        '{"error":{"code":"content_policy_violation",'
        '"message":"prompt rejected"}}'
    )

    assert OpenAIAdapter._format_provider_error(400, body) == (
        "API 錯誤 (400): content_policy_violation - prompt rejected"
    )


def test_openai_api_base_preserves_configured_v1_path() -> None:
    adapter = OpenAIAdapter(
        AdapterConfig(
            type=AdapterType.OPENAI,
            model="gpt-image-2",
            base_url="https://www.inroi.shop/v1",
        )
    )

    assert adapter._get_api_base_url() == "https://www.inroi.shop/v1"
