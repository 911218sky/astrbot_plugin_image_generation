from __future__ import annotations

from astrbot_plugin_image_generation.core.base_adapter import BaseImageAdapter


def test_retry_policy_skips_non_retryable_provider_errors() -> None:
    assert BaseImageAdapter._is_retryable_error("API 錯誤 (400)") is False
    assert BaseImageAdapter._is_retryable_error("API 錯誤 (401)") is False
    assert BaseImageAdapter._is_retryable_error("API 錯誤 (429)") is True
    assert BaseImageAdapter._is_retryable_error("API 錯誤 (503)") is True


def test_retry_policy_keeps_transport_failures_retryable() -> None:
    assert BaseImageAdapter._is_retryable_error("連線逾時") is True
    assert BaseImageAdapter._is_retryable_error("未配置 API Key") is False
