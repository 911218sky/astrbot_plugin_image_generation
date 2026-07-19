from __future__ import annotations

from astrbot_plugin_image_generation.adapter.openai_adapter import OpenAIAdapter
from astrbot_plugin_image_generation.core.types import (
    AdapterConfig,
    AdapterType,
    GenerationRequest,
)


def test_gpt_image_payload_defaults_to_auto_quality_and_size() -> None:
    adapter = OpenAIAdapter(
        AdapterConfig(type=AdapterType.OPENAI, model="gpt-image-2")
    )

    payload = adapter._build_payload(GenerationRequest(prompt="blue square"))

    assert payload["quality"] == "auto"
    assert payload["size"] == "auto"
