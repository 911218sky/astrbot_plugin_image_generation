from __future__ import annotations

from astrbot_plugin_image_generation.core.llm_tool import (
    ImageGenerationTool,
    _resolve_aspect_ratio,
    _resolve_resolution,
    adjust_tool_parameters,
)
from astrbot_plugin_image_generation.core.types import ImageCapability


def test_llm_dimensions_ignore_non_string_values() -> None:
    assert _resolve_aspect_ratio("cat", 123, None) is None
    assert _resolve_resolution("cat", None, 123) == "1K"


def test_llm_dimensions_infer_from_prompt() -> None:
    assert _resolve_aspect_ratio("手機桌布", None, "自動") == "9:16"
    assert _resolve_resolution("高解析海報", None, "1K") == "2K"


def test_tool_schema_uses_runtime_batch_limit() -> None:
    tool = ImageGenerationTool()

    adjust_tool_parameters(tool, ImageCapability.TEXT_TO_IMAGE, max_batch_count=4)

    assert tool.parameters["properties"]["count"]["maximum"] == 4
