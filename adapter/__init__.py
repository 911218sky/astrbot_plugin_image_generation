"""
Adapter module for image generation plugin
圖像生成插件的適配器模組
"""

from .gemini_adapter import GeminiAdapter
from .gemini_openai_adapter import GeminiOpenAIAdapter
from .grok_adapter import GrokAdapter
from .jimeng2api_adapter import Jimeng2APIAdapter
from .openai_adapter import OpenAIAdapter
from .z_image_adapter import ZImageAdapter

__all__ = [
    "GeminiAdapter",
    "GeminiOpenAIAdapter",
    "OpenAIAdapter",
    "ZImageAdapter",
    "Jimeng2APIAdapter",
    "GrokAdapter",
]
