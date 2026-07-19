from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class AdapterType(str, enum.Enum):
    """支援的圖像生成適配器型別。"""

    GEMINI = "gemini"
    GEMINI_OPENAI = "gemini_openai"
    OPENAI = "openai"
    Z_IMAGE = "z_image_gitee"
    JIMENG2API = "jimeng2api"
    GROK = "grok"


class ImageCapability(enum.Flag):
    """圖像生成適配器支援的功能。"""

    NONE = 0
    TEXT_TO_IMAGE = enum.auto()  # 文生圖
    IMAGE_TO_IMAGE = enum.auto()  # 圖生圖
    RESOLUTION = enum.auto()  # 指定解析度
    ASPECT_RATIO = enum.auto()  # 指定寬高比


@dataclass
class AdapterMetadata:
    """關於適配器能力的後設資料。"""

    name: str
    capabilities: ImageCapability = ImageCapability.TEXT_TO_IMAGE


@dataclass
class AdapterConfig:
    """構造適配器所需的配置。"""

    type: AdapterType = AdapterType.GEMINI
    name: str = ""  # 供應商展示名稱
    base_url: str | None = None
    api_keys: list[str] = field(default_factory=list)
    model: str = ""
    available_models: list[str] = field(default_factory=list)
    proxy: str | None = None
    timeout: int = 180
    max_retry_attempts: int = 5
    safety_settings: str | None = None
    capability_options: dict[str, bool] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)  # 適配器特有配置


@dataclass
class ImageData:
    """帶有 MIME 型別的圖像二進位制資料。"""

    data: bytes
    mime_type: str


@dataclass
class GenerationRequest:
    """使用者生圖請求。"""

    prompt: str
    images: list[ImageData] = field(default_factory=list)
    aspect_ratio: str | None = None
    resolution: str | None = None
    task_id: str | None = None
    count: int = 1


@dataclass
class GenerationResult:
    """生圖嘗試的結果。"""

    images: list[bytes] | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class GenerationProgress:
    completed: int
    total: int
    succeeded: int
    failed: int
    elapsed: float
    last_error: str | None = None
