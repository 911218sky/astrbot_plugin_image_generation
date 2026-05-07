from __future__ import annotations

from astrbot.api import logger

from ..adapter import (
    GeminiAdapter,
    GeminiOpenAIAdapter,
    GrokAdapter,
    Jimeng2APIAdapter,
    OpenAIAdapter,
    ZImageAdapter,
)
from .types import (
    AdapterConfig,
    AdapterType,
    GenerationRequest,
    GenerationResult,
    ImageData,
)
from .utils import convert_images_batch


class ImageGenerator:
    """適配器編排器，負責分發生圖請求。"""

    def __init__(self, adapter_config: AdapterConfig):
        self.adapter_config = adapter_config
        self.adapter = self._create_adapter(adapter_config)

    def _create_adapter(self, config: AdapterConfig):
        """根據配置建立對應的適配器。"""
        adapter_map: dict[AdapterType, type] = {
            AdapterType.GEMINI: GeminiAdapter,
            AdapterType.GEMINI_OPENAI: GeminiOpenAIAdapter,
            AdapterType.OPENAI: OpenAIAdapter,
            AdapterType.Z_IMAGE: ZImageAdapter,
            AdapterType.JIMENG2API: Jimeng2APIAdapter,
            AdapterType.GROK: GrokAdapter,
        }

        adapter_cls = adapter_map.get(config.type)
        if not adapter_cls:
            raise ValueError(f"不支援的適配器型別: {config.type}")
        return adapter_cls(config)

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """執行生圖邏輯。"""
        if not self.adapter:
            return GenerationResult(images=None, error="適配器未初始化")

        # 先將參考圖批次轉換成相容格式，再呼叫下游適配器
        converted_images: list[ImageData] = []
        if request.images:
            converted_images = await convert_images_batch(request.images)

        patched_request = GenerationRequest(
            prompt=request.prompt,
            images=converted_images,
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            task_id=request.task_id,
        )

        try:
            return await self.adapter.generate(patched_request)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[ImageGen] 生成失敗: {exc}", exc_info=True)
            return GenerationResult(images=None, error=str(exc))

    def update_model(self, model: str) -> None:
        """更新適配器使用的模型。"""
        if self.adapter:
            self.adapter.update_model(model)

    async def update_adapter(self, adapter_config: AdapterConfig) -> None:
        """更新適配器配置並重新建立適配器。

        注意: 此方法會關閉舊適配器以釋放資源。
        """
        if self.adapter:
            await self.adapter.close()
        self.adapter_config = adapter_config
        self.adapter = self._create_adapter(adapter_config)

    async def close(self) -> None:
        """關閉適配器。"""
        if self.adapter:
            await self.adapter.close()
