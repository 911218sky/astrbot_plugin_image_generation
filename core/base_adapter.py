from __future__ import annotations

import abc
import asyncio
import re

import aiohttp

from astrbot.api import logger

from .constants import DEFAULT_DOWNLOAD_TIMEOUT
from .types import AdapterConfig, GenerationRequest, GenerationResult, ImageCapability
from .utils import mask_sensitive


class BaseImageAdapter(abc.ABC):
    """圖像生成適配器基類。"""

    def __init__(self, config: AdapterConfig):
        self.config = config
        self.api_keys = config.api_keys or []
        self.current_key_index = 0
        self.base_url = (config.base_url or "").rstrip("/")
        self.model = config.model
        self.proxy = config.proxy
        self.timeout = config.timeout
        self.download_timeout = DEFAULT_DOWNLOAD_TIMEOUT
        self.max_retry_attempts = min(5, max(1, config.max_retry_attempts))
        self.safety_settings = config.safety_settings
        self._session: aiohttp.ClientSession | None = None

    @abc.abstractmethod
    def get_capabilities(self) -> ImageCapability:
        """取得適配器支援的功能。"""

    def _get_configured_capabilities(self) -> ImageCapability:
        """根據配置項構建適配器能力。"""
        capability_map: dict[str, ImageCapability] = {
            "text_to_image": ImageCapability.TEXT_TO_IMAGE,
            "image_to_image": ImageCapability.IMAGE_TO_IMAGE,
            "aspect_ratio": ImageCapability.ASPECT_RATIO,
            "resolution": ImageCapability.RESOLUTION,
        }

        result = ImageCapability.NONE
        for key, capability_flag in capability_map.items():
            if self.config.capability_options.get(key, False):
                result |= capability_flag
        return result

    async def close(self) -> None:
        """關閉底層的 HTTP 會話。"""

        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    def _get_session(self) -> aiohttp.ClientSession:
        """取得或建立 HTTP 會話。"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _get_current_api_key(self) -> str:
        """取得當前使用的 API Key。"""
        if not self.api_keys:
            return ""
        return self.api_keys[self.current_key_index % len(self.api_keys)]

    def _get_masked_api_key(self) -> str:
        """取得脫敏後的當前 API Key，用於日誌輸出。"""
        return mask_sensitive(self._get_current_api_key())

    def _get_log_prefix(self, task_id: str | None = None) -> str:
        """取得統一的日誌字首。"""
        adapter_name = self.__class__.__name__.replace("Adapter", "")
        prefix = f"[ImageGen] [{adapter_name}]"
        if task_id:
            prefix += f" [{task_id}]"
        return prefix

    def _get_timeout(self) -> aiohttp.ClientTimeout:
        """取得統一的請求超時配置。"""
        return aiohttp.ClientTimeout(total=self.timeout)

    def _get_download_timeout(self) -> aiohttp.ClientTimeout:
        """取得統一的下載超時配置。"""
        return aiohttp.ClientTimeout(total=self.download_timeout)

    def _rotate_api_key(self) -> None:
        """輪換 API Key。"""
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            logger.info(
                f"{self._get_log_prefix()} 輪換 API Key -> 索引 {self.current_key_index}"
            )

    def update_model(self, model: str) -> None:
        """更新使用的模型。"""
        self.model = model

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """帶重試邏輯的圖像生成模板方法。

        子類應重寫 `_generate_once()` 方法來實現具體的生成邏輯。
        如需在生成前進行預處理驗證，可重寫 `_pre_generate()` 方法。
        """
        if not self.api_keys:
            return GenerationResult(images=None, error="未配置 API Key")

        # 預處理檢查（子類可重寫）
        pre_result = self._pre_generate(request)
        if pre_result is not None:
            return pre_result

        prefix = self._get_log_prefix(request.task_id)
        last_error = "未配置 API Key"
        for attempt in range(1, self.max_retry_attempts + 1):
            images, err = await self._generate_once(request)
            if images is not None:
                if attempt > 1:
                    logger.info(
                        f"{prefix} 第 {attempt}/{self.max_retry_attempts} 次重試後生成成功"
                    )
                return GenerationResult(images=images, error=None)

            last_error = err or "生成失敗"
            logger.warning(
                f"{prefix} 第 {attempt}/{self.max_retry_attempts} 次嘗試失敗: "
                f"{last_error}"
            )
            if attempt < self.max_retry_attempts and self._is_retryable_error(last_error):
                self._rotate_api_key()
                logger.info(
                    f"{prefix} 已排程第 {attempt + 1}/{self.max_retry_attempts} 次重試"
                )
                # 輪換 Key 時進行指數退避
                if attempt % max(1, len(self.api_keys)) == 0:
                    backoff_seconds = min(2 ** (attempt // len(self.api_keys)), 10)
                    logger.info(f"{prefix} 重試前等待 {backoff_seconds} 秒")
                    await asyncio.sleep(backoff_seconds)

        logger.error(
            f"{prefix} 已達最大重試次數 {self.max_retry_attempts} 次，最終失敗: {last_error}"
        )
        return GenerationResult(images=None, error=last_error)

    @staticmethod
    def _is_retryable_error(error: str) -> bool:
        match = re.search(r"API 錯誤 \((\d{3})\)", error)
        if match:
            status = int(match.group(1))
            return status in {408, 409, 425, 429} or status >= 500
        return not error.startswith(("未配置 API Key", "提示詞未通過審核"))

    def _pre_generate(self, request: GenerationRequest) -> GenerationResult | None:
        """生成前的預處理檢查。

        子類可重寫此方法進行引數驗證。
        返回 None 表示透過檢查，返回 GenerationResult 表示提前返回錯誤。
        """
        return None

    @abc.abstractmethod
    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """執行單次生成請求。

        子類必須實現此方法。
        返回 (images, error) 元組，成功時 images 非空，失敗時 error 非空。
        """
