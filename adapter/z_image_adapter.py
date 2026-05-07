from __future__ import annotations

import base64
import time
from typing import Any

from astrbot.api import logger

from ..core.base_adapter import BaseImageAdapter
from ..core.constants import (
    GITEE_AI_DEFAULT_BASE_URL,
    RESOLUTION_1K_MAP,
    RESOLUTION_2K_MAP,
)
from ..core.types import GenerationRequest, GenerationResult, ImageCapability


class ZImageAdapter(BaseImageAdapter):
    """Gitee AI 圖像生成適配器 (z-image-turbo)。"""

    DEFAULT_BASE_URL = GITEE_AI_DEFAULT_BASE_URL

    def get_capabilities(self) -> ImageCapability:
        """取得適配器支援的功能。"""
        return self._get_configured_capabilities()

    # generate() 方法由基類提供，使用模板方法模式

    def _pre_generate(self, request: GenerationRequest) -> GenerationResult | None:
        """Z-Image 不支援參考圖，在生成前進行檢查。"""
        if request.images:
            return GenerationResult(
                images=None, error="Z-Image 適配器目前僅支援文生圖，請勿上傳圖片。"
            )

        prefix = self._get_log_prefix(request.task_id)
        logger.info(
            f"{prefix} 開始生成: prompt='{request.prompt[:50]}...', model='{self.model or 'z-image-turbo'}'"
        )
        return None

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """執行單次生圖請求。"""
        start_time = time.time()
        payload = self._build_payload(request)
        session = self._get_session()
        prefix = self._get_log_prefix(request.task_id)

        base = self.base_url or self.DEFAULT_BASE_URL
        url = f"{base.rstrip('/')}/v1/images/generations"

        logger.debug(f"{prefix} 請求 URL: {url}, Payload 欄位: {list(payload.keys())}")

        headers = {
            "Authorization": f"Bearer {self._get_current_api_key()}",
            "Content-Type": "application/json",
            "X-Failover-Enabled": "true",
        }

        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                proxy=self.proxy,
                timeout=self._get_timeout(),
            ) as resp:
                duration = time.time() - start_time
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        f"{prefix} API 錯誤 ({resp.status}, 耗時: {duration:.2f}s): {error_text}"
                    )
                    return None, f"API 錯誤 ({resp.status})"

                data = await resp.json()
                logger.info(f"{prefix} 生成成功 (耗時: {duration:.2f}s)")
                return await self._extract_images(data, request.task_id)
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{prefix} 請求異常 (耗時: {duration:.2f}s): {e}")
            return None, str(e)

    def _build_payload(self, request: GenerationRequest) -> dict:
        """構建請求載荷。"""
        prefix = self._get_log_prefix(request.task_id)

        size = "1024x1024"
        aspect_ratio = request.aspect_ratio or "1:1"
        if aspect_ratio == "自動":
            aspect_ratio = "1:1"

        if request.resolution in ("2K", "4K"):
            # 4K 暫時沿用 2K 的邏輯，因為 API 未提供 4K 對映
            size = RESOLUTION_2K_MAP.get(aspect_ratio, "2048x2048")
        else:
            size = RESOLUTION_1K_MAP.get(aspect_ratio, "1024x1024")

        logger.debug(
            f"{prefix} 引數: size={size}, aspect_ratio={aspect_ratio}, resolution={request.resolution or '1K'}"
        )

        payload: dict[str, Any] = {
            "model": self.model or "z-image-turbo",
            "prompt": request.prompt,
            "size": size,
            "num_inference_steps": 9,
        }

        return payload

    async def _extract_images(
        self, data: dict, task_id: str | None = None
    ) -> tuple[list[bytes] | None, str | None]:
        """從 API 響應中提取圖像資料。"""
        prefix = self._get_log_prefix(task_id)
        # Gitee 的響應格式通常遵循 OpenAI 規範
        if "data" not in data:
            return None, f"響應格式錯誤: {data}"

        images = []
        for item in data["data"]:
            if "b64_json" in item:
                images.append(base64.b64decode(item["b64_json"]))
            elif "url" in item:
                # 如果返回的是 URL，需要下載
                logger.debug(f"{prefix} 正在下載圖像: {item['url'][:50]}...")
                img_bytes = await self._download_image(item["url"], task_id)
                if img_bytes:
                    images.append(img_bytes)
            else:
                logger.warning(f"{prefix} 無法從響應項中提取圖像: {item}")

        if not images:
            return None, "未生成任何圖像"

        logger.info(f"{prefix} 成功提取 {len(images)} 張圖像")
        return images, None

    async def _download_image(
        self, url: str, task_id: str | None = None
    ) -> bytes | None:
        """下載圖像。"""
        session = self._get_session()
        prefix = self._get_log_prefix(task_id)
        try:
            async with session.get(
                url, proxy=self.proxy, timeout=self._get_download_timeout()
            ) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    logger.debug(f"{prefix} 圖像下載成功: {len(data)} bytes")
                    return data
                logger.error(f"{prefix} 下載圖像失敗 ({resp.status}): {url}")
        except Exception as e:
            logger.error(f"{prefix} 下載圖像異常: {e}")
        return None
