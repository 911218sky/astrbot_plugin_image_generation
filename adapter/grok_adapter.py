from __future__ import annotations

import base64
import time
from typing import Any

from astrbot.api import logger

from ..core.base_adapter import BaseImageAdapter
from ..core.types import GenerationRequest, ImageCapability


class GrokAdapter(BaseImageAdapter):
    """Grok（xAI）圖像生成適配器。"""

    def get_capabilities(self) -> ImageCapability:
        """取得適配器支援的功能。"""
        return self._get_configured_capabilities()

    # generate() 方法由基類提供，使用模板方法模式

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """執行單次生圖請求。"""
        start_time = time.time()
        prefix = self._get_log_prefix(request.task_id)

        payload = self._build_payload(request)
        session = self._get_session()

        if request.images:
            end_point = "/images/edits"
        else:
            end_point = "/images/generations"

        if not self.base_url:
            url = f"https://api.x.ai/v1{end_point}"
        else:
            # 考慮到 main.py 會清理掉 /v1，這裡統一加上
            url = f"{self.base_url.rstrip('/')}/v1{end_point}"

        headers = {
            "Authorization": f"Bearer {self._get_current_api_key()}",
            "Content-Type": "application/json",
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
                return await self._extract_images(data)
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{prefix} 請求異常 (耗時: {duration:.2f}s): {e}")
            return None, str(e)

    def _build_payload(self, request: GenerationRequest) -> dict:
        """構建請求載荷。"""

        accept_ratio = [
            "auto",
            "1:1",
            "16:9",
            "9:16",
            "4:3",
            "3:4",
            "3:2",
            "2:3",
            "1:2",
            "2:1",
            "19.5:9",
            "9:19.5",
            "20:9",
            "9:20",
        ]
        accept_resolution = ["1k", "2k"]

        ratio = "auto"
        if request.aspect_ratio in accept_ratio:
            ratio = request.aspect_ratio
        if request.aspect_ratio == "自動":
            ratio = "auto"

        resolution = "2k"
        if request.resolution.lower() in accept_resolution:
            resolution = request.resolution.lower()

        images_ref = []
        for image in request.images:
            b64_data = base64.b64encode(image.data).decode("utf-8")
            images_ref.append(
                {
                    "type": "image_url",
                    "url": f"data:{image.mime_type};base64,{b64_data}",
                }
            )

        payload: dict[str, Any] = {
            "model": self.model or "grok-imagine-image",
            "prompt": request.prompt,
            "aspect_ratio": ratio,
            "resolution": resolution,
            "response_format": "b64_json",
        }

        if len(images_ref) > 0:
            payload.update({"images": images_ref})

        return payload

    async def _extract_images(
        self, response: dict
    ) -> tuple[list[bytes] | None, str | None]:
        """從響應中提取圖片資料。"""
        if "data" not in response:
            return None, "響應中未找到 data 欄位"

        images = []
        for item in response["data"]:
            if "b64_json" in item:
                images.append(base64.b64decode(item["b64_json"]))
            elif "url" in item:
                # 如果返回的是 URL，需要下載（雖然我們請求的是 b64_json）
                async with self._get_session().get(
                    item["url"], proxy=self.proxy, timeout=self._get_download_timeout()
                ) as resp:
                    if resp.status == 200:
                        images.append(await resp.read())

        if not images:
            return None, "未找到有效的圖片資料"

        return images, None
