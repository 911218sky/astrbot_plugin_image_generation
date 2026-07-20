from __future__ import annotations

import time
from typing import Any

import aiohttp

from astrbot.api import logger

from ..core.base_adapter import BaseImageAdapter
from ..core.provider_transport import (
    MAX_PROVIDER_JSON_BYTES,
    decode_provider_base64,
    download_provider_image,
    read_provider_json,
)
from ..core.types import GenerationRequest, ImageCapability


class OpenAIAdapter(BaseImageAdapter):
    """OpenAI image generation adapter for GPT Image models only."""

    def get_capabilities(self) -> ImageCapability:
        """取得適配器支援的功能。"""
        return self._get_configured_capabilities()

    def _is_gpt_image_model(self) -> bool:
        """This adapter only supports GPT Image models."""
        return True

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """執行單次生圖請求。"""
        start_time = time.time()
        prefix = self._get_log_prefix(request.task_id)

        use_edit = bool(request.images)
        session = self._get_session()
        base = self._get_api_base_url()
        headers = {"Authorization": f"Bearer {self._get_current_api_key()}"}

        if use_edit:
            url = f"{base}/images/edits"
            form = aiohttp.FormData()
            form.add_field("model", self.model or "gpt-image-1")
            form.add_field("prompt", request.prompt)
            form.add_field("n", "1")
            if size := self._map_aspect_ratio_to_size(request.aspect_ratio):
                form.add_field("size", size)
            for img in request.images:
                form.add_field(
                    "image[]",
                    img.data,
                    content_type=img.mime_type,
                    filename="image",
                )
            kwargs: dict = {"data": form}
        else:
            url = f"{base}/images/generations"
            headers["Content-Type"] = "application/json"
            kwargs = {"json": self._build_payload(request)}

        try:
            async with session.post(
                url,
                headers=headers,
                proxy=self.proxy,
                timeout=self._get_timeout(),
                **kwargs,
            ) as resp:
                duration = time.time() - start_time
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        f"{prefix} API 錯誤 ({resp.status}, 耗時: {duration:.2f}s): {error_text}"
                    )
                    return None, f"API 錯誤 ({resp.status})"
                data, error = await self._read_json_response(
                    resp, prefix=prefix, duration=duration
                )
                if error:
                    return None, error
                logger.info(f"{prefix} 生成成功 (耗時: {duration:.2f}s)")
                return await self._extract_images(data)
        except TimeoutError:
            duration = time.time() - start_time
            logger.error(f"{prefix} 請求逾時 (耗時: {duration:.2f}s)")
            return None, "請求逾時，請稍後重試"
        except aiohttp.ClientError as e:
            duration = time.time() - start_time
            error_message = str(e).strip() or e.__class__.__name__
            logger.error(f"{prefix} 請求異常 (耗時: {duration:.2f}s): {error_message}")
            return None, error_message
        except Exception as e:
            duration = time.time() - start_time
            error_message = str(e).strip() or e.__class__.__name__
            logger.error(f"{prefix} 請求異常 (耗時: {duration:.2f}s): {error_message}")
            return None, error_message

    async def _read_json_response(
        self,
        resp: aiohttp.ClientResponse,
        *,
        prefix: str,
        duration: float,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Read JSON response with tolerant handling for non-standard content types."""
        data = await read_provider_json(resp)
        if data is not None:
            return data, None
        logger.error(
            f"{prefix} API 回應不是有效 JSON 或超過 "
            f"{MAX_PROVIDER_JSON_BYTES // (1024 * 1024)} MiB (耗時: {duration:.2f}s)"
        )
        return None, (
            "API 回應格式錯誤，無法解析生成結果"
            f"（回應上限 {MAX_PROVIDER_JSON_BYTES // (1024 * 1024)} MiB）"
        )

    def _get_api_base_url(self) -> str:
        base = self.base_url or "https://api.openai.com"
        return base if base.endswith("/v1") else f"{base}/v1"

    def _build_payload(self, request: GenerationRequest) -> dict:
        """構建請求載荷。"""
        model = self.model or "gpt-image-1"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "n": 1,
            "response_format": "b64_json",
        }

        if model.lower().startswith("gpt-image-"):
            payload["quality"] = "auto"

        if size := self._map_aspect_ratio_to_size(request.aspect_ratio):
            payload["size"] = size
        return payload

    def _map_aspect_ratio_to_size(self, aspect_ratio: str | None) -> str | None:
        """Map aspect ratio to a GPT Image size."""
        if not aspect_ratio or aspect_ratio == "自動":
            return "auto"

        mapping = {
            "1:1": "1024x1024",
            "3:2": "1536x1024",
            "16:9": "1536x1024",
            "4:3": "1536x1024",
            "5:4": "1536x1024",
            "21:9": "1536x1024",
            "2:3": "1024x1536",
            "3:4": "1024x1536",
            "9:16": "1024x1536",
            "4:5": "1024x1536",
        }
        return mapping.get(aspect_ratio)

    async def _extract_images(
        self, response: dict
    ) -> tuple[list[bytes] | None, str | None]:
        """從響應中提取圖片資料。"""
        if "data" not in response:
            return None, "響應中未找到 data 欄位"

        images = []
        for item in response["data"]:
            if decoded := decode_provider_base64(item.get("b64_json")):
                images.append(decoded)
            elif url := item.get("url"):
                if downloaded := await download_provider_image(url):
                    images.append(downloaded)

        if not images:
            return None, "未找到有效的圖片資料"

        return images, None
