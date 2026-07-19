from __future__ import annotations

import base64
import re
import time
from typing import Any

import aiohttp

from astrbot.api import logger

from ..core.base_adapter import BaseImageAdapter
from ..core.constants import GEMINI_DEFAULT_BASE_URL
from ..core.provider_transport import (
    decode_provider_base64,
    download_provider_image,
    read_provider_json,
)
from ..core.types import GenerationRequest, ImageCapability


class GeminiOpenAIAdapter(BaseImageAdapter):
    """透過 OpenAI 相容的聊天補全接口進行 Gemini 圖像生成。"""

    DEFAULT_BASE_URL = GEMINI_DEFAULT_BASE_URL

    def get_capabilities(self) -> ImageCapability:
        """取得適配器支援的功能。"""
        return self._get_configured_capabilities()

    # generate() 方法由基類提供，使用模板方法模式

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """執行單次生圖請求。"""
        payload = self._build_payload(request)
        session = self._get_session()
        response, error = await self._make_request(session, payload, request.task_id)
        if response is None:
            return None, error or "API 請求失敗"

        images = await self._extract_images(response, request.task_id)
        if images:
            return images, None

        # 嘗試提取文字錯誤資訊
        if "choices" in response and response["choices"]:
            content = response["choices"][0].get("message", {}).get("content")
            if isinstance(content, str) and content.strip():
                return None, f"未生成圖片，API 返回文字: {content[:100]}"
        return None, "響應中未找到圖片 data"

    def _build_payload(self, request: GenerationRequest) -> dict:
        """構建請求載荷。"""
        message_content: list[dict] = [
            {"type": "text", "text": f"Generate an image: {request.prompt}"}
        ]

        for image in request.images:
            b64_data = base64.b64encode(image.data).decode("utf-8")
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image.mime_type};base64,{b64_data}"},
                }
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": message_content}],
            "modalities": ["image", "text"],
            "stream": False,
        }

        image_config: dict[str, Any] = {}
        generation_config: dict[str, Any] = {}

        if request.aspect_ratio and not request.images:
            image_config["aspectRatio"] = request.aspect_ratio
        if request.resolution:
            image_config["imageSize"] = request.resolution
        if image_config:
            generation_config["imageConfig"] = image_config
        if generation_config:
            payload["generationConfig"] = generation_config

        return payload

    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        payload: dict,
        task_id: str | None,
    ) -> tuple[dict | None, str | None]:
        """傳送 API 請求。"""
        start_time = time.time()
        url = f"{self.base_url or self.DEFAULT_BASE_URL}/v1/chat/completions"
        api_key = self._get_current_api_key()
        masked_key = self._get_masked_api_key()
        prefix = self._get_log_prefix(task_id)
        logger.debug(f"{prefix} 請求 -> {url}, key={masked_key}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self._get_timeout(),
                proxy=self.proxy,
            ) as response:
                duration = time.time() - start_time
                logger.debug(
                    f"{prefix} 狀態 -> {response.status} (耗時: {duration:.2f}s)"
                )
                if response.status != 200:
                    error_text = await response.text()
                    preview = (
                        error_text[:200] + "..."
                        if len(error_text) > 200
                        else error_text
                    )
                    logger.error(
                        f"{prefix} 錯誤 {response.status} (耗時: {duration:.2f}s): {preview}"
                    )
                    return None, f"API 錯誤 ({response.status})"
                data = await read_provider_json(response)
                return data, None if data is not None else "API 回應格式錯誤"
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{prefix} 請求異常 (耗時: {duration:.2f}s): {e}")
            return None, str(e).strip() or "API 請求失敗"

    async def _download_image_from_url(
        self, url: str, task_id: str | None = None
    ) -> bytes | None:
        """從 URL 下載圖像。"""
        prefix = self._get_log_prefix(task_id)
        data = await download_provider_image(url)
        if data is None:
            logger.error(f"{prefix} 下載圖像失敗")
        return data

    async def _extract_images(
        self, response_data: dict[str, Any], task_id: str | None = None
    ) -> list[bytes] | None:
        """從響應資料中提取圖像。"""
        images: list[bytes] = []
        prefix = self._get_log_prefix(task_id)

        # DALL-E 風格
        if isinstance(response_data.get("data"), list):
            for item in response_data["data"]:
                if not isinstance(item, dict):
                    continue
                if b64 := item.get("b64_json"):
                    if decoded := decode_provider_base64(b64):
                        images.append(decoded)
                    else:
                        logger.warning(f"{prefix} Base64 解碼失敗 (b64_json)")
                elif url := item.get("url"):
                    if url.startswith("http"):
                        if content := await self._download_image_from_url(url, task_id):
                            images.append(content)
                    else:
                        decoded = self._decode_image_url(url, task_id)
                        if decoded:
                            images.append(decoded)

        # 聊天補全風格
        if choices := response_data.get("choices"):
            message = (
                choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            )
            content = message.get("content")

            if isinstance(content, str):
                markdown_matches = re.findall(r"!\[.*?\]\((.*?)\)", content)
                for url in markdown_matches:
                    if url.startswith("http"):
                        if data := await self._download_image_from_url(url, task_id):
                            images.append(data)
                    else:
                        decoded = self._decode_image_url(url, task_id)
                        if decoded:
                            images.append(decoded)

                content_without_md = re.sub(r"!\[.*?\]\(.*?\)", "", content)
                pattern = re.compile(
                    r"data\s*:\s*image/([a-zA-Z0-9.+-]+)\s*;\s*base64\s*,\s*([-A-Za-z0-9+/=_\s]+)",
                    flags=re.IGNORECASE,
                )
                for _, b64_str in pattern.findall(content_without_md):
                    if decoded := decode_provider_base64(b64_str):
                        images.append(decoded)
                    else:
                        logger.warning(f"{prefix} Base64 解碼失敗 (inline)")

            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_url = part.get("image_url", {}).get("url")
                        if not image_url:
                            continue
                        if image_url.startswith("http"):
                            if data := await self._download_image_from_url(
                                image_url, task_id
                            ):
                                images.append(data)
                        else:
                            decoded = self._decode_image_url(image_url, task_id)
                            if decoded:
                                images.append(decoded)

            if message.get("images"):
                for img_item in message["images"]:
                    url = None
                    if isinstance(img_item, dict):
                        url = img_item.get("url") or img_item.get("image_url", {}).get(
                            "url"
                        )
                    elif isinstance(img_item, str):
                        url = img_item
                    if not url:
                        continue
                    if url.startswith("http"):
                        if data := await self._download_image_from_url(url, task_id):
                            images.append(data)
                    else:
                        decoded = self._decode_image_url(url, task_id)
                        if decoded:
                            images.append(decoded)

        return images or None

    def _decode_image_url(self, url: str, task_id: str | None = None) -> bytes | None:
        """解碼 Data URL 形式的圖像。"""
        if url.startswith("data:image/") and ";base64," in url:
            try:
                _, _, data_part = url.partition(";base64,")
                return decode_provider_base64(data_part)
            except (AttributeError, ValueError):
                prefix = self._get_log_prefix(task_id)
                logger.error(f"{prefix} Base64 解碼失敗")
        return None
