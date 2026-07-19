from __future__ import annotations

import base64
import time
from typing import Any

from astrbot.api import logger

from ..core.base_adapter import BaseImageAdapter
from ..core.provider_transport import (
    decode_provider_base64,
    download_provider_image,
    read_provider_json,
)
from ..core.types import GenerationRequest, ImageCapability


class Jimeng2APIAdapter(BaseImageAdapter):
    """Jimeng2API 圖像生成適配器。"""

    def get_capabilities(self) -> ImageCapability:
        """取得適配器支援的功能。"""
        return self._get_configured_capabilities()

    # generate() 方法由基類提供，使用模板方法模式

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """執行單次生圖請求。"""
        start_time = time.time()
        session = self._get_session()
        prefix = self._get_log_prefix(request.task_id)

        prompt_text = request.prompt
        if prompt_text is None:
            return None, "缺少提示詞"
        if not isinstance(prompt_text, str):
            logger.warning(f"{prefix} prompt 非字串型別: {type(prompt_text)}")
            prompt_text = str(prompt_text)

        base_url = self.base_url or "http://localhost:5100"
        headers = {
            "Authorization": f"Bearer {self._get_current_api_key()}",
        }

        try:
            if request.images:
                # 圖生圖：改為 JSON，images 作為 data URL（服務端宣告只接受 URL 或本地檔案）
                url = f"{base_url.rstrip('/')}/v1/images/compositions"
                headers["Content-Type"] = "application/json"

                images_as_urls: list[str] = []
                for img in request.images:
                    mime = img.mime_type or "image/jpeg"
                    b64 = base64.b64encode(img.data).decode("ascii")
                    images_as_urls.append(f"data:{mime};base64,{b64}")

                payload: dict[str, object] = {
                    "model": self.model or "jimeng-4.5",
                    "prompt": prompt_text,
                    "images": images_as_urls,
                }
                if request.aspect_ratio:
                    if request.aspect_ratio == "自動":
                        payload["intelligent_ratio"] = True
                    else:
                        payload["ratio"] = request.aspect_ratio
                if request.resolution:
                    payload["resolution"] = request.resolution.lower()

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
                            f"{prefix} Compositions 錯誤 ({resp.status}, 耗時: {duration:.2f}s): {error_text}"
                        )
                        return None, f"API 錯誤 ({resp.status})"

                    data_json = await read_provider_json(resp)
                    if data_json is None:
                        return None, "API 回應格式錯誤"
                    logger.debug(f"{prefix} Compositions 響應: {data_json}")
                    logger.info(f"{prefix} Compositions 成功 (耗時: {duration:.2f}s)")
                    return await self._extract_images(data_json, request.task_id)
            else:
                # 文生圖
                url = f"{base_url.rstrip('/')}/v1/images/generations"
                headers["Content-Type"] = "application/json"

                payload = {
                    "model": self.model or "jimeng-4.5",
                    "prompt": prompt_text,
                    "response_format": "url",  # 預設使用 url，然後下載
                }
                if request.aspect_ratio:
                    if request.aspect_ratio == "自動":
                        payload["intelligent_ratio"] = True
                    else:
                        payload["ratio"] = request.aspect_ratio
                if request.resolution:
                    payload["resolution"] = request.resolution.lower()

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
                            f"{prefix} Generations 錯誤 ({resp.status}, 耗時: {duration:.2f}s): {error_text}"
                        )
                        return None, f"API 錯誤 ({resp.status})"

                    data_json = await read_provider_json(resp)
                    if data_json is None:
                        return None, "API 回應格式錯誤"
                    logger.debug(f"{prefix} Generations 響應: {data_json}")
                    logger.info(f"{prefix} Generations 成功 (耗時: {duration:.2f}s)")
                    return await self._extract_images(data_json, request.task_id)

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{prefix} 請求異常 (耗時: {duration:.2f}s): {e}")
            return None, str(e)

    async def _extract_images(
        self, response: dict, task_id: str | None = None
    ) -> tuple[list[bytes] | None, str | None]:
        """從響應中提取圖片資料。"""
        prefix = self._get_log_prefix(task_id)
        if response is None:
            return None, "響應為空"
        if "data" not in response:
            return None, f"響應中未找到 data 欄位: {response}"

        data = response.get("data")
        if data is None:
            return None, "data 欄位為 None"

        images = []
        for item in data:
            if decoded := decode_provider_base64(item.get("b64_json")):
                images.append(decoded)
            elif url := item.get("url"):
                if downloaded := await download_provider_image(url):
                    images.append(downloaded)
                else:
                    logger.error(f"{prefix} 下載圖像失敗")

        if not images:
            return None, "未找到有效的圖片資料"

        return images, None

    async def receive_token(self) -> dict[str, Any]:
        """為所有 API Key 自動領取積分。"""
        results = {}
        if not self.api_keys:
            return {"error": "未配置 API Key"}

        base_url = self.base_url or "http://localhost:5100"
        url = f"{base_url.rstrip('/')}/token/receive"

        for i, key in enumerate(self.api_keys):
            headers = {
                "Authorization": f"Bearer {key}",
            }
            try:
                async with self._get_session().post(
                    url,
                    headers=headers,
                    proxy=self.proxy,
                    timeout=self._get_download_timeout(),
                ) as resp:
                    resp_json = await read_provider_json(resp)
                    if resp_json is None:
                        results[f"key_{i}"] = {"error": "API 回應格式錯誤"}
                        continue
                    status_code = resp.status
                    results[f"key_{i}"] = {"status": status_code, "data": resp_json}
                    if status_code == 200:
                        logger.info(
                            f"{self._get_log_prefix()} API Key (索引 {i}) 積分領取成功: {resp_json}"
                        )
                    else:
                        logger.warning(
                            f"{self._get_log_prefix()} API Key (索引 {i}) 積分領取失敗 ({status_code}): {resp_json}"
                        )
            except Exception as e:
                logger.error(
                    f"{self._get_log_prefix()} API Key (索引 {i}) 積分領取請求異常: {e}"
                )
                results[f"key_{i}"] = {"error": str(e)}

        return results
