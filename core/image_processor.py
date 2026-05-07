"""
圖片處理模組 - 下載、提取、快取管理
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.core.utils.io import download_image_by_url

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent


class ImageProcessor:
    """圖片處理器 - 負責圖片下載、提取和快取管理。"""

    def __init__(self, cache_dir: str, max_image_size_mb: int, max_cache_count: int):
        self._cache_dir = cache_dir
        self._max_image_size_mb = max_image_size_mb
        self._max_cache_count = max_cache_count
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """確保快取目錄存在。"""
        os.makedirs(self._cache_dir, exist_ok=True)

    def update_settings(
        self, max_image_size_mb: int | None = None, max_cache_count: int | None = None
    ) -> None:
        """更新設定。"""
        if max_image_size_mb is not None:
            self._max_image_size_mb = max_image_size_mb
        if max_cache_count is not None:
            self._max_cache_count = max_cache_count

    @property
    def cache_dir(self) -> str:
        """取得快取目錄路徑。"""
        return self._cache_dir

    async def download_image(self, url: str) -> tuple[bytes, str] | None:
        """下載圖片並返回二進位制資料和 MIME 型別。"""
        try:
            data: bytes | None = None
            if os.path.exists(url) and os.path.isfile(url):
                with open(url, "rb") as f:
                    data = f.read()
            else:
                # 使用插件快取目錄
                file_name = f"ref_{hashlib.md5(url.encode()).hexdigest()[:10]}"
                path = os.path.join(self._cache_dir, file_name)
                path = await download_image_by_url(url, path=path)
                if path:
                    with open(path, "rb") as f:
                        data = f.read()

            if not data:
                return None

            if len(data) > self._max_image_size_mb * 1024 * 1024:
                logger.warning(
                    f"[ImageGen] 圖片超過大小限制 ({self._max_image_size_mb}MB)"
                )
                return None

            mime = self._detect_mime_type(data)
            return data, mime
        except Exception as exc:
            logger.error(f"[ImageGen] 取得圖片失敗 (URL/Path: {url}): {exc}")
        return None

    def _detect_mime_type(self, data: bytes) -> str:
        """檢測圖片 MIME 型別。"""
        if data.startswith(b"\xff\xd8"):
            return "image/jpeg"
        elif data.startswith(b"GIF"):
            return "image/gif"
        elif data.startswith(b"RIFF") and b"WEBP" in data[:16]:
            return "image/webp"
        return "image/png"

    async def get_avatar(self, user_id: str) -> bytes | None:
        """取得使用者頭像。"""
        url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        try:
            file_name = f"avatar_{user_id}.jpg"
            path = os.path.join(self._cache_dir, file_name)
            path = await download_image_by_url(url, path=path)
            if path:
                with open(path, "rb") as f:
                    return f.read()
        except Exception as e:
            logger.debug(f"[ImageGen] 取得頭像失敗 (user_id={user_id}): {e}")
        return None

    async def fetch_images_from_event(
        self, event: AstrMessageEvent
    ) -> list[tuple[bytes, str]]:
        """從訊息事件中提取圖片（包括直接傳送的圖片、引用訊息中的圖片、被@使用者的頭像）。"""
        images_data: list[tuple[bytes, str]] = []

        if not event.message_obj or not event.message_obj.message:
            return images_data

        # 預掃描：記錄引用訊息的傳送者以及各個 @ 出現次數，用於過濾自動 @
        reply_sender_id = None
        at_counts: dict[str, int] = {}

        for component in event.message_obj.message:
            if isinstance(component, Comp.Reply):
                if hasattr(component, "sender_id") and component.sender_id:
                    reply_sender_id = str(component.sender_id)
            elif isinstance(component, Comp.At):
                if hasattr(component, "qq") and component.qq != "all":
                    uid = str(component.qq)
                    at_counts[uid] = at_counts.get(uid, 0) + 1

        for component in event.message_obj.message:
            try:
                if isinstance(component, Comp.Image):
                    # 處理直接傳送的圖片
                    url = component.url or component.file
                    if url and (data := await self.download_image(url)):
                        images_data.append(data)
                elif isinstance(component, Comp.Reply):
                    # 處理引用訊息中的圖片
                    if component.chain:
                        for sub_comp in component.chain:
                            if isinstance(sub_comp, Comp.Image):
                                url = sub_comp.url or sub_comp.file
                                if url and (data := await self.download_image(url)):
                                    images_data.append(data)
                elif isinstance(component, Comp.At):
                    # 處理 @ 使用者的頭像
                    if hasattr(component, "qq") and component.qq != "all":
                        uid = str(component.qq)
                        # 引用訊息帶來的單次自動 @ 預設忽略頭像，除非使用者再次顯式 @
                        if reply_sender_id and uid == reply_sender_id:
                            if at_counts.get(uid, 0) == 1:
                                continue
                        self_id = str(event.get_self_id()).strip()
                        # 機器人單次被 @ 多為觸發字首，預設不取機器人頭像
                        if self_id and uid == self_id and at_counts.get(uid, 0) == 1:
                            continue
                        if avatar_data := await self.get_avatar(uid):
                            images_data.append((avatar_data, "image/jpeg"))
            except Exception as e:
                logger.error(f"[ImageGen] 提取訊息元件圖片失敗: {e}")
                continue
        return images_data

    async def cleanup_cache(self) -> None:
        """執行快取清理。"""
        if not os.path.exists(self._cache_dir):
            return

        files = []
        for f in os.listdir(self._cache_dir):
            path = os.path.join(self._cache_dir, f)
            if os.path.isfile(path):
                files.append((path, os.path.getmtime(path)))

        # 按修改時間排序（舊的在前）
        files.sort(key=lambda x: x[1])

        # 按數量清理
        if len(files) > self._max_cache_count:
            to_delete = files[: len(files) - self._max_cache_count]
            deleted_count = 0
            for path, _ in to_delete:
                try:
                    os.remove(path)
                    deleted_count += 1
                except OSError as e:
                    logger.debug(f"[ImageGen] 刪除快取檔案失敗: {path} - {e}")
            logger.info(
                f"[ImageGen] 已清理 {deleted_count}/{len(to_delete)} 箇舊快取檔案 (按數量)"
            )

    def save_generated_image(self, task_id: str, img_bytes: bytes) -> str | None:
        """儲存生成的圖片到快取目錄，返回檔案路徑。"""
        try:
            import time

            file_name = f"gen_{task_id}_{int(time.time())}_{hashlib.md5(img_bytes).hexdigest()[:6]}.png"
            file_path = os.path.join(self._cache_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(img_bytes)
            return file_path
        except Exception as exc:
            logger.error(f"[ImageGen] 儲存圖片失敗: {exc}")
            return None
