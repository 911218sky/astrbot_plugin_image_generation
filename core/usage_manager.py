"""
使用者使用資料管理模組
"""

from __future__ import annotations

import datetime
import json
import os
import time
from typing import TYPE_CHECKING

from astrbot.api import logger

from .constants import USAGE_DATA_RETENTION_DAYS

if TYPE_CHECKING:
    from .config_manager import UsageSettings


class UsageManager:
    """使用者使用資料管理器。"""

    def __init__(self, data_dir: str, settings: UsageSettings):
        self._data_dir = data_dir
        self._settings = settings
        self._usage_file = os.path.join(data_dir, "usage.json")
        self._usage_data: dict[str, dict[str, int]] = {}  # {date: {user_id: count}}
        self._user_request_timestamps: dict[str, float] = {}  # 用於頻率限制
        self._load_usage_data()

    def update_settings(self, settings: UsageSettings) -> None:
        """更新設定。"""
        self._settings = settings

    def _load_usage_data(self) -> None:
        """載入使用者使用資料。"""
        if os.path.exists(self._usage_file):
            try:
                with open(self._usage_file, encoding="utf-8") as f:
                    self._usage_data = json.load(f)

                # 清理舊資料，只保留最近 N 天（由 USAGE_DATA_RETENTION_DAYS 控制）
                today = datetime.date.today()
                keys_to_delete = []
                for date_str in self._usage_data:
                    try:
                        date_obj = datetime.date.fromisoformat(date_str)
                        if (today - date_obj).days > USAGE_DATA_RETENTION_DAYS:
                            keys_to_delete.append(date_str)
                    except ValueError:
                        keys_to_delete.append(date_str)

                if keys_to_delete:
                    for key in keys_to_delete:
                        del self._usage_data[key]
                    self._save_usage_data()
            except Exception as exc:
                logger.error(f"[ImageGen] 載入使用資料失敗: {exc}")
                self._usage_data = {}

    def _save_usage_data(self) -> None:
        """儲存使用者使用資料。"""
        try:
            os.makedirs(os.path.dirname(self._usage_file), exist_ok=True)
            with open(self._usage_file, "w", encoding="utf-8") as f:
                json.dump(self._usage_data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error(f"[ImageGen] 儲存使用資料失敗: {exc}")

    def is_session_blocked(self, user_id: str) -> bool:
        """Check whether the current session UMO is blocked."""
        uid = user_id.strip()
        if not uid:
            return False
        return uid in self._settings.umo_blacklist

    def check_rate_limit(self, user_id: str) -> bool | str:
        """檢查使用者請求頻率限制和每日限制。

        返回:
            - True: 檢查透過
            - str: 錯誤訊息
        """
        # 1. 檢查頻率限制
        if self.is_session_blocked(user_id):
            return self._settings.blacklist_block_message

        if self._settings.rate_limit_seconds > 0:
            now = time.time()
            last_ts = self._user_request_timestamps.get(user_id, 0)
            if now - last_ts < self._settings.rate_limit_seconds:
                remaining = int(self._settings.rate_limit_seconds - (now - last_ts))
                return f"❌ 請求過於頻繁，請在 {remaining} 秒後再試"
            self._user_request_timestamps[user_id] = now

        # 2. 檢查每日限制
        if self._settings.enable_daily_limit:
            today = datetime.date.today().isoformat()
            if today not in self._usage_data:
                self._usage_data[today] = {}

            count = self._usage_data[today].get(user_id, 0)
            if count >= self._settings.daily_limit_count:
                return f"❌ 您今日的生圖額度已用完 ({self._settings.daily_limit_count}次)，請明天再試"

        return True

    def record_usage(self, user_id: str) -> None:
        """記錄使用者使用次數。"""
        if self._settings.enable_daily_limit:
            today = datetime.date.today().isoformat()
            if today not in self._usage_data:
                self._usage_data[today] = {}
            self._usage_data[today][user_id] = (
                self._usage_data[today].get(user_id, 0) + 1
            )
            self._save_usage_data()

    def get_usage_count(self, user_id: str) -> int:
        """取得使用者今日使用次數。"""
        today = datetime.date.today().isoformat()
        return self._usage_data.get(today, {}).get(user_id, 0)

    def get_daily_limit(self) -> int:
        """取得每日限制次數。"""
        return self._settings.daily_limit_count

    def is_daily_limit_enabled(self) -> bool:
        """是否啟用每日限制。"""
        return self._settings.enable_daily_limit
