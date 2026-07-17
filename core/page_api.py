from __future__ import annotations

import base64
import time
from typing import Any

from quart import request

from astrbot.api import logger

from . import page_images

PLUGIN_NAME = "astrbot_plugin_image_generation_911218sky"
PAGE_API_PREFIX = f"/{PLUGIN_NAME}/page"


class PluginPageApi:
    IMAGE_SUFFIXES = page_images.IMAGE_SUFFIXES
    DEFAULT_IMAGE_PAGE_SIZE = 12
    MAX_IMAGE_PAGE_SIZE = 48
    ORIGINAL_DATA_MAX_MB = 24

    def __init__(self, plugin) -> None:
        self.plugin = plugin
        self._images = page_images.PageImageCatalog(
            plugin.cache_dir, plugin.image_metadata_store
        )

    def register_routes(self) -> None:
        register = self.plugin.context.register_web_api
        routes = (
            ("overview", self.get_overview),
            ("image", self.get_image),
            ("images", self.get_images),
        )
        for suffix, handler in routes:
            path = f"{PAGE_API_PREFIX}/{suffix}"
            register(path, handler, ["GET"], f"ImageGen Page {suffix}")

    async def get_overview(self):
        try:
            return self._ok(self._build_snapshot())
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[ImageGen] Page API 取得概覽失敗: {exc}", exc_info=True)
            return self._error(str(exc))

    async def get_image(self):
        try:
            file_name = str(request.args.get("name", "")).strip()
            snapshot = self._images.inspect(file_name)
            if snapshot is None:
                return self._error("圖片不存在或檔名無效")

            max_bytes = self._original_data_max_bytes()
            if snapshot.size > max_bytes:
                return self._error(
                    "圖片檔案過大，無法透過插件頁直接預覽或下載。"
                    f"目前上限 {self._format_size(max_bytes)}。"
                )
            payload = self._images.read_original(file_name, max_bytes)
            if payload is None:
                return self._error("圖片不存在或檔名無效")
            encoded = base64.b64encode(payload.data).decode("ascii")
            image = payload.snapshot
            return self._ok(
                {
                    "name": image.name,
                    "mime_type": payload.mime_type,
                    "size": image.size,
                    "size_label": self._format_size(image.size),
                    "modified_at": self._format_timestamp(image.modified_at),
                    "data_url": f"data:{payload.mime_type};base64,{encoded}",
                    "metadata": page_images.format_metadata(image.metadata, self),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[ImageGen] Page API 取得圖片失敗: {exc}", exc_info=True)
            return self._error(str(exc))

    async def get_images(self):
        try:
            page = self._positive_int(
                request.args.get("page"), default=1, maximum=1_000_000
            )
            page_size = self._positive_int(
                request.args.get("page_size"),
                default=self.DEFAULT_IMAGE_PAGE_SIZE,
                maximum=self.MAX_IMAGE_PAGE_SIZE,
            )
            image_page = self._images.page(
                page, page_size, self._original_data_max_bytes()
            )
            return self._ok(page_images.format_page(image_page, self))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[ImageGen] Page API 取得圖片列表失敗: {exc}", exc_info=True)
            return self._error(str(exc))

    def _build_snapshot(self) -> dict[str, Any]:
        now_ts = time.time()
        active_tasks = [
            self._normalize_task(task_id, raw, now_ts)
            | {"finished_at": None, "duration": None}
            for task_id, raw in self.plugin._active_generation_tasks.items()
        ]
        active_tasks.sort(key=lambda item: item.get("started_at_ts") or 0)
        recent_tasks = self._collect_recent_tasks(now_ts)
        cache_file_count = self._images.count_images()
        running_count = sum(1 for task in active_tasks if task["status"] == "running")
        queued_count = sum(1 for task in active_tasks if task["status"] == "queued")
        cfg = self.plugin.config_manager
        adapter = cfg.adapter_config
        usage_settings = cfg.usage_settings
        task_manager = self.plugin.task_manager

        summary = {
            "initialized": bool(self.plugin.generator and adapter),
            "provider": adapter.name if adapter else "",
            "model": adapter.model if adapter else "",
            "model_full": f"{adapter.name}/{adapter.model}" if adapter else "",
            "max_concurrent": cfg.max_concurrent_tasks,
            "active_count": len(active_tasks),
            "running_count": running_count,
            "queued_count": queued_count,
            "recent_count": len(recent_tasks),
            "cache_file_count": cache_file_count,
            "background_tasks": len(task_manager.background_tasks),
            "loop_tasks": len(getattr(task_manager, "_loop_tasks", {})),
            "daily_tasks": len(getattr(task_manager, "_daily_tasks", {})),
            "daily_limit_enabled": self.plugin.usage_manager.is_daily_limit_enabled(),
            "daily_limit": self.plugin.usage_manager.get_daily_limit(),
            "rate_limit_seconds": usage_settings.rate_limit_seconds,
            "generated_at": self._format_timestamp(now_ts),
        }

        providers = []
        if adapter:
            available = adapter.available_models or []
            providers = [
                {
                    "name": model.split("/", 1)[0] if "/" in model else adapter.name,
                    "model": model.split("/", 1)[1] if "/" in model else model,
                    "full": model,
                    "current": model == f"{adapter.name}/{adapter.model}",
                }
                for model in available
            ]

        settings = {
            "default_aspect_ratio": cfg.default_aspect_ratio,
            "default_resolution": cfg.default_resolution,
            "show_generation_info": cfg.show_generation_info,
            "show_model_info": cfg.show_model_info,
            "cache_dir": str(self.plugin.cache_dir),
            "max_cache_count": cfg.cache_settings.max_cache_count,
            "cleanup_interval_hours": cfg.cache_settings.cleanup_interval_hours,
            "max_image_size_mb": usage_settings.max_image_size_mb,
            "blocked_sessions": len(usage_settings.umo_blacklist),
            "audit_whitelist": len(cfg.safety_audit_settings.umo_whitelist),
        }

        return {
            "summary": summary,
            "tasks": active_tasks,
            "recent_tasks": recent_tasks,
            "recent_images": [],
            "providers": providers,
            "settings": settings,
        }

    def _collect_recent_tasks(self, now_ts: float) -> list[dict[str, Any]]:
        tasks = []
        for raw in self.plugin._recent_generation_tasks[:50]:
            task_id = str(raw.get("task_id") or "")
            task = self._normalize_task(task_id, raw, now_ts)
            task["finished_at"] = self._format_timestamp(raw.get("finished_at"))
            task["finished_at_ts"] = self._as_float(raw.get("finished_at"))
            task["duration"] = self._as_float(raw.get("duration"))
            task["image_count"] = int(raw.get("image_count") or 0)
            task["files"] = list(raw.get("files") or [])
            task["error"] = str(raw.get("error") or "")
            tasks.append(task)
        return tasks

    def _normalize_task(
        self, task_id: str, raw: dict[str, Any], now_ts: float
    ) -> dict[str, Any]:
        started_at = self._as_float(raw.get("started_at"))
        running_at = self._as_float(raw.get("running_at"))
        status = str(raw.get("status") or "queued")
        prompt = str(raw.get("prompt_preview") or "")
        user = str(raw.get("user") or "")
        return {
            "task_id": task_id,
            "status": status,
            "status_label": self._status_label(status),
            "mode": str(raw.get("mode") or "文生圖"),
            "prompt_preview": prompt,
            "user": user,
            "user_label": self._compact_user(user),
            "aspect_ratio": str(raw.get("aspect_ratio") or "自動"),
            "resolution": str(raw.get("resolution") or "1K"),
            "reference_count": int(raw.get("reference_count") or 0),
            "started_at": self._format_timestamp(started_at),
            "started_at_ts": started_at,
            "running_at": self._format_timestamp(running_at),
            "running_at_ts": running_at,
            "elapsed": max(0.0, now_ts - started_at) if started_at else 0.0,
        }

    def _original_data_max_bytes(self) -> int:
        config = getattr(self.plugin, "config_manager", None)
        usage_settings = getattr(config, "usage_settings", None)
        configured_mb = getattr(
            usage_settings,
            "max_image_size_mb",
            self.ORIGINAL_DATA_MAX_MB,
        )
        try:
            max_mb = min(float(configured_mb or 0), float(self.ORIGINAL_DATA_MAX_MB))
        except (TypeError, ValueError):
            max_mb = float(self.ORIGINAL_DATA_MAX_MB)
        return int(max(1.0, max_mb) * 1024 * 1024)

    @staticmethod
    def _status_label(status: str) -> str:
        return {
            "queued": "正在排隊",
            "running": "生成中",
            "success": "已完成",
            "failed": "失敗",
            "blocked": "審核阻擋",
            "generated": "已生成",
        }.get(status, status or "未知")

    @staticmethod
    def _compact_user(user: str) -> str:
        if not user:
            return "-"
        if len(user) <= 32:
            return user
        return f"{user[:16]}...{user[-10:]}"

    @staticmethod
    def _format_timestamp(value: Any) -> str:
        ts = PluginPageApi._as_float(value)
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else ""

    @staticmethod
    def _as_float(value: Any) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _format_size(size: int) -> str:
        units = ("B", "KB", "MB", "GB")
        value = float(size)
        for unit in units:
            if value < 1024 or unit == units[-1]:
                return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
            value /= 1024
        return f"{size} B"

    @staticmethod
    def _positive_int(value: Any, *, default: int, maximum: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        return min(max(1, number), maximum)

    @staticmethod
    def _ok(data: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok", "data": data}

    @staticmethod
    def _error(message: str) -> dict[str, Any]:
        return {"status": "error", "message": message}
