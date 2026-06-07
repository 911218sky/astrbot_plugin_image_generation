"""AstrBot 官方插件頁 API。

這層只負責把插件執行期狀態整理成前端可讀的快照；不直接觸碰 AstrBot 核心資料。
"""

from __future__ import annotations

import base64
import io
import mimetypes
import time
from pathlib import Path
from typing import Any

from quart import request

from astrbot.api import logger

PLUGIN_NAME = "astrbot_plugin_image_generation_911218sky"
PAGE_API_PREFIX = f"/{PLUGIN_NAME}/page"


class PluginPageApi:
    """Image Generation 官方插件頁 API。"""

    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    PREVIEW_LIMIT = 24
    ORIGINAL_DATA_MAX_MB = 24

    def __init__(self, plugin) -> None:
        self.plugin = plugin

    def register_routes(self) -> None:
        """註冊官方插件頁需要的 API。"""
        register = self.plugin.context.register_web_api
        register(
            f"{PAGE_API_PREFIX}/overview",
            self.get_overview,
            ["GET"],
            "ImageGen Page overview",
        )
        register(
            f"{PAGE_API_PREFIX}/image",
            self.get_image,
            ["GET"],
            "ImageGen Page cached image",
        )

    async def get_overview(self):
        """回傳任務、模型、快取與限制摘要。"""
        try:
            return self._ok(self._build_snapshot())
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[ImageGen] Page API 取得概覽失敗: {exc}", exc_info=True)
            return self._error(str(exc))

    async def get_image(self):
        """回傳快取原圖 data URL，供頁面預覽與下載使用。"""
        try:
            file_name = str(request.args.get("name", "")).strip()
            path = self._resolve_cache_image(file_name)
            if path is None:
                return self._error("圖片不存在或檔名無效")

            stat = path.stat()
            max_bytes = self._original_data_max_bytes()
            if stat.st_size > max_bytes:
                return self._error(
                    "圖片檔案過大，無法透過插件頁直接預覽或下載。"
                    f"目前上限 {self._format_size(max_bytes)}。"
                )
            mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            metadata = self.plugin.image_metadata_store.get(path.name)
            return self._ok(
                {
                    "name": path.name,
                    "mime_type": mime_type,
                    "size": stat.st_size,
                    "size_label": self._format_size(stat.st_size),
                    "modified_at": self._format_timestamp(stat.st_mtime),
                    "data_url": f"data:{mime_type};base64,{encoded}",
                    "metadata": self._format_image_metadata(metadata),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[ImageGen] Page API 取得圖片失敗: {exc}", exc_info=True)
            return self._error(str(exc))

    def _build_snapshot(self) -> dict[str, Any]:
        now_ts = time.time()
        active_tasks = self._collect_active_tasks(now_ts)
        recent_tasks = self._collect_recent_tasks(now_ts)
        recent_images = self._collect_recent_images()
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
            "cache_file_count": len(recent_images),
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
            "recent_images": recent_images,
            "providers": providers,
            "settings": settings,
        }

    def _collect_active_tasks(self, now_ts: float) -> list[dict[str, Any]]:
        tasks = []
        for task_id, raw in self.plugin._active_generation_tasks.items():
            task = self._normalize_task(task_id, raw, now_ts)
            task["finished_at"] = None
            task["duration"] = None
            tasks.append(task)
        tasks.sort(key=lambda item: item.get("started_at_ts") or 0)
        return tasks

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

    def _collect_recent_images(self) -> list[dict[str, Any]]:
        cache_dir = Path(self.plugin.cache_dir)
        if not cache_dir.exists():
            return []

        metadata_by_name = self.plugin.image_metadata_store.get_all()
        image_rows: list[tuple[Path, dict[str, Any]]] = []
        for path in cache_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in self.IMAGE_SUFFIXES:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            metadata = metadata_by_name.get(path.name, {})
            image_rows.append(
                (
                    path,
                    {
                        "name": path.name,
                        "size": stat.st_size,
                        "size_label": self._format_size(stat.st_size),
                        "modified_at": self._format_timestamp(stat.st_mtime),
                        "modified_at_ts": stat.st_mtime,
                        "kind": "generated"
                        if path.name.startswith("gen_")
                        else "reference",
                        "metadata": self._format_image_metadata(metadata),
                    },
                )
            )
        image_rows.sort(key=lambda item: item[1]["modified_at_ts"], reverse=True)
        self.plugin.image_metadata_store.prune_missing(
            {path.name for path, _ in image_rows}
        )

        images = []
        for index, (path, image) in enumerate(image_rows[:60]):
            if (
                index < self.PREVIEW_LIMIT
                and image["size"] <= self._thumbnail_max_bytes()
            ):
                image["preview_data_url"] = self._make_preview_data_url(path)
            images.append(image)
        return images

    def _resolve_cache_image(self, file_name: str) -> Path | None:
        if not file_name or "/" in file_name or "\\" in file_name:
            return None
        if file_name != Path(file_name).name:
            return None
        cache_dir = Path(self.plugin.cache_dir).resolve()
        path = (cache_dir / file_name).resolve()
        if path.parent != cache_dir:
            return None
        if path.suffix.lower() not in self.IMAGE_SUFFIXES:
            return None
        if not path.is_file():
            return None
        return path

    def _format_image_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        status = str(metadata.get("status") or "")
        return {
            "known": bool(metadata),
            "task_id": str(metadata.get("task_id") or ""),
            "status": status,
            "status_label": self._status_label(status) if status else "",
            "prompt": str(metadata.get("prompt") or ""),
            "prompt_preview": str(metadata.get("prompt_preview") or ""),
            "model_full": str(metadata.get("model_full") or ""),
            "provider": str(metadata.get("provider") or ""),
            "model": str(metadata.get("model") or ""),
            "user": str(metadata.get("user") or ""),
            "user_label": self._compact_user(str(metadata.get("user") or "")),
            "mode": str(metadata.get("mode") or ""),
            "aspect_ratio": str(metadata.get("aspect_ratio") or ""),
            "resolution": str(metadata.get("resolution") or ""),
            "created_at": self._format_timestamp(metadata.get("created_at")),
            "created_at_ts": self._as_float(metadata.get("created_at")),
        }

    def _original_data_max_bytes(self) -> int:
        configured_mb = getattr(
            self.plugin.config_manager.usage_settings,
            "max_image_size_mb",
            self.ORIGINAL_DATA_MAX_MB,
        )
        try:
            max_mb = min(float(configured_mb or 0), float(self.ORIGINAL_DATA_MAX_MB))
        except (TypeError, ValueError):
            max_mb = float(self.ORIGINAL_DATA_MAX_MB)
        return int(max(1.0, max_mb) * 1024 * 1024)

    def _thumbnail_max_bytes(self) -> int:
        return self._original_data_max_bytes()

    @staticmethod
    def _make_preview_data_url(path: Path) -> str:
        try:
            from PIL import Image
        except ImportError:
            return ""

        try:
            with Image.open(path) as image:
                image.thumbnail((320, 220))
                if image.mode not in {"RGB", "L"}:
                    image = image.convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=72, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[ImageGen] 產生縮圖失敗: {path.name} - {exc}")
            return ""

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
        if not ts:
            return ""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

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
    def _ok(data: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok", "data": data}

    @staticmethod
    def _error(message: str) -> dict[str, Any]:
        return {"status": "error", "message": message}
