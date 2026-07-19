"""
AstrBot 圖像生成插件主模組

"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import datetime
import hashlib
import inspect
import json
import time
from collections.abc import Coroutine
from typing import Any, assert_never

import anyio

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.star.filter.custom_filter import CustomFilter
from astrbot.core.star.star_tools import StarTools

from .core.config_manager import ConfigManager
from .core.admission import (
    AdmissionController,
    AdmissionDenied,
    AdmissionLimits,
    AdmissionTicket,
)
from .core.base_adapter import BaseImageAdapter
from .core.generator import ImageGenerator
from .core.image_metadata_store import ImageMetadataStore
from .core.image_processor import ImageProcessor
from .core.llm_tool import ImageGenerationTool, adjust_tool_parameters
from .core.reference_collector import (
    CollectedReferences,
    ReferenceCollector,
    ReferenceRejected,
)
from .core.safety_auditor import SafetyAuditor
from .core.task_manager import TaskManager
from .core.types import (
    GenerationProgress,
    GenerationRequest,
    ImageCapability,
    ImageData,
)
from .core.usage_manager import UsageManager
from .core.utils import (
    extract_self_avatar_alias,
    mask_sensitive,
    normalize_batch_count,
    validate_aspect_ratio,
    validate_resolution,
)


class LegacyImageCommandFilter(CustomFilter):
    """Only match legacy commands when the original message used a wake prefix."""

    LEGACY_COMMANDS = {"img_model", "img_tasks", "preset"}

    def filter(self, event: AstrMessageEvent, cfg: AstrBotConfig) -> bool:
        raw_message = (
            getattr(getattr(event, "message_obj", None), "message_str", "") or ""
        ).strip()
        if not raw_message:
            raw_message = (event.get_message_str() or "").strip()

        wake_prefixes = []
        try:
            wake_prefixes = cfg.get("wake_prefix", []) or []
        except AttributeError:
            wake_prefixes = []

        for prefix in sorted(
            (prefix for prefix in wake_prefixes if isinstance(prefix, str) and prefix),
            key=len,
            reverse=True,
        ):
            if raw_message.startswith(prefix):
                command_text = raw_message[len(prefix) :].strip()
                command = command_text.split(maxsplit=1)[0] if command_text else ""
                return command in self.LEGACY_COMMANDS

        return False


class ImageGenerationPlugin(Star):
    """圖像生成插件主類"""

    LLM_TOOL_SYSTEM_PROMPT = """
# 圖像生成工具規則

當使用者希望最終產出是一張圖片時，優先呼叫 `generate_image` 工具，而不是只回覆文字。

當使用者要求以下內容時，應呼叫 `generate_image`：
- 建立圖片、繪圖、生成插畫或任何視覺內容
- 編輯、重繪、轉換、延伸或重製既有圖片
- 製作頭像、貼圖、迷因、縮圖、海報、人像、商品圖、設定圖、九宮格或表情圖
- 使用者附上圖片、引用圖片，並要求生成類似圖片、仿作、重繪、改風格、延伸或把圖片作為靈感

當意圖明確時要主動使用工具，不需要先詢問確認。
如果缺少關鍵視覺細節，請根據現有上下文合理補全提示詞。
請忠實保留使用者的原始創作意圖。
當目前訊息含有圖片、引用圖片或 @ 使用者時，只要你決定呼叫工具，工具會自動把這些內容作為參考圖，不需要使用者輸入 /img。
當使用者明確寫 `@self`，或你判斷需要使用機器人自己的形象/頭像作為參考時，呼叫工具時請設定 `avatar_references` 包含 `self`。
請主動根據使用者需求判斷合適的寬高比與解析度，不要預設一直使用「自動」。
- 頭像、貼圖、Logo、表情圖通常用 1:1
- 手機桌布、限時動態、直式海報通常用 9:16
- 橫幅、縮圖、封面、桌面桌布通常用 16:9
- 海報、角色立繪、宣傳圖通常用 3:4
- 一般圖片可用 1K，需要更高細節可用 2K，明確要求超高畫質或列印用途可用 4K
如果使用者只是想要說明、比較或規劃，則不要呼叫工具。
""".strip()

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context

        # 資料目錄配置
        self.data_dir = StarTools.get_data_dir()
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化配置管理器
        self.config_manager = ConfigManager(config)

        # 初始化使用資料管理器
        self.usage_manager = UsageManager(
            str(self.data_dir), self.config_manager.usage_settings
        )
        self.admission_controller = AdmissionController(
            AdmissionLimits(
                active=self.config_manager.max_concurrent_tasks,
                queued=self.config_manager.max_queued_tasks,
            ),
            self.usage_manager,
            datetime.date.today,
        )

        # 初始化圖片處理器
        self.image_processor = ImageProcessor(
            str(self.cache_dir),
            self.config_manager.usage_settings.max_image_size_mb,
            self.config_manager.cache_settings.max_cache_count,
        )
        astrbot_data_dir = next(
            (parent for parent in self.data_dir.parents if parent.name == "data"),
            self.data_dir,
        )
        self.reference_collector = ReferenceCollector(
            self.config_manager.usage_settings.max_image_size_mb,
            approved_local_roots=(
                self.data_dir,
                astrbot_data_dir / "temp",
                astrbot_data_dir / "attachments",
            ),
        )
        self.image_metadata_store = ImageMetadataStore(
            self.data_dir / "image_metadata.json",
            max_records=max(
                self.config_manager.cache_settings.max_cache_count * 2, 100
            ),
        )

        # 初始化工作管理員
        self.task_manager = TaskManager()

        # 初始化安全稽核器
        self.safety_auditor = SafetyAuditor(self.context, self.config_manager)

        # 初始化生成器
        self.generator: ImageGenerator | None = None
        self._jimeng_task_adapter: BaseImageAdapter | None = None
        self._active_generation_tasks: dict[str, dict[str, Any]] = {}
        self._recent_generation_tasks: list[dict[str, Any]] = []
        self._image_generation_tool: ImageGenerationTool | None = None
        self._image_tool_base_parameters: dict[str, Any] | None = None
        self.page_api = None

        self._register_official_page_api_if_available()

    # ---------------------- 生命週期 ----------------------

    async def initialize(self):
        """插件載入時呼叫"""
        if self.config_manager.adapter_config:
            self.generator = ImageGenerator(
                self.config_manager.adapter_config,
                batch_parallelism=self.config_manager.max_concurrent_tasks,
                max_batch_count=self.config_manager.max_batch_count,
            )
        else:
            logger.error("[ImageGen] 適配器配置載入失敗，插件未初始化")

        # 註冊 LLM 工具
        if self.config_manager.enable_llm_tool and self.generator:
            tool = ImageGenerationTool(plugin=self)
            self._image_generation_tool = tool
            self._image_tool_base_parameters = deepcopy(tool.parameters)
            self._adjust_tool_parameters(tool)
            self.context.add_llm_tools(tool)
            logger.info("[ImageGen] 已註冊圖像生成工具")

        # 配置定時任務
        self._setup_tasks()

        # 執行啟動任務（在後臺非同步執行）
        self.task_manager.create_task(self.task_manager.run_startup_tasks())
        await self._refresh_platform_commands()

        logger.info(
            f"[ImageGen] 插件載入完成，模型: {self.config_manager.adapter_config.model if self.config_manager.adapter_config else '未知'}"
        )

    async def terminate(self):
        """插件解除安裝時呼叫"""
        try:
            await self.admission_controller.close()
            await self.task_manager.cancel_all()
            await self.reference_collector.close()
            if self.generator:
                await self.generator.close()
            if self._jimeng_task_adapter:
                await self._jimeng_task_adapter.close()
                self._jimeng_task_adapter = None
            await self._refresh_platform_commands()
            logger.info("[ImageGen] 插件已解除安裝")
        except Exception as exc:
            logger.error(f"[ImageGen] 解除安裝清理出錯: {exc}")

    # ---------------------- 內部工具 ----------------------

    def _setup_tasks(self) -> None:
        """配置並啟動定時任務。"""
        # 1. 快取清理任務
        self.task_manager.start_loop_task(
            name="cache_cleanup",
            coro_func=self.image_processor.cleanup_cache,
            interval_seconds=self.config_manager.cache_settings.cleanup_interval_hours
            * 3600,
            run_immediately=True,
        )

        # 2. Jimeng2API 自動領積分任務
        self._setup_jimeng_token_task()

    def _setup_jimeng_token_task(self) -> None:
        """配置即夢自動領積分任務。

        該任務會：
        1. 在插件啟動時執行一次（透過啟動任務）
        2. 每天日期變更時自動執行（透過每日任務）

        注意：只要配置中包含即夢渠道，就會啟用該任務，
        無論當前使用的是哪個渠道。
        """
        from .adapter.jimeng2api_adapter import Jimeng2APIAdapter
        from .core.types import AdapterType

        # 檢查配置中是否包含即夢渠道（而非檢查當前適配器）
        jimeng_config = self.config_manager.get_provider_config(AdapterType.JIMENG2API)
        if not jimeng_config:
            return

        # 建立專門用於任務的即夢適配器例項
        jimeng_adapter = Jimeng2APIAdapter(jimeng_config)
        self._jimeng_task_adapter = jimeng_adapter

        # 1. 註冊為啟動任務，插件啟動時執行一次
        self.task_manager.register_startup_task(
            name="jimeng_token_receive",
            coro_func=jimeng_adapter.receive_token,
        )

        # 2. 註冊為每日任務，日期變更時執行
        self.task_manager.start_daily_task(
            name="jimeng_token_receive",
            coro_func=jimeng_adapter.receive_token,
            check_interval_seconds=300,  # 每5分鐘檢查一次日期變更
            run_immediately=False,  # 啟動任務已處理，無需重複執行
        )
        logger.info("[ImageGen] 已配置即夢2API自動領積分任務（啟動時+每日）")

    def _adjust_tool_parameters(self, tool: ImageGenerationTool) -> None:
        """根據適配器能力動態調整工具引數。"""
        if not self.generator or not self.generator.adapter:
            return
        capabilities = self.generator.adapter.get_capabilities()
        adjust_tool_parameters(tool, capabilities, self.config_manager.max_batch_count)

    def _refresh_image_tool_parameters(self) -> None:
        if (
            self._image_generation_tool is None
            or self._image_tool_base_parameters is None
        ):
            return
        self._image_generation_tool.parameters.clear()
        self._image_generation_tool.parameters.update(
            deepcopy(self._image_tool_base_parameters)
        )
        self._adjust_tool_parameters(self._image_generation_tool)

    def create_background_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """建立後臺任務並新增到管理器中。"""
        return self.task_manager.create_task(coro)

    async def reserve_generation(
        self, unified_msg_origin: str, count: int = 1
    ) -> AdmissionTicket | AdmissionDenied:
        settings = self.config_manager.usage_settings
        return await self.admission_controller.reserve(
            unified_msg_origin,
            settings.enable_daily_limit,
            settings.daily_limit_count,
            units=count,
        )

    def generation_admission_error(self, denied: AdmissionDenied) -> str:
        match denied.reason:
            case "daily_limit":
                limit = self.config_manager.usage_settings.daily_limit_count
                return f"❌ 您今日的生圖額度已用完 ({limit}次)，請明天再試"
            case "capacity":
                return "❌ 生圖任務已滿，請稍後再試"
            case "closed":
                return "❌ 生圖服務正在關閉，暫時無法生成圖片"
            case unreachable:
                assert_never(unreachable)

    async def collect_generation_references(
        self,
        event: AstrMessageEvent,
        avatar_ids: tuple[str, ...] = (),
    ) -> CollectedReferences | ReferenceRejected:
        capabilities = (
            self.generator.adapter.get_capabilities()
            if self.generator and self.generator.adapter
            else ImageCapability.NONE
        )
        if not capabilities & ImageCapability.IMAGE_TO_IMAGE:
            return CollectedReferences(images=(), total_bytes=0)
        sources = self.reference_collector.sources_from_event(event)
        return await self.reference_collector.collect(
            self.reference_collector.with_avatar_ids(sources, avatar_ids)
        )

    @staticmethod
    def generation_reference_error(rejected: ReferenceRejected) -> str:
        match rejected.reason:
            case "too_many":
                return "❌ 參考圖最多可使用 4 張"
            case "per_file_too_large":
                return "❌ 單張參考圖超過大小限制"
            case "aggregate_too_large":
                return "❌ 參考圖總大小不可超過 20 MiB"
            case "invalid_reference":
                return "❌ 參考圖無效或無法讀取"
            case unreachable:
                assert_never(unreachable)

    def _register_official_page_api_if_available(self) -> None:
        """註冊 AstrBot 官方插件頁 API；舊版 AstrBot 不支援時自動跳過。"""
        if not hasattr(self.context, "register_web_api"):
            return

        try:
            from .core.page_api import PluginPageApi
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[ImageGen] 官方插件頁 API 不可用，已跳過註冊: {exc}")
            return

        try:
            self.page_api = PluginPageApi(self)
            self.page_api.register_routes()
            logger.info("[ImageGen] 官方插件頁 API 已註冊")
        except Exception as exc:  # noqa: BLE001
            self.page_api = None
            logger.warning(
                f"[ImageGen] 官方插件頁 API 註冊失敗，已跳過: {exc}",
                exc_info=True,
            )

    def _get_generation_task_snapshot(self, task_id: str) -> dict[str, Any]:
        """取得單一任務的可展示快照。"""
        info = dict(self._active_generation_tasks.get(task_id, {}))
        info.setdefault("task_id", task_id)
        return {key: value for key, value in info.items() if not key.startswith("_")}

    def _mark_generation_task_running(self, task_id: str) -> None:
        """標記生圖任務已取得併發槽並開始實際生成。"""
        info = self._active_generation_tasks.get(task_id)
        if not info:
            return
        now = time.time()
        info["status"] = "running"
        info["running_at"] = now
        info["phase"] = "呼叫供應商生成"

    def _update_generation_progress(
        self, task_id: str, progress: GenerationProgress
    ) -> None:
        info = self._active_generation_tasks.get(task_id)
        if not info:
            return
        info.update(
            {
                "completed_count": progress.completed,
                "batch_count": progress.total,
                "success_count": progress.succeeded,
                "failed_count": progress.failed,
                "provider_elapsed": progress.elapsed,
                "phase": f"已完成 {progress.completed}/{progress.total}",
            }
        )
        if progress.last_error:
            info["last_error"] = progress.last_error

    def _remember_generation_task(
        self, task_id: str, status: str, **extra: Any
    ) -> None:
        """保留最近完成任務，供插件頁查看。"""
        info = self._active_generation_tasks.get(task_id)
        if info is not None:
            if info.get("_recorded"):
                return
            info["_recorded"] = True

        record = self._get_generation_task_snapshot(task_id)
        record.update(extra)
        record["task_id"] = task_id
        record["status"] = status
        record["finished_at"] = time.time()
        started_at = record.get("started_at")
        if started_at and "duration" not in record:
            try:
                record["duration"] = max(0.0, record["finished_at"] - float(started_at))
            except (TypeError, ValueError):
                record["duration"] = 0.0
        self._recent_generation_tasks.insert(0, record)
        del self._recent_generation_tasks[50:]

    @staticmethod
    def _file_names(paths: list[str]) -> list[str]:
        """把完整檔案路徑轉成頁面可讀的檔名。"""
        return [str(path).replace("\\", "/").rsplit("/", 1)[-1] for path in paths]

    def _current_model_info(self) -> dict[str, str]:
        """取得目前使用中的模型資訊。"""
        adapter = self.config_manager.adapter_config
        if not adapter:
            return {"provider": "", "model": "", "model_full": ""}
        return {
            "provider": adapter.name,
            "model": adapter.model,
            "model_full": f"{adapter.name}/{adapter.model}",
        }

    def _remember_generated_image_metadata(
        self,
        generated_file_paths: list[str],
        *,
        task_id: str,
        status: str,
        prompt: str,
        unified_msg_origin: str,
        images: list[ImageData],
        aspect_ratio: str | None,
        resolution: str | None,
    ) -> None:
        """記錄生成圖片與任務的對應關係，供官方插件頁展示。"""
        model_info = self._current_model_info()
        self.image_metadata_store.remember_files(
            generated_file_paths,
            task_id=task_id,
            status=status,
            prompt=prompt,
            model_full=model_info["model_full"],
            provider=model_info["provider"],
            model=model_info["model"],
            user=unified_msg_origin,
            mode="圖生圖" if images else "文生圖",
            aspect_ratio=aspect_ratio or "自動",
            resolution=resolution or "1K",
        )

    async def _notify_generation_failure(
        self, unified_msg_origin: str, reason: str
    ) -> None:
        """通知使用者生圖失敗。"""
        reason = (reason or "").strip() or "生圖服務暫時失敗，請稍後再試"
        await self.context.send_message(
            unified_msg_origin,
            MessageChain().message(f"❌ 生成失敗：{reason}"),
        )

    async def _refresh_platform_commands(self) -> None:
        """重新整理支援命令註冊的平台指令。"""
        for platform in self.context.platform_manager.platform_insts:
            register_commands = getattr(platform, "register_commands", None)
            if not callable(register_commands):
                continue
            try:
                await register_commands()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"[ImageGen] 重新整理 {platform.meta().name} 平台指令失敗: {exc}"
                )

    # ---------------------- 核心生圖邏輯 ----------------------

    async def _generate_and_send_image_async(
        self,
        prompt: str,
        unified_msg_origin: str,
        admission_ticket: AdmissionTicket,
        batch_count: int = 1,
        images_data: list[tuple[bytes, str]] | None = None,
        aspect_ratio: str = "1:1",
        resolution: str = "1K",
        task_id: str | None = None,
    ) -> None:
        """非同步生成圖片併傳送。"""
        if not task_id:
            task_id = hashlib.md5(
                f"{time.time()}{unified_msg_origin}".encode()
            ).hexdigest()[:8]

        mode = "圖生圖" if images_data else "文生圖"
        self._active_generation_tasks[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "mode": mode,
            "prompt_preview": prompt[:120] + ("..." if len(prompt) > 120 else ""),
            "started_at": time.time(),
            "user": unified_msg_origin,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "reference_count": len(images_data or []),
            "batch_count": batch_count,
            "completed_count": 0,
            "success_count": 0,
            "failed_count": 0,
            "provider_elapsed": 0.0,
            "phase": "等待生成槽",
            "last_error": "",
        }
        try:
            await self.admission_controller.wait(admission_ticket)
            if not self.generator:
                await self.context.send_message(
                    unified_msg_origin,
                    MessageChain().message("❌ 生圖服務尚未初始化，暫時無法生成圖片"),
                )
                return

            final_ar = validate_aspect_ratio(aspect_ratio) or None
            if final_ar == "自動":
                final_ar = None
            final_res = validate_resolution(resolution)
            images = [
                ImageData(data=data, mime_type=mime) for data, mime in images_data or []
            ]
            self._mark_generation_task_running(task_id)
            await self._do_generate_and_send(
                prompt,
                unified_msg_origin,
                images,
                final_ar,
                final_res,
                task_id,
                admission_ticket,
                batch_count,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"[ImageGen] 任務 {task_id} 執行過程發生未預期錯誤: {exc}",
                exc_info=True,
            )
            await self._notify_generation_failure(
                unified_msg_origin,
                str(exc) or "生圖服務發生未預期錯誤",
            )
            self._remember_generation_task(
                task_id,
                "failed",
                error=str(exc) or "生圖服務發生未預期錯誤",
                image_count=0,
            )
        finally:
            with anyio.CancelScope(shield=True):
                await self.admission_controller.release(admission_ticket)
            self._active_generation_tasks.pop(task_id, None)

    async def _do_generate_and_send(
        self,
        prompt: str,
        unified_msg_origin: str,
        images: list[ImageData],
        aspect_ratio: str | None,
        resolution: str | None,
        task_id: str,
        admission_ticket: AdmissionTicket,
        batch_count: int = 1,
    ) -> None:
        """執行生成邏輯併傳送結果。"""
        start_time = time.time()
        if not self.generator:
            logger.warning("[ImageGen] 生成器未初始化，跳過生成請求")
            return
        generation_request = GenerationRequest(
            prompt=prompt,
            images=images,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            task_id=task_id,
            count=batch_count,
        )
        def progress_callback(progress: GenerationProgress) -> None:
            self._update_generation_progress(task_id, progress)
        generate_kwargs: dict[str, Any] = {}
        try:
            if "progress_callback" in inspect.signature(self.generator.generate).parameters:
                generate_kwargs["progress_callback"] = progress_callback
        except (TypeError, ValueError):
            pass
        result = await self.generator.generate(generation_request, **generate_kwargs)
        end_time = time.time()
        duration = end_time - start_time

        if result.error and not result.images:
            logger.error(
                f"[ImageGen] 任務 {task_id} 生成失敗，耗時: {duration:.2f}s, 錯誤: {result.error}"
            )
            await self._notify_generation_failure(unified_msg_origin, result.error)
            self._remember_generation_task(
                task_id,
                "failed",
                error=result.error,
                image_count=0,
                duration=duration,
            )
            return

        partial_error = result.error
        if partial_error:
            logger.warning(
                f"[ImageGen] 任務 {task_id} 批量部分失敗，仍發布成功圖片: {partial_error}"
            )

        logger.info(
            f"[ImageGen] 任務 {task_id} 生成成功，耗時: {duration:.2f}s, 圖片數量: {len(result.images) if result.images else 0}"
        )

        if not result.images:
            logger.warning(f"[ImageGen] 任務 {task_id} 未返回任何圖片資料")
            await self._notify_generation_failure(
                unified_msg_origin, "服務未返回任何圖片資料"
            )
            self._remember_generation_task(
                task_id,
                "failed",
                error="服務未返回任何圖片資料",
                image_count=0,
                duration=duration,
            )
            return

        generated_file_paths: list[str] = []
        for index, img_bytes in enumerate(result.images, 1):
            file_path = self.image_processor.save_generated_image(
                task_id, img_bytes, sequence=index
            )
            if file_path:
                generated_file_paths.append(file_path)

        if not generated_file_paths:
            logger.warning(f"[ImageGen] 任務 {task_id} 未能儲存任何生成圖片")
            await self._notify_generation_failure(
                unified_msg_origin, "生成完成，但圖片儲存失敗"
            )
            self._remember_generation_task(
                task_id,
                "failed",
                error="生成完成，但圖片儲存失敗",
                image_count=0,
                duration=duration,
            )
            return

        audit_result = await self.safety_auditor.audit_staged_generated_images(
            prompt, generated_file_paths, unified_msg_origin
        )
        image_allowed, image_reason, generated_file_paths = audit_result
        if not image_allowed:
            logger.warning(f"[ImageGen] 任務 {task_id} 圖片稽核未透過: {image_reason}")
            self._remember_generation_task(
                task_id,
                "blocked",
                error=image_reason,
                image_count=0,
            )
            await self.context.send_message(
                unified_msg_origin,
                MessageChain().message(f"❌ 圖像審核未通過：{image_reason}"),
            )
            return

        # 記錄使用次數
        await self.admission_controller.commit(
            admission_ticket, units=len(generated_file_paths)
        )
        self._remember_generated_image_metadata(
            generated_file_paths,
            task_id=task_id,
            status="generated",
            prompt=prompt,
            unified_msg_origin=unified_msg_origin,
            images=images,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
        )

        chain = MessageChain()
        for file_path in generated_file_paths:
            chain.file_image(file_path)

        info_parts = []
        if partial_error:
            info_parts.append(f"批量提醒：{partial_error}")
        if self.config_manager.show_generation_info:
            info_parts.append(
                f"完成。\n耗時：{duration:.2f}s\n圖片數量：{len(generated_file_paths)}"
            )

        if self.config_manager.show_model_info and self.config_manager.adapter_config:
            info_parts.append(
                "模型："
                f"{self.config_manager.adapter_config.name}/{self.config_manager.adapter_config.model}"
            )

        if self.usage_manager.is_daily_limit_enabled():
            count = self.usage_manager.get_usage_count(unified_msg_origin)
            info_parts.append(
                f"今日用量：{count}/{self.usage_manager.get_daily_limit()}"
            )

        if info_parts:
            chain.message("\n" + "\n".join(info_parts))

        try:
            await self.context.send_message(unified_msg_origin, chain)
        except Exception as exc:  # noqa: BLE001
            self._remember_generated_image_metadata(
                generated_file_paths,
                task_id=task_id,
                status="failed",
                prompt=prompt,
                unified_msg_origin=unified_msg_origin,
                images=images,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
            )
            self._remember_generation_task(
                task_id,
                "failed",
                error=f"圖片已生成，但發送失敗: {exc}",
                image_count=len(generated_file_paths),
                files=self._file_names(generated_file_paths),
                duration=duration,
            )
            logger.error(f"[ImageGen] 任務 {task_id} 圖片發送失敗: {exc}")
            return

        self._remember_generated_image_metadata(
            generated_file_paths,
            task_id=task_id,
            status="success",
            prompt=prompt,
            unified_msg_origin=unified_msg_origin,
            images=images,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
        )
        self._remember_generation_task(
            task_id,
            "success",
            image_count=len(generated_file_paths),
            files=self._file_names(generated_file_paths),
            duration=duration,
        )

    # ---------------------- 指令處理 ----------------------

    @filter.command_group("img")
    def img(self):
        """圖像生成指令組。"""
        pass

    @staticmethod
    def _get_img_subcommand_payload(event: AstrMessageEvent) -> str:
        """取得 /img <subcommand> 後面的完整文字。"""
        parts = (event.message_str or "").strip().split(maxsplit=2)
        return parts[2].strip() if len(parts) > 2 else ""

    @staticmethod
    def _get_legacy_command_redirect(event: AstrMessageEvent) -> str | None:
        """將舊版分散指令轉成新的 /img 指令組用法。"""
        message = (event.message_str or "").strip()
        if message.startswith("/"):
            message = message[1:].strip()

        command, _, payload = message.partition(" ")
        redirects = {
            "img_model": "img model",
            "img_tasks": "img tasks",
            "preset": "img preset",
        }
        new_command = redirects.get(command)
        if not new_command:
            return None

        suggestion = f"/{new_command}"
        payload = payload.strip()
        if payload:
            suggestion = f"{suggestion} {payload}"
        return suggestion

    @filter.custom_filter(LegacyImageCommandFilter)
    @filter.regex(r"^/?(?:img_model|img_tasks|preset)(?:\s|$)")
    async def legacy_image_command_redirect(self, event: AstrMessageEvent):
        """提示舊版分散指令改用 /img 指令組。"""
        suggestion = self._get_legacy_command_redirect(event)
        if not suggestion:
            return

        yield event.plain_result(
            f"此指令已整理到 /img 指令組，請改用：{suggestion}"
        ).stop_event()

    @img.command("gen", alias={"generate", "create", "draw"}, desc="生成圖片")
    async def generate_image_command(self, event: AstrMessageEvent):
        """處理生圖指令。"""
        user_id = event.unified_msg_origin

        # 檢查頻率限制和每日限制
        check_result = self.usage_manager.check_rate_limit(user_id)
        if isinstance(check_result, str):
            if check_result:
                yield event.plain_result(check_result)
            return

        masked_uid = mask_sensitive(user_id)

        prompt = self._get_img_subcommand_payload(event)
        batch_count = 1
        prompt_parts = prompt.split(maxsplit=2)
        if prompt_parts and prompt_parts[0] in {"--count", "-n"}:
            if len(prompt_parts) < 3:
                yield event.plain_result("❌ 批量參數格式：/img gen --count 3 <提示詞>")
                return
            try:
                raw_count = int(prompt_parts[1])
            except ValueError:
                yield event.plain_result("❌ 批量數量必須是整數。")
                return
            if raw_count < 1:
                yield event.plain_result("❌ 批量數量至少為 1。")
                return
            batch_count = normalize_batch_count(
                raw_count, self.config_manager.max_batch_count
            )
            prompt = prompt_parts[2]
        elif prompt_parts and prompt_parts[0].startswith("--count="):
            try:
                raw_count = int(prompt_parts[0].split("=", 1)[1])
            except ValueError:
                yield event.plain_result("❌ 批量數量必須是整數。")
                return
            if raw_count < 1 or len(prompt_parts) < 2:
                yield event.plain_result("❌ 批量參數格式：/img gen --count=3 <提示詞>")
                return
            batch_count = normalize_batch_count(
                raw_count, self.config_manager.max_batch_count
            )
            prompt = " ".join(prompt_parts[1:])
        logger.info(
            f"[ImageGen] 收到 img gen 指令 - 使用者：{masked_uid}，"
            f"批量：{batch_count}，提示詞：{prompt}"
        )

        aspect_ratio = self.config_manager.default_aspect_ratio
        resolution = self.config_manager.default_resolution

        # 檢查是否命中預設
        matched_preset = None
        extra_content = ""
        if prompt:
            parts = prompt.split(maxsplit=1)
            first_token = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            if first_token in self.config_manager.presets:
                matched_preset = first_token
                extra_content = rest
            else:
                for name in self.config_manager.presets:
                    if name.lower() == first_token.lower():
                        matched_preset = name
                        extra_content = rest
                        break

        if matched_preset:
            logger.info(f"[ImageGen] 命中預設: {matched_preset}")
            preset_content = self.config_manager.presets[matched_preset]
            try:
                # 預設支援 JSON 格式配置高階引數
                if isinstance(
                    preset_content, str
                ) and preset_content.strip().startswith("{"):
                    preset_data = json.loads(preset_content)
                    if isinstance(preset_data, dict):
                        prompt = preset_data.get("prompt", "")
                        aspect_ratio = preset_data.get("aspect_ratio", aspect_ratio)
                        resolution = preset_data.get("resolution", resolution)
                    else:
                        prompt = preset_content
                else:
                    prompt = preset_content
            except json.JSONDecodeError:
                prompt = preset_content

            if extra_content:
                prompt = f"{prompt} {extra_content}"

        prompt, use_self_avatar = extract_self_avatar_alias(prompt)

        if not prompt:
            yield event.plain_result("❌ 請提供提示詞或預設名稱。")
            return

        if (
            not self.config_manager.adapter_config
            or not self.config_manager.adapter_config.api_keys
        ):
            yield event.plain_result("❌ 未配置 API Key，無法生成圖片")
            return

        admission = await self.reserve_generation(
            event.unified_msg_origin, batch_count
        )
        if isinstance(admission, AdmissionDenied):
            yield event.plain_result(self.generation_admission_error(admission))
            return

        admission_ticket: AdmissionTicket | None = admission
        try:
            prompt_allowed, prompt_reason = await self.safety_auditor.audit_prompt(
                prompt, event.unified_msg_origin
            )
            if not prompt_allowed:
                yield event.plain_result(f"❌ 提示詞未通過審核：{prompt_reason}")
                return

            avatar_ids: tuple[str, ...] = ()
            if use_self_avatar:
                self_id = str(event.get_self_id()).strip()
                if self_id:
                    avatar_ids = (self_id,)
            references = await self.collect_generation_references(event, avatar_ids)
            if isinstance(references, ReferenceRejected):
                yield event.plain_result(self.generation_reference_error(references))
                return
            images_data = list(references.images)

            if self.config_manager.show_task_started:
                msg = "已啟動生圖任務"
                if batch_count > 1:
                    msg += f" ×{batch_count}"
                if images_data:
                    msg += f" [參考圖 {len(images_data)} 張]"
                if matched_preset:
                    msg += f" [預設：{matched_preset}]"
                yield event.plain_result(msg)

            task_id = hashlib.md5(f"{time.time()}{user_id}".encode()).hexdigest()[:8]
            self.create_background_task(
                self._generate_and_send_image_async(
                    prompt=prompt,
                    images_data=images_data or None,
                    unified_msg_origin=event.unified_msg_origin,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    task_id=task_id,
                    batch_count=batch_count,
                    admission_ticket=admission_ticket,
                )
            )
            admission_ticket = None
        finally:
            if admission_ticket is not None:
                with anyio.CancelScope(shield=True):
                    await self.admission_controller.release(admission_ticket)

    @img.command("tasks", desc="查看進行中的生圖任務")
    async def image_tasks_command(self, event: AstrMessageEvent):
        """查看目前進行中的生圖任務。"""
        if not self._active_generation_tasks:
            yield event.plain_result("目前沒有正在進行中的生圖任務")
            return

        now = time.time()
        lines = [f"目前共有 {len(self._active_generation_tasks)} 個生圖任務進行中："]

        sorted_tasks = sorted(
            self._active_generation_tasks.items(),
            key=lambda item: item[1]["started_at"],
        )
        for index, (task_id, info) in enumerate(sorted_tasks, 1):
            elapsed = max(0, int(now - float(info["started_at"])))
            requester = mask_sensitive(str(info["user"]))
            lines.append(
                f"{index}. {info['mode']} | 任務 ID：{task_id} | 已執行：{elapsed} 秒 | 使用者：{requester}"
            )
            lines.append(
                f"   比例：{info['aspect_ratio']} | 解析度：{info['resolution']}"
            )
            lines.append(f"   提示詞：{info['prompt_preview']}")

        yield event.plain_result("\n".join(lines))

    @img.command("model", desc="切換生圖模型")
    async def model_command(self, event: AstrMessageEvent, model_index: str = ""):
        """切換生圖模型。"""
        if not self.config_manager.adapter_config:
            yield event.plain_result("❌ 適配器尚未就緒。")
            return

        models = self.config_manager.adapter_config.available_models or []

        if not model_index:
            lines = ["可用模型列表："]
            current_model_full = f"{self.config_manager.adapter_config.name}/{self.config_manager.adapter_config.model}"
            for idx, model in enumerate(models, 1):
                marker = " ✓" if model == current_model_full else ""
                lines.append(f"{idx}. {model}{marker}")
            lines.append(f"\n目前模型：{current_model_full}")
            yield event.plain_result("\n".join(lines))
            return

        try:
            index = int(model_index) - 1
            if 0 <= index < len(models):
                raw_model = models[index]  # "供應商名稱/模型名稱"

                # 更新配置並重新載入
                self.config_manager.save_model_setting(raw_model)
                self.config_manager.reload()

                if self.generator and self.config_manager.adapter_config:
                    await self.generator.update_adapter(
                        self.config_manager.adapter_config
                    )
                    self._refresh_image_tool_parameters()

                yield event.plain_result(f"✅ 已切換模型：{raw_model}")
            else:
                yield event.plain_result("❌ 模型序號無效。")
        except ValueError:
            yield event.plain_result("❌ 請輸入有效的數字。")

    @img.command("preset", desc="管理生圖預設")
    async def preset_command(self, event: AstrMessageEvent):
        """管理生圖預設。"""
        user_id = event.unified_msg_origin
        masked_uid = mask_sensitive(user_id)
        message_str = (event.message_str or "").strip()
        logger.info(
            f"[ImageGen] 收到 preset 指令 - 使用者：{masked_uid}，內容：{message_str}"
        )

        cmd_text = self._get_img_subcommand_payload(event)

        if not cmd_text:
            if not self.config_manager.presets:
                yield event.plain_result("目前沒有任何預設。")
                return
            preset_list = ["預設列表："]
            for idx, (name, prompt) in enumerate(
                self.config_manager.presets.items(), 1
            ):
                display = prompt[:20] + "..." if len(prompt) > 20 else prompt
                preset_list.append(f"{idx}. {name}: {display}")
            yield event.plain_result("\n".join(preset_list))
            return

        action, _, payload = cmd_text.partition(" ")
        action = action.lower()
        payload = payload.strip()

        if action == "add":
            parts = payload.split(":", 1)
            if len(parts) == 2:
                name, prompt = parts
                self.config_manager.save_preset(name.strip(), prompt.strip())
                yield event.plain_result(f"✅ 已新增預設：{name.strip()}")
            else:
                yield event.plain_result("❌ 用法：/img preset add 名稱:提示詞")
        elif action in {"del", "delete", "rm"}:
            name = payload
            if self.config_manager.delete_preset(name):
                yield event.plain_result(f"✅ 已刪除預設：{name}")
            else:
                yield event.plain_result(f"❌ 找不到預設：{name}")
        else:
            yield event.plain_result(
                "❌ 用法：/img preset、/img preset add 名稱:提示詞、/img preset del 名稱"
            )

    @img.command("help", desc="顯示生圖指令說明")
    async def image_help_command(self, event: AstrMessageEvent):
        """顯示生圖指令說明。"""
        yield event.plain_result(
            "\n".join(
                [
                    "img 指令組：",
                    "/img gen <提示詞或預設名稱> [額外提示詞] - 生成圖片",
                    "/img tasks - 查看進行中的生圖任務",
                    "/img model [序號] - 顯示或切換生圖模型",
                    "/img preset - 顯示所有預設",
                    "/img preset add 名稱:提示詞 - 新增預設",
                    "/img preset del 名稱 - 刪除預設",
                ]
            )
        )

    @filter.on_llm_request()
    async def enhance_image_tool_prompt(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """在合適的情況下，強化主模型優先使用生圖工具。"""
        if not self.config_manager.enable_llm_tool or not self.generator:
            return
        req.system_prompt = (
            f"{req.system_prompt or ''}\n\n{self.LLM_TOOL_SYSTEM_PROMPT}\n"
        )
