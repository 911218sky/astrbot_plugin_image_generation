"""
AstrBot 圖像生成插件主模組

"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Coroutine
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.star.star_tools import StarTools

from .core.config_manager import ConfigManager
from .core.generator import ImageGenerator
from .core.image_processor import ImageProcessor
from .core.llm_tool import ImageGenerationTool, adjust_tool_parameters
from .core.safety_auditor import SafetyAuditor
from .core.task_manager import TaskManager
from .core.types import GenerationRequest, ImageCapability, ImageData
from .core.usage_manager import UsageManager
from .core.utils import (
    extract_self_avatar_alias,
    mask_sensitive,
    validate_aspect_ratio,
    validate_resolution,
)


class ImageGenerationPlugin(Star):
    """圖像生成插件主類"""

    LLM_TOOL_SYSTEM_PROMPT = """
# 圖像生成工具規則

當使用者希望最終產出是一張圖片時，優先呼叫 `generate_image` 工具，而不是只回覆文字。

當使用者要求以下內容時，應呼叫 `generate_image`：
- 建立圖片、繪圖、生成插畫或任何視覺內容
- 編輯、重繪、轉換、延伸或重製既有圖片
- 製作頭像、貼圖、迷因、縮圖、海報、人像、商品圖、設定圖、九宮格或表情圖

當意圖明確時要主動使用工具，不需要先詢問確認。
如果缺少關鍵視覺細節，請根據現有上下文合理補全提示詞。
請忠實保留使用者的原始創作意圖。
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

        # 初始化圖片處理器
        self.image_processor = ImageProcessor(
            str(self.cache_dir),
            self.config_manager.usage_settings.max_image_size_mb,
            self.config_manager.cache_settings.max_cache_count,
        )

        # 初始化工作管理員
        self.task_manager = TaskManager()

        # 初始化安全稽核器
        self.safety_auditor = SafetyAuditor(self.context, self.config_manager)

        # 初始化生成器
        self.generator: ImageGenerator | None = None
        self.semaphore: asyncio.Semaphore | None = None
        self._active_generation_tasks: dict[str, dict[str, Any]] = {}

    # ---------------------- 生命週期 ----------------------

    async def initialize(self):
        """插件載入時呼叫"""
        if self.config_manager.adapter_config:
            self.generator = ImageGenerator(self.config_manager.adapter_config)
            self.semaphore = asyncio.Semaphore(self.config_manager.max_concurrent_tasks)
        else:
            logger.error("[ImageGen] 適配器配置載入失敗，插件未初始化")

        # 註冊 LLM 工具
        if self.config_manager.enable_llm_tool and self.generator:
            tool = ImageGenerationTool(plugin=self)
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
            if self.generator:
                await self.generator.close()
            await self.task_manager.cancel_all()
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
        adjust_tool_parameters(tool, capabilities)

    def create_background_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """建立後臺任務並新增到管理器中。"""
        return self.task_manager.create_task(coro)

    async def _notify_generation_failure(
        self, unified_msg_origin: str, reason: str
    ) -> None:
        """通知使用者生圖失敗。"""
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
        images_data: list[tuple[bytes, str]] | None = None,
        aspect_ratio: str = "1:1",
        resolution: str = "1K",
        task_id: str | None = None,
    ) -> None:
        """非同步生成圖片併傳送。"""
        if not self.generator or not self.generator.adapter:
            await self.context.send_message(
                unified_msg_origin,
                MessageChain().message("❌ 生圖服務尚未初始化，暫時無法生成圖片"),
            )
            return

        capabilities = self.generator.adapter.get_capabilities()

        # 檢查並清理不支援的引數
        if not (capabilities & ImageCapability.IMAGE_TO_IMAGE) and images_data:
            logger.warning(
                f"[ImageGen] 當前適配器不支援參考圖，已忽略 {len(images_data)} 張圖片"
            )
            images_data = None

        if not (capabilities & ImageCapability.ASPECT_RATIO) and aspect_ratio != "自動":
            logger.info(
                f"[ImageGen] 當前適配器不支援指定比例，已忽略參數: {aspect_ratio}"
            )
            aspect_ratio = "自動"

        if not (capabilities & ImageCapability.RESOLUTION) and resolution != "1K":
            logger.info(
                f"[ImageGen] 當前適配器不支援指定解析度，已忽略參數: {resolution}"
            )
            resolution = "1K"

        if not task_id:
            task_id = hashlib.md5(
                f"{time.time()}{unified_msg_origin}".encode()
            ).hexdigest()[:8]

        mode = "圖生圖" if images_data else "文生圖"
        self._active_generation_tasks[task_id] = {
            "mode": mode,
            "prompt_preview": prompt[:40] + ("..." if len(prompt) > 40 else ""),
            "started_at": time.time(),
            "user": unified_msg_origin,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
        }

        final_ar = validate_aspect_ratio(aspect_ratio) or None
        if final_ar == "自動":
            final_ar = None
        final_res = validate_resolution(resolution)

        images: list[ImageData] = []
        if images_data:
            for data, mime in images_data:
                images.append(ImageData(data=data, mime_type=mime))

        # 使用訊號量控制併發
        try:
            if self.semaphore is None:
                await self._do_generate_and_send(
                    prompt, unified_msg_origin, images, final_ar, final_res, task_id
                )
                return

            async with self.semaphore:
                await self._do_generate_and_send(
                    prompt, unified_msg_origin, images, final_ar, final_res, task_id
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
        finally:
            self._active_generation_tasks.pop(task_id, None)

    async def _do_generate_and_send(
        self,
        prompt: str,
        unified_msg_origin: str,
        images: list[ImageData],
        aspect_ratio: str | None,
        resolution: str | None,
        task_id: str,
    ) -> None:
        """執行生成邏輯併傳送結果。"""
        start_time = time.time()
        if not self.generator:
            logger.warning("[ImageGen] 生成器未初始化，跳過生成請求")
            return
        result = await self.generator.generate(
            GenerationRequest(
                prompt=prompt,
                images=images,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                task_id=task_id,
            )
        )
        end_time = time.time()
        duration = end_time - start_time

        if result.error:
            logger.error(
                f"[ImageGen] 任務 {task_id} 生成失敗，耗時: {duration:.2f}s, 錯誤: {result.error}"
            )
            await self._notify_generation_failure(unified_msg_origin, result.error)
            return

        logger.info(
            f"[ImageGen] 任務 {task_id} 生成成功，耗時: {duration:.2f}s, 圖片數量: {len(result.images) if result.images else 0}"
        )

        if not result.images:
            logger.warning(f"[ImageGen] 任務 {task_id} 未返回任何圖片資料")
            await self._notify_generation_failure(
                unified_msg_origin, "服務未返回任何圖片資料"
            )
            return

        generated_file_paths: list[str] = []
        for img_bytes in result.images:
            file_path = self.image_processor.save_generated_image(task_id, img_bytes)
            if file_path:
                generated_file_paths.append(file_path)

        if not generated_file_paths:
            logger.warning(f"[ImageGen] 任務 {task_id} 未能儲存任何生成圖片")
            await self._notify_generation_failure(
                unified_msg_origin, "生成完成，但圖片儲存失敗"
            )
            return

        # 生圖後圖片稽核
        image_allowed, image_reason = await self.safety_auditor.audit_generated_images(
            prompt=prompt,
            image_paths=generated_file_paths,
            unified_msg_origin=unified_msg_origin,
        )
        if not image_allowed:
            logger.warning(f"[ImageGen] 任務 {task_id} 圖片稽核未透過: {image_reason}")
            await self.context.send_message(
                unified_msg_origin,
                MessageChain().message(f"❌ 圖像審核未通過：{image_reason}"),
            )
            return

        # 記錄使用次數
        self.usage_manager.record_usage(unified_msg_origin)

        chain = MessageChain()
        for file_path in generated_file_paths:
            chain.file_image(file_path)

        info_parts = []
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

        await self.context.send_message(unified_msg_origin, chain)

    # ---------------------- 指令處理 ----------------------

    @filter.command("img", desc="生成圖片")
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

        user_input = (event.message_str or "").strip()
        logger.info(
            f"[ImageGen] 收到 img 指令 - 使用者：{masked_uid}，輸入：{user_input}"
        )

        cmd_parts = user_input.split(maxsplit=1)
        if not cmd_parts:
            return

        prompt = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
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

        prompt_allowed, prompt_reason = await self.safety_auditor.audit_prompt(
            prompt, event.unified_msg_origin
        )
        if not prompt_allowed:
            yield event.plain_result(f"❌ 提示詞未通過審核：{prompt_reason}")
            return

        # 取得參考圖
        images_data = None
        if (
            self.generator
            and self.generator.adapter
            and (
                self.generator.adapter.get_capabilities()
                & ImageCapability.IMAGE_TO_IMAGE
            )
        ):
            images_data = await self.image_processor.fetch_images_from_event(event)
            if use_self_avatar:
                self_id = str(event.get_self_id()).strip()
                if self_id:
                    avatar_data = await self.image_processor.get_avatar(self_id)
                    if avatar_data:
                        images_data.append((avatar_data, "image/jpeg"))
                    else:
                        logger.warning(
                            "[ImageGen] @self alias was used, but the bot avatar could not be loaded"
                        )

        msg = "已啟動生圖任務"
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
            )
        )

    @filter.command("img_tasks", desc="查看進行中的生圖任務")
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

    @filter.command("img_model", desc="切換生圖模型")
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

                if self.generator:
                    await self.generator.update_adapter(
                        self.config_manager.adapter_config
                    )

                yield event.plain_result(f"✅ 已切換模型：{raw_model}")
            else:
                yield event.plain_result("❌ 模型序號無效。")
        except ValueError:
            yield event.plain_result("❌ 請輸入有效的數字。")

    @filter.command("preset", desc="管理生圖預設")
    async def preset_command(self, event: AstrMessageEvent):
        """管理生圖預設。"""
        user_id = event.unified_msg_origin
        masked_uid = mask_sensitive(user_id)
        message_str = (event.message_str or "").strip()
        logger.info(
            f"[ImageGen] 收到 preset 指令 - 使用者：{masked_uid}，內容：{message_str}"
        )

        parts = message_str.split(maxsplit=1)
        cmd_text = parts[1].strip() if len(parts) > 1 else ""

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
                yield event.plain_result("❌ 用法：/preset add 名稱:提示詞")
        elif action in {"del", "delete", "rm"}:
            name = payload
            if self.config_manager.delete_preset(name):
                yield event.plain_result(f"✅ 已刪除預設：{name}")
            else:
                yield event.plain_result(f"❌ 找不到預設：{name}")
        else:
            yield event.plain_result(
                "❌ 用法：/preset、/preset add 名稱:提示詞、/preset del 名稱"
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
