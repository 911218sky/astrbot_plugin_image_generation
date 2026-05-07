"""
LLM 可呼叫的圖像生成工具模組
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Any

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from astrbot.api import logger
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

from .types import ImageCapability

if TYPE_CHECKING:
    pass


@pydantic_dataclass
class ImageGenerationTool(FunctionTool[AstrAgentContext]):
    """LLM 可呼叫的圖像生成工具。"""

    name: str = "generate_image"
    description: str = (
        "生成或編輯圖片。當使用者希望實際產出圖片時就應使用這個工具，"
        "包括繪圖、重繪、風格轉換、製作頭像、貼圖、迷因、海報、縮圖、"
        "人像、表情圖或各種圖片變體。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "請填入最終的生圖提示詞，忠實保留使用者的視覺意圖；如果細節不足，可根據上下文合理補全。",
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "目標圖片寬高比；若無法確定請使用「自動」。",
                    "enum": [
                        "自動",
                        "1:1",
                        "2:3",
                        "3:2",
                        "3:4",
                        "4:3",
                        "4:5",
                        "5:4",
                        "9:16",
                        "16:9",
                        "21:9",
                    ],
                    "default": "自動",
                },
                "resolution": {
                    "type": "string",
                    "description": "目標圖片品質或解析度，預設為「1K」。",
                    "enum": ["1K", "2K", "4K"],
                    "default": "1K",
                },
                "avatar_references": {
                    "type": "array",
                    "description": "可選的頭像參考來源，用於圖生圖或角色對齊。可填入 `self` 代表機器人頭像、`sender` 代表當前使用者，或直接填使用者 ID。",
                    "items": {"type": "string"},
                },
            },
            "required": ["prompt"],
        }
    )

    # 使用 Any 避免 Pydantic 迴圈引用問題
    # 實際型別為 ImageGenerationPlugin，在 TYPE_CHECKING 中定義
    plugin: Any = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> ToolExecResult:
        """執行工具呼叫。"""
        # 取得提示詞
        prompt = kwargs.get("prompt", "").strip()
        if not prompt:
            return "❌ 請提供圖片生成的提示詞"

        plugin = self.plugin
        if not plugin:
            return "❌ 插件未正確初始化"

        # 取得事件上下文
        event = None
        if hasattr(context, "context") and isinstance(
            context.context, AstrAgentContext
        ):
            event = context.context.event
        elif isinstance(context, dict):
            event = context.get("event")

        if not event:
            logger.warning(
                f"[ImageGen] 工具呼叫上下文缺少事件。上下文型別: {type(context)}"
            )
            return "❌ 無法取得目前訊息上下文"

        # 檢查頻率限制和每日限制
        check_result = plugin.usage_manager.check_rate_limit(event.unified_msg_origin)
        if isinstance(check_result, str):
            if check_result:
                logger.warning(
                    f"[ImageGen] 工具呼叫觸發限制: {check_result} (使用者: {event.unified_msg_origin})"
                )
            return check_result

        if (
            not plugin.config_manager.adapter_config
            or not plugin.config_manager.adapter_config.api_keys
        ):
            logger.warning(
                f"[ImageGen] 工具呼叫失敗: 未配置 API Key (使用者: {event.unified_msg_origin})"
            )
            return "❌ 未配置 API Key，無法生成圖片"

        prompt_allowed, prompt_reason = await plugin.safety_auditor.audit_prompt(
            prompt, event.unified_msg_origin
        )
        if not prompt_allowed:
            return f"❌ 提示詞審核未通過：{prompt_reason}"

        # 工具呼叫同樣支援取得上下文參考圖（訊息/引用/頭像）
        images_data = []
        capabilities = (
            plugin.generator.adapter.get_capabilities()
            if plugin.generator and plugin.generator.adapter
            else ImageCapability.NONE
        )

        try:
            if capabilities & ImageCapability.IMAGE_TO_IMAGE:
                images_data = await plugin.image_processor.fetch_images_from_event(
                    event
                )

                # 處理頭像引用引數
                avatar_refs = kwargs.get("avatar_references", [])
                if avatar_refs and isinstance(avatar_refs, list):
                    for ref in avatar_refs:
                        if not isinstance(ref, str):
                            continue
                        ref = ref.strip().lower()
                        user_id = None
                        if ref == "self":
                            user_id = str(event.get_self_id())
                        elif ref == "sender":
                            user_id = str(
                                event.get_sender_id() or event.unified_msg_origin
                            )
                        else:
                            # 簡單的 QQ 號校驗（可選）
                            if ref.isdigit():
                                user_id = ref

                        if user_id:
                            avatar_data = await plugin.image_processor.get_avatar(
                                user_id
                            )
                            if avatar_data:
                                images_data.append((avatar_data, "image/jpeg"))
                                logger.info(
                                    f"[ImageGen] 已新增 {user_id} 的頭像作為參考圖"
                                )
        except Exception as e:
            logger.error(f"[ImageGen] 處理參考圖失敗: {e}", exc_info=True)
            # 參考圖處理失敗不影響文生圖流程，記錄日誌繼續執行

        # 生成任務 ID
        task_id = hashlib.md5(
            f"{time.time()}{event.unified_msg_origin}".encode()
        ).hexdigest()[:8]

        # 建立後臺任務進行生圖
        plugin.create_background_task(
            plugin._generate_and_send_image_async(
                prompt=prompt,
                images_data=images_data or None,
                unified_msg_origin=event.unified_msg_origin,
                aspect_ratio=kwargs.get("aspect_ratio")
                or plugin.config_manager.default_aspect_ratio,
                resolution=kwargs.get("resolution")
                or plugin.config_manager.default_resolution,
                task_id=task_id,
            )
        )

        mode = "圖生圖" if images_data else "文生圖"
        return f"✅ 已啟動{mode}任務 (任務ID: {task_id})"


def adjust_tool_parameters(
    tool: ImageGenerationTool, capabilities: ImageCapability
) -> None:
    """根據適配器能力動態調整工具參數。"""
    props = tool.parameters["properties"]

    if not (capabilities & ImageCapability.ASPECT_RATIO):
        if "aspect_ratio" in props:
            del props["aspect_ratio"]
            logger.debug("[ImageGen] 適配器不支援寬高比，已從工具參數中移除")

    if not (capabilities & ImageCapability.RESOLUTION):
        if "resolution" in props:
            del props["resolution"]
            logger.debug("[ImageGen] 適配器不支援解析度，已從工具參數中移除")

    if not (capabilities & ImageCapability.IMAGE_TO_IMAGE):
        if "avatar_references" in props:
            del props["avatar_references"]
            logger.debug("[ImageGen] 適配器不支援參考圖，已從工具參數中移除頭像引用")
