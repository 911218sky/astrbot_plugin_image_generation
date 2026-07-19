"""
LLM 可呼叫的圖像生成工具模組
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Any

import anyio
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from astrbot.api import logger
from astrbot.api.event import MessageChain
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

from .admission import AdmissionDenied
from .reference_collector import ReferenceRejected
from .types import ImageCapability
from .utils import extract_self_avatar_alias

if TYPE_CHECKING:
    pass


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _normalize_avatar_references(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    refs: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        ref = item.strip().lower()
        if not ref:
            continue
        refs.append(ref)
    return refs


def _resolve_aspect_ratio(
    prompt: str, requested: str | None, fallback: str | None
) -> str | None:
    requested_value = (requested or "").strip()
    if requested_value and requested_value != "自動":
        return requested_value

    normalized = prompt.lower()

    if _contains_any(
        normalized,
        (
            "21:9",
            "ultrawide",
            "ultra-wide",
            "cinematic",
            "panorama",
            "panoramic",
            "全景",
            "超寬",
            "寬銀幕",
            "電影感",
        ),
    ):
        return "21:9"

    if _contains_any(
        normalized,
        (
            "9:16",
            "手機桌布",
            "手機壁紙",
            "手機背景",
            "直式",
            "豎式",
            "縱向",
            "限時動態",
            "story",
            "stories",
            "reels",
            "shorts",
            "tiktok",
            "phone wallpaper",
            "mobile wallpaper",
            "vertical poster",
        ),
    ):
        return "9:16"

    if _contains_any(
        normalized,
        (
            "1:1",
            "頭像",
            "大頭貼",
            "avatar",
            "icon",
            "logo",
            "貼圖",
            "sticker",
            "表情圖",
            "emoji",
            "方形",
        ),
    ):
        return "1:1"

    if _contains_any(
        normalized,
        (
            "16:9",
            "橫幅",
            "banner",
            "封面",
            "縮圖",
            "thumbnail",
            "header",
            "hero image",
            "desktop wallpaper",
            "桌面壁紙",
            "桌布",
            "橫式",
            "youtube",
        ),
    ):
        return "16:9"

    if _contains_any(
        normalized,
        (
            "海報",
            "poster",
            "宣傳圖",
            "角色卡",
            "立繪",
            "全身像",
            "book cover",
        ),
    ):
        return "3:4"

    fallback_value = (fallback or "").strip()
    if fallback_value and fallback_value != "自動":
        return fallback_value
    if requested_value:
        return requested_value
    return fallback_value or None


def _resolve_resolution(
    prompt: str, requested: str | None, fallback: str | None
) -> str | None:
    requested_value = (requested or "").strip()
    if requested_value:
        return requested_value

    normalized = prompt.lower()

    if _contains_any(
        normalized,
        (
            "4k",
            "uhd",
            "超高解析",
            "超高畫質",
            "超清",
            "列印",
            "印刷",
            "print",
        ),
    ):
        return "4K"

    if _contains_any(
        normalized,
        (
            "2k",
            "高解析",
            "高畫質",
            "高清",
            "細節",
            "detail",
            "detailed",
            "桌布",
            "壁紙",
            "海報",
            "banner",
            "封面",
            "縮圖",
            "wallpaper",
        ),
    ):
        return "2K"

    if _contains_any(
        normalized,
        (
            "頭像",
            "大頭貼",
            "avatar",
            "icon",
            "logo",
            "貼圖",
            "sticker",
            "表情圖",
            "emoji",
        ),
    ):
        return "1K"

    fallback_value = (fallback or "").strip()
    return fallback_value or "1K"


@pydantic_dataclass
class ImageGenerationTool(FunctionTool[AstrAgentContext]):
    """LLM 可呼叫的圖像生成工具。"""

    name: str = "generate_image"
    description: str = (
        "生成或編輯圖片。當使用者希望實際產出圖片時就應使用這個工具，"
        "包括繪圖、重繪、風格轉換、製作頭像、貼圖、迷因、海報、縮圖、"
        "人像、表情圖或各種圖片變體。若當前訊息含有圖片、引用圖片或 @ 使用者，"
        "工具會自動把這些內容作為參考圖。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "請填入最終的生圖提示詞，忠實保留使用者的視覺意圖；如果使用者附圖或引用圖片並要求生成、重繪、仿作、改風格或做類似圖片，應直接呼叫工具，參考圖會由工具自動帶入。",
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "目標圖片寬高比；請根據使用情境主動選擇，例如頭像或貼圖用 1:1、手機桌布用 9:16、橫幅或縮圖用 16:9、海報用 3:4。只有完全無法判斷時才使用「自動」。",
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
                    "description": "目標圖片品質或解析度；一般圖片可用 1K，需要更高細節的桌布、海報、封面可優先選 2K，明確要求超高畫質或列印用途時可選 4K。",
                    "enum": ["1K", "2K", "4K"],
                    "default": "1K",
                },
                "avatar_references": {
                    "type": "array",
                    "description": "可選的頭像參考來源，用於圖生圖或角色對齊。可填入 `self` 代表機器人頭像、`sender` 代表當前使用者，或直接填使用者 ID。當使用者寫 @self，或需求明確需要機器人自身形象時，填入 `self`。",
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
        prompt, use_self_avatar = extract_self_avatar_alias(prompt)

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

        admission = await plugin.reserve_generation(event.unified_msg_origin)
        if isinstance(admission, AdmissionDenied):
            return plugin.generation_admission_error(admission)

        admission_ticket = admission
        try:
            prompt_allowed, prompt_reason = await plugin.safety_auditor.audit_prompt(
                prompt, event.unified_msg_origin
            )
            if not prompt_allowed:
                return f"❌ 提示詞審核未通過：{prompt_reason}"

            avatar_refs = _normalize_avatar_references(
                kwargs.get("avatar_references", [])
            )
            if use_self_avatar:
                avatar_refs.append("self")
            avatar_ids: list[str] = []
            for ref in avatar_refs:
                if ref == "self":
                    user_id = str(event.get_self_id()).strip()
                elif ref == "sender":
                    user_id = str(
                        event.get_sender_id() or event.unified_msg_origin
                    ).strip()
                elif ref.isdigit():
                    user_id = ref
                else:
                    user_id = ""
                if user_id:
                    avatar_ids.append(user_id)

            references = await plugin.collect_generation_references(
                event, tuple(avatar_ids)
            )
            if isinstance(references, ReferenceRejected):
                return plugin.generation_reference_error(references)
            images_data = list(references.images)

            task_id = hashlib.md5(
                f"{time.time()}{event.unified_msg_origin}".encode()
            ).hexdigest()[:8]
            aspect_ratio = _resolve_aspect_ratio(
                prompt,
                kwargs.get("aspect_ratio"),
                plugin.config_manager.default_aspect_ratio,
            )
            resolution = _resolve_resolution(
                prompt,
                kwargs.get("resolution"),
                plugin.config_manager.default_resolution,
            )
            logger.info(
                f"[ImageGen] 工具呼叫解析參數: aspect_ratio={aspect_ratio}, resolution={resolution}"
            )

            plugin.create_background_task(
                plugin._generate_and_send_image_async(
                    prompt=prompt,
                    images_data=images_data or None,
                    unified_msg_origin=event.unified_msg_origin,
                    aspect_ratio=aspect_ratio
                    or plugin.config_manager.default_aspect_ratio,
                    resolution=resolution or plugin.config_manager.default_resolution,
                    task_id=task_id,
                    admission_ticket=admission_ticket,
                )
            )
            admission_ticket = None

            mode = "圖生圖" if images_data else "文生圖"
            ref_info = f" [參考圖 {len(images_data)} 張]" if images_data else ""
            notice = f"✅ 已啟動{mode}任務{ref_info}（任務 ID：{task_id}）"
            if plugin.config_manager.show_task_started:
                await plugin.context.send_message(
                    event.unified_msg_origin,
                    MessageChain().message(notice),
                )
                return notice
            return f"已建立{mode}任務，背景生成中。"
        finally:
            if admission_ticket is not None:
                with anyio.CancelScope(shield=True):
                    await plugin.admission_controller.release(admission_ticket)


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
