"""
插件配置管理模組
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from astrbot.api import logger
from astrbot.core.config.astrbot_config import AstrBotConfig

from .constants import (
    DEFAULT_MAX_IMAGE_SIZE_MB,
    DEFAULT_MAX_RETRY_ATTEMPTS,
)
from .types import AdapterConfig, AdapterType

type JsonValue = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


def _bounded_int(value: JsonValue, default: int, limits: tuple[int, int]) -> int:
    if type(value) is not int:
        return default
    return min(limits[1], max(limits[0], value))


@dataclass
class UsageSettings:
    """使用者使用限制設定。"""

    rate_limit_seconds: int = 0
    enable_daily_limit: bool = False
    daily_limit_count: int = 10
    max_image_size_mb: int = 10
    umo_blacklist: list[str] = field(default_factory=list)
    blacklist_block_message: str = "❌ 當前會話未啟用生圖功能"


@dataclass
class CacheSettings:
    """快取設定。"""

    max_cache_count: int = 100
    cleanup_interval_hours: int = 24


@dataclass
class GenerationSettings:
    """生成設定。"""

    default_aspect_ratio: str = "自動"
    default_resolution: str = "1K"
    max_concurrent_tasks: int = 3
    max_queued_tasks: int = 6
    show_task_started: bool = False
    show_generation_info: bool = False
    show_model_info: bool = False


@dataclass
class PromptAuditSettings:
    """生圖前提示詞稽核設定。"""

    blocked_words: list[str] = field(default_factory=list)
    enable_ai_audit: bool = False
    ai_provider_id: str = ""
    ai_prompt: str = (
        "你是生圖安全審核員。請判斷以下使用者提示詞是否安全，且可用於一般圖像生成。\n"
        "使用者提示詞：{prompt}\n"
        '僅輸出 JSON：{"allow": true/false, "reason": "簡短原因"}。'
    )


@dataclass
class ImageAuditSettings:
    """生圖後圖片稽核設定。"""

    enable_ai_audit: bool = False
    ai_provider_id: str = ""
    ai_prompt: str = (
        "你是圖像內容安全審核員。請判斷輸入圖片是否安全，且可傳送給一般使用者。"
        '僅輸出 JSON：{"allow": true/false, "reason": "簡短原因"}。'
    )


@dataclass
class SafetyAuditSettings:
    """安全稽核總設定。"""

    umo_whitelist: list[str] = field(default_factory=list)
    prompt_audit: PromptAuditSettings = field(default_factory=PromptAuditSettings)
    image_audit: ImageAuditSettings = field(default_factory=ImageAuditSettings)


@dataclass
class PluginConfig:
    """完整的插件配置。"""

    adapter_config: AdapterConfig | None = None
    usage_settings: UsageSettings = field(default_factory=UsageSettings)
    cache_settings: CacheSettings = field(default_factory=CacheSettings)
    generation_settings: GenerationSettings = field(default_factory=GenerationSettings)
    safety_audit_settings: SafetyAuditSettings = field(
        default_factory=SafetyAuditSettings
    )
    presets: dict[str, Any] = field(default_factory=dict)
    enable_llm_tool: bool = True


class ConfigManager:
    """插件配置管理器。"""

    def __init__(self, config: AstrBotConfig):
        self._config = config
        self._plugin_config: PluginConfig = PluginConfig()
        self._all_provider_configs: list[AdapterConfig] = []  # 儲存所有供應商配置
        self.load()

    def load(self) -> PluginConfig:
        """載入並解析插件配置。"""
        gen_cfg = self._config.get("generation", {})
        user_limits_cfg = self._config.get("user_limits", {})
        cache_cfg = self._config.get("cache", {})
        safety_cfg = self._config.get("safety_audit", {})
        api_providers_raw = self._config.get("api_providers", [])

        self._plugin_config.enable_llm_tool = self._config.get("enable_llm_tool", True)
        max_retry_attempts = _bounded_int(
            gen_cfg.get("max_retry_attempts", DEFAULT_MAX_RETRY_ATTEMPTS),
            DEFAULT_MAX_RETRY_ATTEMPTS,
            (1, 5),
        )

        provider_name_counts: dict[str, int] = {}
        for provider_item in api_providers_raw:
            if not isinstance(provider_item, dict):
                continue
            raw_name = provider_item.get("name")
            if not isinstance(raw_name, str) or not raw_name.strip():
                continue
            name = raw_name.strip()
            provider_name_counts[name] = provider_name_counts.get(name, 0) + 1

        # 1. 收集所有供應商配置
        all_provider_configs: list[AdapterConfig] = []
        for provider_item in api_providers_raw:
            if not isinstance(provider_item, dict):
                continue

            adapter_type_str = provider_item.get("__template_key")
            if not adapter_type_str:
                continue

            try:
                adapter_type = AdapterType(adapter_type_str)
            except ValueError:
                logger.warning(f"[ImageGen] 忽略未知適配器型別: {adapter_type_str}")
                continue

            raw_name = provider_item.get("name")
            if not isinstance(raw_name, str):
                continue
            name = raw_name.strip()
            if not name or provider_name_counts.get(name) != 1:
                continue
            base_url = (provider_item.get("base_url") or "").strip()
            api_keys = [k for k in provider_item.get("api_keys", []) if k]
            available_models = provider_item.get("available_models") or []
            proxy = (provider_item.get("proxy") or "").strip() or None
            capability_options = self._parse_capability_options(provider_item)

            # 解析適配器特有配置
            extra: dict[str, Any] = {}
            if adapter_type == AdapterType.OPENAI:
                available_models = [
                    model
                    for model in available_models
                    if isinstance(model, str) and "gpt-image" in model
                ]
                if not available_models:
                    available_models = ["gpt-image-1"]

            all_provider_configs.append(
                AdapterConfig(
                    type=adapter_type,
                    name=name,
                    base_url=self._clean_base_url(base_url),
                    api_keys=api_keys,
                    available_models=available_models,
                    proxy=proxy,
                    timeout=gen_cfg.get("timeout", 180),
                    max_retry_attempts=max_retry_attempts,
                    capability_options=capability_options,
                    extra=extra,
                )
            )

        # 儲存所有供應商配置供後續使用
        self._all_provider_configs = all_provider_configs

        # 2. 取得當前選擇的模型
        model_setting = gen_cfg.get("model", "")

        # 3. 匹配當前適配器
        matched_config = None
        current_model = ""

        if "/" in model_setting:
            target_provider_name, target_model = (
                part.strip() for part in model_setting.split("/", 1)
            )
            for cfg in all_provider_configs:
                if (
                    cfg.name == target_provider_name
                    and target_model in cfg.available_models
                ):
                    matched_config = cfg
                    current_model = target_model
                    break

        # 如果沒匹配到（或者沒設定），取第一個可用的
        if not matched_config and all_provider_configs:
            matched_config = all_provider_configs[0]
            current_model = (
                matched_config.available_models[0]
                if matched_config.available_models
                else ""
            )
            logger.info(
                f"[ImageGen] 未匹配到當前模型配置，預設使用: {matched_config.name}/{current_model}"
            )

        if matched_config:
            self._plugin_config.adapter_config = matched_config
            self._plugin_config.adapter_config.model = current_model
            # 將所有可用模型彙總，供切換指令使用，格式為 "供應商名稱/模型名稱"
            all_available_models = []
            for cfg in all_provider_configs:
                for m in cfg.available_models:
                    all_available_models.append(f"{cfg.name}/{m}")
            self._plugin_config.adapter_config.available_models = all_available_models
        else:
            self._plugin_config.adapter_config = None
            logger.error("[ImageGen] 未找到任何有效的生圖模型配置")

        # 使用者限制設定
        umo_blacklist_raw = user_limits_cfg.get("umo_blacklist", [])
        umo_blacklist: list[str] = []
        if isinstance(umo_blacklist_raw, list):
            umo_blacklist = [
                str(umo).strip() for umo in umo_blacklist_raw if str(umo).strip()
            ]
        blacklist_block_message = user_limits_cfg.get(
            "blacklist_block_message", UsageSettings.blacklist_block_message
        )
        if not isinstance(blacklist_block_message, str):
            blacklist_block_message = UsageSettings.blacklist_block_message
        blacklist_block_message = blacklist_block_message.strip()
        max_image_size_mb = _bounded_int(
            user_limits_cfg.get("max_image_size_mb", DEFAULT_MAX_IMAGE_SIZE_MB),
            DEFAULT_MAX_IMAGE_SIZE_MB,
            (1, 30),
        )

        self._plugin_config.usage_settings = UsageSettings(
            rate_limit_seconds=max(0, user_limits_cfg.get("rate_limit_seconds", 0)),
            max_image_size_mb=max_image_size_mb,
            enable_daily_limit=user_limits_cfg.get("enable_daily_limit", False),
            daily_limit_count=max(1, user_limits_cfg.get("daily_limit_count", 10)),
            umo_blacklist=umo_blacklist,
            blacklist_block_message=blacklist_block_message,
        )

        # 快取設定
        self._plugin_config.cache_settings = CacheSettings(
            max_cache_count=max(1, cache_cfg.get("max_cache_count", 100)),
            cleanup_interval_hours=max(1, cache_cfg.get("cleanup_interval_hours", 24)),
        )

        # 生成設定
        raw_show_task_started = gen_cfg.get("show_task_started", False)
        show_task_started = (
            raw_show_task_started if isinstance(raw_show_task_started, bool) else False
        )

        self._plugin_config.generation_settings = GenerationSettings(
            default_aspect_ratio=gen_cfg.get("default_aspect_ratio", "自動"),
            default_resolution=gen_cfg.get("default_resolution", "1K"),
            max_concurrent_tasks=max(1, gen_cfg.get("max_concurrent_tasks", 3)),
            max_queued_tasks=_bounded_int(
                gen_cfg.get("max_queued_tasks", 6),
                6,
                (0, 100),
            ),
            show_task_started=show_task_started,
            show_generation_info=gen_cfg.get("show_generation_info", False),
            show_model_info=gen_cfg.get("show_model_info", False),
        )

        # 安全稽核設定
        prompt_audit_cfg = safety_cfg.get("prompt_audit", {})
        image_audit_cfg = safety_cfg.get("image_audit", {})
        umo_whitelist_raw = safety_cfg.get("umo_whitelist", [])

        blocked_words_raw = prompt_audit_cfg.get("blocked_words", [])
        blocked_words: list[str] = []
        if isinstance(blocked_words_raw, list):
            blocked_words = [
                str(word).strip() for word in blocked_words_raw if str(word).strip()
            ]

        umo_whitelist: list[str] = []
        if isinstance(umo_whitelist_raw, list):
            umo_whitelist = [
                str(umo).strip() for umo in umo_whitelist_raw if str(umo).strip()
            ]

        self._plugin_config.safety_audit_settings = SafetyAuditSettings(
            umo_whitelist=umo_whitelist,
            prompt_audit=PromptAuditSettings(
                blocked_words=blocked_words,
                enable_ai_audit=bool(prompt_audit_cfg.get("enable_ai_audit", False)),
                ai_provider_id=str(prompt_audit_cfg.get("ai_provider_id", "")).strip(),
                ai_prompt=str(
                    prompt_audit_cfg.get(
                        "ai_prompt",
                        PromptAuditSettings.ai_prompt,
                    )
                ).strip(),
            ),
            image_audit=ImageAuditSettings(
                enable_ai_audit=bool(image_audit_cfg.get("enable_ai_audit", False)),
                ai_provider_id=str(image_audit_cfg.get("ai_provider_id", "")).strip(),
                ai_prompt=str(
                    image_audit_cfg.get(
                        "ai_prompt",
                        ImageAuditSettings.ai_prompt,
                    )
                ).strip(),
            ),
        )

        # 預設
        self._plugin_config.presets = self._load_presets(
            self._config.get("presets", [])
        )

        return self._plugin_config

    def reload(self) -> PluginConfig:
        """重新載入配置。"""
        return self.load()

    def _parse_capability_options(
        self, provider_item: dict[str, Any]
    ) -> dict[str, bool]:
        """解析供應商能力配置（完全由配置驅動）。"""
        raw = provider_item.get("capability_options", [])

        supported_keys = (
            "text_to_image",
            "image_to_image",
            "aspect_ratio",
            "resolution",
        )

        if not isinstance(raw, list):
            logger.warning("[ImageGen] capability_options 配置格式錯誤，已按空列表處理")
            raw = []

        capability_alias_map = {
            "文生圖": "text_to_image",
            "圖生圖": "image_to_image",
            "寬高比": "aspect_ratio",
            "解析度": "resolution",
            # 允許英文值，便於手動配置檔案時相容
            "text_to_image": "text_to_image",
            "image_to_image": "image_to_image",
            "aspect_ratio": "aspect_ratio",
            "resolution": "resolution",
        }

        selected: set[str] = set()
        for item in raw:
            if not isinstance(item, str):
                continue
            key = capability_alias_map.get(item.strip())
            if key:
                selected.add(key)

        return {key: key in selected for key in supported_keys}

    def _clean_base_url(self, url: str) -> str:
        """清理 Base URL，移除末尾的 /v1*"""
        if not url:
            return ""
        url = url.rstrip("/")
        if "/v1" in url:
            url = url.split("/v1", 1)[0]
        return url.rstrip("/")

    def _load_presets(self, presets_config: list[Any]) -> dict[str, Any]:
        """載入預設配置。"""
        presets: dict[str, Any] = {}
        if not isinstance(presets_config, list):
            return presets

        for preset_str in presets_config:
            if isinstance(preset_str, str) and ":" in preset_str:
                name, prompt = preset_str.split(":", 1)
                if name.strip() and prompt.strip():
                    presets[name.strip()] = prompt.strip()
        return presets

    def save_model_setting(self, model: str) -> None:
        """儲存模型設定。"""
        self._config.setdefault("generation", {})["model"] = model
        self._config.save_config()

    def save_preset(self, name: str, content: str) -> None:
        """儲存預設。"""
        self._plugin_config.presets[name] = content
        self._config["presets"] = [
            f"{k}:{v}" for k, v in self._plugin_config.presets.items()
        ]
        self._config.save_config()

    def delete_preset(self, name: str) -> bool:
        """刪除預設，返回是否成功。"""
        if name in self._plugin_config.presets:
            del self._plugin_config.presets[name]
            self._config["presets"] = [
                f"{k}:{v}" for k, v in self._plugin_config.presets.items()
            ]
            self._config.save_config()
            return True
        return False

    # ---------------------- 便捷屬性訪問 ----------------------
    @property
    def adapter_config(self) -> AdapterConfig | None:
        """取得適配器配置。"""
        return self._plugin_config.adapter_config

    @property
    def presets(self) -> dict[str, Any]:
        """取得預設字典。"""
        return self._plugin_config.presets

    @property
    def enable_llm_tool(self) -> bool:
        """是否啟用 LLM 工具。"""
        return self._plugin_config.enable_llm_tool

    @property
    def default_aspect_ratio(self) -> str:
        """預設寬高比。"""
        return self._plugin_config.generation_settings.default_aspect_ratio

    @property
    def default_resolution(self) -> str:
        """預設解析度。"""
        return self._plugin_config.generation_settings.default_resolution

    @property
    def max_concurrent_tasks(self) -> int:
        """最大併發任務數。"""
        return self._plugin_config.generation_settings.max_concurrent_tasks

    @property
    def max_queued_tasks(self) -> int:
        return self._plugin_config.generation_settings.max_queued_tasks

    @property
    def show_task_started(self) -> bool:
        return self._plugin_config.generation_settings.show_task_started

    @property
    def show_generation_info(self) -> bool:
        """是否顯示生成資訊。"""
        return self._plugin_config.generation_settings.show_generation_info

    @property
    def show_model_info(self) -> bool:
        """是否顯示模型資訊。"""
        return self._plugin_config.generation_settings.show_model_info

    @property
    def usage_settings(self) -> UsageSettings:
        """使用者使用限制設定。"""
        return self._plugin_config.usage_settings

    @property
    def cache_settings(self) -> CacheSettings:
        """快取設定。"""
        return self._plugin_config.cache_settings

    @property
    def safety_audit_settings(self) -> SafetyAuditSettings:
        """安全稽核設定。"""
        return self._plugin_config.safety_audit_settings

    # ---------------------- 供應商查詢方法 ----------------------
    def has_provider_type(self, adapter_type: AdapterType) -> bool:
        """檢查配置中是否包含指定型別的供應商。

        Args:
            adapter_type: 要檢查的適配器型別。

        Returns:
            如果配置中包含該型別的供應商則返回 True，否則返回 False。
        """
        return any(cfg.type == adapter_type for cfg in self._all_provider_configs)

    def get_provider_config(self, adapter_type: AdapterType) -> AdapterConfig | None:
        """取得指定型別的供應商配置。

        Args:
            adapter_type: 要取得的適配器型別。

        Returns:
            匹配的供應商配置，如果沒有則返回 None。
        """
        for cfg in self._all_provider_configs:
            if cfg.type == adapter_type:
                return cfg
        return None
