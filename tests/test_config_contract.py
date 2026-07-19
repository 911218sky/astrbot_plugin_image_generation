from __future__ import annotations

import json
from pathlib import Path

import pytest

from astrbot_plugin_image_generation.core.config_manager import ConfigManager

type JsonValue = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)

SCHEMA_PATH = Path(__file__).parents[1] / "_conf_schema.json"


class RecordingConfig(dict[str, JsonValue]):
    def __init__(self, initial: dict[str, JsonValue] | None = None) -> None:
        super().__init__(initial or {})
        self.save_calls = 0

    def save_config(self) -> None:
        self.save_calls += 1


def provider(name: JsonValue, models: list[str]) -> dict[str, JsonValue]:
    return {
        "__template_key": "gemini",
        "name": name,
        "available_models": models,
    }


def load_config(initial: dict[str, JsonValue]) -> tuple[ConfigManager, RecordingConfig]:
    config = RecordingConfig(initial)
    try:
        manager = ConfigManager(config)
    except (TypeError, ValueError) as exc:
        pytest.fail(
            f"IMG-101 RED: configuration load raised {type(exc).__name__}",
            pytrace=False,
        )
    return manager, config


def test_baseline_config_manager_imports_and_preserves_active_concurrency() -> None:
    config = RecordingConfig()

    manager = ConfigManager(config)

    assert manager.max_concurrent_tasks == 3
    assert manager.max_batch_count == 4
    assert manager.usage_settings.max_image_size_mb == 10
    assert manager.show_generation_info is False
    assert config.save_calls == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(True, True, id="enabled"),
        pytest.param(False, False, id="disabled"),
        pytest.param("true", False, id="string-is-disabled"),
        pytest.param(1, False, id="integer-is-disabled"),
    ],
)
def test_task_started_notice_requires_a_boolean(
    raw: JsonValue, expected: bool
) -> None:
    manager, _ = load_config({"generation": {"show_task_started": raw}})

    assert manager.show_task_started is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(True, True, id="enabled"),
        pytest.param(False, False, id="disabled"),
        pytest.param("false", True, id="string-falls-back-to-enabled"),
        pytest.param(0, True, id="integer-falls-back-to-enabled"),
        pytest.param(None, True, id="null-falls-back-to-enabled"),
    ],
)
def test_llm_tool_setting_requires_a_boolean(raw: JsonValue, expected: bool) -> None:
    manager, _ = load_config({"enable_llm_tool": raw})

    assert manager.enable_llm_tool is expected


def test_baseline_schema_is_valid_json_with_generation_object() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["generation"]["type"] == "object"
    assert schema["user_limits"]["type"] == "object"


def test_missing_values_use_aligned_runtime_defaults() -> None:
    manager, config = load_config({})

    queue = getattr(manager, "max_queued_tasks", None)
    assert queue == 6, "IMG-101 RED: missing queue must default to six"
    assert manager.max_concurrent_tasks == 3, (
        "IMG-101 RED: active concurrency default must remain three"
    )
    assert manager.usage_settings.max_image_size_mb == 10, (
        "IMG-101 RED: missing per-file limit must default to ten"
    )
    assert manager.usage_settings.umo_blacklist == [], (
        "IMG-101 RED: missing blacklist must default to empty"
    )
    assert config.save_calls == 0, "IMG-101 RED: loading defaults must not write config"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(-1, 0, id="negative"),
        pytest.param(101, 100, id="oversize"),
        pytest.param("8", 6, id="string"),
        pytest.param(True, 6, id="bool"),
        pytest.param(None, 6, id="null"),
    ],
)
def test_queue_limit_is_strictly_bounded(raw: JsonValue, expected: int) -> None:
    manager, _ = load_config({"generation": {"max_queued_tasks": raw}})

    actual = getattr(manager, "max_queued_tasks", None)
    assert actual == expected, (
        f"IMG-101 RED: queue {raw!r} must normalize to {expected}, got {actual!r}"
    )


def test_queue_setting_does_not_replace_active_concurrency() -> None:
    manager, _ = load_config(
        {"generation": {"max_concurrent_tasks": 7, "max_queued_tasks": 4}}
    )

    assert manager.max_concurrent_tasks == 7, (
        "IMG-101 RED: queue setting must retain explicit active concurrency"
    )
    assert getattr(manager, "max_queued_tasks", None) == 4, (
        "IMG-101 RED: explicit queue limit must remain independently available"
    )


def test_malformed_numeric_settings_fall_back_without_crashing() -> None:
    manager, _ = load_config(
        {
            "generation": {
                "timeout": "slow",
                "max_concurrent_tasks": "many",
            },
            "user_limits": {
                "rate_limit_seconds": "soon",
                "daily_limit_count": "many",
            },
            "cache": {
                "max_cache_count": "lots",
                "cleanup_interval_hours": "often",
            },
        }
    )

    assert manager.max_concurrent_tasks == 3
    assert manager.usage_settings.rate_limit_seconds == 0
    assert manager.usage_settings.daily_limit_count == 10
    assert manager.cache_settings.max_cache_count == 100
    assert manager.cache_settings.cleanup_interval_hours == 24


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(-1, 1, id="negative"),
        pytest.param(31, 30, id="oversize"),
        pytest.param("12", 10, id="string"),
        pytest.param(True, 10, id="bool"),
        pytest.param(None, 10, id="null"),
    ],
)
def test_per_file_limit_is_strictly_bounded(raw: JsonValue, expected: int) -> None:
    manager, _ = load_config({"user_limits": {"max_image_size_mb": raw}})

    actual = manager.usage_settings.max_image_size_mb
    assert actual == expected, (
        f"IMG-101 RED: per-file limit {raw!r} must normalize to {expected}, "
        f"got {actual!r}"
    )


def test_provider_receives_configured_image_limit() -> None:
    manager, _ = load_config(
        {
            "api_providers": [provider("primary", ["model-a"])],
            "user_limits": {"max_image_size_mb": 7},
        }
    )

    adapter = manager.adapter_config
    assert adapter is not None
    assert adapter.max_image_size_mb == 7


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(-1, 1, id="negative"),
        pytest.param(6, 5, id="oversize"),
        pytest.param("3", 5, id="string"),
        pytest.param(True, 5, id="bool"),
        pytest.param(None, 5, id="null"),
    ],
)
def test_retry_limit_is_strictly_bounded(raw: JsonValue, expected: int) -> None:
    manager, _ = load_config(
        {
            "generation": {"max_retry_attempts": raw},
            "api_providers": [provider("primary", ["model-a"])],
        }
    )

    adapter = manager.adapter_config
    assert adapter is not None, "IMG-101 RED: valid provider must remain available"
    assert adapter.max_retry_attempts == expected, (
        f"IMG-101 RED: retry {raw!r} must normalize to {expected}, "
        f"got {adapter.max_retry_attempts!r}"
    )


def test_schema_defaults_and_ranges_match_runtime_contract() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    generation = schema["generation"]["items"]
    user_limits = schema["user_limits"]["items"]

    queue = generation.get("max_queued_tasks")
    assert queue is not None, "IMG-101 RED: schema must expose the queue limit"
    assert queue["default"] == 6, "IMG-101 RED: schema queue default must be six"
    assert queue["slider"] == {"min": 0, "max": 100, "step": 1}, (
        "IMG-101 RED: schema queue range must be zero through one hundred"
    )
    batch = generation.get("max_batch_count")
    assert batch is not None and batch["default"] == 4
    assert batch["slider"] == {"min": 1, "max": 10, "step": 1}
    assert generation["max_retry_attempts"]["slider"]["max"] == 5, (
        "IMG-101 RED: schema retry maximum must match runtime maximum five"
    )
    assert generation["show_generation_info"]["default"] is False, (
        "IMG-101 RED: generation-info schema default must be false"
    )
    assert user_limits["max_image_size_mb"]["slider"] == {
        "min": 1,
        "max": 30,
        "step": 1,
    }, "IMG-101 RED: schema per-file range must be one through thirty"


@pytest.mark.parametrize("raw", [None, 17], ids=["null", "non_string"])
def test_blacklist_message_invalid_values_use_aligned_default(raw: JsonValue) -> None:
    manager, config = load_config({"user_limits": {"blacklist_block_message": raw}})

    assert manager.usage_settings.blacklist_block_message == (
        "❌ 當前會話未啟用生圖功能"
    ), "IMG-101 RED: invalid blacklist message must use the documented default"
    assert config.save_calls == 0, (
        "IMG-101 RED: blacklist normalization must not write configuration"
    )


def test_blank_provider_name_is_rejected() -> None:
    manager, _ = load_config({"api_providers": [provider("   ", ["blank-model"])]})

    assert manager.adapter_config is None, (
        "IMG-101 RED: whitespace-only provider names must be rejected"
    )


def test_duplicate_normalized_provider_names_are_all_rejected() -> None:
    manager, _ = load_config(
        {
            "api_providers": [
                provider(" repeated ", ["first"]),
                provider("repeated", ["second"]),
                provider("unique", ["actual"]),
            ]
        }
    )

    adapter = manager.adapter_config
    assert adapter is not None, "IMG-101 RED: unique provider must remain available"
    assert adapter.name == "unique", (
        "IMG-101 RED: every ambiguous duplicate provider must be rejected"
    )


def test_stale_current_model_falls_back_to_first_available_model() -> None:
    manager, _ = load_config(
        {
            "generation": {"model": "primary/stale"},
            "api_providers": [provider("primary", ["actual"])],
        }
    )

    adapter = manager.adapter_config
    assert adapter is not None, "IMG-101 RED: valid provider must remain available"
    assert adapter.model == "actual", (
        "IMG-101 RED: current model must be compared with provider availability"
    )


def test_reload_replaces_stale_provider_and_model_state() -> None:
    manager, config = load_config(
        {
            "generation": {"model": "primary/model-a"},
            "api_providers": [provider("primary", ["model-a"])],
        }
    )
    config["generation"] = {"model": "secondary/model-b"}
    config["api_providers"] = [provider("secondary", ["model-b"])]

    manager.reload()

    adapter = manager.adapter_config
    assert adapter is not None
    assert adapter.name == "secondary"
    assert adapter.model == "model-b"
    assert config.save_calls == 0
