from __future__ import annotations

from astrbot_plugin_image_generation.core.config_manager import ConfigManager


class RecordingConfig(dict[str, object]):
    def save_config(self) -> None:
        pass


def test_task_started_notice_is_disabled_by_default() -> None:
    manager = ConfigManager(RecordingConfig())

    assert manager.show_task_started is False


def test_task_started_notice_can_be_enabled() -> None:
    manager = ConfigManager(
        RecordingConfig({"generation": {"show_task_started": True}})
    )

    assert manager.show_task_started is True
