from __future__ import annotations

import argparse
import os
import re
from collections.abc import Callable
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import anyio

from astrbot_plugin_image_generation.core.admission import (
    AdmissionController,
    AdmissionDenied,
    AdmissionLimits,
    AdmissionTicket,
)
from astrbot_plugin_image_generation.core.config_manager import ConfigManager
from astrbot_plugin_image_generation.core.image_metadata_store import ImageMetadataStore
from astrbot_plugin_image_generation.core.page_api import PluginPageApi
from astrbot_plugin_image_generation.core.reference_collector import (
    ReferenceCollector,
    ReferenceRejected,
    ReferenceSource,
)
from astrbot_plugin_image_generation.core.safety_auditor import (
    delete_blocked_generated_files,
)

type JsonValue = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


class ProbeFailure(RuntimeError):
    pass


class ProbeConfig(dict[str, JsonValue]):
    def __init__(self, initial: dict[str, JsonValue]) -> None:
        super().__init__(initial)
        self.save_calls = 0

    def save_config(self) -> None:
        self.save_calls += 1


class ProbeLedger:
    def __init__(self) -> None:
        self.counts: dict[tuple[str, str], int] = {}

    def get_usage_count_for(self, user_id: str, date_bucket: str) -> int:
        return self.counts.get((date_bucket, user_id), 0)

    def record_usage_for(self, user_id: str, date_bucket: str) -> None:
        key = (date_bucket, user_id)
        self.counts[key] = self.counts.get(key, 0) + 1


class ProbeRemoteReader:
    def __init__(self) -> None:
        self.calls = 0

    async def read(self, url: str, max_bytes: int) -> bytes | None:
        self.calls += 1
        return None

    async def close(self) -> None:
        return None


def provider() -> dict[str, JsonValue]:
    return {
        "__template_key": "gemini",
        "name": "probe",
        "available_models": ["model-a"],
    }


def selected_config() -> dict[str, JsonValue]:
    return {
        "generation": {"model": "probe/model-a"},
        "api_providers": [provider()],
    }


def run_happy() -> None:
    config = ProbeConfig(selected_config())
    manager = ConfigManager(config)

    if (
        manager.max_queued_tasks != 6
        or manager.max_concurrent_tasks != 3
        or manager.usage_settings.max_image_size_mb != 10
        or config.save_calls != 0
    ):
        raise ProbeFailure("happy config contract mismatch")
    print("IMG-101 PASS case=happy queue=6 per_file_min=1 per_file_max=30")


def run_invalid() -> None:
    raw_config = selected_config()
    raw_config["generation"] = {
        "model": "probe/model-a",
        "max_queued_tasks": "invalid",
        "max_retry_attempts": True,
    }
    raw_config["user_limits"] = {
        "max_image_size_mb": 300,
        "blacklist_block_message": None,
    }
    config = ProbeConfig(raw_config)
    manager = ConfigManager(config)
    adapter = manager.adapter_config

    if (
        manager.max_queued_tasks != 6
        or manager.usage_settings.max_image_size_mb != 30
        or adapter is None
        or adapter.max_retry_attempts != 5
        or config.save_calls != 0
    ):
        raise ProbeFailure("invalid config was not rejected or clamped")
    print("IMG-101 PASS case=invalid rejected_or_clamped=1 config_writes=0")


async def run_admission_happy() -> None:
    controller = AdmissionController(
        AdmissionLimits(active=3, queued=6), ProbeLedger(), date.today
    )
    tickets: list[AdmissionTicket] = []
    for index in range(9):
        result = await controller.reserve(f"probe:{index}", False, 10)
        if not isinstance(result, AdmissionTicket):
            raise ProbeFailure("nine reservations were not admitted")
        tickets.append(result)
    snapshot = await controller.snapshot()
    if (snapshot.active, snapshot.queued) != (3, 6):
        raise ProbeFailure("admission capacity snapshot mismatch")
    for ticket in tickets:
        await controller.release(ticket)
    final_snapshot = await controller.snapshot()
    if (final_snapshot.active, final_snapshot.queued) != (0, 0):
        raise ProbeFailure("admission capacity was not released")
    print("IMG-102 PASS case=happy admitted=9 queued_max=6")


async def run_admission_full() -> None:
    controller = AdmissionController(
        AdmissionLimits(active=1, queued=0), ProbeLedger(), date.today
    )
    first = await controller.reserve("probe:first", False, 10)
    rejected = await controller.reserve("probe:full", False, 10)
    provider_calls = int(isinstance(rejected, AdmissionTicket))
    if not isinstance(first, AdmissionTicket) or not isinstance(
        rejected, AdmissionDenied
    ):
        raise ProbeFailure("full admission did not reject synchronously")

    reader = ProbeRemoteReader()
    collector = ReferenceCollector(30, reader)
    references = tuple(
        ReferenceSource.location(f"https://loopback.invalid/{index}")
        for index in range(5)
    )
    reference_result = await collector.collect(references)
    if (
        not isinstance(reference_result, ReferenceRejected)
        or provider_calls != 0
        or reader.calls != 0
    ):
        raise ProbeFailure("full reference request performed a download")
    await collector.close()
    await controller.release(first)
    print(
        f"IMG-102 PASS case=full provider_calls={provider_calls} downloads={reader.calls}"
    )


def blocked_probe_paths() -> tuple[Path, ImageMetadataStore, PluginPageApi]:
    root = Path(os.environ["ASTRBOT_ROOT"])
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    blocked_path = cache_dir / "gen_blocked.png"
    blocked_path.write_bytes(b"blocked-image-bytes")
    metadata_store = ImageMetadataStore(root / "image_metadata.json")
    page = PluginPageApi(
        SimpleNamespace(cache_dir=cache_dir, image_metadata_store=metadata_store)
    )
    return blocked_path, metadata_store, page


def assert_blocked_probe_clean(
    blocked_path: Path,
    metadata_store: ImageMetadataStore,
    page: PluginPageApi,
) -> tuple[int, int]:
    cache_visible = (
        len(page._images.list_images())
        + page._images.count_images()
        + int(page._images.inspect(blocked_path.name) is not None)
    )
    metadata_paths = len(metadata_store.get_all())
    if blocked_path.exists() or cache_visible or metadata_paths:
        raise ProbeFailure("blocked output remained visible")
    return cache_visible, metadata_paths


def run_blocked_cleanup_happy() -> None:
    blocked_path, metadata_store, page = blocked_probe_paths()
    delete_blocked_generated_files([str(blocked_path)])
    cache_visible, metadata_paths = assert_blocked_probe_clean(
        blocked_path, metadata_store, page
    )
    print(
        "IMG-104 PASS case=happy "
        f"cache_visible={cache_visible} metadata_paths={metadata_paths}"
    )


def run_blocked_cleanup_unlink_failure() -> None:
    blocked_path, metadata_store, page = blocked_probe_paths()
    original_unlink = Path.unlink
    original_replace = os.replace
    direct_failures = 0
    operations: list[tuple[str, Path]] = []

    def fail_direct_unlink(path: Path, missing_ok: bool = False) -> None:
        nonlocal direct_failures
        operations.append(("unlink", path))
        if path == blocked_path:
            direct_failures += 1
            raise OSError("injected direct unlink failure")
        original_unlink(path, missing_ok=missing_ok)

    def track_quarantine_rename(source: Path, target: Path) -> None:
        operations.append(("replace", Path(target)))
        original_replace(source, target)

    with (
        patch.object(Path, "unlink", fail_direct_unlink),
        patch.object(os, "replace", track_quarantine_rename),
    ):
        delete_blocked_generated_files([str(blocked_path)])
    quarantine = operations[1][1]
    exact_name = re.fullmatch(
        rf"\.{re.escape(blocked_path.name)}\.[0-9a-f]{{32}}\.imagegen-blocked",
        quarantine.name,
    )
    expected = [
        ("unlink", blocked_path),
        ("replace", quarantine),
        ("unlink", quarantine),
    ]
    if direct_failures != 1 or not exact_name or operations != expected:
        raise ProbeFailure("direct unlink failure did not complete quarantine deletion")
    cache_visible, metadata_paths = assert_blocked_probe_clean(
        blocked_path, metadata_store, page
    )
    print(
        "IMG-104 PASS case=unlink_failure "
        f"cache_visible={cache_visible} metadata_paths={metadata_paths}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", choices=("config", "admission", "blocked_cleanup"), required=True
    )
    parser.add_argument(
        "--case", choices=("happy", "invalid", "full", "unlink_failure"), required=True
    )
    args = parser.parse_args()

    runners: dict[tuple[str, str], Callable[[], None]] = {
        ("config", "happy"): run_happy,
        ("config", "invalid"): run_invalid,
        ("admission", "happy"): lambda: anyio.run(run_admission_happy),
        ("admission", "full"): lambda: anyio.run(run_admission_full),
        ("blocked_cleanup", "happy"): run_blocked_cleanup_happy,
        ("blocked_cleanup", "unlink_failure"): run_blocked_cleanup_unlink_failure,
    }
    runner = runners.get((args.scenario, args.case))
    if runner is None:
        raise ProbeFailure("scenario and case do not match")
    runner()


if __name__ == "__main__":
    main()
