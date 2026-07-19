from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Literal, assert_never

import astrbot.api.message_components as Comp
import pytest

from astrbot_plugin_image_generation.core.image_processor import ImageProcessor
from astrbot_plugin_image_generation.core.llm_tool import _normalize_avatar_references

MIB = 1024 * 1024


class FakeRemoteReader:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, int]] = []
        self.closed = False

    async def read(self, url: str, max_bytes: int) -> bytes | None:
        self.calls.append((url, max_bytes))
        return self.payloads.get(url)

    async def close(self) -> None:
        self.closed = True


class FakeMessageObject:
    def __init__(self, message: list) -> None:
        self.message = message


class FakeEvent:
    def __init__(self, message: list) -> None:
        self.message_obj = FakeMessageObject(message)

    def get_self_id(self) -> str:
        return "42"


def _reference_module() -> ModuleType:
    module_name = "astrbot_plugin_image_generation.core.reference_collector"
    spec = importlib.util.find_spec(module_name)
    assert spec is not None, "IMG-102 RED: core.reference_collector must exist"
    return importlib.import_module(module_name)


@pytest.mark.asyncio
async def test_baseline_local_reference_below_per_file_limit_is_loaded(
    tmp_path: Path,
) -> None:
    # Given
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nsmall")
    processor = ImageProcessor(str(tmp_path / "cache"), 1, 10)

    # When
    loaded = await processor.download_image(str(image_path))

    # Then
    assert loaded == (b"\x89PNG\r\n\x1a\nsmall", "image/png")


@pytest.mark.asyncio
async def test_fifth_raw_source_rejects_before_any_download() -> None:
    # Given
    module = _reference_module()
    reader = FakeRemoteReader({})
    collector = module.ReferenceCollector(30, reader)
    event = FakeEvent(
        [
            Comp.Image(file="https://loopback.invalid/direct"),
            Comp.Reply(
                id="reply",
                chain=[Comp.Image(file="https://loopback.invalid/reply")],
                sender_id="99",
            ),
            Comp.At(qq="7"),
            Comp.At(qq="7"),
        ]
    )
    sources = collector.with_avatar_ids(collector.sources_from_event(event), ("8",))

    # When
    result = await collector.collect(sources)

    # Then
    assert isinstance(result, module.ReferenceRejected), (
        "IMG-102 RED: a fifth message/reply/mention/tool source must reject"
    )
    assert reader.calls == [], (
        "IMG-102 RED: source count rejection must happen before downloads"
    )
    await collector.close()


@pytest.mark.asyncio
async def test_exact_twenty_mib_passes_and_plus_one_rejects_whole_request(
    tmp_path: Path,
) -> None:
    # Given
    module = _reference_module()
    first = tmp_path / "first.bin"
    exact_second = tmp_path / "exact-second.bin"
    oversized_second = tmp_path / "oversized-second.bin"
    first.write_bytes(b"a" * (10 * MIB))
    exact_second.write_bytes(b"b" * (10 * MIB))
    oversized_second.write_bytes(b"c" * (10 * MIB + 1))
    collector = module.ReferenceCollector(
        30,
        FakeRemoteReader({}),
        approved_local_roots=(tmp_path,),
    )

    # When
    exact = await collector.collect(
        (
            module.ReferenceSource.location(str(first)),
            module.ReferenceSource.location(str(exact_second)),
        )
    )
    oversized = await collector.collect(
        (
            module.ReferenceSource.location(str(first)),
            module.ReferenceSource.location(str(oversized_second)),
        )
    )

    # Then
    assert isinstance(exact, module.CollectedReferences), (
        "IMG-102 RED: exactly twenty MiB aggregate must be accepted"
    )
    assert exact.total_bytes == 20 * MIB
    assert isinstance(oversized, module.ReferenceRejected), (
        "IMG-102 RED: twenty MiB plus one byte must reject the whole request"
    )
    await collector.close()


@pytest.mark.asyncio
async def test_duplicate_sources_count_toward_aggregate_limit() -> None:
    # Given
    module = _reference_module()
    url = "https://loopback.invalid/duplicate"
    reader = FakeRemoteReader({url: b"d" * (10 * MIB + 1)})
    collector = module.ReferenceCollector(30, reader)
    duplicate = module.ReferenceSource.location(url)

    # When
    result = await collector.collect((duplicate, duplicate))

    # Then
    assert isinstance(result, module.ReferenceRejected), (
        "IMG-102 RED: duplicate bytes must count toward the aggregate cap"
    )
    assert len(reader.calls) == 2, (
        "IMG-102 RED: duplicate references must not be deduplicated"
    )
    await collector.close()


@pytest.mark.asyncio
async def test_local_oversize_is_rejected_after_stat_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    module = _reference_module()
    path = tmp_path / "too-large.bin"
    with path.open("wb") as file_handle:
        file_handle.truncate(31 * MIB)
    collector = module.ReferenceCollector(
        30,
        FakeRemoteReader({}),
        approved_local_roots=(tmp_path,),
    )

    def fail_read(_path: Path) -> bytes:
        raise AssertionError("IMG-102 RED: oversized local file was fully read")

    monkeypatch.setattr(Path, "read_bytes", fail_read)

    # When
    result = await collector.collect((module.ReferenceSource.location(str(path)),))

    # Then
    assert isinstance(result, module.ReferenceRejected), (
        "IMG-102 RED: per-file limit must reject local data from stat metadata"
    )
    await collector.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("local_kind", ("directory", "missing", "symlink", "fifo"))
async def test_unsafe_local_reference_kind_is_rejected_without_remote_fallback(
    local_kind: Literal["directory", "missing", "symlink", "fifo"],
    tmp_path: Path,
) -> None:
    # Given
    module = _reference_module()
    path = tmp_path / local_kind
    match local_kind:
        case "directory":
            path.mkdir()
        case "missing":
            pass
        case "symlink":
            target = tmp_path / "target.png"
            target.write_bytes(b"local")
            path.symlink_to(target)
        case "fifo":
            if not hasattr(os, "mkfifo"):
                pytest.skip("FIFO creation is unavailable on this platform")
            os.mkfifo(path)
        case unreachable:
            assert_never(unreachable)
    reader = FakeRemoteReader({})
    collector = module.ReferenceCollector(
        30,
        reader,
        approved_local_roots=(tmp_path,),
    )

    # When
    result = await collector.collect((module.ReferenceSource.location(str(path)),))

    # Then
    assert reader.calls == [], (
        f"IMG-102 RED: {local_kind} local references must never use remote loading"
    )
    assert isinstance(result, module.ReferenceRejected), (
        f"IMG-102 RED: {local_kind} local references must reject the whole request"
    )
    assert result.reason == "invalid_reference", (
        "IMG-102 RED: unsafe local references must use the generic rejection"
    )
    await collector.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "open_error", (PermissionError("denied"), OSError("open failed"))
)
async def test_local_descriptor_open_failure_is_rejected_without_remote_fallback(
    open_error: OSError, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    module = _reference_module()
    path = tmp_path / "reference.png"
    path.write_bytes(b"local")
    original_open = os.open

    def fail_target_open(candidate, flags, mode=0o777, *, dir_fd=None):
        if str(candidate) == path.name and dir_fd is not None:
            raise open_error
        return original_open(candidate, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", fail_target_open)
    reader = FakeRemoteReader({})
    collector = module.ReferenceCollector(
        30,
        reader,
        approved_local_roots=(tmp_path,),
    )

    # When
    result = await collector.collect((module.ReferenceSource.location(str(path)),))

    # Then
    assert reader.calls == [], (
        "IMG-102 RED: local stat failures must never use remote loading"
    )
    assert isinstance(result, module.ReferenceRejected), (
        "IMG-102 RED: local stat failures must reject the whole request"
    )
    assert result.reason == "invalid_reference", (
        "IMG-102 RED: local stat failures must use the generic rejection"
    )
    await collector.close()


@pytest.mark.asyncio
async def test_local_read_failure_is_rejected_without_remote_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    module = _reference_module()
    path = tmp_path / "reference.png"
    path.write_bytes(b"local")
    original_read = os.read

    def fail_target_read(descriptor: int, size: int) -> bytes:
        if size:
            raise OSError("read failed")
        return original_read(descriptor, size)

    monkeypatch.setattr(os, "read", fail_target_read)
    reader = FakeRemoteReader({})
    collector = module.ReferenceCollector(
        30,
        reader,
        approved_local_roots=(tmp_path,),
    )

    # When
    try:
        result = await collector.collect((module.ReferenceSource.location(str(path)),))
    except OSError:
        pytest.fail("IMG-102 RED: local read failures must become a generic rejection")

    # Then
    assert reader.calls == [], (
        "IMG-102 RED: local read failures must never use remote loading"
    )
    assert isinstance(result, module.ReferenceRejected), (
        "IMG-102 RED: local read failures must reject the whole request"
    )
    assert result.reason == "invalid_reference", (
        "IMG-102 RED: local read failures must use the generic rejection"
    )
    await collector.close()


def test_tool_avatar_duplicates_remain_raw_references() -> None:
    # Given / When
    normalized = _normalize_avatar_references(["self", " self ", "sender"])

    # Then
    assert normalized == ["self", "self", "sender"], (
        "IMG-102 RED: duplicate tool avatars must count as raw references"
    )
