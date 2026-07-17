from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType

import pytest


class FakeRemoteReader:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def read(self, url: str, _max_bytes: int) -> bytes | None:
        self.calls.append(url)
        return None

    async def close(self) -> None:
        return None


def _module() -> ModuleType:
    return importlib.import_module(
        "astrbot_plugin_image_generation.core.reference_collector"
    )


async def _collect(path: str, root: Path):
    module = _module()
    remote = FakeRemoteReader()
    collector = module.ReferenceCollector(
        1,
        remote,
        approved_local_roots=(root,),
    )
    result = await collector.collect((module.ReferenceSource.location(path),))
    await collector.close()
    return module, remote, result


@pytest.mark.asyncio
async def test_approved_local_file_is_read_from_descriptor(tmp_path: Path) -> None:
    # Given
    root = tmp_path / "approved"
    root.mkdir()
    image = root / "image.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\napproved")

    # When
    module, remote, result = await _collect(str(image), root)

    # Then
    assert isinstance(result, module.CollectedReferences)
    assert result.images == ((b"\x89PNG\r\n\x1a\napproved", "image/png"),)
    assert remote.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("run", range(2))
async def test_outside_and_traversal_paths_are_rejected(
    run: int, tmp_path: Path
) -> None:
    # Given
    root = tmp_path / "approved"
    nested = root / "nested"
    nested.mkdir(parents=True)
    outside = tmp_path / f"outside-{run}.png"
    outside.write_bytes(b"secret")
    traversal = nested / ".." / ".." / outside.name

    # When
    module, remote, direct = await _collect(str(outside), root)
    _, _, escaped = await _collect(str(traversal), root)

    # Then
    assert all(
        isinstance(result, module.ReferenceRejected)
        and result.reason == "invalid_reference"
        for result in (direct, escaped)
    )
    assert remote.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("link_kind", ("final", "parent", "hard"))
async def test_symlink_and_hardlink_references_are_rejected(
    link_kind: str, tmp_path: Path
) -> None:
    # Given
    root = tmp_path / "approved"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"secret")
    match link_kind:
        case "final":
            candidate = root / "image.png"
            candidate.symlink_to(outside)
        case "parent":
            linked_parent = root / "linked"
            linked_parent.symlink_to(tmp_path, target_is_directory=True)
            candidate = linked_parent / outside.name
        case "hard":
            candidate = root / "image.png"
            try:
                os.link(outside, candidate)
            except OSError as exc:
                pytest.skip(f"hardlinks unavailable: {exc}")
        case unreachable:
            raise AssertionError(f"unexpected link kind: {unreachable}")

    # When
    module, remote, result = await _collect(str(candidate), root)

    # Then
    assert isinstance(result, module.ReferenceRejected)
    assert result.reason == "invalid_reference" and remote.calls == []


@pytest.mark.asyncio
async def test_path_swap_after_validation_reads_original_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    root = tmp_path / "approved"
    root.mkdir()
    candidate = root / "image.png"
    candidate.write_bytes(b"original-owner")
    original_stat = Path.stat
    original_open = os.open
    swapped = False

    def swap_after_path_stat(path: Path, *, follow_symlinks: bool = True):
        nonlocal swapped
        info = original_stat(path, follow_symlinks=follow_symlinks)
        if path == candidate and not swapped:
            candidate.unlink()
            candidate.write_bytes(b"replacement-owner")
            swapped = True
        return info

    def swap_after_descriptor_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if str(path) == candidate.name and dir_fd is not None and not swapped:
            candidate.unlink()
            candidate.write_bytes(b"replacement-owner")
            swapped = True
        return descriptor

    monkeypatch.setattr(Path, "stat", swap_after_path_stat)
    monkeypatch.setattr(os, "open", swap_after_descriptor_open)

    # When
    module, _, result = await _collect(str(candidate), root)

    # Then
    assert isinstance(result, module.ReferenceRejected)
    assert result.reason == "invalid_reference"
    assert candidate.read_bytes() == b"replacement-owner"


@pytest.mark.asyncio
async def test_descriptor_open_permission_failure_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    root = tmp_path / "approved"
    root.mkdir()
    candidate = root / "image.png"
    candidate.write_bytes(b"secret")
    original_open = os.open

    def deny_file(path, flags, mode=0o777, *, dir_fd=None):
        if str(path) == candidate.name and dir_fd is not None:
            raise PermissionError("denied")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", deny_file)

    # When
    module, _, result = await _collect(str(candidate), root)

    # Then
    assert isinstance(result, module.ReferenceRejected)
    assert result.reason == "invalid_reference"


@pytest.mark.asyncio
async def test_platform_without_descriptor_guards_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    root = tmp_path / "approved"
    root.mkdir()
    candidate = root / "image.png"
    candidate.write_bytes(b"secret")
    module = _module()
    monkeypatch.setattr(
        module,
        "_DESCRIPTOR_LOCAL_READ_SUPPORTED",
        False,
        raising=False,
    )

    # When
    _, _, result = await _collect(str(candidate), root)

    # Then
    assert isinstance(result, module.ReferenceRejected)
    assert result.reason == "invalid_reference"
