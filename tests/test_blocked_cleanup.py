from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
from conftest import FakeMetadataStore, create_multi_file_page

from astrbot_plugin_image_generation import main as plugin_main
from astrbot_plugin_image_generation.core.admission import AdmissionTicket
from astrbot_plugin_image_generation.core.image_processor import ImageProcessor
from astrbot_plugin_image_generation.core.page_api import PluginPageApi
from astrbot_plugin_image_generation.core import safety_auditor as safety_module
from astrbot_plugin_image_generation.core.types import GenerationResult

stage_files = safety_module.stage_generated_files_for_audit


def test_legacy_blocked_files_are_excluded_from_all_page_surfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    approved = cache_dir / "gen_approved.png"
    blocked_file = cache_dir / "gen_blocked.png"
    untracked = cache_dir / "gen_untracked.png"
    outside = tmp_path / "gen_outside.png"
    for path in (approved, blocked_file, untracked, outside):
        path.write_bytes(b"generated-owner")
    blocked_link = cache_dir / "gen_blocked_link.png"
    outside_link = cache_dir / "gen_outside_link.png"
    hardlink = cache_dir / "gen_hardlink.png"
    blocked_link.symlink_to(blocked_file)
    outside_link.symlink_to(outside)
    os.link(outside, hardlink)
    generated = (approved, blocked_link, outside_link, hardlink)
    store = FakeMetadataStore(
        {path.name: {"status": "generated"} for path in generated}
    )
    store.records[blocked_file.name] = {"status": "blocked"}
    page = PluginPageApi(
        SimpleNamespace(cache_dir=cache_dir, image_metadata_store=store)
    )
    catalog = getattr(page, "_images", None)
    assert catalog is not None, "IMG-104 RED: Page image reads need an owned catalog"
    original_open = os.open
    replace_owner = False

    def intercept_open(path: str | bytes | os.PathLike[str], flags: int) -> int:
        assert flags & os.O_NOFOLLOW
        descriptor = original_open(path, flags)
        if replace_owner and Path(path) == approved:
            approved.unlink()
            approved.write_bytes(b"replacement-owner")
        return descriptor

    monkeypatch.setattr(os, "open", intercept_open)

    # When
    image_page = page._images.page(1, 12, 1024)
    payload = catalog.read_original(approved.name, 1024)

    # Then
    assert page._images.count_images() == image_page.total == 1
    assert all(
        page._images.inspect(path.name) is None
        for path in (*generated[1:], blocked_file, untracked)
    )
    assert [item.snapshot.name for item in image_page.items] == [approved.name]
    assert payload is not None and payload.data == b"generated-owner"
    replace_owner = True
    assert catalog.read_original(approved.name, 1024) is None
    assert approved.read_bytes() == b"replacement-owner"


class FakeGenerator:
    async def generate(self, _request) -> GenerationResult:
        return GenerationResult(images=[b"blocked-image-bytes"], error=None)


class FakeImageProcessor:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    def save_generated_image(self, _task_id: str, data: bytes) -> str:
        self.output_path.write_bytes(data)
        return str(self.output_path)


class FakeSafetyAuditor(safety_module.SafetyAuditor):
    async def audit_generated_images(self, **_kwargs) -> tuple[bool, str]:
        return False, "blocked"


class BarrierSafetyAuditor(safety_module.SafetyAuditor):
    def __init__(self) -> None:
        self.entered = anyio.Event()
        self.release = anyio.Event()
        self.paths: list[str] = []

    async def audit_generated_images(
        self, *, image_paths: list[str], **_kwargs
    ) -> tuple[bool, str]:
        self.paths = image_paths
        self.entered.set()
        await self.release.wait()
        return False, "blocked"


class FakeContext:
    async def send_message(self, _umo: str, _message) -> None:
        pass


def _build_barrier_generation(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    output_path = cache_dir / "gen_pending.png"
    auditor = BarrierSafetyAuditor()
    plugin = object.__new__(plugin_main.ImageGenerationPlugin)
    plugin.generator = FakeGenerator()
    plugin.image_processor = FakeImageProcessor(output_path)
    plugin.safety_auditor = auditor
    plugin.context = FakeContext()
    plugin._remember_generation_task = lambda *_args, **_kwargs: None
    page = PluginPageApi(
        SimpleNamespace(cache_dir=cache_dir, image_metadata_store=FakeMetadataStore({}))
    )
    return plugin, auditor, page, output_path


def _assert_page_hidden(page: PluginPageApi, names: list[str]) -> None:
    surfaces = page._images.list_images(), page._images.count_images()
    resolved = tuple(page._images.inspect(name) for name in names)
    assert (*surfaces, resolved) == ((), 0, (None,) * len(names)), (
        "IMG-104 RED: unapproved files must be absent from Page list, count, "
        "and direct access surfaces"
    )


async def _run_generation(plugin: plugin_main.ImageGenerationPlugin) -> None:
    ticket = AdmissionTicket(1, "umo:test", "2026-07-17", False, 1)
    await plugin._do_generate_and_send(
        "prompt", "umo:test", [], None, None, "task", ticket
    )


@pytest.mark.asyncio
async def test_pending_audit_never_exposes_recognized_output(tmp_path: Path) -> None:
    # Given
    plugin, auditor, page, output_path = _build_barrier_generation(tmp_path)

    # When
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(_run_generation, plugin)
        await auditor.entered.wait()

        # Then
        _assert_page_hidden(page, [output_path.name])
        assert all(
            Path(path).suffix.lower() not in PluginPageApi.IMAGE_SUFFIXES
            for path in auditor.paths
        ), "IMG-104 RED: audit must receive staged, unrecognized paths"
        auditor.release.set()


def test_page_excludes_restart_residue_before_and_during_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    residues = [
        tmp_path / f".gen_{state}.png.{index * 32}.imagegen-{state}"
        for index, state in (("1", "pending"), ("2", "blocked"))
    ]
    for residue in residues:
        residue.write_bytes(b"owned-residue")
    page = PluginPageApi(
        SimpleNamespace(cache_dir=tmp_path, image_metadata_store=FakeMetadataStore({}))
    )
    original_unlink = Path.unlink
    observed: list[Path] = []

    def observe_unlink(path: Path, missing_ok: bool = False) -> None:
        if path in residues:
            _assert_page_hidden(page, [item.name for item in residues])
            observed.append(path)
        original_unlink(path, missing_ok=missing_ok)

    _assert_page_hidden(page, [item.name for item in residues])
    monkeypatch.setattr(Path, "unlink", observe_unlink)

    # When
    ImageProcessor(str(tmp_path), 1, 10)

    # Then
    assert observed == residues and not any(tmp_path.iterdir()), (
        "IMG-104 RED: Page must exclude exact residue before and throughout restart "
        "reconciliation"
    )


def test_multi_file_publication_has_no_partial_page_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    files, page = create_multi_file_page(tmp_path)
    staged = stage_files([str(path) for path in files])
    original_link = os.link
    visible_during_publish: list[list[Path]] = []

    def observe_publication(source: Path, target: Path, **kwargs) -> None:
        original_link(source, target, **kwargs)
        if Path(target) in files:
            _assert_page_hidden(page, [path.name for path in files])
            visible_during_publish.append([])

    monkeypatch.setattr(safety_module.os, "link", observe_publication)

    # When
    published = safety_module.publish_audited_generated_files(staged)

    # Then
    assert visible_during_publish == [[], [], []], (
        "IMG-104 RED: Page must expose no partial batch during publication"
    )
    rows = [snapshot.path for snapshot in page._images.list_images()]
    assert published == [str(path) for path in files] and sorted(rows) == sorted(
        files
    ), "IMG-104 RED: successful publication must expose every file exactly once"
    assert set(tmp_path.iterdir()) == set(files)


def test_publication_rollback_preserves_later_owner_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public = [tmp_path / f"gen_replace_{index}.png" for index in range(2)]
    for path in public:
        path.write_bytes(b"generated-owner")
    staged = stage_files([str(path) for path in public])
    original_link = os.link

    def replace_then_fail(source: Path, target: Path, **kwargs) -> None:
        if Path(target) == public[0]:
            original_link(source, target, **kwargs)
            return
        public[0].unlink()
        public[0].write_bytes(b"replacement-owner")
        raise OSError("injected second publication failure")

    monkeypatch.setattr(safety_module.os, "link", replace_then_fail)
    with pytest.raises(OSError, match="second publication failure"):
        safety_module.publish_audited_generated_files(staged)

    assert public[0].read_bytes() == b"replacement-owner" and not public[1].exists()
    assert list(tmp_path.iterdir()) == [public[0]], (
        "IMG-104 RED: rollback must preserve replacements and remove only owned bytes"
    )


@pytest.mark.parametrize("failure_index", range(3))
@pytest.mark.parametrize("error_type", [OSError, InterruptedError])
def test_cleanup_final_unlink_failure_excludes_all_files_and_reports_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_index: int,
    error_type: type[OSError],
) -> None:
    # Given
    files, page = create_multi_file_page(tmp_path)
    original_unlink = Path.unlink

    def interrupt_one_tombstone(path: Path, missing_ok: bool = False) -> None:
        if path in files:
            raise OSError("force quarantine path")
        if f"gen_{failure_index}.png" in path.name:
            raise error_type("injected final unlink failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", interrupt_one_tombstone)

    # When
    with pytest.raises(OSError, match="injected final unlink failure"):
        safety_module.delete_blocked_generated_files([str(path) for path in files])

    # Then
    _assert_page_hidden(page, [path.name for path in files])
    remaining = list(tmp_path.iterdir())
    assert len(remaining) == 1 and remaining[0].suffix not in page.IMAGE_SUFFIXES, (
        "IMG-104 RED: cleanup must continue after interruption and must not "
        "report success while tombstoned bytes remain"
    )


@pytest.mark.asyncio
async def test_blocked_generation_deletes_before_zero_path_metadata(
    tmp_path: Path,
) -> None:
    # Given
    output_path = tmp_path / "gen_blocked.png"
    plugin = object.__new__(plugin_main.ImageGenerationPlugin)
    plugin.generator = FakeGenerator()
    plugin.image_processor = FakeImageProcessor(output_path)
    plugin.safety_auditor = object.__new__(FakeSafetyAuditor)
    plugin.context = FakeContext()
    metadata_calls: list[list[str]] = []
    task_calls: list[tuple[str, dict[str, str | int]]] = []

    def remember_metadata(paths: list[str], **_kwargs) -> None:
        metadata_calls.append(paths)

    def remember_task(_task_id: str, status: str, **extra) -> None:
        task_calls.append((status, extra))

    plugin._remember_generated_image_metadata = remember_metadata
    plugin._remember_generation_task = remember_task
    ticket = AdmissionTicket(1, "umo:test", "2026-07-17", False, 1)

    # When
    await plugin._do_generate_and_send(
        "prompt", "umo:test", [], None, None, "task", ticket
    )

    # Then
    assert not output_path.exists(), "IMG-104 RED: blocked bytes survived"
    assert metadata_calls == [], "IMG-104 RED: blocked metadata must contain zero paths"
    expected = [("blocked", {"error": "blocked", "image_count": 0})]
    assert task_calls == expected, "IMG-104 RED: blocked task retained file paths"
