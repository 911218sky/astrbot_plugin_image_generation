from __future__ import annotations

import os
import re
from pathlib import Path

import anyio
import pytest
from conftest import (
    AllowingAuditor,
    assert_second_publication_failure_rolls_back,
    create_restart_residue_matrix,
    residue_name,
)

from astrbot_plugin_image_generation.core import safety_auditor as safety_module
from astrbot_plugin_image_generation.core.image_processor import (
    ImageProcessor,
    is_generated_file_publication_hidden,
)
from astrbot_plugin_image_generation.core.safety_auditor import SafetyAuditor

INVALID_AUDIT_REASON = "安全稽核異常：模型回應格式無效"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"allow": true, "reason": "approved"}', (True, "approved")),
        ('{"reason": "blocked", "allow": false}', (False, "blocked")),
    ],
    ids=("allow", "deny"),
)
def test_exact_safety_json_is_accepted(
    response: str, expected: tuple[bool, str]
) -> None:
    # Given
    auditor = object.__new__(SafetyAuditor)

    # When
    result = auditor._parse_audit_response(response)

    # Then
    assert result == expected


@pytest.mark.parametrize(
    "response",
    [
        "",
        "unsafe",
        "not safe",
        "不安全",
        '```json\n{"allow": true, "reason": "approved"}\n```',
        'prefix {"allow": true, "reason": "approved"}',
        '{"allow": true, "reason": "approved"} trailing',
        '{"allow": false, "allow": true, "reason": "duplicate"}',
        '{"allow": "true", "reason": "coerced"}',
        '{"allow": 1, "reason": "coerced"}',
        '{"allow": true, "reason": 1}',
        '{"allow": true}',
        '{"reason": "safe"}',
        '{"allow": true, "reason": "approved", "extra": false}',
        '[true, "allow"]',
        "null",
        "{malformed",
    ],
)
def test_invalid_safety_responses_fail_closed_without_echo(response: str) -> None:
    # Given
    auditor = object.__new__(SafetyAuditor)

    # When
    try:
        result = auditor._parse_audit_response(response)
    except ValueError as exc:
        pytest.fail(f"IMG-104 RED: decoder boundary exception escaped: {exc}")

    # Then
    assert result == (False, INVALID_AUDIT_REASON), (
        f"IMG-104 RED: invalid safety response must fail closed with one generic "
        f"reason; response_class={response!r} result={result!r}"
    )
    if response:
        assert response not in result[1], (
            "IMG-104 RED: invalid response must not be echoed"
        )


def test_oversized_valid_reason_fails_closed_without_echo() -> None:
    # Given
    auditor = object.__new__(SafetyAuditor)
    reason = "provider-controlled-secret" * 100_000
    response = '{"allow": false, "reason": "' + reason + '"}'

    # When
    result = auditor._parse_audit_response(response)

    # Then
    assert result == (False, INVALID_AUDIT_REASON), (
        "IMG-104 RED: oversized valid reasons must fail closed generically"
    )
    assert reason[:100] not in result[1]


@pytest.mark.asyncio
async def test_provider_exception_text_is_generic_and_bounded() -> None:
    secret = "provider-secret-that-must-not-be-echoed"

    class FailingProvider:
        async def text_chat(self, **_kwargs):
            raise RuntimeError(secret)

    class ProviderContext:
        def get_using_provider(self, _origin: str):
            return FailingProvider()

    auditor = object.__new__(SafetyAuditor)
    auditor._context = ProviderContext()
    allowed, reason = await auditor._audit_with_model(
        unified_msg_origin="umo:test",
        review_prompt="review",
        provider_id="",
        image_urls=None,
    )

    assert allowed is False and secret not in reason and len(reason) <= 64, (
        "IMG-104 RED: provider errors must fail closed without exception-text echo"
    )


def test_direct_unlink_failure_renames_to_exact_quarantine_then_unlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocked_file = tmp_path / "gen_blocked.png"
    blocked_file.write_bytes(b"blocked-image-bytes")
    operations: list[tuple[str, Path]] = []
    original_unlink = Path.unlink
    original_replace = os.replace

    def fail_initial_unlink(path: Path, missing_ok: bool = False) -> None:
        operations.append(("unlink", path))
        if path == blocked_file:
            raise OSError("injected direct unlink failure")
        original_unlink(path, missing_ok=missing_ok)

    def track_quarantine_rename(source: Path, target: Path) -> None:
        operations.append(("replace", Path(target)))
        original_replace(source, target)

    monkeypatch.setattr(Path, "unlink", fail_initial_unlink)
    monkeypatch.setattr(safety_module.os, "replace", track_quarantine_rename)
    safety_module.delete_blocked_generated_files([str(blocked_file)])

    quarantine = operations[1][1]
    assert [operation for operation, _path in operations] == [
        "unlink",
        "replace",
        "unlink",
    ], "IMG-104 RED: cleanup must try direct unlink before quarantine fallback"
    assert re.fullmatch(
        rf"\.{re.escape(blocked_file.name)}\.[0-9a-f]{{32}}\.imagegen-blocked",
        quarantine.name,
    ), "IMG-104 RED: quarantine must use the exact unrecognized owned name"
    assert operations == [
        ("unlink", blocked_file),
        ("replace", quarantine),
        ("unlink", quarantine),
    ], "IMG-104 RED: quarantine fallback must complete its final unlink"
    assert not any(tmp_path.iterdir()), (
        "IMG-104 RED: direct unlink fallback must leave zero residue"
    )


def test_restart_reconciliation_removes_only_exact_owned_regular_residue(
    tmp_path: Path,
) -> None:
    matrix = create_restart_residue_matrix(tmp_path)

    ImageProcessor(str(matrix.cache_dir), 1, 10)

    assert not any(path.exists() for path in matrix.exact), (
        "IMG-104 RED: restart must remove exact owned pending and blocked residue"
    )
    assert (
        all(path.exists() for path in matrix.malformed) and matrix.symlink.is_symlink()
    ), "IMG-104 RED: restart must preserve malformed, unrelated, and symlink entries"
    assert matrix.target.read_bytes() == b"target" and matrix.outside.exists(), (
        "IMG-104 RED: reconciliation must stay inside the owned cache root"
    )


def test_restart_reconciliation_continues_after_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    residues = [
        cache_dir / residue_name(f"gen_{index}.png", "blocked", str(index) * 32)
        for index in (1, 2)
    ]
    for residue in residues:
        residue.write_bytes(b"owned-residue")
    original_unlink = Path.unlink

    def fail_one(path: Path, missing_ok: bool = False) -> None:
        if path == residues[0]:
            raise OSError("injected restart cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_one)
    ImageProcessor(str(cache_dir), 1, 10)

    assert residues[0].exists() and not residues[1].exists(), (
        "IMG-104 RED: restart cleanup failure must not stop later residue cleanup"
    )


@pytest.mark.asyncio
async def test_second_publication_error_rolls_back_entire_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await assert_second_publication_failure_rolls_back(
        tmp_path, monkeypatch, OSError("injected second publication error")
    )


@pytest.mark.asyncio
async def test_second_publication_cancellation_propagates_and_rolls_back_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cancellation = anyio.get_cancelled_exc_class()
    await assert_second_publication_failure_rolls_back(
        tmp_path, monkeypatch, cancellation()
    )


@pytest.mark.parametrize("alias_kind", ["duplicate", "hardlink"])
def test_duplicate_or_hardlink_inputs_fail_before_staging_without_data_loss(
    tmp_path: Path, alias_kind: str
) -> None:
    first = tmp_path / "gen_first.png"
    first.write_bytes(b"first-owner")
    paths = [first, first]
    if alias_kind == "hardlink":
        alias = tmp_path / "gen_alias.png"
        alias.hardlink_to(first)
        paths = [first, alias]

    with pytest.raises(OSError):
        safety_module.stage_generated_files_for_audit([str(path) for path in paths])

    assert all(path.read_bytes() == b"first-owner" for path in paths), (
        "IMG-104 RED: duplicate identities must be rejected before destructive staging"
    )
    assert set(tmp_path.iterdir()) == set(paths)


def test_staging_private_name_collision_preserves_both_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public = tmp_path / "gen_owner.png"
    private = tmp_path / residue_name(public.name, "pending")
    public.write_bytes(b"generated-owner")
    private.write_bytes(b"existing-private-owner")
    monkeypatch.setattr(
        safety_module, "make_private_generated_path", lambda *_: private
    )

    with pytest.raises(OSError):
        safety_module.stage_generated_files_for_audit([str(public)])

    assert public.read_bytes() == b"generated-owner"
    assert private.read_bytes() == b"existing-private-owner"


def test_publication_collision_fails_closed_without_overwrite(tmp_path: Path) -> None:
    public = tmp_path / "gen_collision.png"
    public.write_bytes(b"generated-owner")
    staged = safety_module.stage_generated_files_for_audit([str(public)])
    public.write_bytes(b"replacement-owner")

    with pytest.raises(OSError):
        safety_module.publish_audited_generated_files(staged)

    assert public.read_bytes() == b"replacement-owner", (
        "IMG-104 RED: publication must never overwrite a recreated public path"
    )
    assert list(tmp_path.iterdir()) == [public]


@pytest.mark.asyncio
async def test_cancellation_survives_cleanup_failure_and_keeps_batch_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public = tmp_path / "gen_cancel.png"
    public.write_bytes(b"generated-owner")
    cancellation = anyio.get_cancelled_exc_class()
    warnings: list[str] = []

    class CancellingAuditor(AllowingAuditor):
        async def audit_generated_images(self, **_kwargs) -> tuple[bool, str]:
            raise cancellation()

    def fail_cleanup(_files: list[object]) -> None:
        raise safety_module.BlockedFileCleanupError(
            [OSError("injected cleanup failure")]
        )

    monkeypatch.setattr(safety_module, "cleanup_staged_files", fail_cleanup)
    monkeypatch.setattr(safety_module.logger, "warning", warnings.append)
    auditor = object.__new__(CancellingAuditor)

    with pytest.raises(cancellation):
        await auditor.audit_staged_generated_images("prompt", [str(public)], "umo:test")

    assert is_generated_file_publication_hidden(public) and warnings, (
        "IMG-104 RED: incomplete cancellation cleanup must remain hidden and reported"
    )
