from __future__ import annotations

import ast
import importlib
import importlib.util
from datetime import date
from types import ModuleType
from pathlib import Path

import anyio
import pytest

from astrbot_plugin_image_generation.core.config_manager import UsageSettings
from astrbot_plugin_image_generation.core.usage_manager import UsageManager


class FakeLedger:
    def __init__(self) -> None:
        self.counts: dict[tuple[str, str], int] = {}
        self.recorded: list[tuple[str, str]] = []

    def get_usage_count_for(self, user_id: str, date_bucket: str) -> int:
        return self.counts.get((date_bucket, user_id), 0)

    def record_usage_for(self, user_id: str, date_bucket: str) -> None:
        key = (date_bucket, user_id)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.recorded.append(key)


class MutableClock:
    def __init__(self, value: date) -> None:
        self.value = value

    def today(self) -> date:
        return self.value


class AsyncBarrier:
    def __init__(self, parties: int) -> None:
        self._remaining = parties
        self._lock = anyio.Lock()
        self._open = anyio.Event()

    async def wait(self) -> None:
        async with self._lock:
            self._remaining -= 1
            if self._remaining == 0:
                self._open.set()
        await self._open.wait()


def _admission_module() -> ModuleType:
    module_name = "astrbot_plugin_image_generation.core.admission"
    spec = importlib.util.find_spec(module_name)
    assert spec is not None, "IMG-102 RED: core.admission must exist"
    return importlib.import_module(module_name)


def _method_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        (item for item in ast.walk(tree) if getattr(item, "name", None) == name),
        None,
    )
    assert node is not None, f"IMG-102 RED: method {name} must remain available"
    return ast.get_source_segment(source, node) or ""


def test_baseline_committed_usage_is_counted_after_recording(tmp_path: Path) -> None:
    # Given
    manager = UsageManager(
        str(tmp_path),
        UsageSettings(enable_daily_limit=True, daily_limit_count=1),
    )

    # When
    before = manager.get_usage_count("umo:baseline")
    manager.record_usage("umo:baseline")

    # Then
    assert before == 0
    assert manager.get_usage_count("umo:baseline") == 1


@pytest.mark.asyncio
async def test_atomic_capacity_admits_three_active_and_six_fifo_queued() -> None:
    # Given
    module = _admission_module()
    controller = module.AdmissionController(
        module.AdmissionLimits(active=3, queued=6),
        FakeLedger(),
        MutableClock(date(2026, 7, 17)).today,
    )
    barrier = AsyncBarrier(9)
    results = []

    async def reserve_one(index: int) -> None:
        await barrier.wait()
        results.append(await controller.reserve(f"umo:{index}", False, 10))

    # When
    async with anyio.create_task_group() as task_group:
        for index in range(9):
            task_group.start_soon(reserve_one, index)
    rejected = await controller.reserve("umo:tenth", False, 10)
    snapshot = await controller.snapshot()

    # Then
    tickets = [item for item in results if isinstance(item, module.AdmissionTicket)]
    assert len(tickets) == 9, "IMG-102 RED: 3+6 must admit exactly nine"
    assert (snapshot.active, snapshot.queued) == (3, 6), (
        "IMG-102 RED: capacity must atomically split active and FIFO queued tickets"
    )
    assert isinstance(rejected, module.AdmissionDenied), (
        "IMG-102 RED: the tenth request must be rejected synchronously"
    )
    for ticket in tickets:
        await controller.release(ticket)


@pytest.mark.asyncio
async def test_queue_zero_accepts_only_an_immediately_free_active_slot() -> None:
    # Given
    module = _admission_module()
    controller = module.AdmissionController(
        module.AdmissionLimits(active=1, queued=0),
        FakeLedger(),
        MutableClock(date(2026, 7, 17)).today,
    )

    # When
    first = await controller.reserve("umo:first", False, 10)
    second = await controller.reserve("umo:second", False, 10)

    # Then
    assert isinstance(first, module.AdmissionTicket)
    assert isinstance(second, module.AdmissionDenied), (
        "IMG-102 RED: queue zero must reject when the active slot is occupied"
    )
    await controller.release(first)
    third = await controller.reserve("umo:third", False, 10)
    assert isinstance(third, module.AdmissionTicket)
    await controller.release(third)


@pytest.mark.asyncio
async def test_daily_limit_one_race_reserves_exactly_one_pending_quota() -> None:
    # Given
    module = _admission_module()
    ledger = FakeLedger()
    controller = module.AdmissionController(
        module.AdmissionLimits(active=3, queued=6),
        ledger,
        MutableClock(date(2026, 7, 17)).today,
    )
    barrier = AsyncBarrier(2)
    results = []

    async def reserve_one() -> None:
        await barrier.wait()
        results.append(await controller.reserve("umo:same", True, 1))

    # When
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(reserve_one)
        task_group.start_soon(reserve_one)

    # Then
    tickets = [item for item in results if isinstance(item, module.AdmissionTicket)]
    assert len(tickets) == 1, (
        "IMG-102 RED: pending daily quota must admit exactly one provider request"
    )
    await controller.release(tickets[0])


@pytest.mark.asyncio
async def test_batch_quota_reservation_refunds_unsuccessful_outputs() -> None:
    module = _admission_module()
    ledger = FakeLedger()
    controller = module.AdmissionController(
        module.AdmissionLimits(active=2, queued=1),
        ledger,
        MutableClock(date(2026, 7, 17)).today,
    )

    ticket = await controller.reserve("umo:batch", True, 3, units=3)
    assert isinstance(ticket, module.AdmissionTicket)
    denied = await controller.reserve("umo:batch", True, 3, units=1)
    assert isinstance(denied, module.AdmissionDenied)

    await controller.commit(ticket, units=2)
    await controller.release(ticket)

    assert ledger.get_usage_count_for("umo:batch", "2026-07-17") == 2
    next_ticket = await controller.reserve("umo:batch", True, 3, units=1)
    assert isinstance(next_ticket, module.AdmissionTicket)
    await controller.release(next_ticket)


@pytest.mark.asyncio
async def test_ticket_snapshot_survives_midnight_and_quota_reload() -> None:
    # Given
    module = _admission_module()
    ledger = FakeLedger()
    clock = MutableClock(date(2026, 7, 17))
    controller = module.AdmissionController(
        module.AdmissionLimits(active=2, queued=1), ledger, clock.today
    )
    old_ticket = await controller.reserve("umo:snapshot", True, 1)
    assert isinstance(old_ticket, module.AdmissionTicket)

    # When
    clock.value = date(2026, 7, 18)
    await controller.commit(old_ticket)
    await controller.release(old_ticket)
    new_ticket = await controller.reserve("umo:snapshot", False, 99)

    # Then
    assert ledger.recorded == [("2026-07-17", "umo:snapshot")], (
        "IMG-102 RED: commit must use the ticket's frozen date and quota policy"
    )
    assert isinstance(new_ticket, module.AdmissionTicket)
    assert (
        new_ticket.date_bucket,
        new_ticket.quota_enabled,
        new_ticket.daily_limit,
    ) == (
        "2026-07-18",
        False,
        99,
    ), "IMG-102 RED: a new request must observe new date and quota settings"
    await controller.release(new_ticket)


@pytest.mark.asyncio
async def test_queued_cancellation_preserves_fifo_and_releases_pending_quota() -> None:
    # Given
    module = _admission_module()
    controller = module.AdmissionController(
        module.AdmissionLimits(active=1, queued=2),
        FakeLedger(),
        MutableClock(date(2026, 7, 17)).today,
    )
    first = await controller.reserve("umo:first", False, 10)
    cancelled = await controller.reserve("umo:cancelled", True, 1)
    survivor = await controller.reserve("umo:survivor", False, 10)
    assert isinstance(first, module.AdmissionTicket)
    assert isinstance(cancelled, module.AdmissionTicket)
    assert isinstance(survivor, module.AdmissionTicket)
    started = anyio.Event()
    finished = anyio.Event()
    scope = anyio.CancelScope()

    async def wait_cancelled() -> None:
        try:
            with scope:
                started.set()
                await controller.wait(cancelled)
        finally:
            finished.set()

    # When
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(wait_cancelled)
        await started.wait()
        scope.cancel()
        await finished.wait()
        await controller.release(first)
        with anyio.fail_after(1):
            await controller.wait(survivor)

    # Then
    snapshot = await controller.snapshot()
    assert (snapshot.active, snapshot.queued, snapshot.pending_quota) == (1, 0, 0), (
        "IMG-102 RED: cancellation must remove only its queued ticket and quota"
    )
    await controller.release(survivor)


def test_command_and_tool_share_admission_and_reference_gate_order() -> None:
    # Given
    root = Path(__file__).parents[1]
    command = _method_source(root / "main.py", "generate_image_command")
    tool = _method_source(root / "core" / "llm_tool.py", "call")

    # When
    command_positions = tuple(
        command.find(token)
        for token in (
            "reserve_generation",
            "audit_prompt",
            "collect_generation_references",
            "plain_result(msg)",
        )
    )
    tool_positions = tuple(
        tool.find(token)
        for token in (
            "reserve_generation",
            "audit_prompt",
            "collect_generation_references",
            "create_background_task",
        )
    )

    # Then
    assert (
        command_positions == tuple(sorted(command_positions))
        and min(command_positions) >= 0
    ), "IMG-102 RED: command must admit before audit/reference/ack"
    assert (
        tool_positions == tuple(sorted(tool_positions)) and min(tool_positions) >= 0
    ), "IMG-102 RED: tool must admit before audit/reference/task creation"
