from __future__ import annotations

import asyncio

import pytest
from astrbot_plugin_image_generation.core.task_manager import TaskManager


@pytest.mark.asyncio
async def test_background_task_exception_is_consumed_and_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    manager = TaskManager()
    errors: list[str] = []
    monkeypatch.setattr(
        "astrbot_plugin_image_generation.core.task_manager.logger.error",
        lambda message, **_kwargs: errors.append(message),
    )

    async def fail() -> None:
        raise RuntimeError("boom")

    # When
    task = manager.create_task(fail(), name="failing")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Then
    assert task.done()
    assert task not in manager.background_tasks
    assert any("failing" in message and "boom" in message for message in errors)


@pytest.mark.asyncio
async def test_cancel_all_has_a_bound_when_task_ignores_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TaskManager()
    monkeypatch.setattr(
        "astrbot_plugin_image_generation.core.task_manager.BACKGROUND_TASK_CANCEL_TIMEOUT_SECONDS",
        0.01,
    )
    cancellation_seen = asyncio.Event()

    async def slow_cleanup() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await asyncio.sleep(0.1)

    task = manager.create_task(slow_cleanup(), name="slow-cleanup")
    await asyncio.sleep(0)

    started = asyncio.get_running_loop().time()
    await manager.cancel_all()
    elapsed = asyncio.get_running_loop().time() - started

    assert cancellation_seen.is_set()
    assert elapsed < 0.08
    await task


@pytest.mark.asyncio
async def test_replacing_loop_task_keeps_new_task_registered() -> None:
    # Given
    manager = TaskManager()

    async def work() -> None:
        return

    # When
    manager.start_loop_task("refresh", work, interval_seconds=3600)
    old_task = manager._loop_tasks["refresh"]
    manager.start_loop_task("refresh", work, interval_seconds=3600)
    new_task = manager._loop_tasks["refresh"]
    await asyncio.sleep(0)

    # Then
    assert old_task is not new_task
    assert manager._loop_tasks["refresh"] is new_task
    manager.stop_loop_task("refresh")
    await asyncio.gather(old_task, new_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_replacing_daily_task_keeps_new_task_registered() -> None:
    # Given
    manager = TaskManager()

    async def work() -> None:
        return

    # When
    manager.start_daily_task("daily", work, check_interval_seconds=3600)
    old_task = manager._daily_tasks["daily"]
    manager.start_daily_task("daily", work, check_interval_seconds=3600)
    new_task = manager._daily_tasks["daily"]
    await asyncio.sleep(0)

    # Then
    assert old_task is not new_task
    assert manager._daily_tasks["daily"] is new_task
    manager.stop_daily_task("daily")
    await asyncio.gather(old_task, new_task, return_exceptions=True)
