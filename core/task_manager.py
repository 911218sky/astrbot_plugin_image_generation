from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any

from astrbot.api import logger


class TaskManager:
    """統一的任務管理器，管理插件的背景任務與定時任務。"""

    def __init__(self):
        self.background_tasks: set[asyncio.Task] = set()
        self._loop_tasks: dict[str, asyncio.Task] = {}
        self._daily_tasks: dict[str, asyncio.Task] = {}
        self._last_run_dates: dict[str, str] = {}  # 記錄每日任務上次執行的日期
        self._startup_tasks: list[Callable[[], Coroutine[Any, Any, Any]]] = []
        self._startup_completed: bool = False

    def create_task(
        self, coro: Coroutine[Any, Any, Any], name: str | None = None
    ) -> asyncio.Task:
        """建立一個一般背景任務。"""
        task = asyncio.create_task(coro)
        if name:
            task.set_name(name)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    def start_loop_task(
        self,
        name: str,
        coro_func: Callable[[], Coroutine[Any, Any, Any]],
        interval_seconds: float,
        run_immediately: bool = True,
    ) -> None:
        """啟動一個週期性的定時任務。

        Args:
            name: 任務名稱，用於唯一標識和日誌記錄。
            coro_func: 返回協程的函式（任務的主邏輯）。
            interval_seconds: 執行間隔（秒）。
            run_immediately: 是否在啟動時立即執行一次。
        """
        if name in self._loop_tasks:
            self.stop_loop_task(name)

        async def _loop():
            if run_immediately:
                try:
                    await coro_func()
                except Exception as e:
                    logger.error(
                        f"[ImageGen] [TaskManager] 定時任務 {name} 初始執行失敗: {e}",
                        exc_info=True,
                    )

            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    await coro_func()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(
                        f"[ImageGen] [TaskManager] 定時任務 {name} 執行出錯: {e}",
                        exc_info=True,
                    )

        task = asyncio.create_task(_loop(), name=f"loop_{name}")
        self._loop_tasks[name] = task
        self.background_tasks.add(task)
        task.add_done_callback(functools.partial(self._on_loop_task_done, name))
        logger.info(
            f"[ImageGen] [TaskManager] 定時任務 {name} 已啟動 (間隔: {interval_seconds}s)"
        )

    def stop_loop_task(self, name: str) -> None:
        """停止指定的定時任務。"""
        if task := self._loop_tasks.pop(name, None):
            if not task.done():
                task.cancel()
            logger.info(f"[ImageGen] [TaskManager] 定時任務 {name} 已停止")

    def _on_loop_task_done(self, name: str, task: asyncio.Task) -> None:
        """定時任務結束時的回呼。"""
        self.background_tasks.discard(task)
        self._loop_tasks.pop(name, None)

    def register_startup_task(
        self,
        name: str,
        coro_func: Callable[[], Coroutine[Any, Any, Any]],
    ) -> None:
        """註冊一個啟動時執行的任務。

        Args:
            name: 任務名稱，用於日誌記錄。
            coro_func: 返回協程的函式（任務的主邏輯）。
        """
        self._startup_tasks.append((name, coro_func))
        logger.info(f"[ImageGen] [TaskManager] 已註冊啟動任務: {name}")

    async def run_startup_tasks(self) -> None:
        """執行所有註冊的啟動任務。

        此方法應在插件初始化完成後呼叫一次。
        """
        if self._startup_completed:
            logger.warning("[ImageGen] [TaskManager] 啟動任務已執行過，跳過重複執行")
            return

        if not self._startup_tasks:
            logger.info("[ImageGen] [TaskManager] 沒有註冊的啟動任務")
            self._startup_completed = True
            return

        logger.info(
            f"[ImageGen] [TaskManager] 開始執行 {len(self._startup_tasks)} 個啟動任務"
        )

        for name, coro_func in self._startup_tasks:
            try:
                logger.info(f"[ImageGen] [TaskManager] 執行啟動任務: {name}")
                await coro_func()
                logger.info(f"[ImageGen] [TaskManager] 啟動任務 {name} 執行完成")
            except Exception as e:
                logger.error(
                    f"[ImageGen] [TaskManager] 啟動任務 {name} 執行失敗: {e}",
                    exc_info=True,
                )

        self._startup_completed = True
        logger.info("[ImageGen] [TaskManager] 所有啟動任務執行完畢")

    def start_daily_task(
        self,
        name: str,
        coro_func: Callable[[], Coroutine[Any, Any, Any]],
        check_interval_seconds: float = 60.0,
        run_immediately: bool = False,
    ) -> None:
        """啟動一個每日任務，在日期變更時執行。

        Args:
            name: 任務名稱，用於唯一標識和日誌記錄。
            coro_func: 返回協程的函式（任務的主邏輯）。
            check_interval_seconds: 檢查日期變更的間隔（秒），預設 60 秒。
            run_immediately: 是否在啟動時立即執行一次（無論日期）。
        """
        if name in self._daily_tasks:
            self.stop_daily_task(name)

        async def _daily_loop():
            # 初始化上次執行日期
            if run_immediately:
                try:
                    await coro_func()
                    self._last_run_dates[name] = datetime.now().strftime("%Y-%m-%d")
                    logger.info(
                        f"[ImageGen] [TaskManager] 每日任務 {name} 初始執行完成"
                    )
                except Exception as e:
                    logger.error(
                        f"[ImageGen] [TaskManager] 每日任務 {name} 初始執行失敗: {e}",
                        exc_info=True,
                    )
            else:
                # 記錄當前日期，避免啟動當天重複執行
                self._last_run_dates[name] = datetime.now().strftime("%Y-%m-%d")

            while True:
                try:
                    await asyncio.sleep(check_interval_seconds)
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    last_run_date = self._last_run_dates.get(name)

                    if current_date != last_run_date:
                        logger.info(
                            f"[ImageGen] [TaskManager] 檢測到日期變更 ({last_run_date} -> {current_date})，執行每日任務 {name}"
                        )
                        try:
                            await coro_func()
                            self._last_run_dates[name] = current_date
                            logger.info(
                                f"[ImageGen] [TaskManager] 每日任務 {name} 執行完成"
                            )
                        except Exception as e:
                            logger.error(
                                f"[ImageGen] [TaskManager] 每日任務 {name} 執行出錯: {e}",
                                exc_info=True,
                            )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(
                        f"[ImageGen] [TaskManager] 每日任務 {name} 迴圈出錯: {e}",
                        exc_info=True,
                    )

        task = asyncio.create_task(_daily_loop(), name=f"daily_{name}")
        self._daily_tasks[name] = task
        self.background_tasks.add(task)
        task.add_done_callback(functools.partial(self._on_daily_task_done, name))
        logger.info(
            f"[ImageGen] [TaskManager] 每日任務 {name} 已啟動 (檢查間隔: {check_interval_seconds}s)"
        )

    def stop_daily_task(self, name: str) -> None:
        """停止指定的每日任務。"""
        if task := self._daily_tasks.pop(name, None):
            if not task.done():
                task.cancel()
            self._last_run_dates.pop(name, None)
            logger.info(f"[ImageGen] [TaskManager] 每日任務 {name} 已停止")

    def _on_daily_task_done(self, name: str, task: asyncio.Task) -> None:
        """每日任務結束時的回呼。"""
        self.background_tasks.discard(task)
        self._daily_tasks.pop(name, None)
        self._last_run_dates.pop(name, None)

    async def cancel_all(self):
        """取消所有正在執行中的任務。"""
        for task in list(self.background_tasks):
            if not task.done():
                task.cancel()

        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        self.background_tasks.clear()
        self._loop_tasks.clear()
        self._daily_tasks.clear()
        self._last_run_dates.clear()
        logger.info("[ImageGen] [TaskManager] 所有背景任務已取消")
