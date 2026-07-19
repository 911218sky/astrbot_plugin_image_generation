from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

import anyio


class UsageLedger(Protocol):
    def get_usage_count_for(self, user_id: str, date_bucket: str) -> int: ...

    def record_usage_for(self, user_id: str, date_bucket: str) -> None: ...


@dataclass(frozen=True, slots=True)
class AdmissionLimits:
    active: int
    queued: int


@dataclass(frozen=True, slots=True)
class AdmissionTicket:
    token: int
    umo: str
    date_bucket: str
    quota_enabled: bool
    daily_limit: int


@dataclass(frozen=True, slots=True)
class AdmissionDenied:
    reason: Literal["capacity", "daily_limit", "closed"]


@dataclass(frozen=True, slots=True)
class AdmissionSnapshot:
    active: int
    queued: int
    pending_quota: int


class _TicketState:
    __slots__ = ("committed", "event", "phase", "ticket")

    def __init__(
        self,
        ticket: AdmissionTicket,
        phase: Literal["active", "queued"],
        event: anyio.Event,
    ) -> None:
        self.ticket = ticket
        self.phase: Literal["active", "queued", "released"] = phase
        self.event = event
        self.committed = False


class AdmissionController:
    def __init__(
        self,
        limits: AdmissionLimits,
        ledger: UsageLedger,
        today_callable: Callable[[], date],
    ) -> None:
        self._limits = AdmissionLimits(
            active=max(0, limits.active), queued=max(0, limits.queued)
        )
        self._ledger = ledger
        self._today = today_callable
        self._lock = anyio.Lock()
        self._states: dict[int, _TicketState] = {}
        self._queue: deque[int] = deque()
        self._pending: dict[tuple[str, str], int] = {}
        self._active = 0
        self._next_token = 0
        self._closed = False

    async def reserve(
        self, umo: str, quota_enabled: bool, daily_limit: int
    ) -> AdmissionTicket | AdmissionDenied:
        date_bucket = self._today().isoformat()
        effective_limit = max(1, daily_limit)
        quota_key = (date_bucket, umo)

        async with self._lock:
            if self._closed:
                return AdmissionDenied(reason="closed")

            if quota_enabled:
                committed = self._ledger.get_usage_count_for(umo, date_bucket)
                pending = self._pending.get(quota_key, 0)
                if committed + pending >= effective_limit:
                    return AdmissionDenied(reason="daily_limit")

            if self._active < self._limits.active:
                phase: Literal["active", "queued"] = "active"
            elif len(self._queue) < self._limits.queued:
                phase = "queued"
            else:
                return AdmissionDenied(reason="capacity")

            self._next_token += 1
            ticket = AdmissionTicket(
                token=self._next_token,
                umo=umo,
                date_bucket=date_bucket,
                quota_enabled=quota_enabled,
                daily_limit=effective_limit,
            )
            state = _TicketState(ticket, phase, anyio.Event())
            self._states[ticket.token] = state

            if phase == "active":
                self._active += 1
                state.event.set()
            else:
                self._queue.append(ticket.token)

            if quota_enabled:
                self._pending[quota_key] = self._pending.get(quota_key, 0) + 1
            return ticket

    async def wait(self, ticket: AdmissionTicket) -> None:
        async with self._lock:
            state = self._states.get(ticket.token)
            if state is None or state.phase == "released":
                return
            event = state.event

        try:
            await event.wait()
        except anyio.get_cancelled_exc_class():
            with anyio.CancelScope(shield=True):
                await self.release(ticket)
            raise

    async def commit(self, ticket: AdmissionTicket) -> None:
        async with self._lock:
            state = self._states.get(ticket.token)
            if state is None or state.committed or state.phase == "released":
                return
            state.committed = True
            if ticket.quota_enabled:
                self._remove_pending(ticket)
                self._ledger.record_usage_for(ticket.umo, ticket.date_bucket)

    async def release(self, ticket: AdmissionTicket) -> None:
        async with self._lock:
            state = self._states.get(ticket.token)
            if state is None or state.phase == "released":
                return

            previous_phase = state.phase
            state.phase = "released"
            if previous_phase == "queued":
                if ticket.token in self._queue:
                    self._queue.remove(ticket.token)
            else:
                self._active -= 1

            if ticket.quota_enabled and not state.committed:
                self._remove_pending(ticket)
            del self._states[ticket.token]

            if previous_phase == "active":
                self._promote_next()

    async def snapshot(self) -> AdmissionSnapshot:
        async with self._lock:
            return AdmissionSnapshot(
                active=self._active,
                queued=len(self._queue),
                pending_quota=sum(self._pending.values()),
            )

    async def close(self) -> None:
        async with self._lock:
            self._closed = True

    def _remove_pending(self, ticket: AdmissionTicket) -> None:
        key = (ticket.date_bucket, ticket.umo)
        remaining = self._pending.get(key, 0) - 1
        if remaining > 0:
            self._pending[key] = remaining
        else:
            self._pending.pop(key, None)

    def _promote_next(self) -> None:
        while self._queue and self._active < self._limits.active:
            token = self._queue.popleft()
            state = self._states.get(token)
            if state is None or state.phase != "queued":
                continue
            state.phase = "active"
            self._active += 1
            state.event.set()
            return
