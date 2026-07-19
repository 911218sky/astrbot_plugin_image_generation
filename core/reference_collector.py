from __future__ import annotations

import mimetypes
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, assert_never
from urllib.parse import quote, urlsplit
from urllib.request import url2pathname

import astrbot.api.message_components as Comp
from astrbot.api.event import AstrMessageEvent

from .local_reference import (
    ApprovedLocalReader,
    LocalReadLimits,
    LocalReferenceInvalid,
    LocalReferenceTooLarge,
    _DESCRIPTOR_LOCAL_READ_SUPPORTED,
)
from .reference_transport import (
    AiohttpRemoteReader,
    RemoteLimitExceeded,
    RemoteReferenceDenied,
)

MIB = 1024 * 1024
MAX_REFERENCE_COUNT = 4
MAX_AGGREGATE_BYTES = 20 * MIB


class RemoteReader(Protocol):
    async def read(self, url: str, max_bytes: int) -> bytes | None: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ReferenceSource:
    kind: Literal["location", "avatar"]
    value: str

    @classmethod
    def location(cls, value: str) -> ReferenceSource:
        return cls(kind="location", value=value)

    @classmethod
    def avatar(cls, user_id: str) -> ReferenceSource:
        return cls(kind="avatar", value=user_id)


@dataclass(frozen=True, slots=True)
class CollectedReferences:
    images: tuple[tuple[bytes, str], ...]
    total_bytes: int


@dataclass(frozen=True, slots=True)
class ReferenceRejected:
    reason: Literal[
        "too_many",
        "per_file_too_large",
        "aggregate_too_large",
        "invalid_reference",
    ]


class ReferenceCollector:
    def __init__(
        self,
        per_file_limit_mb: int,
        remote_reader: RemoteReader | None = None,
        *,
        approved_local_roots: Iterable[str | Path] = (),
    ) -> None:
        self._per_file_limit = max(1, per_file_limit_mb) * MIB
        self._remote_reader = remote_reader or AiohttpRemoteReader()
        self._local_reader = ApprovedLocalReader(approved_local_roots)

    def sources_from_event(
        self, event: AstrMessageEvent
    ) -> tuple[ReferenceSource, ...]:
        message_obj = getattr(event, "message_obj", None)
        components = getattr(message_obj, "message", None)
        if not components:
            return ()

        reply_sender_id: str | None = None
        at_counts: dict[str, int] = {}
        for component in components:
            if isinstance(component, Comp.Reply) and component.sender_id:
                reply_sender_id = str(component.sender_id)
            elif isinstance(component, Comp.At) and component.qq != "all":
                user_id = str(component.qq)
                at_counts[user_id] = at_counts.get(user_id, 0) + 1

        self_id = str(event.get_self_id()).strip()
        sources: list[ReferenceSource] = []
        for component in components:
            if isinstance(component, Comp.Image):
                location = (
                    component.url or getattr(component, "path", None) or component.file
                )
                if location:
                    sources.append(ReferenceSource.location(str(location)))
            elif isinstance(component, Comp.Reply) and component.chain:
                for nested in component.chain:
                    if isinstance(nested, Comp.Image):
                        location = (
                            nested.url or getattr(nested, "path", None) or nested.file
                        )
                        if location:
                            sources.append(ReferenceSource.location(str(location)))
            elif isinstance(component, Comp.At) and component.qq != "all":
                user_id = str(component.qq)
                auto_reply_mention = (
                    reply_sender_id == user_id and at_counts.get(user_id, 0) == 1
                )
                auto_self_mention = (
                    self_id == user_id and at_counts.get(user_id, 0) == 1
                )
                if not auto_reply_mention and not auto_self_mention:
                    sources.append(ReferenceSource.avatar(user_id))
        return tuple(sources)

    def with_avatar_ids(
        self, sources: Iterable[ReferenceSource], avatar_ids: Iterable[str]
    ) -> tuple[ReferenceSource, ...]:
        appended = list(sources)
        appended.extend(
            ReferenceSource.avatar(user_id)
            for raw_user_id in avatar_ids
            if (user_id := str(raw_user_id).strip())
        )
        return tuple(appended)

    async def collect(
        self, sources: Iterable[ReferenceSource]
    ) -> CollectedReferences | ReferenceRejected:
        raw_sources = tuple(sources)
        if len(raw_sources) > MAX_REFERENCE_COUNT:
            return ReferenceRejected(reason="too_many")

        images: list[tuple[bytes, str]] = []
        total_bytes = 0
        for source in raw_sources:
            remaining = MAX_AGGREGATE_BYTES - total_bytes
            read_limit = min(self._per_file_limit, remaining)
            try:
                read_result = await self._read(source, read_limit)
            except RemoteLimitExceeded:
                reason = (
                    "per_file_too_large"
                    if read_limit == self._per_file_limit
                    else "aggregate_too_large"
                )
                return ReferenceRejected(reason=reason)
            match read_result:
                case ReferenceRejected():
                    return read_result
                case None:
                    continue
                case bytes() as payload:
                    pass
                case unreachable:
                    assert_never(unreachable)
            if len(payload) > self._per_file_limit:
                return ReferenceRejected(reason="per_file_too_large")
            if len(payload) > remaining:
                return ReferenceRejected(reason="aggregate_too_large")
            images.append((payload, self._mime_type(source, payload)))
            total_bytes += len(payload)
        return CollectedReferences(images=tuple(images), total_bytes=total_bytes)

    async def close(self) -> None:
        await self._remote_reader.close()

    async def _read(
        self, source: ReferenceSource, read_limit: int
    ) -> bytes | ReferenceRejected | None:
        match source.kind:
            case "avatar":
                user_id = quote(source.value, safe="")
                url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
                try:
                    return await self._remote_reader.read(url, read_limit)
                except RemoteReferenceDenied:
                    return ReferenceRejected(reason="invalid_reference")
            case "location":
                pass
            case unreachable:
                assert_never(unreachable)

        try:
            location = urlsplit(source.value)
            remote_port = location.port
        except ValueError:
            location = None
            remote_port = None
        if (
            location is not None
            and location.scheme in {"http", "https"}
            and location.hostname is not None
            and (remote_port is None or 0 <= remote_port <= 65535)
        ):
            try:
                return await self._remote_reader.read(source.value, read_limit)
            except RemoteReferenceDenied:
                return ReferenceRejected(reason="invalid_reference")

        path = self._local_path(source.value, location)
        if path is None or not _DESCRIPTOR_LOCAL_READ_SUPPORTED:
            return ReferenceRejected(reason="invalid_reference")
        try:
            return await self._local_reader.read(
                path,
                LocalReadLimits(
                    per_file=self._per_file_limit,
                    aggregate_remaining=read_limit,
                ),
            )
        except LocalReferenceTooLarge as exc:
            return ReferenceRejected(reason=exc.reason)
        except LocalReferenceInvalid:
            return ReferenceRejected(reason="invalid_reference")

    @staticmethod
    def _local_path(source: str, location) -> Path | None:
        if location is None or not location.scheme:
            return Path(source)
        if location.scheme != "file" or location.hostname not in {
            None,
            "",
            "localhost",
        }:
            return None
        return Path(url2pathname(location.path))

    @staticmethod
    def _mime_type(source: ReferenceSource, payload: bytes) -> str:
        if payload.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if payload.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
            return "image/webp"
        if payload.startswith(b"\xff\xd8\xff") or source.kind == "avatar":
            return "image/jpeg"
        guessed, _ = mimetypes.guess_type(source.value)
        return guessed or "image/jpeg"
