from __future__ import annotations

import io
import mimetypes
import os
import stat
import base64
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol

from astrbot.api import logger

from .image_processor import is_generated_file_publication_hidden

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})
VISIBLE_IMAGE_STATUSES = frozenset({"generated", "success"})
_OPEN_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


class ImageMetadataReader(Protocol):
    def get(self, file_name: str) -> dict[str, object]: ...

    def get_all(self) -> dict[str, dict[str, object]]: ...


class PageImagePresenter(Protocol):
    def _format_size(self, size: int) -> str: ...

    def _format_timestamp(self, value: object) -> str: ...

    def _status_label(self, status: str) -> str: ...

    def _compact_user(self, user: str) -> str: ...

    def _as_float(self, value: object) -> float: ...


@dataclass(frozen=True, slots=True)
class PageImageSnapshot:
    path: Path
    size: int
    modified_at: float
    modified_at_ns: int
    device: int
    inode: int
    metadata: Mapping[str, object]

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def identity(self) -> tuple[int, int, int, int]:
        return self.device, self.inode, self.size, self.modified_at_ns


@dataclass(frozen=True, slots=True)
class PageImagePayload:
    snapshot: PageImageSnapshot
    mime_type: str
    data: bytes


@dataclass(frozen=True, slots=True)
class PageImagePage:
    items: tuple[PageImagePayload, ...]
    page: int
    page_size: int
    total: int
    total_pages: int


def format_page(
    page: PageImagePage, presenter: PageImagePresenter
) -> dict[str, object]:
    items: list[dict[str, object]] = []
    for payload in page.items:
        image = payload.snapshot
        item = {
            "name": image.name,
            "size": image.size,
            "size_label": presenter._format_size(image.size),
            "modified_at": presenter._format_timestamp(image.modified_at),
            "modified_at_ts": image.modified_at,
            "kind": "generated" if image.name.startswith("gen_") else "reference",
            "metadata": format_metadata(image.metadata, presenter),
        }
        if payload.data:
            encoded = base64.b64encode(payload.data).decode("ascii")
            item["preview_data_url"] = f"data:{payload.mime_type};base64,{encoded}"
        items.append(item)
    return {
        "items": items,
        "page": page.page,
        "page_size": page.page_size,
        "total": page.total,
        "total_pages": page.total_pages,
        "has_prev": page.page > 1,
        "has_next": page.page < page.total_pages,
    }


def format_metadata(
    metadata: Mapping[str, object], presenter: PageImagePresenter
) -> dict[str, object]:
    status = str(metadata.get("status") or "")
    return {
        "known": bool(metadata),
        "task_id": str(metadata.get("task_id") or ""),
        "status": status,
        "status_label": presenter._status_label(status) if status else "",
        "prompt": str(metadata.get("prompt") or ""),
        "prompt_preview": str(metadata.get("prompt_preview") or ""),
        "model_full": str(metadata.get("model_full") or ""),
        "provider": str(metadata.get("provider") or ""),
        "model": str(metadata.get("model") or ""),
        "user": str(metadata.get("user") or ""),
        "user_label": presenter._compact_user(str(metadata.get("user") or "")),
        "mode": str(metadata.get("mode") or ""),
        "aspect_ratio": str(metadata.get("aspect_ratio") or ""),
        "resolution": str(metadata.get("resolution") or ""),
        "created_at": presenter._format_timestamp(metadata.get("created_at")),
        "created_at_ts": presenter._as_float(metadata.get("created_at")),
    }


class PageImageCatalog:
    def __init__(self, cache_dir: str | Path, metadata: ImageMetadataReader) -> None:
        self._cache_dir = Path(os.path.abspath(cache_dir))
        self._metadata = metadata

    def list_images(self) -> tuple[PageImageSnapshot, ...]:
        try:
            paths = tuple(self._cache_dir.iterdir())
        except OSError:
            return ()
        records = self._metadata.get_all()
        snapshots = []
        for path in paths:
            payload = self._capture(path.name, records.get(path.name, {}), None)
            if payload is not None:
                snapshots.append(payload.snapshot)
        snapshots.sort(key=lambda item: item.modified_at_ns, reverse=True)
        return tuple(snapshots)

    def count_images(self) -> int:
        return len(self.list_images())

    def inspect(self, file_name: str) -> PageImageSnapshot | None:
        payload = self._capture(file_name, self._metadata.get(file_name), None)
        return payload.snapshot if payload else None

    def read_original(self, file_name: str, max_bytes: int) -> PageImagePayload | None:
        return self._capture(file_name, self._metadata.get(file_name), max_bytes)

    def thumbnail(self, file_name: str, max_bytes: int) -> PageImagePayload | None:
        payload = self.read_original(file_name, max_bytes)
        if payload is None:
            return None
        try:
            from PIL import Image

            with Image.open(io.BytesIO(payload.data)) as image:
                image.thumbnail((320, 220))
                if image.mode not in {"RGB", "L"}:
                    image = image.convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=72, optimize=True)
        except (ImportError, OSError, ValueError) as exc:
            logger.debug(f"[ImageGen] 產生縮圖失敗: {file_name} ({type(exc).__name__})")
            return None
        return PageImagePayload(payload.snapshot, "image/jpeg", buffer.getvalue())

    def page(self, page: int, page_size: int, thumbnail_limit: int) -> PageImagePage:
        snapshots = self.list_images()
        total = len(snapshots)
        total_pages = max(1, (total + page_size - 1) // page_size)
        current = min(max(1, page), total_pages)
        start = (current - 1) * page_size
        items: list[PageImagePayload] = []
        for snapshot in snapshots[start : start + page_size]:
            preview = None
            if snapshot.size <= thumbnail_limit:
                preview = self.thumbnail(snapshot.name, thumbnail_limit)
            if preview is None or preview.snapshot.identity != snapshot.identity:
                preview = PageImagePayload(snapshot, "", b"")
            items.append(preview)
        return PageImagePage(tuple(items), current, page_size, total, total_pages)

    def _capture(
        self,
        file_name: str,
        metadata: Mapping[str, object],
        read_limit: int | None,
    ) -> PageImagePayload | None:
        status = metadata.get("status")
        if not self._valid_name(file_name) or status not in VISIBLE_IMAGE_STATUSES:
            return None
        path = self._cache_dir / file_name
        try:
            before = path.lstat()
            if not self._owned_regular(before):
                return None
            if is_generated_file_publication_hidden(path):
                return None
            descriptor = os.open(path, _OPEN_FLAGS)
        except OSError:
            return None
        try:
            opened = os.fstat(descriptor)
            if not self._same_file(before, opened) or not self._owned_regular(opened):
                return None
            size = opened.st_size
            if read_limit is not None and size > read_limit:
                return None
            data = b"" if read_limit is None else self._read_all(descriptor, size)
            if data is None:
                return None
            after_descriptor = os.fstat(descriptor)
        except OSError:
            return None
        finally:
            os.close(descriptor)
        try:
            after_path = path.lstat()
        except OSError:
            return None
        if not self._same_file(opened, after_descriptor, after_path):
            return None
        if not self._owned_regular(after_path):
            return None
        if is_generated_file_publication_hidden(path):
            return None
        snapshot = PageImageSnapshot(
            path=path,
            size=after_path.st_size,
            modified_at=after_path.st_mtime,
            modified_at_ns=after_path.st_mtime_ns,
            device=after_path.st_dev,
            inode=after_path.st_ino,
            metadata=MappingProxyType(dict(metadata)),
        )
        mime_type = mimetypes.guess_type(file_name)[0] or "image/png"
        return PageImagePayload(snapshot, mime_type, data)

    @staticmethod
    def _valid_name(file_name: str) -> bool:
        return bool(
            file_name
            and "/" not in file_name
            and "\\" not in file_name
            and file_name == Path(file_name).name
            and Path(file_name).suffix.lower() in IMAGE_SUFFIXES
        )

    @staticmethod
    def _owned_regular(info: os.stat_result) -> bool:
        return stat.S_ISREG(info.st_mode) and info.st_nlink == 1

    @staticmethod
    def _same_file(*items: os.stat_result) -> bool:
        identities = {
            (
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_nlink,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )
            for item in items
        }
        return len(identities) == 1

    @staticmethod
    def _read_all(descriptor: int, size: int) -> bytes | None:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return None if os.read(descriptor, 1) else b"".join(chunks)
