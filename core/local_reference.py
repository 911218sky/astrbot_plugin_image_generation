from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import anyio

READ_CHUNK_SIZE = 64 * 1024
_DESCRIPTOR_LOCAL_READ_SUPPORTED = (
    hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_DIRECTORY")
    and os.open in os.supports_dir_fd
)
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


class LocalReferenceInvalid(Exception):
    def __str__(self) -> str:
        return "local reference is outside approved roots or changed during reading"


@dataclass(frozen=True, slots=True)
class LocalReferenceTooLarge(Exception):
    reason: Literal["per_file_too_large", "aggregate_too_large"]

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class LocalReadLimits:
    per_file: int
    aggregate_remaining: int


@dataclass(frozen=True, slots=True)
class _ApprovedRoot:
    path: Path
    device: int
    inode: int


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
    )


class ApprovedLocalReader:
    def __init__(self, roots: Iterable[str | Path]) -> None:
        approved: dict[Path, _ApprovedRoot] = {}
        for value in roots:
            path = Path(os.path.realpath(os.path.abspath(value)))
            try:
                info = path.stat()
            except OSError:
                continue
            if stat.S_ISDIR(info.st_mode):
                approved[path] = _ApprovedRoot(path, info.st_dev, info.st_ino)
        self._roots = tuple(
            sorted(
                approved.values(), key=lambda item: len(item.path.parts), reverse=True
            )
        )

    async def read(self, path: Path, limits: LocalReadLimits) -> bytes:
        if not _DESCRIPTOR_LOCAL_READ_SUPPORTED:
            raise LocalReferenceInvalid
        return await anyio.to_thread.run_sync(self._read_sync, path, limits)

    def _read_sync(self, path: Path, limits: LocalReadLimits) -> bytes:
        candidate = Path(os.path.abspath(path))
        root = next(
            (
                item
                for item in self._roots
                if candidate != item.path and candidate.is_relative_to(item.path)
            ),
            None,
        )
        if root is None:
            raise LocalReferenceInvalid
        descriptor = self._open_beneath(root, candidate.relative_to(root.path).parts)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise LocalReferenceInvalid
            if before.st_size > limits.per_file:
                raise LocalReferenceTooLarge(reason="per_file_too_large")
            if before.st_size > limits.aggregate_remaining:
                raise LocalReferenceTooLarge(reason="aggregate_too_large")
            payload = bytearray()
            while len(payload) <= limits.aggregate_remaining:
                remaining = limits.aggregate_remaining + 1 - len(payload)
                chunk = os.read(descriptor, min(READ_CHUNK_SIZE, remaining))
                if not chunk:
                    break
                payload.extend(chunk)
            after = os.fstat(descriptor)
            if _identity(before) != _identity(after):
                raise LocalReferenceInvalid
            if len(payload) > limits.aggregate_remaining:
                raise LocalReferenceTooLarge(reason="aggregate_too_large")
            return bytes(payload)
        except OSError as exc:
            raise LocalReferenceInvalid from exc
        finally:
            os.close(descriptor)

    @staticmethod
    def _open_beneath(root: _ApprovedRoot, parts: tuple[str, ...]) -> int:
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise LocalReferenceInvalid
        directories: list[int] = []
        try:
            current = os.open(root.path, _DIRECTORY_FLAGS)
            directories.append(current)
            root_info = os.fstat(current)
            if not stat.S_ISDIR(root_info.st_mode) or (
                root_info.st_dev,
                root_info.st_ino,
            ) != (root.device, root.inode):
                raise LocalReferenceInvalid
            for part in parts[:-1]:
                current = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
                directories.append(current)
                if not stat.S_ISDIR(os.fstat(current).st_mode):
                    raise LocalReferenceInvalid
            return os.open(parts[-1], _FILE_FLAGS, dir_fd=current)
        except OSError as exc:
            raise LocalReferenceInvalid from exc
        finally:
            for directory in reversed(directories):
                os.close(directory)
