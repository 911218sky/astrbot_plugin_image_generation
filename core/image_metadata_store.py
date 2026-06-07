"""Generated image metadata storage.

The image files stay in the existing cache directory. This store only keeps
small searchable facts for the dashboard, keyed by cache file name.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from astrbot.api import logger


class ImageMetadataStore:
    """Tiny JSON store for generated image metadata."""

    def __init__(self, path: Path, max_records: int = 500) -> None:
        self.path = Path(path)
        self.max_records = max(50, max_records)
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def remember_files(
        self,
        file_paths: list[str],
        *,
        task_id: str,
        status: str,
        prompt: str,
        model_full: str,
        provider: str,
        model: str,
        user: str,
        mode: str,
        aspect_ratio: str,
        resolution: str,
    ) -> None:
        """Record metadata for generated cache files."""
        if not file_paths:
            return

        now = time.time()
        with self._lock:
            records = self._read_unlocked()
            for raw_path in file_paths:
                file_name = Path(str(raw_path)).name
                if not file_name:
                    continue
                records[file_name] = {
                    "file_name": file_name,
                    "task_id": task_id,
                    "status": status,
                    "prompt": prompt,
                    "prompt_preview": self._preview(prompt),
                    "model_full": model_full,
                    "provider": provider,
                    "model": model,
                    "user": user,
                    "mode": mode,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "created_at": now,
                }
            self._write_unlocked(self._trim(records))

    def get(self, file_name: str) -> dict[str, Any]:
        """Return metadata for one file, or an empty dict."""
        with self._lock:
            record = self._read_unlocked().get(file_name, {})
        return dict(record) if isinstance(record, dict) else {}

    def get_all(self) -> dict[str, dict[str, Any]]:
        """Return all known metadata records keyed by file name."""
        with self._lock:
            records = self._read_unlocked()
        return {
            name: dict(record)
            for name, record in records.items()
            if isinstance(name, str) and isinstance(record, dict)
        }

    def prune_missing(self, existing_file_names: set[str]) -> None:
        """Drop records whose image file no longer exists."""
        with self._lock:
            records = self._read_unlocked()
            kept = {
                name: record
                for name, record in records.items()
                if name in existing_file_names
            }
            if len(kept) != len(records):
                self._write_unlocked(kept)

    def _read_unlocked(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"[ImageGen] 讀取圖片 metadata 失敗: {exc}")
            return {}

        if not isinstance(data, dict):
            return {}
        records = data.get("images", data)
        return records if isinstance(records, dict) else {}

    def _write_unlocked(self, records: dict[str, dict[str, Any]]) -> None:
        payload = {"version": 1, "updated_at": time.time(), "images": records}
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(self.path.parent),
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temp_path = file.name
                json.dump(payload, file, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.path)
        except (OSError, TypeError) as exc:
            logger.warning(f"[ImageGen] 寫入圖片 metadata 失敗: {exc}")
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _trim(self, records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        records = {
            name: record
            for name, record in records.items()
            if isinstance(name, str) and isinstance(record, dict)
        }
        if len(records) <= self.max_records:
            return records
        ordered = sorted(
            records.items(),
            key=lambda item: float(item[1].get("created_at") or 0),
            reverse=True,
        )
        return dict(ordered[: self.max_records])

    @staticmethod
    def _preview(prompt: str, limit: int = 180) -> str:
        prompt = (prompt or "").strip()
        return prompt[:limit] + ("..." if len(prompt) > limit else "")
