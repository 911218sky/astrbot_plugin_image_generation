from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import TypeAlias

from astrbot.api import logger
from astrbot.api.star import Context

from .config_manager import ConfigManager
from .image_processor import (
    BlockedFileCleanupError,
    StagedGeneratedFile,
    begin_generated_file_publication,
    cleanup_staged_files,
    delete_blocked_generated_files,
    end_generated_file_publication,
    make_private_generated_path,
    owns_staged_path,
)

_JsonValue: TypeAlias = (
    str | int | float | bool | None | list["_JsonValue"] | dict[str, "_JsonValue"]
)
_INVALID_AUDIT_REASON = "安全稽核異常：模型回應格式無效"
_MODEL_CALL_FAILURE = "安全稽核異常：模型呼叫失敗"
_MAX_AUDIT_RESPONSE_CHARS = 16_384
_MAX_AUDIT_REASON_CHARS = 512


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, _JsonValue]],
) -> dict[str, _JsonValue]:
    payload: dict[str, _JsonValue] = {}
    for key, value in pairs:
        if key in payload:
            raise _DuplicateJsonKeyError(key)
        payload[key] = value
    return payload


def _rollback_staged_files(staged_files: list[StagedGeneratedFile], phase: str) -> bool:
    try:
        cleanup_staged_files(staged_files)
    except BlockedFileCleanupError as error:
        logger.warning(f"[ImageGen] {phase}回滾未完成: failures={len(error.failures)}")
        return False
    return True


def stage_generated_files_for_audit(
    file_paths: list[str],
) -> list[StagedGeneratedFile]:
    public_paths = [Path(file_path) for file_path in file_paths]
    if len(set(map(os.path.abspath, public_paths))) != len(public_paths):
        raise FileExistsError("duplicate generated file path")
    identities: list[tuple[int, int]] = []
    for path in public_paths:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OSError("generated file must be an owned regular file")
        identities.append((info.st_dev, info.st_ino))

    staged_files: list[StagedGeneratedFile] = []
    begin_generated_file_publication(public_paths)
    succeeded = False
    try:
        for public_path, identity in zip(public_paths, identities, strict=True):
            audit_path = make_private_generated_path(public_path, "pending")
            os.link(public_path, audit_path, follow_symlinks=False)
            item = StagedGeneratedFile(public_path, audit_path, *identity)
            staged_files.append(item)
            if not owns_staged_path(public_path, item):
                raise OSError("generated file ownership changed during staging")
            public_path.unlink()
        succeeded = True
        return staged_files
    finally:
        if not succeeded and _rollback_staged_files(staged_files, "暫存"):
            end_generated_file_publication(public_paths)


def publish_audited_generated_files(
    staged_files: list[StagedGeneratedFile],
) -> list[str]:
    public_paths = [item.public_path for item in staged_files]
    succeeded = False
    try:
        for item in staged_files:
            if not owns_staged_path(item.audit_path, item):
                raise OSError("staged generated file ownership changed")
            os.link(item.audit_path, item.public_path, follow_symlinks=False)
            if not owns_staged_path(item.public_path, item):
                raise OSError("published generated file ownership changed")
        for item in staged_files:
            delete_blocked_generated_files([str(item.audit_path)])
        succeeded = True
        return [str(path) for path in public_paths]
    finally:
        if succeeded:
            end_generated_file_publication(public_paths)
        elif _rollback_staged_files(staged_files, "發佈"):
            end_generated_file_publication(public_paths)


class SafetyAuditor:
    PROMPT_PLACEHOLDER = "{prompt}"

    def __init__(self, context: Context, config_manager: ConfigManager):
        self._context = context
        self._config_manager = config_manager

    async def audit_prompt(
        self, prompt: str, unified_msg_origin: str
    ) -> tuple[bool, str]:
        if self._is_umo_whitelisted(unified_msg_origin):
            return True, ""

        settings = self._config_manager.safety_audit_settings.prompt_audit

        hit = self._match_blocked_word(prompt, settings.blocked_words)
        if hit:
            return False, f"命中遮蔽詞: {hit}"

        if not settings.enable_ai_audit:
            return True, ""

        review_prompt = self._build_review_prompt(
            settings.ai_prompt,
            prompt,
            append_prompt_if_missing_placeholder=True,
        )
        return await self._audit_with_model(
            unified_msg_origin=unified_msg_origin,
            review_prompt=review_prompt,
            provider_id=settings.ai_provider_id,
            image_urls=None,
        )

    async def audit_generated_images(
        self,
        prompt: str,
        image_paths: list[str],
        unified_msg_origin: str,
    ) -> tuple[bool, str]:
        if self._is_umo_whitelisted(unified_msg_origin):
            return True, ""

        settings = self._config_manager.safety_audit_settings.image_audit
        if not settings.enable_ai_audit:
            return True, ""

        review_prompt = self._build_review_prompt(
            settings.ai_prompt,
            prompt,
            append_prompt_if_missing_placeholder=False,
        )
        return await self._audit_with_model(
            unified_msg_origin=unified_msg_origin,
            review_prompt=review_prompt,
            provider_id=settings.ai_provider_id,
            image_urls=image_paths,
        )

    async def audit_staged_generated_images(
        self,
        prompt: str,
        image_paths: list[str],
        unified_msg_origin: str,
    ) -> tuple[bool, str, list[str]]:
        staged_files = stage_generated_files_for_audit(image_paths)
        owns_staged_files = True
        try:
            image_allowed, image_reason = await self.audit_generated_images(
                prompt=prompt,
                image_paths=[str(item.audit_path) for item in staged_files],
                unified_msg_origin=unified_msg_origin,
            )
            if not image_allowed:
                cleanup_staged_files(staged_files)
                end_generated_file_publication(
                    [item.public_path for item in staged_files]
                )
                owns_staged_files = False
                return False, image_reason, []

            owns_staged_files = False
            published_paths = publish_audited_generated_files(staged_files)
            return True, image_reason, published_paths
        finally:
            if owns_staged_files and _rollback_staged_files(staged_files, "稽核"):
                end_generated_file_publication(
                    [item.public_path for item in staged_files]
                )

    def _is_umo_whitelisted(self, unified_msg_origin: str) -> bool:
        umo = unified_msg_origin.strip()
        if not umo:
            return False
        return umo in self._config_manager.safety_audit_settings.umo_whitelist

    def _build_review_prompt(
        self,
        template: str,
        prompt: str,
        *,
        append_prompt_if_missing_placeholder: bool,
    ) -> str:
        review_prompt = template.strip() or (
            "請根據輸入內容完成安全稽核。"
            '僅輸出 JSON：{"allow": true/false, "reason": "簡短原因"}。'
        )
        prompt = prompt.strip()
        if self.PROMPT_PLACEHOLDER in review_prompt:
            return review_prompt.replace(self.PROMPT_PLACEHOLDER, prompt)
        if append_prompt_if_missing_placeholder and prompt:
            return f"{review_prompt}\n\n使用者提示詞：\n{prompt}"
        return review_prompt

    async def _audit_with_model(
        self,
        *,
        unified_msg_origin: str,
        review_prompt: str,
        provider_id: str,
        image_urls: list[str] | None,
    ) -> tuple[bool, str]:
        provider = (
            self._context.get_provider_by_id(provider_id) if provider_id else None
        )
        if provider_id and not provider:
            logger.warning(
                f"[ImageGen] 未找到稽核 Provider ID: {provider_id}，將回退到當前會話模型"
            )
        if provider is None:
            provider = self._context.get_using_provider(unified_msg_origin)
        if not provider:
            msg = "安全稽核異常：未找到可用稽核模型"
            logger.warning(f"[ImageGen] {msg}")
            return False, msg

        try:
            response = await provider.text_chat(
                prompt=review_prompt,
                image_urls=image_urls or [],
                persist=False,
            )
            completion_text = (response.completion_text or "").strip()
            return self._parse_audit_response(completion_text)
        except Exception as exc:
            logger.warning(f"[ImageGen] {_MODEL_CALL_FAILURE}: {type(exc).__name__}")
            return False, _MODEL_CALL_FAILURE

    def _match_blocked_word(self, prompt: str, blocked_words: list[str]) -> str:
        content = prompt.lower()
        for word in blocked_words:
            if word and word.lower() in content:
                return word
        return ""

    def _parse_audit_response(self, text: str) -> tuple[bool, str]:
        if len(text) > _MAX_AUDIT_RESPONSE_CHARS:
            return False, _INVALID_AUDIT_REASON
        try:
            payload = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
        except (ValueError, RecursionError):
            return False, _INVALID_AUDIT_REASON

        if not isinstance(payload, dict) or set(payload) != {"allow", "reason"}:
            return False, _INVALID_AUDIT_REASON

        allow = payload["allow"]
        reason = payload["reason"]
        if (
            not isinstance(allow, bool)
            or not isinstance(reason, str)
            or len(reason) > _MAX_AUDIT_REASON_CHARS
        ):
            return False, _INVALID_AUDIT_REASON
        return allow, reason
