from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from types import ModuleType
from pathlib import Path
from typing import Generic, TypeVar

import pytest


class _Logger:
    @staticmethod
    def debug(_message: str, *_args, **_kwargs) -> None:
        pass

    @staticmethod
    def info(_message: str, *_args, **_kwargs) -> None:
        pass

    @staticmethod
    def warning(_message: str, *_args, **_kwargs) -> None:
        pass

    @staticmethod
    def error(_message: str, *_args, **_kwargs) -> None:
        pass


class Image:
    def __init__(self, file: str | None = None, url: str | None = None) -> None:
        self.file = file
        self.url = url


class At:
    def __init__(self, qq: str | int) -> None:
        self.qq = qq


class Reply:
    def __init__(
        self,
        id: str,
        chain: list[Image | At | Reply] | None = None,
        sender_id: str | int | None = None,
    ) -> None:
        self.id = id
        self.chain = chain
        self.sender_id = sender_id


class AstrMessageEvent:
    pass


class MessageChain:
    def message(self, _content: str) -> MessageChain:
        return self

    def file_image(self, _path: str) -> MessageChain:
        return self


class AstrBotConfig:
    pass


class StarContext:
    pass


class Star:
    def __init__(self, context: StarContext) -> None:
        self.context = context


class ProviderRequest:
    pass


class CustomFilter:
    pass


class StarTools:
    @staticmethod
    def get_data_dir() -> Path:
        return Path("/tmp/astrbot-test-data")


class _Filter:
    @staticmethod
    def _decorate(function):
        return function

    def command_group(self, *_args, **_kwargs):
        def decorator(function):
            function.command = self.command
            return function

        return decorator

    def command(self, *_args, **_kwargs):
        return self._decorate

    def custom_filter(self, *_args, **_kwargs):
        return self._decorate

    def regex(self, *_args, **_kwargs):
        return self._decorate

    def on_llm_request(self, *_args, **_kwargs):
        return self._decorate


ContextT = TypeVar("ContextT")


class ContextWrapper(Generic[ContextT]):
    pass


class FunctionTool(Generic[ContextT]):
    pass


class ToolExecResult:
    pass


class AstrAgentContext:
    pass


async def _download_image_by_url(_url: str, *, path: str) -> str:
    return path


def _package(name: str) -> ModuleType:
    package = ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
    return package


def _module(name: str) -> ModuleType:
    module = ModuleType(name)
    sys.modules[name] = module
    return module


astrbot = _package("astrbot")
api = _package("astrbot.api")
core = _package("astrbot.core")
config = _package("astrbot.core.config")
agent = _package("astrbot.core.agent")
utils = _package("astrbot.core.utils")
core_star = _package("astrbot.core.star")
core_star_filter = _package("astrbot.core.star.filter")

components = _module("astrbot.api.message_components")
components.Image = Image
components.At = At
components.Reply = Reply

event = _module("astrbot.api.event")
event.AstrMessageEvent = AstrMessageEvent
event.MessageChain = MessageChain
event.filter = _Filter()

star = _module("astrbot.api.star")
star.Context = StarContext
star.Star = Star

provider = _module("astrbot.api.provider")
provider.ProviderRequest = ProviderRequest

astrbot_config = _module("astrbot.core.config.astrbot_config")
astrbot_config.AstrBotConfig = AstrBotConfig

run_context = _module("astrbot.core.agent.run_context")
run_context.ContextWrapper = ContextWrapper

agent_tool = _module("astrbot.core.agent.tool")
agent_tool.FunctionTool = FunctionTool
agent_tool.ToolExecResult = ToolExecResult

agent_context = _module("astrbot.core.astr_agent_context")
agent_context.AstrAgentContext = AstrAgentContext

core_io = _module("astrbot.core.utils.io")
core_io.download_image_by_url = _download_image_by_url

custom_filter = _module("astrbot.core.star.filter.custom_filter")
custom_filter.CustomFilter = CustomFilter

star_tools = _module("astrbot.core.star.star_tools")
star_tools.StarTools = StarTools

api.logger = _Logger()
api.message_components = components
api.event = event
api.star = star
api.provider = provider
astrbot.api = api
astrbot.core = core
core.config = config
core.agent = agent
core.utils = utils
core.star = core_star
config.astrbot_config = astrbot_config
agent.run_context = run_context
agent.tool = agent_tool
core.astr_agent_context = agent_context
utils.io = core_io
core_star.filter = core_star_filter
core_star_filter.custom_filter = custom_filter
core_star.star_tools = star_tools

from astrbot_plugin_image_generation.core.page_api import PluginPageApi  # noqa: E402
from astrbot_plugin_image_generation.core import safety_auditor as safety_module  # noqa: E402
from astrbot_plugin_image_generation.core.safety_auditor import (  # noqa: E402
    SafetyAuditor,
)


class FakeMetadataStore:
    def __init__(self, records: dict[str, dict[str, str]]) -> None:
        self.records = records
        self.pruned_with: set[str] | None = None

    def get_all(self) -> dict[str, dict[str, str]]:
        return self.records

    def get(self, file_name: str) -> dict[str, str]:
        return self.records.get(file_name, {})

    def prune_missing(self, existing_file_names: set[str]) -> None:
        self.pruned_with = existing_file_names
        self.records = {
            name: metadata
            for name, metadata in self.records.items()
            if name in existing_file_names
        }


def residue_name(public_name: str, state: str, nonce: str = "0" * 32) -> str:
    return f".{public_name}.{nonce}.imagegen-{state}"


@dataclass(frozen=True, slots=True)
class RestartResidueMatrix:
    cache_dir: Path
    exact: tuple[Path, ...]
    malformed: tuple[Path, ...]
    symlink: Path
    target: Path
    outside: Path


def create_restart_residue_matrix(tmp_path: Path) -> RestartResidueMatrix:
    cache_dir = tmp_path / "cache"
    outside_dir = tmp_path / "outside"
    cache_dir.mkdir()
    outside_dir.mkdir()
    exact = (
        cache_dir / residue_name("gen_pending.png", "pending"),
        cache_dir / residue_name("gen_blocked.webp", "blocked", "a" * 32),
    )
    malformed = (
        cache_dir / residue_name("ref_unowned.png", "pending"),
        cache_dir / residue_name("gen_short.png", "blocked", "f" * 31),
        cache_dir / f".gen_extra.png.{'0' * 32}.imagegen-pending.extra",
        cache_dir / "unrelated.txt",
    )
    for path in (*exact, *malformed):
        path.write_bytes(b"residue")
    target = outside_dir / "target.bin"
    target.write_bytes(b"target")
    symlink = cache_dir / residue_name("gen_link.png", "pending", "b" * 32)
    symlink.symlink_to(target)
    outside = outside_dir / residue_name("gen_outside.png", "blocked", "c" * 32)
    outside.write_bytes(b"outside")
    return RestartResidueMatrix(cache_dir, exact, malformed, symlink, target, outside)


def create_multi_file_page(tmp_path: Path) -> tuple[list[Path], PluginPageApi]:
    files = [tmp_path / f"gen_{index}.png" for index in range(3)]
    for path in files:
        path.write_bytes(b"blocked-image-bytes")
    records = {path.name: {"status": "generated"} for path in files}
    page = PluginPageApi(
        SimpleNamespace(
            cache_dir=tmp_path, image_metadata_store=FakeMetadataStore(records)
        )
    )
    return files, page


class AllowingAuditor(SafetyAuditor):
    async def audit_generated_images(self, **_kwargs) -> tuple[bool, str]:
        return True, "allowed"


async def assert_second_publication_failure_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    public_paths = [tmp_path / f"gen_{index}.png" for index in range(2)]
    for path in public_paths:
        path.write_bytes(b"allowed-image-bytes")
    auditor = object.__new__(AllowingAuditor)
    original_link = os.link
    published = 0

    def fail_second_publication(source: Path, target: Path, **kwargs) -> None:
        nonlocal published
        if Path(target) in public_paths:
            published += 1
            if published == 2:
                raise failure
        original_link(source, target, **kwargs)

    monkeypatch.setattr(safety_module.os, "link", fail_second_publication)
    with pytest.raises(type(failure)):
        await auditor.audit_staged_generated_images(
            "prompt", [str(path) for path in public_paths], "umo:test"
        )
    assert not any(tmp_path.iterdir()), (
        "IMG-104 RED: second-publication failure must propagate after removing all "
        "public originals and private residue"
    )
