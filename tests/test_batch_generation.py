from __future__ import annotations

import anyio
import anyio.lowlevel
import pytest
from astrbot_plugin_image_generation.core.generator import ImageGenerator
from astrbot_plugin_image_generation.core.types import (
    GenerationRequest,
    GenerationResult,
    ImageCapability,
)


class FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[str | None] = []
        self.active = 0
        self.peak = 0

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request.task_id)
        self.active += 1
        self.peak = max(self.peak, self.active)
        await anyio.sleep(0.01)
        self.active -= 1
        if request.task_id and request.task_id.endswith("-2"):
            return GenerationResult(images=None, error="fake provider failure")
        return GenerationResult(images=[f"image:{request.task_id}".encode()])

    def get_capabilities(self) -> ImageCapability:
        return ImageCapability.TEXT_TO_IMAGE


class BlockingAdapter(FakeAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.started = anyio.Event()
        self.release = anyio.Event()
        self.closed = False

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.started.set()
        await self.release.wait()
        return GenerationResult(images=[f"image:{request.task_id}".encode()])

    async def close(self) -> None:
        self.closed = True


class TextOnlyRecordingAdapter(FakeAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[GenerationRequest] = []

    def get_capabilities(self) -> ImageCapability:
        return ImageCapability.TEXT_TO_IMAGE

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return await super().generate(request)


class SingleFlightAdapter(FakeAdapter):
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await anyio.lowlevel.checkpoint()
        overloaded = self.active > 1
        self.active -= 1
        if overloaded:
            return GenerationResult(images=None, error="provider overloaded")
        return GenerationResult(images=[f"image:{request.task_id}".encode()])


def make_generator(adapter: FakeAdapter, parallelism: int = 2) -> ImageGenerator:
    generator = object.__new__(ImageGenerator)
    generator.adapter = adapter
    generator._batch_limiter = anyio.CapacityLimiter(parallelism)
    return generator


@pytest.mark.asyncio
async def test_batch_fanout_preserves_order_and_reports_partial_success() -> None:
    adapter = FakeAdapter()
    generator = make_generator(adapter)

    result = await generator.generate(
        GenerationRequest(prompt="cat", task_id="task", count=4)
    )

    assert result.images == [b"image:task-1", b"image:task-3", b"image:task-4"]
    assert result.error is not None and "部分成功" in result.error
    assert adapter.peak == 1
    assert adapter.calls == ["task-1", "task-2", "task-3", "task-4"]


@pytest.mark.asyncio
async def test_batch_generation_serializes_single_flight_provider_requests() -> None:
    adapter = SingleFlightAdapter()
    generator = make_generator(adapter, parallelism=2)

    result = await generator.generate(
        GenerationRequest(prompt="cat", task_id="task", count=4)
    )

    assert result.images == [
        b"image:task-1",
        b"image:task-2",
        b"image:task-3",
        b"image:task-4",
    ]
    assert result.error is None
    assert adapter.peak == 1


@pytest.mark.asyncio
async def test_batch_generation_reports_progress_as_items_finish() -> None:
    adapter = FakeAdapter()
    generator = make_generator(adapter)
    progress = []

    await generator.generate(
        GenerationRequest(prompt="cat", task_id="task", count=3),
        progress_callback=progress.append,
    )

    assert len(progress) == 3
    assert progress[-1].completed == 3
    assert progress[-1].total == 3
    assert progress[-1].succeeded == 2
    assert progress[-1].failed == 1


@pytest.mark.asyncio
async def test_single_generation_keeps_original_request_shape() -> None:
    adapter = FakeAdapter()
    generator = make_generator(adapter)

    result = await generator.generate(
        GenerationRequest(prompt="cat", task_id="task", count=1)
    )

    assert result.images == [b"image:task"]
    assert result.error is None
    assert adapter.calls == ["task"]


@pytest.mark.asyncio
async def test_direct_generation_clamps_untrusted_count() -> None:
    adapter = FakeAdapter()
    generator = make_generator(adapter)

    result = await generator.generate(
        GenerationRequest(prompt="cat", task_id="task", count=999999)
    )

    assert result.images is not None
    assert len(adapter.calls) == 4


@pytest.mark.asyncio
async def test_provider_multiple_images_are_reduced_to_one_per_request() -> None:
    class MultiImageAdapter(FakeAdapter):
        async def generate(self, request: GenerationRequest) -> GenerationResult:
            return GenerationResult(images=[b"first", b"second"])

    generator = make_generator(MultiImageAdapter())

    result = await generator.generate(
        GenerationRequest(prompt="cat", task_id="task", count=2)
    )

    assert result.images == [b"first", b"first"]


@pytest.mark.asyncio
async def test_adapter_switch_waits_for_active_generation() -> None:
    old_adapter = BlockingAdapter()
    new_adapter = FakeAdapter()
    generator = make_generator(old_adapter)
    generator._create_adapter = lambda _config: new_adapter

    switched = anyio.Event()

    async def switch() -> None:
        await generator.update_adapter("new-config")
        switched.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(
            generator.generate,
            GenerationRequest(prompt="cat", task_id="task"),
        )
        await old_adapter.started.wait()
        task_group.start_soon(switch)
        await anyio.lowlevel.checkpoint()
        assert not switched.is_set()
        old_adapter.release.set()
        await switched.wait()

    assert old_adapter.closed is True
    assert generator.adapter is new_adapter


@pytest.mark.asyncio
async def test_generation_waits_for_adapter_replacement() -> None:
    old_adapter = FakeAdapter()
    new_adapter = FakeAdapter()
    generator = make_generator(old_adapter)
    generator._ensure_lifecycle()

    async with generator._lifecycle_condition:
        generator._adapter_updating = True

    result_holder: list[GenerationResult] = []

    async def run_generation() -> None:
        result_holder.append(
            await generator.generate(
                GenerationRequest(prompt="cat", task_id="task")
            )
        )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_generation)
        await anyio.lowlevel.checkpoint()
        assert not old_adapter.calls
        async with generator._lifecycle_condition:
            generator.adapter = new_adapter
            generator._adapter_updating = False
            generator._lifecycle_condition.notify_all()

    assert result_holder[0].images == [b"image:task"]
    assert new_adapter.calls == ["task"]


@pytest.mark.asyncio
async def test_get_capabilities_uses_lifecycle_guard() -> None:
    generator = make_generator(FakeAdapter())

    capabilities = await generator.get_capabilities()

    assert capabilities == ImageCapability.TEXT_TO_IMAGE


@pytest.mark.asyncio
async def test_generation_filters_unsupported_parameters_under_adapter_guard() -> None:
    adapter = TextOnlyRecordingAdapter()
    generator = make_generator(adapter)

    result = await generator.generate(
        GenerationRequest(
            prompt="cat",
            task_id="task",
            aspect_ratio="16:9",
            resolution="4K",
        )
    )

    assert result.images == [b"image:task"]
    assert adapter.requests[0].aspect_ratio is None
    assert adapter.requests[0].resolution is None
