from __future__ import annotations

import anyio
import pytest

from astrbot_plugin_image_generation.core.generator import ImageGenerator
from astrbot_plugin_image_generation.core.types import GenerationRequest, GenerationResult


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
    assert adapter.peak == 2
    assert adapter.calls == ["task-1", "task-2", "task-3", "task-4"]


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
