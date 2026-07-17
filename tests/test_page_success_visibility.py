from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from astrbot_plugin_image_generation.core.page_api import PluginPageApi
from conftest import FakeMetadataStore


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_success_image_is_visible_on_every_page_surface(tmp_path: Path) -> None:
    # Given
    payload = _png_bytes()
    approved = tmp_path / "gen_success.png"
    approved.write_bytes(payload)
    hidden = {}
    for status in ("pending", "blocked", "failed"):
        path = tmp_path / f"gen_{status}.png"
        path.write_bytes(payload)
        hidden[path.name] = {"status": status}
    records = {approved.name: {"status": "success"}, **hidden}
    catalog = PluginPageApi.__new__(PluginPageApi)
    catalog._images = __import__(
        "astrbot_plugin_image_generation.core.page_images",
        fromlist=["PageImageCatalog"],
    ).PageImageCatalog(tmp_path, FakeMetadataStore(records))

    # When
    listed = catalog._images.list_images()
    page = catalog._images.page(1, 12, 1024 * 1024)
    original = catalog._images.read_original(approved.name, 1024 * 1024)
    thumbnail = catalog._images.thumbnail(approved.name, 1024 * 1024)

    # Then
    assert [item.name for item in listed] == [approved.name]
    assert catalog._images.count_images() == page.total == 1
    assert catalog._images.inspect(approved.name) is not None
    assert original is not None and original.data == payload
    assert thumbnail is not None and thumbnail.data
    for file_name in hidden:
        assert catalog._images.inspect(file_name) is None
        assert catalog._images.read_original(file_name, 1024 * 1024) is None
        assert catalog._images.thumbnail(file_name, 1024 * 1024) is None
