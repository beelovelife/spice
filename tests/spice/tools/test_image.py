from __future__ import annotations

import asyncio
import base64
from io import BytesIO

import pytest
from PIL import Image

from spice.llm import images
from spice.sandbox.policy import WorkspacePolicy
from spice.tools.base import ToolContext
from spice.tools.image import view_image
from spice.tools.tool_registry import create_coding_tools, create_read_only_tools


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "screen.png"
    Image.new("RGB", (64, 32), "red").save(path)
    return path


def test_view_image_returns_actual_pixels(image_path):
    result = asyncio.run(view_image({"path": image_path.name}, ToolContext(cwd=image_path.parent, supports_vision=True)))
    assert not result.is_error
    assert len(result.images) == 1
    image = result.images[0]
    assert (image.width, image.height) == (64, 32)
    with Image.open(BytesIO(base64.b64decode(image.data))) as decoded:
        assert decoded.getpixel((0, 0)) == (255, 0, 0, 255)
    assert image.data not in result.content
    assert image.source == str(image_path)


def test_view_image_rejects_text_model_without_reading(image_path, monkeypatch):
    def unexpected_read(path):
        raise AssertionError("must not read without vision support")
    monkeypatch.setattr("spice.tools.image.load_image", unexpected_read)
    result = asyncio.run(view_image({"path": image_path.name}, ToolContext(cwd=image_path.parent)))
    assert result.is_error and result.error_code == "vision_unsupported"
    assert not result.images


@pytest.mark.parametrize("kind", ["outside", "symlink", "secret"])
def test_view_image_obeys_workspace_policy(tmp_path, image_path, kind):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if kind == "outside":
        path = str(image_path)
    elif kind == "symlink":
        (workspace / "link.png").symlink_to(image_path)
        path = "link.png"
    else:
        (workspace / ".env").write_bytes(image_path.read_bytes())
        path = ".env"
    context = ToolContext(cwd=workspace, workspace=WorkspacePolicy(workspace), supports_vision=True)
    result = asyncio.run(view_image({"path": path}, context))
    assert result.is_error
    assert result.error_code == "workspace_policy_denied"
    assert not result.images


@pytest.mark.parametrize("name,data", [("bad.png", b"not an image"), ("x.svg", b"<svg/>"), ("missing.png", None)])
def test_view_image_invalid_files_are_recoverable(tmp_path, name, data):
    if data is not None:
        (tmp_path / name).write_bytes(data)
    result = asyncio.run(view_image({"path": name}, ToolContext(cwd=tmp_path, supports_vision=True)))
    assert result.is_error and result.error_code == "invalid_image"
    assert result.disposition == "recoverable"


def test_image_decoder_resizes_and_rejects_oversized_data(tmp_path, monkeypatch):
    path = tmp_path / "large.png"
    Image.new("RGB", (3000, 1000), "blue").save(path)
    image = images.load_image(path)
    assert image.width == 2048 and image.height < 2048
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 5)
    with pytest.raises(ValueError, match="input limit"):
        images.load_image(path)


def test_image_decoder_rejects_pixel_bomb(image_path, monkeypatch):
    monkeypatch.setattr(images, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError, match="megapixel limit"):
        images.load_image(image_path)


def test_image_decoder_uses_first_gif_frame(tmp_path):
    path = tmp_path / "animation.gif"
    Image.new("RGB", (10, 10), "red").save(path, save_all=True, append_images=[Image.new("RGB", (10, 10), "blue")])
    result = images.load_image(path)
    with Image.open(BytesIO(base64.b64decode(result.data))) as decoded:
        assert decoded.getpixel((0, 0)) == (255, 0, 0, 255)


def test_image_tool_is_available_in_coding_and_read_only_modes():
    for tools in (create_coding_tools(), create_read_only_tools()):
        assert "view_image" in {tool.name for tool in tools}
