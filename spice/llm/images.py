"""Bounded image snapshots shared by user attachments and tools."""

from __future__ import annotations

import base64
import warnings
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from spice.llm.messages import ImageContent

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_EDGE = 2048
MAX_ENCODED_IMAGE_BYTES = 4 * 1024 * 1024
MAX_PROMPT_IMAGES = 5
SUPPORTED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "GIF"}


def load_image(path: Path) -> ImageContent:
    if not path.is_file():
        raise ValueError(f"Image file does not exist or is not a regular file: {path}")
    with path.open("rb") as handle:
        data = handle.read(MAX_IMAGE_BYTES + 1)
    return image_from_bytes(data, source=str(path))


def image_from_bytes(data: bytes, *, source: str | None = None) -> ImageContent:
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 20 MiB input limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as original:
                if original.format not in SUPPORTED_IMAGE_FORMATS:
                    raise ValueError("Unsupported image format; use PNG, JPEG, WebP, or GIF.")
                if original.width * original.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds the 40 megapixel limit.")
                original.seek(0)
                image = ImageOps.exif_transpose(original).convert("RGBA")
                image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
                output = BytesIO()
                image.save(output, format="PNG")
                media_type = "image/png"
                if output.tell() > MAX_ENCODED_IMAGE_BYTES:
                    background = Image.new("RGB", image.size, "white")
                    background.paste(image, mask=image.getchannel("A"))
                    output = BytesIO()
                    background.save(output, format="JPEG", quality=85)
                    media_type = "image/jpeg"
                encoded = output.getvalue()
                if len(encoded) > MAX_ENCODED_IMAGE_BYTES:
                    raise ValueError("Processed image exceeds the 4 MiB limit.")
                return ImageContent(
                    media_type=media_type,
                    data=base64.b64encode(encoded).decode("ascii"),
                    width=image.width,
                    height=image.height,
                    source=source,
                )
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Invalid, damaged, or oversized image.") from exc
