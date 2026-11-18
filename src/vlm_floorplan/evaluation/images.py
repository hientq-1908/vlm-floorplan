"""Load a plan image, shrink it to the resolution under test, and encode it for the API.

The resolution knob is the longest side in pixels (for example 768 versus 1280).
The image is resized here, on the client, so the same pixels reach every endpoint and
the run directory can store exactly what the model saw. Images are never enlarged.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


def prepare_image(path: Path, max_long_side: int) -> Image.Image:
    """Open an image as RGB and shrink it so its longest side is at most `max_long_side`."""
    image = Image.open(path)
    image.load()
    # Palette/alpha images (common for plans) must be flattened onto white, not black.
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)
    image = image.convert("RGB")
    long_side = max(image.size)
    if long_side > max_long_side:
        scale = max_long_side / long_side
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def to_png_bytes(image: Image.Image) -> bytes:
    """PNG, not JPEG: thin lines and small labels are what the model has to read."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def to_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
