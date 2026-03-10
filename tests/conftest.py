"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample test image with a gradient pattern."""
    width, height = 256, 256
    # Create a gradient image
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)

    # Create RGB image with different patterns per channel
    r = xv
    g = yv
    b = ((xv.astype(np.uint16) + yv.astype(np.uint16)) // 2).astype(np.uint8)

    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


@pytest.fixture
def solid_white_image() -> Image.Image:
    """Create a solid white test image."""
    return Image.new("RGB", (64, 64), color=(255, 255, 255))


@pytest.fixture
def solid_black_image() -> Image.Image:
    """Create a solid black test image."""
    return Image.new("RGB", (64, 64), color=(0, 0, 0))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
