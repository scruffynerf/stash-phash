"""
Video processing for perceptual hash generation.

This module handles extracting frames from videos and generating sprite images
for perceptual hashing, replicating the exact behavior of Stash's videophash
package.

The process:
1. Extract 25 frames (5x5 grid) from the video
2. Skip first 5% and last 5% of video duration (avoid intros/outros)
3. Each frame is resized to 160x160 pixels
4. Frames are combined into a single sprite image
5. Sprite is passed to the phash algorithm
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from stash_phash.phash import perception_hash

if TYPE_CHECKING:
    pass

# Sprite configuration (matches Stash exactly)
SCREENSHOT_SIZE = 160
COLUMNS = 5
ROWS = 5
CHUNK_COUNT = COLUMNS * ROWS  # 25 frames


def get_video_duration(video_path: str | Path, ffprobe_path: str | None = None) -> float:
    """
    Get the duration of a video file in seconds.

    Args:
        video_path: Path to the video file
        ffprobe_path: Optional path to ffprobe executable. If None, uses 'ffprobe' from PATH.

    Returns:
        Duration in seconds as a float

    Raises:
        subprocess.CalledProcessError: If ffprobe fails
        ValueError: If duration cannot be parsed
    """
    ffprobe = ffprobe_path or "ffprobe"
    video_path = Path(video_path)

    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    duration_str = data.get("format", {}).get("duration")
    if duration_str is None:
        raise ValueError(f"Could not determine duration for {video_path}")

    return float(duration_str)


def extract_frame(
    video_path: str | Path,
    timestamp: float,
    size: int = SCREENSHOT_SIZE,
    ffmpeg_path: str | None = None,
) -> Image.Image:
    """
    Extract a single frame from a video at the specified timestamp.

    Args:
        video_path: Path to the video file
        timestamp: Time in seconds to extract frame from
        size: Width to resize the frame to (height auto-calculated)
        ffmpeg_path: Optional path to ffmpeg executable. If None, uses 'ffmpeg' from PATH.

    Returns:
        PIL Image of the extracted frame

    Raises:
        subprocess.CalledProcessError: If ffmpeg fails
    """
    ffmpeg = ffmpeg_path or "ffmpeg"
    video_path = Path(video_path)

    # Use a temporary file for the frame
    # We use BMP format like Stash for lossless extraction
    with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            ffmpeg,
            "-ss",
            str(timestamp),
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-vf",
            f"scale={size}:-1",
            "-y",  # Overwrite output file
            tmp_path,
        ]

        subprocess.run(
            cmd,
            capture_output=True,
            check=True,
        )

        # Load and return the image
        img = Image.open(tmp_path)
        # Make a copy so we can delete the temp file
        img_copy = img.copy()
        img.close()

        return img_copy

    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


def generate_sprite(
    video_path: str | Path,
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
) -> Image.Image:
    """
    Generate a sprite image from video frames for perceptual hashing.

    This replicates Stash's sprite generation exactly:
    - Extracts 25 frames (5x5 grid)
    - Skips first and last 5% of video duration
    - Each frame is 160 pixels wide
    - Frames are arranged in a grid

    Args:
        video_path: Path to the video file
        ffmpeg_path: Optional path to ffmpeg executable
        ffprobe_path: Optional path to ffprobe executable

    Returns:
        PIL Image containing the sprite

    Raises:
        ValueError: If video duration cannot be determined
        subprocess.CalledProcessError: If ffmpeg/ffprobe fails
    """
    video_path = Path(video_path)

    # Get video duration
    duration = get_video_duration(video_path, ffprobe_path)

    # Calculate frame timestamps (skip first/last 5%)
    # This matches Stash: offset = 0.05 * duration, stepSize = (0.9 * duration) / chunkCount
    offset = 0.05 * duration
    step_size = (0.9 * duration) / CHUNK_COUNT

    # Extract frames
    frames: list[Image.Image] = []
    for i in range(CHUNK_COUNT):
        timestamp = offset + (i * step_size)
        frame = extract_frame(video_path, timestamp, SCREENSHOT_SIZE, ffmpeg_path)
        frames.append(frame)

    # Combine frames into sprite
    return combine_images(frames)


def combine_images(images: list[Image.Image]) -> Image.Image:
    """
    Combine multiple images into a sprite grid.

    This replicates Stash's combineImages function exactly.

    Args:
        images: List of PIL Images (should be 25 for video phash)

    Returns:
        PIL Image containing the combined sprite
    """
    if not images:
        raise ValueError("No images to combine")

    # Get dimensions from first image
    width = images[0].width
    height = images[0].height

    # Create canvas
    canvas_width = width * COLUMNS
    canvas_height = height * ROWS

    # Create new RGBA image (like imaging.New in Go)
    sprite = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))

    # Paste images in grid
    for index, img in enumerate(images):
        x = width * (index % COLUMNS)
        y = height * (index // COLUMNS)  # Note: Go uses rows but same calculation

        # Convert to RGBA if needed
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        sprite.paste(img, (x, y))

    return sprite


def generate_video_phash(
    video_path: str | Path,
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
) -> int:
    """
    Generate a perceptual hash for a video file.

    This is the main entry point for video phash generation. It:
    1. Generates a sprite image from video frames
    2. Computes the perceptual hash of the sprite

    The resulting hash is fully compatible with Stash's phash format.

    Args:
        video_path: Path to the video file
        ffmpeg_path: Optional path to ffmpeg executable
        ffprobe_path: Optional path to ffprobe executable

    Returns:
        64-bit perceptual hash as an unsigned integer

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video duration cannot be determined
        subprocess.CalledProcessError: If ffmpeg/ffprobe fails

    Example:
        >>> phash = generate_video_phash("video.mp4")
        >>> print(f"{phash:016x}")  # Print as hex
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Generate sprite from video frames
    sprite = generate_sprite(video_path, ffmpeg_path, ffprobe_path)

    # Compute and return perceptual hash
    return perception_hash(sprite)
