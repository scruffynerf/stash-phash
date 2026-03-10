# stash-phash

A Python implementation of perceptual hash (pHash) generation that is fully compatible with [Stash](https://github.com/stashapp/stash).

## Features

- **Stash-compatible**: Generates identical phashes to Stash's built-in generator
- **Video support**: Extracts frames and creates sprite images for video hashing
- **CLI tool**: Easy command-line interface for generating hashes
- **Library**: Import and use in your own Python projects

## Installation

```bash
# Using uv (recommended)
uv pip install stash-phash

# Or install from source
git clone https://github.com/yourusername/stash-phash
cd stash-phash
uv pip install -e .
```

### Requirements

- Python 3.10+
- FFmpeg (for video processing) - must be in PATH or specified via `--ffmpeg`

## Usage

### Command Line

```bash
# Generate phash for a video file
stash-phash video.mp4

# Process multiple files
stash-phash video1.mp4 video2.mp4

# Output as integer (for database storage)
stash-phash --format int video.mp4

# Output as signed int64 (as stored in Stash's database)
stash-phash --format int-signed video.mp4

# Quiet mode (only output hashes, no filenames)
stash-phash -q video.mp4

# Specify custom FFmpeg path
stash-phash --ffmpeg /path/to/ffmpeg video.mp4

# Verbose output
stash-phash -v video.mp4
```

### Python Library

```python
from stash_phash import generate_video_phash, phash_to_string, hamming_distance

# Generate phash for a video
phash = generate_video_phash("video.mp4")

# Convert to hex string (Stash display format)
hex_string = phash_to_string(phash)
print(f"pHash: {hex_string}")  # e.g., "1a2b3c4d5e6f7890"

# Compare two phashes
phash1 = generate_video_phash("video1.mp4")
phash2 = generate_video_phash("video2.mp4")
distance = hamming_distance(phash1, phash2)

if distance < 10:
    print("Videos are likely duplicates!")
```

### Working with Images

```python
from PIL import Image
from stash_phash import perception_hash, hamming_distance

# Generate phash for an image
img = Image.open("image.jpg")
phash = perception_hash(img)

# Compare images
img1 = Image.open("image1.jpg")
img2 = Image.open("image2.jpg")
distance = hamming_distance(perception_hash(img1), perception_hash(img2))
```

## Algorithm

This implementation matches Stash's phash algorithm exactly:

### For Videos

1. Extract 25 frames (5×5 grid) from the video
2. Skip first/last 5% of duration to avoid intros/outros
3. Each frame is resized to 160×160 pixels
4. Frames are combined into a sprite image
5. Apply perception hash to the sprite

### Perception Hash (pHash)

1. Resize image to 64×64 using bilinear interpolation
2. Convert to grayscale using luminosity formula: `0.299*R + 0.587*G + 0.114*B`
3. Apply 2D Discrete Cosine Transform (DCT-II)
4. Extract top-left 8×8 DCT coefficients (64 values)
5. Calculate median of these 64 values
6. Generate 64-bit hash: bit is 1 if coefficient > median, else 0

### Hamming Distance

The similarity between two phashes is measured using Hamming distance (number of differing bits):

- **0**: Identical images
- **1-10**: Very similar (likely duplicates)
- **10-20**: Somewhat similar
- **20+**: Different images

## Compatibility

This implementation produces hashes that are:
- Identical to Stash's built-in phash generator
- Compatible with Stash's duplicate detection
- Storable in Stash's database format (signed int64)
- Displayable in Stash's hex format (16-character lowercase)

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/stash-phash
cd stash-phash

# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
ruff format .
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Stash](https://github.com/stashapp/stash) - The original implementation
- [goimagehash](https://github.com/corona10/goimagehash) - Go library used by Stash
