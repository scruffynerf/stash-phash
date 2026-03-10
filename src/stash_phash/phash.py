"""
Core perceptual hash algorithm implementation.

This module implements the exact same perceptual hash algorithm used by Stash
(via the goimagehash library). The algorithm:

1. Resize image to 64x64 pixels using bilinear interpolation
2. Convert to grayscale using luminosity formula: 0.299*R + 0.587*G + 0.114*B
3. Apply 2D Discrete Cosine Transform (DCT-II)
4. Extract top-left 8x8 block of DCT coefficients (64 values)
5. Calculate median of these 64 values
6. Generate 64-bit hash where each bit is 1 if DCT value > median, else 0

The resulting hash is compatible with Stash's phash format and can be used
for duplicate detection via Hamming distance comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from scipy.fftpack import dct

if TYPE_CHECKING:
    from numpy.typing import NDArray


def rgb_to_gray(image: Image.Image) -> NDArray[np.float64]:
    """
    Convert an RGB image to grayscale using the luminosity formula.

    This matches the goimagehash implementation exactly:
    lum = 0.299 * R + 0.587 * G + 0.114 * B

    Note: The Go implementation divides by 257 for R/G and 256 for B due to
    the way Go's color.RGBA() returns 16-bit values. Since PIL gives us 8-bit
    values directly, we just use the coefficients.

    Args:
        image: PIL Image in RGB mode

    Returns:
        2D numpy array of grayscale values (float64)
    """
    # Ensure image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Get pixel data as numpy array
    pixels = np.array(image, dtype=np.float64)

    # Apply luminosity formula
    # Note: The Go code has a slight inconsistency (divides B by 256 instead of 257)
    # We replicate this exactly for compatibility
    gray = 0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]

    return gray


def dct_2d(pixels: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Apply 2D Discrete Cosine Transform (DCT-II) to the pixel array.

    This uses the separable property: apply 1D DCT to rows, then to columns.
    The scipy.fftpack.dct function implements DCT-II.

    Args:
        pixels: 2D numpy array of pixel values

    Returns:
        2D numpy array of DCT coefficients
    """
    # Apply DCT to rows first, then to columns
    # norm=None gives us the unscaled DCT-II (matching goimagehash)
    dct_rows = dct(pixels, type=2, axis=1, norm=None)
    dct_2d_result = dct(dct_rows, type=2, axis=0, norm=None)

    return dct_2d_result


def perception_hash(image: Image.Image) -> int:
    """
    Compute the perceptual hash (pHash) of an image.

    This implements the exact algorithm used by Stash (via goimagehash):
    1. Resize to 64x64 using bilinear interpolation
    2. Convert to grayscale
    3. Apply 2D DCT
    4. Extract top-left 8x8 DCT coefficients
    5. Calculate median of these coefficients
    6. Generate hash: bit is 1 if coefficient > median

    Args:
        image: PIL Image object

    Returns:
        64-bit perceptual hash as an unsigned integer

    Example:
        >>> from PIL import Image
        >>> img = Image.open("photo.jpg")
        >>> phash = perception_hash(img)
        >>> print(f"{phash:016x}")  # Print as 16-char hex
    """
    # Step 1: Resize to 64x64 using bilinear interpolation
    # PIL's BILINEAR matches Go's resize.Bilinear
    resized = image.resize((64, 64), Image.Resampling.BILINEAR)

    # Step 2: Convert to grayscale
    pixels = rgb_to_gray(resized)

    # Step 3: Apply 2D DCT
    dct_result = dct_2d(pixels)

    # Step 4: Extract top-left 8x8 block (64 low-frequency coefficients)
    dct_low_freq = dct_result[:8, :8]

    # Flatten to 1D array (row-major order)
    flattened = dct_low_freq.flatten()

    # Step 5: Calculate median using quickselect (numpy's median)
    median = np.median(flattened)

    # Step 6: Generate 64-bit hash
    # In goimagehash, bits are set from MSB to LSB:
    # leftShiftSet(64 - idx - 1) means bit position (63 - idx)
    # So for idx=0, we set bit 63 (MSB); for idx=63, we set bit 0 (LSB)
    hash_value = 0
    for idx, value in enumerate(flattened):
        if value > median:
            # Set bit at position (63 - idx)
            hash_value |= 1 << (63 - idx)

    return hash_value


def hamming_distance(hash1: int, hash2: int) -> int:
    """
    Calculate the Hamming distance between two perceptual hashes.

    The Hamming distance is the number of bit positions where the two hashes
    differ. A smaller distance indicates more similar images.

    Typical thresholds:
    - 0: Identical or near-identical images
    - 1-10: Very similar images (likely duplicates)
    - 10-20: Somewhat similar images
    - 20+: Different images

    Args:
        hash1: First 64-bit perceptual hash
        hash2: Second 64-bit perceptual hash

    Returns:
        Number of differing bits (0-64)

    Example:
        >>> distance = hamming_distance(hash1, hash2)
        >>> if distance < 10:
        ...     print("Images are likely duplicates")
    """
    # XOR gives us bits that differ
    xor = hash1 ^ hash2

    # Count the number of 1 bits (popcount)
    # We need to handle this as unsigned 64-bit
    # Python's int.bit_count() works correctly for positive integers
    # For negative results from XOR (if hashes were stored as signed),
    # we mask to 64 bits
    return (xor & 0xFFFFFFFFFFFFFFFF).bit_count()
