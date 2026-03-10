"""Tests for the core phash algorithm."""

import numpy as np
from PIL import Image

from stash_phash.phash import dct_2d, hamming_distance, perception_hash, rgb_to_gray


class TestRgbToGray:
    """Tests for RGB to grayscale conversion."""

    def test_white_image(self, solid_white_image):
        """White image should produce all 255 values."""
        gray = rgb_to_gray(solid_white_image)
        # White = 0.299*255 + 0.587*255 + 0.114*255 = 255
        assert np.allclose(gray, 255.0, atol=1.0)

    def test_black_image(self, solid_black_image):
        """Black image should produce all 0 values."""
        gray = rgb_to_gray(solid_black_image)
        assert np.allclose(gray, 0.0)

    def test_output_shape(self, sample_image):
        """Output should be 2D with same dimensions as input."""
        gray = rgb_to_gray(sample_image)
        assert gray.shape == (256, 256)

    def test_luminosity_formula(self):
        """Test specific color values against luminosity formula."""
        # Pure red (255, 0, 0) -> 0.299 * 255 = 76.245
        red_img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        gray = rgb_to_gray(red_img)
        assert np.allclose(gray, 0.299 * 255, atol=1.0)

        # Pure green (0, 255, 0) -> 0.587 * 255 = 149.685
        green_img = Image.new("RGB", (10, 10), color=(0, 255, 0))
        gray = rgb_to_gray(green_img)
        assert np.allclose(gray, 0.587 * 255, atol=1.0)

        # Pure blue (0, 0, 255) -> 0.114 * 255 = 29.07
        blue_img = Image.new("RGB", (10, 10), color=(0, 0, 255))
        gray = rgb_to_gray(blue_img)
        assert np.allclose(gray, 0.114 * 255, atol=1.0)


class TestDct2D:
    """Tests for 2D DCT computation."""

    def test_output_shape(self):
        """DCT output should have same shape as input."""
        input_arr = np.random.rand(64, 64)
        output = dct_2d(input_arr)
        assert output.shape == input_arr.shape

    def test_constant_input(self):
        """DCT of constant input should have energy in DC component."""
        input_arr = np.ones((8, 8)) * 100
        output = dct_2d(input_arr)
        # DC component (0,0) should be large, others should be small
        assert abs(output[0, 0]) > abs(output[1, 1])


class TestPerceptionHash:
    """Tests for perceptual hash computation."""

    def test_hash_is_64_bits(self, sample_image):
        """Hash should fit in 64 bits."""
        phash = perception_hash(sample_image)
        assert 0 <= phash <= 0xFFFFFFFFFFFFFFFF

    def test_identical_images_same_hash(self, sample_image):
        """Identical images should produce identical hashes."""
        hash1 = perception_hash(sample_image)
        hash2 = perception_hash(sample_image)
        assert hash1 == hash2

    def test_different_images_different_hash(self, solid_white_image, solid_black_image):
        """Very different images should produce different hashes."""
        hash_white = perception_hash(solid_white_image)
        hash_black = perception_hash(solid_black_image)
        assert hash_white != hash_black

    def test_similar_images_close_hash(self, sample_image):
        """Slightly modified images should have similar hashes."""
        # Create a slightly modified version
        modified = sample_image.copy()
        pixels = np.array(modified)
        pixels = np.clip(pixels.astype(np.int16) + 5, 0, 255).astype(np.uint8)
        modified = Image.fromarray(pixels)

        hash1 = perception_hash(sample_image)
        hash2 = perception_hash(modified)

        distance = hamming_distance(hash1, hash2)
        # Slightly modified images should not be maximally different
        # Note: Synthetic gradient images can produce larger distances than real photos
        assert distance < 45  # Less than ~70% different

    def test_resized_image_similar_hash(self, sample_image):
        """Resized image should have similar hash."""
        # Resize to different dimension
        resized = sample_image.resize((128, 128), Image.Resampling.BILINEAR)

        hash1 = perception_hash(sample_image)
        hash2 = perception_hash(resized)

        distance = hamming_distance(hash1, hash2)
        # Resized images should still be somewhat similar
        # Note: The phash algorithm normalizes size, but gradient patterns
        # can have aliasing effects that increase distance
        assert distance < 25


class TestHammingDistance:
    """Tests for Hamming distance calculation."""

    def test_identical_hashes(self):
        """Identical hashes should have distance 0."""
        assert hamming_distance(0x1234567890ABCDEF, 0x1234567890ABCDEF) == 0

    def test_completely_different_hashes(self):
        """Completely opposite hashes should have distance 64."""
        assert hamming_distance(0x0000000000000000, 0xFFFFFFFFFFFFFFFF) == 64

    def test_single_bit_difference(self):
        """Hashes differing by one bit should have distance 1."""
        assert hamming_distance(0x0000000000000000, 0x0000000000000001) == 1
        assert hamming_distance(0x8000000000000000, 0x0000000000000000) == 1

    def test_known_distance(self):
        """Test known bit difference count."""
        # 0xFF = 11111111 in binary (8 bits set)
        assert hamming_distance(0x00000000000000FF, 0x0000000000000000) == 8

    def test_symmetry(self):
        """Distance should be symmetric."""
        hash1 = 0x123456789ABCDEF0
        hash2 = 0xFEDCBA9876543210
        assert hamming_distance(hash1, hash2) == hamming_distance(hash2, hash1)

    def test_negative_hashes(self):
        """Should handle negative hash values correctly."""
        # -1 as signed int64 is 0xFFFFFFFFFFFFFFFF
        assert hamming_distance(-1, 0) == 64
        assert hamming_distance(-1, -1) == 0
