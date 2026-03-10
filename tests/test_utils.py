"""Tests for utility functions."""

import pytest

from stash_phash.utils import (
    phash_to_signed,
    phash_to_string,
    phash_to_unsigned,
    string_to_phash,
)


class TestPhashToString:
    """Tests for phash_to_string conversion."""

    def test_zero(self):
        """Zero should produce 16 zeros."""
        assert phash_to_string(0) == "0000000000000000"

    def test_max_value(self):
        """Max uint64 should produce all f's."""
        assert phash_to_string(0xFFFFFFFFFFFFFFFF) == "ffffffffffffffff"

    def test_negative_one(self):
        """Negative one (as signed int64) should produce all f's."""
        assert phash_to_string(-1) == "ffffffffffffffff"

    def test_known_value(self):
        """Test a known hash value."""
        assert phash_to_string(0x1234567890ABCDEF) == "1234567890abcdef"

    def test_lowercase_output(self):
        """Output should be lowercase hex."""
        result = phash_to_string(0xABCDEF)
        assert result == result.lower()
        assert "abcdef" in result

    def test_always_16_chars(self):
        """Output should always be 16 characters."""
        assert len(phash_to_string(0)) == 16
        assert len(phash_to_string(1)) == 16
        assert len(phash_to_string(0xFFFFFFFFFFFFFFFF)) == 16


class TestStringToPhash:
    """Tests for string_to_phash conversion."""

    def test_zero(self):
        """Zero hex should produce 0."""
        assert string_to_phash("0000000000000000") == 0
        assert string_to_phash("0") == 0

    def test_known_value(self):
        """Test a known hash value."""
        assert string_to_phash("1234567890abcdef") == 0x1234567890ABCDEF

    def test_case_insensitive(self):
        """Should handle both upper and lowercase hex."""
        assert string_to_phash("ABCDEF") == string_to_phash("abcdef")

    def test_with_0x_prefix(self):
        """Should handle '0x' prefix."""
        assert string_to_phash("0x1234") == string_to_phash("1234")

    def test_high_bit_set_returns_negative(self):
        """Values with high bit set should return negative (signed int64)."""
        # 0xFFFFFFFFFFFFFFFF as signed int64 is -1
        assert string_to_phash("ffffffffffffffff") == -1

        # 0x8000000000000000 is the minimum int64 value
        assert string_to_phash("8000000000000000") < 0

    def test_invalid_hex(self):
        """Should raise ValueError for invalid hex."""
        with pytest.raises(ValueError):
            string_to_phash("not_hex")

    def test_too_large(self):
        """Should raise ValueError for values > 64 bits."""
        with pytest.raises(ValueError):
            string_to_phash("10000000000000000")  # 65 bits


class TestPhashToUnsigned:
    """Tests for phash_to_unsigned conversion."""

    def test_positive_unchanged(self):
        """Positive values should be unchanged."""
        assert phash_to_unsigned(12345) == 12345

    def test_negative_to_unsigned(self):
        """Negative values should convert to unsigned representation."""
        assert phash_to_unsigned(-1) == 0xFFFFFFFFFFFFFFFF

    def test_zero(self):
        """Zero should remain zero."""
        assert phash_to_unsigned(0) == 0


class TestPhashToSigned:
    """Tests for phash_to_signed conversion."""

    def test_small_positive_unchanged(self):
        """Small positive values should be unchanged."""
        assert phash_to_signed(12345) == 12345

    def test_high_bit_to_negative(self):
        """Values with high bit set should become negative."""
        assert phash_to_signed(0xFFFFFFFFFFFFFFFF) == -1
        assert phash_to_signed(0x8000000000000000) == -9223372036854775808  # min int64

    def test_zero(self):
        """Zero should remain zero."""
        assert phash_to_signed(0) == 0


class TestRoundTrip:
    """Test round-trip conversions."""

    def test_string_roundtrip(self):
        """Converting to string and back should preserve value."""
        original = 0x1234567890ABCDEF
        as_string = phash_to_string(original)
        back = string_to_phash(as_string)
        # Note: roundtrip may change sign, but bit pattern should match
        assert phash_to_unsigned(back) == original

    def test_signed_unsigned_roundtrip(self):
        """Converting between signed and unsigned should preserve bits."""
        positive = 12345
        assert phash_to_unsigned(phash_to_signed(positive)) == positive

        negative = -12345
        unsigned = phash_to_unsigned(negative)
        assert phash_to_signed(unsigned) == negative
