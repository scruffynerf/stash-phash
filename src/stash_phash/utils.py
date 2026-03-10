"""
Utility functions for perceptual hash conversion and manipulation.

This module provides functions for converting perceptual hashes between
different representations (integer, hex string) in a way that's fully
compatible with Stash's format.

Stash stores phashes as:
- int64 (signed) in the database
- 16-character lowercase hex string for display

The conversion handles the signed/unsigned distinction correctly.
"""

from __future__ import annotations

import ctypes


def phash_to_string(phash: int) -> str:
    """
    Convert a perceptual hash integer to a 16-character hex string.

    This matches Stash's PhashToString function exactly:
    `strconv.FormatUint(uint64(phash), 16)`

    Args:
        phash: 64-bit perceptual hash (can be signed or unsigned)

    Returns:
        16-character lowercase hex string

    Example:
        >>> phash_to_string(0x1234567890abcdef)
        '1234567890abcdef'
        >>> phash_to_string(-1)  # All bits set
        'ffffffffffffffff'
    """
    # Handle as unsigned 64-bit integer
    unsigned_value = phash & 0xFFFFFFFFFFFFFFFF
    return f"{unsigned_value:016x}"


def string_to_phash(s: str) -> int:
    """
    Convert a hex string to a perceptual hash integer.

    This matches Stash's StringToPhash function exactly:
    `strconv.ParseUint(s, 16, 64)`

    Note: Returns as signed int64 to match Stash's database storage,
    where values with the high bit set become negative.

    Args:
        s: Hex string (with or without '0x' prefix)

    Returns:
        64-bit perceptual hash as signed int64

    Raises:
        ValueError: If the string is not a valid hex value

    Example:
        >>> string_to_phash('1234567890abcdef')
        1311768467294899695
        >>> string_to_phash('ffffffffffffffff')
        -1  # High bit set, so negative as signed int64
    """
    # Remove '0x' prefix if present
    s = s.lower().strip()
    if s.startswith("0x"):
        s = s[2:]

    # Parse as unsigned
    unsigned_value = int(s, 16)

    # Validate it fits in 64 bits
    if unsigned_value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"Value too large for 64-bit hash: {s}")

    # Convert to signed int64 (matching Go's int64 behavior)
    # This makes values with high bit set negative
    return ctypes.c_int64(unsigned_value).value


def phash_to_unsigned(phash: int) -> int:
    """
    Ensure a phash is represented as an unsigned 64-bit integer.

    Args:
        phash: Perceptual hash (may be negative if stored as signed int64)

    Returns:
        Unsigned 64-bit integer representation

    Example:
        >>> phash_to_unsigned(-1)
        18446744073709551615  # 0xFFFFFFFFFFFFFFFF
    """
    return phash & 0xFFFFFFFFFFFFFFFF


def phash_to_signed(phash: int) -> int:
    """
    Convert a phash to signed int64 representation (as stored in Stash's database).

    Args:
        phash: Perceptual hash (unsigned)

    Returns:
        Signed int64 representation

    Example:
        >>> phash_to_signed(0xFFFFFFFFFFFFFFFF)
        -1
    """
    return ctypes.c_int64(phash & 0xFFFFFFFFFFFFFFFF).value
