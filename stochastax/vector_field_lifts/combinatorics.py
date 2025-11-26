"""
Small combinatorial helpers shared by vector field lifts.

Currently contains utilities for enumerating node colourings of trees/forests.
"""


def unrank_base_d(index: int, num_digits: int, base: int) -> list[int]:
    """
    Represent a nonnegative integer in base ``base`` with fixed length ``num_digits``.

    The most-significant digit comes first in the returned list.
    """
    digits: list[int] = [0] * num_digits
    x = int(index)
    for k in range(num_digits - 1, -1, -1):
        digits[k] = x % base
        x //= base
    return digits


