"""Types for integrators.

Public API:
- LieSeries
"""

from typing import NewType
import jax

LieSeries = NewType("LieSeries", jax.Array)

Series = LieSeries
