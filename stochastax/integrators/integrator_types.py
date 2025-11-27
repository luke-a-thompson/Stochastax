"""Types for integrators.

Public API:
- ButcherSeries
- LieSeries
- LieButcherSeries
"""

from typing import NewType
import jax

ButcherSeries = NewType("ButcherSeries", jax.Array)
LieSeries = NewType("LieSeries", jax.Array)
LieButcherSeries = NewType("LieButcherSeries", jax.Array)
