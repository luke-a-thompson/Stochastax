"""
Controls for stochastic and rough differential equations.

## Public API
### Paths:
- Path: A path in a state space.
- pathify: Convert a JAX array of a data stream into a Path object.

### Drivers:
- bm_driver: Generate a Brownian motion path.
- correlate_bm_driver_against_reference: Correlate a Brownian motion path against a reference path.
- correlated_bm_drivers: Generate a new Brownian motion path that is correlated with a reference path.
- riemann_liouville_driver: Generate a Riemann-Liouville fractional Brownian motion path.
- fractional_bm_driver: Generate a fractional Brownian motion path.

### Augmentations:
- augment_path: Augment a path with a list of augmentations.
- basepoint_augmentation: Augment a path with a basepoint.
- time_augmentation: Augment a path with a time shift.
- lead_lag_augmentation: Augment a path with a lead-lag shift.
- non_overlapping_windower: Augment a path with a non-overlapping window.
- dyadic_windower: Augment a path with a dyadic windower.

"""

from .paths_types import Path, pathify
from .augmentations import (
    augment_path,
    basepoint_augmentation,
    time_augmentation,
    lead_lag_augmentation,
    non_overlapping_windower,
    dyadic_windower,
)
from .drivers import (
    bm_driver,
    correlate_bm_driver_against_reference,
    correlated_bm_drivers,
    riemann_liouville_driver,
    fractional_bm_driver,
)

__all__ = [
    "Path",
    "pathify",
    "augment_path",
    "basepoint_augmentation",
    "time_augmentation",
    "lead_lag_augmentation",
    "non_overlapping_windower",
    "dyadic_windower",
    "bm_driver",
    "correlate_bm_driver_against_reference",
    "correlated_bm_drivers",
    "riemann_liouville_driver",
    "fractional_bm_driver",
]
