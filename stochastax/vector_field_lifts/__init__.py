"""
Vector field lifts for ordinary differential and stochastic calculus.

## Public API

### Free (pre/post-)Lie Algebra Lifts:
- form_lyndon_brackets_from_words: Lyndon brackets given precomputed Duval tables.
- form_lyndon_lift: Nonlinear Lyndon lift built from vector fields.
- form_bck_brackets: Form BCK brackets for a BCK forest.
- form_mkw_brackets: Form MKW brackets for a MKW forest.

### Butcher Series:
- form_butcher_differentials: Form Butcher differentials for a BCK forest.
- form_lie_butcher_differentials: Form Lie-Butcher differentials for a MKW forest.
"""

from .butcher import (
    form_butcher_differentials,
    form_lie_butcher_differentials,
)
from .lie_lift import (
    form_lyndon_brackets_from_words,
    form_lyndon_lift,
)
from .bck_lift import (
    form_bck_brackets,
)
from .mkw_lift import (
    form_mkw_brackets,
)

__all__ = [
    "form_butcher_differentials",
    "form_lie_butcher_differentials",
    "form_lyndon_brackets_from_words",
    "form_lyndon_lift",
    "form_bck_brackets",
    "form_mkw_brackets",
]
