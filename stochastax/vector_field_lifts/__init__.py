"""
Vector field lifts for ordinary differential and stochastic calculus.

## Public API

### Free (pre/post-)Lie Algebra Lifts:
- form_lyndon_brackets_from_words: Lyndon brackets given precomputed Duval tables.
- form_lyndon_lift: Nonlinear Lyndon lift built from vector fields.
- form_bck_lift: Form BCK lift for a BCK forest.
- form_mkw_lift: Form MKW lift for a MKW forest.

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
    form_bck_lift,
)
from .mkw_lift import (
    form_mkw_lift,
)
from .vector_field_lift_types import (
    ButcherDifferentials,
    LieButcherDifferentials,
    LyndonBrackets,
    BCKBrackets,
    MKWBrackets,
    VectorFieldBrackets,
    VectorFieldLift,
    VectorFieldBracketFunctionLift,
)

__all__ = [
    "form_butcher_differentials",
    "form_lie_butcher_differentials",
    "form_lyndon_brackets_from_words",
    "form_lyndon_lift",
    "form_bck_lift",
    "form_mkw_lift",
    # Vector field lift types
    "ButcherDifferentials",
    "LieButcherDifferentials",
    "LyndonBrackets",
    "BCKBrackets",
    "MKWBrackets",
    "VectorFieldBrackets",
    "VectorFieldLift",
    "VectorFieldBracketFunctionLift",
]
