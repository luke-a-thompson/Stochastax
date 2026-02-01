"""
Vector field lifts for ordinary differential and stochastic calculus.

## Public API

### Free (pre/post-)Lie Algebra Lifts:
- form_lyndon_brackets_from_words: Lyndon brackets given precomputed Duval tables.

### Butcher Series:
- form_butcher_differentials: Form Butcher differentials for a BCK forest.
- form_lie_butcher_differentials: Form Lie-Butcher differentials for a MKW forest.
"""

from .butcher import (
    form_butcher_differentials,
    form_lie_butcher_differentials,
)
from .lie_lift import (
    form_lyndon_bracket_functions,
)
from .bck_lift import (
    form_bck_bracket_functions,
)
from .mkw_lift import (
    form_mkw_bracket_functions,
)
from .vector_field_lift_types import (
    ButcherDifferentials,
    LieButcherDifferentials,
    LyndonBrackets,
    BCKBrackets,
    MKWBrackets,
    VectorFieldBracketFunctionLift,
)

__all__ = [
    # Butcher differentials
    "form_butcher_differentials",
    "form_lie_butcher_differentials",
    # Function lifts
    "form_lyndon_bracket_functions",
    "form_bck_bracket_functions",
    "form_mkw_bracket_functions",
    # Vector field lift types
    "ButcherDifferentials",
    "LieButcherDifferentials",
    "LyndonBrackets",
    "BCKBrackets",
    "MKWBrackets",
    "VectorFieldBracketFunctionLift",
]
