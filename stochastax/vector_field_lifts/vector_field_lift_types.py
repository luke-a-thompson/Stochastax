"""Types for vector field lifts.

## Public API

### For use in log-ODE:
- LyndonBrackets: The Lyndon brackets (commutators) of vector fields.
- BCKBrackets: The BCK brackets of vector fields.
- MKWBrackets: The MKW brackets of vector fields. Suitable for manifolds.

### For use in Butcher/Lie-Butcher series:
- ButcherDifferentials: The elementary differentials of a vector field evaluated as a BCK forest.
- LieButcherDifferentials: The elementary differentials of a vector field evaluated as a MKW forest. Suitable for manifolds.
"""

import jax
from typing import NewType, Protocol, Callable

from stochastax.hopf_algebras.hopf_algebras import HopfAlgebraT
from stochastax.manifolds.manifolds import Manifold, EuclideanSpace

# Elementary differentials for Butcher/Lie-Butcher are kept as a single stacked array
# because series formation code expects a flat concatenation contract.
ButcherDifferentials = NewType("ButcherDifferentials", jax.Array)
LieButcherDifferentials = NewType("LieButcherDifferentials", jax.Array)

# Bracket matrices are per-degree lists, mirroring signature inputs.
# Each entry k stores a [Nk, n, n] stack of matrices for degree k+1.
LyndonBrackets = NewType("LyndonBrackets", list[jax.Array])
BCKBrackets = NewType("BCKBrackets", list[jax.Array])
MKWBrackets = NewType("MKWBrackets", list[jax.Array])

VectorFieldBrackets = LyndonBrackets | BCKBrackets | MKWBrackets

# Bracket functions: per-degree lists of callables V_w(y): R^n -> R^n.
LyndonBracketFunctions = NewType(
    "LyndonBracketFunctions", list[list[Callable[[jax.Array], jax.Array]]]
)
BCKBracketFunctions = NewType("BCKBracketFunctions", list[list[Callable[[jax.Array], jax.Array]]])
MKWBracketFunctions = NewType("MKWBracketFunctions", list[list[Callable[[jax.Array], jax.Array]]])
VectorFieldBracketFunctions = LyndonBracketFunctions | BCKBracketFunctions | MKWBracketFunctions


class VectorFieldLift(Protocol[HopfAlgebraT]):
    def __call__(
        self,
        vector_fields: list[Callable[[jax.Array], jax.Array]],
        base_point: jax.Array,
        hopf: HopfAlgebraT,
        manifold: Manifold = EuclideanSpace(),
    ) -> VectorFieldBrackets: ...


class VectorFieldBracketFunctionLift(Protocol[HopfAlgebraT]):
    def __call__(
        self,
        vector_fields: list[Callable[[jax.Array], jax.Array]],
        hopf: HopfAlgebraT,
        manifold: Manifold = EuclideanSpace(),
    ) -> VectorFieldBracketFunctions: ...
