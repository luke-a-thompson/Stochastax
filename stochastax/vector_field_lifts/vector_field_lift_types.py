"""Types for vector field lifts.

## Public API

### For use in log-ODE:
- LyndonBrackets: The Lyndon brackets (commutators) of vector fields.
- BCKBrackets: The BCK brackets of vector fields.
- MKWBrackets: The MKW brackets of vector fields. Suitable for manifolds.

"""

import jax
from typing import NewType, Protocol, Callable

from stochastax.hopf_algebras.hopf_algebras import HopfAlgebraT
from stochastax.manifolds import Manifold, EuclideanSpace

# Bracket matrices are per-degree lists, mirroring signature inputs.
# Each entry k stores a [Nk, n, n] stack of matrices for degree k+1.
LyndonBrackets = NewType("LyndonBrackets", list[jax.Array])
BCKBrackets = NewType("BCKBrackets", list[jax.Array])
MKWBrackets = NewType("MKWBrackets", list[jax.Array])

# Bracket functions: per-degree lists of callables V_w(y): R^n -> R^n.
LyndonBracketFunctions = NewType(
    "LyndonBracketFunctions", list[list[Callable[[jax.Array], jax.Array]]]
)
BCKBracketFunctions = NewType("BCKBracketFunctions", list[list[Callable[[jax.Array], jax.Array]]])
MKWBracketFunctions = NewType("MKWBracketFunctions", list[list[Callable[[jax.Array], jax.Array]]])
VectorFieldBracketFunctions = LyndonBracketFunctions | BCKBracketFunctions | MKWBracketFunctions


class VectorFieldBracketFunctionLift(Protocol[HopfAlgebraT]):
    def __call__(
        self,
        vector_field: Callable[[jax.Array], jax.Array],
        hopf: HopfAlgebraT,
        manifold: Manifold = EuclideanSpace(),
    ) -> VectorFieldBracketFunctions: ...
