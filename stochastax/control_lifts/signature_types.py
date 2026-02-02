from typing import NewType, Protocol, Literal
from stochastax.hopf_algebras.elements import GroupElement, LieElement

import jax
from stochastax.hopf_algebras.hopf_algebra_types import HopfAlgebraT

# Backward-compatibility type names, now as newtypes over the new element classes.
# Runtime representation remains GroupElement/LieElement.
PathSignature = NewType("PathSignature", GroupElement)
BCKSignature = NewType("BCKSignature", GroupElement)
MKWSignature = NewType("MKWSignature", GroupElement)

# Logged signature types for type-safe pairing with bracket types in log_ode
LogSignature = NewType("LogSignature", LieElement)
BCKLogSignature = NewType("BCKLogSignature", LieElement)
MKWLogSignature = NewType("MKWLogSignature", LieElement)

Signature = PathSignature | BCKSignature | MKWSignature
PrimitiveSignature = LogSignature | BCKLogSignature | MKWLogSignature


class ControlLift(Protocol[HopfAlgebraT]):
    def __call__(
        self,
        path: jax.Array,
        depth: int,
        hopf: HopfAlgebraT,
        mode: Literal["full", "stream", "incremental"],
    ) -> Signature | list[Signature]: ...
