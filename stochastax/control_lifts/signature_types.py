from typing import NewType
from stochastax.hopf_algebras.elements import GroupElement, LieElement

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
