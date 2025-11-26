from typing import NewType
import jax

# Elementary differentials for Butcher/Lie-Butcher are kept as a single stacked array
# because series formation code expects a flat concatenation contract.
ButcherDifferentials = NewType("ButcherDifferentials", jax.Array)
LieButcherDifferentials = NewType("LieButcherDifferentials", jax.Array)

# Bracket matrices are per-degree lists, mirroring signature inputs.
# Each entry k stores a [Nk, n, n] stack of matrices for degree k+1.
LyndonBrackets = NewType("LyndonBrackets", list[jax.Array])
BCKBrackets = NewType("BCKBrackets", list[jax.Array])
MKWBrackets = NewType("MKWBrackets", list[jax.Array])
