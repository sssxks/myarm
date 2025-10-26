from typing import Protocol, runtime_checkable
import sympy as sp

type Num = int | float

@runtime_checkable
class SupportsSympy(Protocol):
    def _sympy_(self) -> sp.Expr: ...
