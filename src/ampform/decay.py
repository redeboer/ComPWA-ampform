"""Definition of a Particle with limited info that is required for amplitude models."""

from __future__ import annotations

import sys
from fractions import Fraction
from functools import total_ordering
from typing import TYPE_CHECKING, Any

import attrs
from attrs import field, frozen
from qrules.transition import State

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if TYPE_CHECKING:
    from IPython.lib.pretty import PrettyPrinter
    from qrules.transition import StateTransition

Parity = Literal[-1, 1]


@total_ordering
@frozen(kw_only=True, order=False, repr=True)
class Particle:
    name: str
    pid: int
    latex: str | None = None
    mass: float = field(converter=float)
    width: float = field(converter=float, default=0.0)
    spin: Fraction = field(converter=Fraction)
    charge: Fraction = field(converter=Fraction, default=Fraction(0))
    isospin: Fraction | None = None
    parity: Parity | None = None
    c_parity: Parity | None = None
    g_parity: Parity | None = None

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, Particle):
            return self.name > other.name
        msg = f"Cannot compare {type(self).__name__} with {type(other).__name__}"
        raise NotImplementedError(msg)

    def _repr_pretty_(self, p: PrettyPrinter, cycle: bool) -> None:
        class_name = type(self).__name__
        if cycle:
            p.text(f"{class_name}(...)")
        else:
            with p.group(indent=2, open=f"{class_name}("):
                for attribute in attrs.fields(type(self)):
                    value = getattr(self, attribute.name)
                    if value != attribute.default:
                        p.breakable()
                        p.text(f"{attribute.name}=")
                        if "parity" in attribute.name:
                            p.text(str(value))
                        else:
                            p.pretty(value)  # type: ignore[attr-defined]
                        p.text(",")
            p.breakable()
            p.text(")")


@total_ordering
@frozen(kw_only=True, order=False, repr=True)
class StateWithID(State):
    """Extension of `~qrules.transition.State` that embeds the state ID."""

    id: int  # noqa: A003

    @classmethod
    def from_transition(cls, transition: StateTransition, state_id: int) -> StateWithID:
        state = transition.states[state_id]
        return cls(
            id=state_id,
            particle=state.particle,  # type: ignore[arg-type]
            spin_projection=state.spin_projection,
        )
