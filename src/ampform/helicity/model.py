"""Data structures that define an amplitude."""
from __future__ import annotations

import collections
import logging
import sys
from collections import abc
from typing import (
    TYPE_CHECKING,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    OrderedDict,
    Union,
    ValuesView,
)

import attrs
import sympy as sp
from attrs import field, frozen
from attrs.validators import instance_of
from qrules.transition import ReactionInfo

from ampform.sympy import PoolSum

from .naming import natural_sorting

if sys.version_info >= (3, 8):
    from functools import singledispatchmethod
else:
    from singledispatchmethod import singledispatchmethod

if TYPE_CHECKING:
    from IPython.lib.pretty import PrettyPrinter

_LOGGER = logging.getLogger(__name__)


def _order_component_mapping(
    mapping: Mapping[str, sp.Expr]
) -> OrderedDict[str, sp.Expr]:
    return collections.OrderedDict(
        [(key, mapping[key]) for key in sorted(mapping, key=natural_sorting)]
    )


def _order_symbol_mapping(
    mapping: Mapping[sp.Symbol, sp.Expr]
) -> OrderedDict[sp.Symbol, sp.Expr]:
    return collections.OrderedDict(
        [
            (symbol, mapping[symbol])
            for symbol in sorted(mapping, key=lambda s: natural_sorting(s.name))
        ]
    )


def _order_amplitudes(
    mapping: Mapping[sp.Indexed, sp.Expr]
) -> OrderedDict[sp.Indexed, sp.Expr]:
    return collections.OrderedDict(
        [
            (key, mapping[key])
            for key in sorted(mapping, key=lambda a: natural_sorting(str(a)))
        ]
    )


def _to_parameter_values(
    mapping: Mapping[sp.Symbol, ParameterValue]
) -> ParameterValues:
    return ParameterValues(mapping)


@frozen
class HelicityModel:  # noqa: R701
    intensity: PoolSum = field(validator=instance_of(PoolSum))
    """Main expression describing the intensity over `kinematic_variables`."""
    amplitudes: OrderedDict[sp.Indexed, sp.Expr] = field(converter=_order_amplitudes)
    """Definitions for the amplitudes that appear in `intensity`.

    The main `intensity` is a sum over amplitudes for each initial and final state
    helicity combination. These amplitudes are indicated with as `sp.Indexed
    <sympy.tensor.indexed.Indexed>` instances and this attribute provides the
    definitions for each of these. See also :ref:`TR-014
    <compwa-org:tr-014-solution-2>`.
    """
    parameter_defaults: ParameterValues = field(converter=_to_parameter_values)
    """A mapping of suggested parameter values.

    Keys are `~sympy.core.symbol.Symbol` instances from the main :attr:`expression` that
    should be interpreted as parameters (as opposed to `kinematic_variables`). The
    symbols are ordered alphabetically by name with natural sort order
    (:func:`.natural_sorting`). Values have been extracted from the input
    `~qrules.transition.ReactionInfo`.
    """
    kinematic_variables: OrderedDict[sp.Symbol, sp.Expr] = field(
        converter=_order_symbol_mapping
    )
    """Expressions for converting four-momenta to kinematic variables."""
    components: OrderedDict[str, sp.Expr] = field(converter=_order_component_mapping)
    """A mapping for identifying main components in the :attr:`expression`.

    Keys are the component names (`str`), formatted as LaTeX, and values are
    sub-expressions in the main :attr:`expression`. The mapping is an
    `~collections.OrderedDict` that orders the component names alphabetically with
    natural sort order (:func:`.natural_sorting`).
    """
    reaction_info: ReactionInfo = field(validator=instance_of(ReactionInfo))

    @property
    def expression(self) -> sp.Expr:
        """Expression for the `intensity` with all amplitudes fully expressed.

        Constructed from `intensity` by substituting its amplitude symbols with the
        definitions with `amplitudes`.
        """

        def unfold_poolsums(expr: sp.Expr) -> sp.Expr:
            new_expr = expr
            for node in sp.postorder_traversal(expr):
                if isinstance(node, PoolSum):
                    new_expr = new_expr.xreplace({node: node.evaluate()})
            return new_expr

        intensity = self.intensity.evaluate()
        intensity = unfold_poolsums(intensity)
        return intensity.subs(self.amplitudes)

    def rename_symbols(  # noqa: R701
        self, renames: Iterable[tuple[str, str]] | Mapping[str, str]
    ) -> HelicityModel:
        """Rename certain symbols in the model.

        Renames all `~sympy.core.symbol.Symbol` instance that appear in `expression`,
        `parameter_defaults`, `components`, and `kinematic_variables`. This method can
        be used to :ref:`couple parameters <usage/modify:Couple parameters>`.

        Args:
            renames: A mapping from old to new names.

        Returns:
            A **new** instance of a `HelicityModel` with symbols in all attributes
            renamed accordingly.
        """
        renames = dict(renames)
        symbols = self.__collect_symbols()
        symbol_names = {s.name for s in symbols}
        for name in renames:
            if name not in symbol_names:
                _LOGGER.warning(f"There is no symbol with name {name}")
        symbol_mapping = {
            s: sp.Symbol(renames[s.name], **s.assumptions0) if s.name in renames else s
            for s in symbols
        }
        return attrs.evolve(
            self,
            intensity=self.intensity.xreplace(symbol_mapping),
            amplitudes={
                amp: expr.xreplace(symbol_mapping)
                for amp, expr in self.amplitudes.items()
            },
            parameter_defaults={
                symbol_mapping[par]: value
                for par, value in self.parameter_defaults.items()
            },
            components={
                name: expr.xreplace(symbol_mapping)
                for name, expr in self.components.items()
            },
            kinematic_variables={
                symbol_mapping[var]: expr.xreplace(symbol_mapping)
                for var, expr in self.kinematic_variables.items()
            },
        )

    def __collect_symbols(self) -> set[sp.Symbol]:
        symbols: set[sp.Symbol] = self.expression.free_symbols  # type: ignore[assignment]
        symbols |= set(self.kinematic_variables)
        for expr in self.kinematic_variables.values():
            symbols |= expr.free_symbols  # type: ignore[arg-type]
        return symbols


class ParameterValues(abc.Mapping):
    """Ordered mapping to `ParameterValue` with convenient getter and setter.

    >>> a, b, c = sp.symbols("a b c")
    >>> parameters = ParameterValues({a: 0.0, b: 1+1j, c: -2})
    >>> parameters[a]
    0.0
    >>> parameters["b"]
    (1+1j)
    >>> parameters["b"] = 3
    >>> parameters[1]
    3
    >>> parameters[2]
    -2
    >>> parameters[2] = 3.14
    >>> parameters[c]
    3.14

    .. automethod:: __getitem__
    .. automethod:: __setitem__
    """

    def __init__(self, parameters: Mapping[sp.Symbol, ParameterValue]) -> None:
        self.__parameters = dict(parameters)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__parameters})"

    def _repr_pretty_(self, p: PrettyPrinter, cycle: bool) -> None:
        class_name = type(self).__name__
        if cycle:
            p.text(f"{class_name}(...)")
        else:
            with p.group(indent=2, open=f"{class_name}({{"):
                p.breakable()
                for par, value in self.items():
                    p.pretty(par)
                    p.text(": ")
                    p.pretty(value)
                    p.text(",")
                    p.breakable()
            p.text("})")

    def __getitem__(self, key: sp.Symbol | int | str) -> ParameterValue:
        par = self._get_parameter(key)
        return self.__parameters[par]

    def __setitem__(self, key: sp.Symbol | int | str, value: ParameterValue) -> None:
        par = self._get_parameter(key)
        self.__parameters[par] = value

    @singledispatchmethod
    def _get_parameter(self, key: sp.Symbol | int | str) -> sp.Symbol:
        raise KeyError(  # no TypeError because of sympy.core.expr.Expr.xreplace
            f"Cannot find parameter for key type {type(key).__name__}"
        )

    @_get_parameter.register(sp.Symbol)
    def _(self, par: sp.Symbol) -> sp.Symbol:
        if par not in self.__parameters:
            raise KeyError(f"{type(self).__name__} has no parameter {par}")
        return par

    @_get_parameter.register(str)
    def _(self, name: str) -> sp.Symbol:
        for parameter in self.__parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(f"No parameter available with name {name}")

    @_get_parameter.register(int)
    def _(self, key: int) -> sp.Symbol:
        for i, parameter in enumerate(self.__parameters):
            if i == key:
                return parameter
        raise KeyError(
            f"Parameter mapping has {len(self)} parameters, but trying to get"
            f" parameter number {key}"
        )

    def __len__(self) -> int:
        return len(self.__parameters)

    def __iter__(self) -> Iterator[sp.Symbol]:
        return iter(self.__parameters)

    def items(self) -> ItemsView[sp.Symbol, ParameterValue]:
        return self.__parameters.items()

    def keys(self) -> KeysView[sp.Symbol]:
        return self.__parameters.keys()

    def values(self) -> ValuesView[ParameterValue]:
        return self.__parameters.values()


ParameterValue = Union[float, complex, int]
"""Allowed value types for parameters."""
