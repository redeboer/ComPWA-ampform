from __future__ import annotations

from collections import abc
from functools import singledispatch
from typing import Iterable

from qrules.topology import Topology
from qrules.transition import ReactionInfo, StateTransition


@singledispatch
def extract_topologies(
    obj: ReactionInfo | Iterable[Topology | StateTransition],
) -> set[Topology]:
    raise TypeError(f"Cannot extract topologies from a {type(obj).__name__}")


@extract_topologies.register(ReactionInfo)
def _(transitions: ReactionInfo) -> set[Topology]:
    return extract_topologies(transitions.transitions)


@extract_topologies.register(abc.Iterable)
def _(transitions: abc.Iterable) -> set[Topology]:
    return {get_topology(t) for t in transitions}


@singledispatch
def get_topology(obj) -> Topology:
    raise TypeError(f"Cannot create a {Topology.__name__} from a {type(obj).__name__}")


@get_topology.register(Topology)
def _(obj: Topology) -> Topology:
    return obj


@get_topology.register(StateTransition)
def _(obj: StateTransition) -> Topology:
    return obj.topology
