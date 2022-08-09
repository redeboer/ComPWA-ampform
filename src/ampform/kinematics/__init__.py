"""Classes and functions for relativistic four-momentum kinematics.

.. autolink-preface::

    import sympy as sp
    from ampform.kinematics import create_four_momentum_symbols
"""
from __future__ import annotations

import itertools

import attrs
import sympy as sp
from attrs import define, field
from qrules.topology import Topology
from qrules.transition import StateTransition

from ampform.helicity.decay import assert_isobar_topology

from ._attrs import extract_topologies, get_topology
from .angles import compute_helicity_angles
from .lorentz import compute_invariant_masses, create_four_momentum_symbols


@define
class HelicityAdapter:
    r"""Converter for four-momenta to kinematic variable data.

    The `.create_expressions` method forms the bridge between four-momentum data for the
    decay you are studying and the kinematic variables that are in the `.HelicityModel`.
    These are invariant mass (see :func:`.get_invariant_mass_symbol`) and the
    :math:`\theta` and :math:`\phi` helicity angles (see
    :func:`.get_helicity_angle_symbols`).
    """

    topologies: set[Topology] = field(converter=extract_topologies)  # type:ignore[misc]

    def __attrs_post_init__(self) -> None:
        for topology in self.topologies:
            assert_isobar_topology(topology)

    def register_transition(self, transition: StateTransition) -> None:
        topology = get_topology(transition)
        self.register_topology(topology)

    def register_topology(self, topology: Topology) -> None:
        assert_isobar_topology(topology)
        if self.topologies:
            existing = next(iter(self.topologies))
            if topology.incoming_edge_ids != existing.incoming_edge_ids:
                raise ValueError(
                    "Initial state ID mismatch those of existing topologies"
                )
            if topology.outgoing_edge_ids != existing.outgoing_edge_ids:
                raise ValueError(
                    "Final state IDs mismatch those of existing topologies"
                )
        self.topologies.add(topology)

    @property
    def registered_topologies(self) -> frozenset[Topology]:
        return frozenset(self.topologies)

    def permutate_registered_topologies(self) -> None:
        """Register outgoing edge permutations of all `registered_topologies`.

        See :ref:`usage/amplitude:Extend kinematic variables`.
        """
        for topology in set(self.topologies):
            final_state_ids = topology.outgoing_edge_ids
            for permutation in itertools.permutations(final_state_ids):
                id_mapping = dict(zip(topology.outgoing_edge_ids, permutation))
                permuted_topology = attrs.evolve(
                    topology,
                    edges={
                        id_mapping.get(i, i): edge for i, edge in topology.edges.items()
                    },
                )
                self.topologies.add(permuted_topology)

    def create_expressions(self) -> dict[sp.Symbol, sp.Expr]:
        output = {}
        for topology in self.topologies:
            momenta = create_four_momentum_symbols(topology)
            output.update(compute_helicity_angles(momenta, topology))
            output.update(compute_invariant_masses(momenta, topology))
        return output
