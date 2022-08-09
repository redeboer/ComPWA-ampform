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
from attrs.validators import deep_iterable, instance_of, optional
from qrules.topology import Topology
from qrules.transition import StateTransition

from ampform.helicity.decay import assert_isobar_topology

from ._attrs import extract_topologies, get_topology, to_optional_set
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
    scalar_initial_state_mass: bool = field(default=False, validator=instance_of(bool))
    r"""Add initial state mass as scalar value to `.parameter_defaults`.

    Put the invariant of the initial state (:math:`m_{012\dots}`) under
    `.HelicityModel.parameter_defaults` (with a *scalar* suggested value) instead of
    `~.HelicityModel.kinematic_variables`. This is useful if four-momenta were generated
    with or kinematically fit to a specific initial state energy.

    .. seealso:: :ref:`usage/amplitude:Scalar masses`
    """
    stable_final_state_ids: set[int] | None = field(
        converter=to_optional_set,
        default=None,
        validator=optional(deep_iterable(member_validator=instance_of(int))),  # type: ignore[arg-type]
    )
    r"""IDs of the final states that should be considered stable.

    Put final state 'invariant' masses (:math:`m_0, m_1, \dots`) under
    `.HelicityModel.parameter_defaults` (with a *scalar* suggested value) instead of
    `~.HelicityModel.kinematic_variables` (which are expressions to compute an
    event-wise array of invariant masses). This is useful if final state particles are
    stable.
    """

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
