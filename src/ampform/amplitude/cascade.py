r"""Amplitude builder for cascade decays with an arbitrary number of final states.

This module implements the helicity formalism for :math:`1 \to n` cascade decays
following the conventions of `10.1103/PhysRevD.111.056015
<https://doi.org/10.1103/PhysRevD.111.056015>`_ :cite:`Habermann:2024sxs`. Amplitudes
for different decay topologies are matched by rotating the final-state spin states
with **relative Wigner rotations**. Unlike in the three-body case (see
`ampform.amplitude.dpd`), the angles that appear in the Wigner-:math:`D` functions
cannot be expressed in closed form in terms of Mandelstam variables. They therefore
remain **free symbols** in the formulated model (see
`CascadeAmplitudeModel.kinematic_variables`) and have to be computed numerically from
four-momenta with the `decayangle <https://pypi.org/project/decayangle>`_ package,
which implements the same conventions.

.. note:: AmpForm itself does **not** depend on :code:`decayangle`, nor on `phasespace
    <https://pypi.org/project/phasespace>`_ for phase-space generation. These packages
    are only required to evaluate a formulated model over data and are therefore
    listed under the :code:`notebooks` dependency group only. See the
    :doc:`/cascade` example, which formulates a four-body decay model and evaluates it
    over a phase-space sample.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from collections.abc import Callable
from itertools import product
from typing import TYPE_CHECKING, Literal

import attrs
import sympy as sp
from attrs import field, frozen
from sympy.core.symbol import Str
from sympy.physics.quantum.spin import Rotation as Wigner

from ampform.amplitude.dpd import (
    _formulate_clebsch_gordan_factors,
    create_mass_symbol_mapping,
)
from ampform.amplitude.helicity import (
    ParameterValue,
    ParameterValues,
    _to_parameter_values,
)
from ampform.decay import (
    Decay,
    DecayChain,
    IsobarNode,
    State,
    generate_helicity_assignments,
    get_final_state_ids,
    to_particle,
)
from ampform.decay.spin import get_spin_projections
from ampform.dynamics.builder import DefinedExpression
from ampform.sympy import PoolSum

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

NodeValue = int | tuple[int, ...]
"""Identifier of a node in a decay topology, as used by :code:`decayangle`.

A final-state node is identified by its integer state ID and a composite node by the
sorted tuple of the final-state IDs it decays into.
"""
TopologyTuple = tuple
"""Nested-tuple descriptor of a decay topology, as used by :code:`decayangle`.

Can be passed directly to :class:`decayangle.decay_topology.Topology` as
:code:`Topology(0, decay_topology=<tuple>)`. See :func:`to_topology`.
"""


@frozen
class HelicityAngle:
    r"""Pointer to a helicity angle that can be computed with :code:`decayangle`.

    Given a mapping :code:`momenta` of final-state IDs to arrays of four-momenta
    :math:`(p_x, p_y, p_z, E)` in the rest frame of the initial state, the angle is
    computed numerically as::

        from decayangle.decay_topology import Topology

        angles = Topology(0, decay_topology=topology).helicity_angles(momenta)
        value = getattr(angles[node], angle)
    """

    topology: TopologyTuple
    """Descriptor of the decay topology in which the angle is defined."""
    node: tuple[NodeValue, NodeValue]
    """Key of the two-body decay node within the topology.

    The first entry is the daughter with respect to which the angles are defined.
    """
    angle: Literal["phi_rf", "theta_rf"]
    """Field name in the named tuple returned by :code:`helicity_angles()`."""


@frozen
class WignerRotation:
    r"""Pointer to a relative Wigner-rotation angle computable with :code:`decayangle`.

    The angle is computed numerically over a set of four-momenta as::

        from decayangle.decay_topology import Topology

        reference = Topology(0, decay_topology=reference_topology)
        angles = reference.relative_wigner_angles(
            Topology(0, decay_topology=topology), momenta
        )
        value = getattr(angles[state_id], angle)
    """

    reference_topology: TopologyTuple
    """Descriptor of the reference topology of the amplitude model."""
    topology: TopologyTuple
    """Descriptor of the topology whose amplitude is rotated to the reference."""
    state_id: int
    """ID of the final-state particle whose spin state is rotated."""
    angle: Literal["phi_rf", "theta_rf", "psi_rf"]
    """Field name in the named tuple returned by :code:`relative_wigner_angles()`."""


@frozen
class InvariantMass:
    r"""Pointer to the invariant mass of a set of final-state particles.

    Compute numerically as :math:`m = \sqrt{E^2 - p_x^2 - p_y^2 - p_z^2}` over the sum
    of the four-momenta of the final states listed in `state_ids`.
    """

    state_ids: tuple[int, ...]


KinematicVariable = HelicityAngle | WignerRotation | InvariantMass
"""Descriptor for computing a kinematic variable of a cascade model numerically."""


@frozen
class CascadeAmplitudeModel:
    """Amplitude model for a cascade decay, formulated by `CascadeAmplitudeBuilder`."""

    decay: Decay
    reference_topology: TopologyTuple
    intensity: sp.Expr = sp.S.One
    amplitudes: dict[sp.Indexed, sp.Expr] = field(factory=dict)
    kinematic_variables: dict[sp.Symbol, KinematicVariable] = field(factory=dict)
    """Free kinematic symbols of the model, with pointers for computing them.

    None of these symbols have a symbolic definition: the helicity angles and Wigner
    rotations of a cascade decay cannot be expressed in terms of Mandelstam variables.
    Compute them numerically from four-momenta with the :code:`decayangle` package, as
    described in `HelicityAngle`, `WignerRotation`, and `InvariantMass`.
    """
    parameter_defaults: ParameterValues = field(
        converter=_to_parameter_values,
        factory=dict,
    )
    masses: dict[sp.Symbol, float] = field(factory=dict)

    @property
    def full_expression(self) -> sp.Expr:
        evaluated_amplitudes = {k: v.doit() for k, v in self.amplitudes.items()}
        return self.intensity.doit().xreplace(evaluated_amplitudes)


class CascadeAmplitudeBuilder:
    def __init__(self, decay: Decay, min_ls: bool = True) -> None:
        r"""Amplitude builder for the helicity formalism for :math:`1 \to n` decays.

        Args:
            decay: The `.Decay` over which to formulate the amplitude model. Its
                chains can have arbitrary decay depth and any number of final states.
            min_ls: Use helicity couplings instead of :math:`LS`-couplings. If
                `False`, each `.IsobarNode` requires an `.LSCoupling` and the helicity
                couplings are expanded into Clebsch-Gordan coefficients (see
                :func:`.dpd._formulate_clebsch_gordan_factors`).
        """
        self.decay = decay
        self.min_ls = min_ls
        self.dynamics_choices = ResonanceDynamicsConfigurator(decay)

    def formulate(
        self,
        reference_topology: TopologyTuple | DecayChain | None = None,
        *,
        cleanup_summations: bool = False,
    ) -> CascadeAmplitudeModel:
        r"""Formulate the amplitude model given the configuration of this builder.

        The intensity is an incoherent sum over the helicities of the initial and
        final state of a coherent sum of topology amplitudes. Amplitudes of chains
        whose topology differs from the **reference topology** are aligned by rotating
        each final-state spin state with a relative Wigner rotation
        :math:`D^{j_i*}_{\lambda'_i,\lambda_i}(\alpha_i, \beta_i, \gamma_i)`, where
        the Euler angles :math:`(\alpha_i, \beta_i, \gamma_i)` correspond to the
        :code:`(phi_rf, theta_rf, psi_rf)` angles returned by :code:`decayangle`'s
        :code:`relative_wigner_angles()` (see `WignerRotation`).

        Args:
            reference_topology: The decay topology with respect to which the spin
                states of all other topologies are aligned, given either as a
                nested-tuple descriptor or as one of the `.DecayChain` objects of the
                decay. If `None`, the topology of the first chain is used.
            cleanup_summations: Whether to remove helicity indices from the summations
                if their corresponding state is spinless.
        """
        reference_topology = self._get_reference_topology(reference_topology)
        state_ids = sorted(self.decay.final_state)
        helicity_symbols: tuple[sp.Symbol, ...] = sp.symbols(
            f"lambda:{len(state_ids) + 1}", rational=True
        )
        allowed_helicities = {
            symbol: get_spin_projections(self.decay.states[i])
            for i, symbol in enumerate(helicity_symbols)
        }
        amplitudes: dict[sp.Indexed, sp.Expr] = {}
        kinematic_variables: dict[sp.Symbol, KinematicVariable] = {}
        parameter_defaults: dict[sp.Basic, ParameterValue] = {}
        topology_groups = _group_chains_by_topology(self.decay.chains)
        for topology, chains in topology_groups.items():
            topology_model = self._formulate_topology_amplitudes(topology, chains)
            amplitudes.update(topology_model.amplitudes)
            kinematic_variables.update(topology_model.kinematic_variables)
            parameter_defaults.update(topology_model.parameter_defaults)
        aligned_amp, wigner_definitions = self.formulate_aligned_amplitude(
            helicity_symbols,
            reference_topology,
            topologies=tuple(topology_groups),
        )
        kinematic_variables.update(wigner_definitions)
        intensity = PoolSum(
            sp.Abs(aligned_amp) ** 2,
            *allowed_helicities.items(),
        )
        if cleanup_summations:
            intensity = intensity.cleanup()
        masses = create_mass_symbol_mapping(self.decay)
        parameter_defaults.update(masses)  # ty:ignore[no-matching-overload]
        return CascadeAmplitudeModel(
            decay=self.decay,
            reference_topology=reference_topology,
            intensity=intensity,
            amplitudes=amplitudes,
            kinematic_variables=kinematic_variables,
            parameter_defaults=parameter_defaults,
            masses=masses,
        )

    def formulate_aligned_amplitude(
        self,
        helicity_symbols: tuple[sp.Symbol, ...],
        reference_topology: TopologyTuple,
        topologies: tuple[TopologyTuple, ...],
    ) -> tuple[sp.Expr, dict[sp.Symbol, KinematicVariable]]:
        r"""Sum the topology amplitudes coherently, aligned to the reference topology.

        The amplitude of the reference topology enters directly, while the amplitudes
        of all other topologies are summed over intermediate helicities
        :math:`\lambda'_i` and rotated to the reference frame orientations with
        relative Wigner rotations.
        """
        λ0, *final_symbols = helicity_symbols
        state_ids = sorted(self.decay.final_state)
        bases = {t: _get_amplitude_base(t) for t in topologies}
        aligned_amp: sp.Expr = sp.S.Zero
        wigner_definitions: dict[sp.Symbol, KinematicVariable] = {}
        if reference_topology in bases:
            aligned_amp += bases[reference_topology][(λ0, *final_symbols)]
        unaligned_topologies = [t for t in topologies if t != reference_topology]
        if unaligned_topologies:
            primed_symbols: list[sp.Symbol] = sp.symbols(
                Rf"\lambda_(1:{len(state_ids) + 1})^{{\prime}}", rational=True
            )
            rotated_sum: sp.Expr = sp.S.Zero
            for topology in unaligned_topologies:
                term: sp.Expr = bases[topology][(λ0, *primed_symbols)]
                for i, λ_prime, λ in zip(
                    state_ids, primed_symbols, final_symbols, strict=True
                ):
                    state = self.decay.final_state[i]
                    if state.spin == 0:
                        continue
                    euler_angles = _create_wigner_rotation_symbols(i, topology)
                    for symbol, angle_name in zip(
                        euler_angles, ("phi_rf", "theta_rf", "psi_rf"), strict=True
                    ):
                        wigner_definitions[symbol] = WignerRotation(
                            reference_topology=reference_topology,
                            topology=topology,
                            state_id=i,
                            angle=angle_name,
                        )
                    term *= sp.conjugate(
                        Wigner.D(sp.Rational(state.spin), λ_prime, λ, *euler_angles)
                    )
                rotated_sum += term
            aligned_amp += PoolSum(
                rotated_sum,
                *[
                    (λ_prime, get_spin_projections(self.decay.final_state[i]))
                    for i, λ_prime in zip(state_ids, primed_symbols, strict=True)
                ],
            )
        return aligned_amp, wigner_definitions

    def _formulate_topology_amplitudes(
        self, topology: TopologyTuple, chains: Iterable[DecayChain]
    ) -> CascadeAmplitudeModel:
        base = _get_amplitude_base(topology)
        kinematic_variables: dict[sp.Symbol, KinematicVariable] = {}
        parameters: dict[sp.Basic, ParameterValue] = {}
        terms: dict[tuple[sp.Rational, ...], list[sp.Expr]] = defaultdict(list)
        state_ids = sorted(self.decay.final_state)
        for chain in chains:
            dynamics = self._formulate_chain_dynamics(chain)
            parameters.update(dynamics.parameters)
            kinematic_variables.update(_collect_invariant_masses(dynamics.expression))
            root_edge = get_final_state_ids(chain.decay)
            for assignment in generate_helicity_assignments(chain):
                helicities = (
                    assignment[root_edge],
                    *(assignment[i,] for i in state_ids),
                )
                term = dynamics.expression
                for node, path in _walk_topology_nodes(chain.decay):
                    node_term, node_parameters, node_variables = (
                        self._formulate_node_term(node, path, topology, assignment)
                    )
                    term *= node_term
                    parameters.update(node_parameters)
                    kinematic_variables.update(node_variables)
                terms[helicities].append(term)
        amplitudes: dict[sp.Indexed, sp.Expr] = {}
        allowed_helicities = [
            get_spin_projections(self.decay.states[i])
            for i in range(len(state_ids) + 1)
        ]
        for helicities in product(*allowed_helicities):
            amplitudes[base[helicities]] = sp.Add(*terms.get(helicities, []))
        return CascadeAmplitudeModel(
            decay=self.decay,
            reference_topology=topology,
            amplitudes=amplitudes,
            kinematic_variables=kinematic_variables,
            parameter_defaults=parameters,
        )

    def _formulate_node_term(  # noqa: PLR0914
        self,
        node: IsobarNode,
        path: tuple[NodeValue, ...],
        topology: TopologyTuple,
        assignment: Mapping[tuple[int, ...], sp.Rational],
    ) -> tuple[sp.Expr, dict[sp.Basic, ParameterValue], dict[sp.Symbol, HelicityAngle]]:
        child1, child2 = _sort_children(node)
        j_parent = sp.Rational(node.parent.spin)
        j2 = sp.Rational(to_particle(child2).spin)
        λ_parent = assignment[get_final_state_ids(node)]
        λ1 = assignment[get_final_state_ids(child1)]
        λ2 = assignment[get_final_state_ids(child2)]
        φ, θ = _create_helicity_angle_symbols(child1, path)
        node_key = (_get_node_value(child1), _get_node_value(child2))
        variables: dict[sp.Symbol, HelicityAngle] = {
            φ: HelicityAngle(topology, node_key, "phi_rf"),
            θ: HelicityAngle(topology, node_key, "theta_rf"),
        }
        term: sp.Expr = sp.conjugate(Wigner.D(j_parent, λ_parent, λ1 - λ2, φ, θ, 0)) * (
            -1
        ) ** (j2 - λ2)
        parameters: dict[sp.Basic, ParameterValue] = {}
        channel = Str(
            Rf"{node.parent.latex} \to"
            Rf" {to_particle(child1).latex} {to_particle(child2).latex}"
        )
        if self.min_ls:
            coupling = _get_helicity_coupling_base()[channel, λ1, λ2]
        else:
            if node.interaction is None:
                msg = (
                    "Cannot formulate the amplitude in the LS-basis if LS-couplings"
                    f" are missing on node {node.parent.name} →"
                    f" {to_particle(node.child1).name} {to_particle(node.child2).name}"
                )
                raise ValueError(msg)
            coupling = _get_ls_coupling_base()[
                channel, node.interaction.L, node.interaction.S
            ]
            sorted_node = attrs.evolve(node, child1=child1, child2=child2)
            term *= _formulate_clebsch_gordan_factors(
                sorted_node,
                helicities={
                    to_particle(child1): λ1,
                    to_particle(child2): λ2,
                },
            )
        parameters[coupling] = 1 + 0j
        term *= coupling
        return term, parameters, variables

    def _formulate_chain_dynamics(self, chain: DecayChain) -> DefinedExpression:
        combined = DefinedExpression()
        for node in chain.nodes:
            if isinstance(node.parent, State):
                continue  # the root node carries no dynamics
            formulate_dynamics = self.dynamics_choices.get_builder(node.parent.name)
            combined *= formulate_dynamics(node)
        expression = combined.expression
        for _ in range(2):  # resolve nested subexpression definitions
            if not set(combined.subexpressions) & expression.free_symbols:
                break
            expression = expression.xreplace(combined.subexpressions)
        return attrs.evolve(combined, expression=expression, subexpressions={})

    def _get_reference_topology(
        self, reference_topology: TopologyTuple | DecayChain | None
    ) -> TopologyTuple:
        if reference_topology is None:
            return to_topology(self.decay.chains[0])  # ty:ignore[invalid-return-type]
        if isinstance(reference_topology, DecayChain):
            return to_topology(reference_topology)  # ty:ignore[invalid-return-type]
        topologies = {to_topology(chain) for chain in self.decay.chains}
        if reference_topology not in topologies:
            msg = (
                f"Reference topology {reference_topology} does not appear in the"
                " decay. Available topologies:"
                f" {', '.join(str(t) for t in sorted(topologies, key=str))}"
            )
            raise ValueError(msg)
        return reference_topology


DynamicsBuilder = Callable[[IsobarNode], DefinedExpression]
"""Function that formulates a lineshape for the resonance that decays via this node.

Use :func:`create_invariant_mass_symbol` with the final-state IDs of the node (see
:func:`.get_final_state_ids`) to formulate the lineshape in terms of an invariant mass
that can be computed numerically from four-momenta.
"""


class ResonanceDynamicsConfigurator:
    """Registry of `DynamicsBuilder` functions per resonance in a `.Decay`."""

    def __init__(self, decay: Decay) -> None:
        self.__decay = decay
        self.__builders: dict[str, DynamicsBuilder] = {}

    def register_builder(self, resonance_name: str, builder: DynamicsBuilder) -> None:
        resonances = {
            particle.name
            for chain in self.__decay.chains
            for particle in chain.resonances
        }
        if resonance_name not in resonances:
            msg = (
                f"Decay does not contain a resonance named {resonance_name}."
                f" Available: {', '.join(sorted(resonances))}"
            )
            raise ValueError(msg)
        self.__builders[resonance_name] = builder

    def get_builder(self, resonance_name: str) -> DynamicsBuilder:
        return self.__builders.get(resonance_name, lambda _: DefinedExpression())


def to_topology(obj: DecayChain | IsobarNode | State) -> TopologyTuple | int:
    """Convert a decay chain into a :code:`decayangle`-style topology descriptor.

    The daughters of each node are ordered following the default ordering of
    :code:`decayangle`: composite nodes before final states, longer composites first,
    and nodes of equal length by their smallest attached final-state ID.

    >>> from ampform.decay import IsobarNode, Particle, State
    >>> def state(index: int) -> State:
    ...     return State("f", "f", spin=0, parity=1, mass=1.0, width=0.0, index=index)
    >>> R = Particle("R", "R", spin=0, parity=1, mass=1.0, width=0.0)
    >>> node = IsobarNode(
    ...     parent=state(0),
    ...     child1=state(4),
    ...     child2=IsobarNode(R, IsobarNode(R, state(1), state(2)), state(3)),
    ... )
    >>> to_topology(node)
    (((1, 2), 3), 4)
    """
    if isinstance(obj, DecayChain):
        return to_topology(obj.decay)
    if isinstance(obj, State):
        return obj.index  # ty:ignore[invalid-return-type]
    child1, child2 = _sort_children(obj)
    return (to_topology(child1), to_topology(child2))


def create_invariant_mass_symbol(state_ids: Iterable[int]) -> sp.Symbol:
    """Create a symbol for the invariant mass of a set of final-state particles.

    >>> create_invariant_mass_symbol((3, 1, 2))
    m_123
    """
    sorted_ids = "".join(map(str, sorted(state_ids)))
    return sp.Symbol(f"m_{sorted_ids}", nonnegative=True)


def _collect_invariant_masses(
    expression: sp.Expr,
) -> dict[sp.Symbol, InvariantMass]:
    """Find symbols named like :code:`m_123` and map them to their state IDs."""
    collected = {}
    for symbol in expression.free_symbols:
        if not isinstance(symbol, sp.Symbol):
            continue
        prefix, _, ids = str(symbol).partition("_")
        if prefix == "m" and ids.isdigit() and len(ids) > 1:
            collected[symbol] = InvariantMass(tuple(int(i) for i in ids))
    return collected


def _sort_children(node: IsobarNode) -> tuple[IsobarNode | State, IsobarNode | State]:
    """Order the children of a node following the default ordering of decayangle."""

    def sort_key(child: IsobarNode | State) -> tuple[int, tuple[int, ...]]:
        edge = get_final_state_ids(child)
        return (-len(edge), edge)

    child1, child2 = sorted(node.children, key=sort_key)
    return child1, child2


def _get_node_value(node: IsobarNode | State) -> NodeValue:
    if isinstance(node, State):
        return node.index  # ty:ignore[invalid-return-type]
    return get_final_state_ids(node)


def _walk_topology_nodes(
    node: IsobarNode, path: tuple[NodeValue, ...] = ()
) -> Iterable[tuple[IsobarNode, tuple[NodeValue, ...]]]:
    """Iterate over all isobar nodes, along with the path that identifies their frame.

    The path of the root node is empty and the path of a deeper node lists the node
    values from the node itself up to, but excluding, the root.
    """
    yield node, path
    for child in node.children:
        if isinstance(child, IsobarNode):
            yield from _walk_topology_nodes(child, (_get_node_value(child), *path))


def _create_helicity_angle_symbols(
    first_child: IsobarNode | State, path: tuple[NodeValue, ...]
) -> tuple[sp.Symbol, sp.Symbol]:
    r"""Create the symbols :math:`\phi, \theta` for the helicity angles of a node.

    The subscript is the first daughter of the node and the superscript lists the
    decaying node and its ancestors, which uniquely identifies the rest frame in which
    the angles are measured. The root node has no superscript, since the rest frame of
    the initial state is shared by all topologies.
    """
    subscript = _to_compact_string(_get_node_value(first_child))
    if path:
        superscript = ",".join(_to_compact_string(value) for value in path)
        suffix = Rf"_{{{subscript}}}^{{{superscript}}}"
    else:
        suffix = Rf"_{{{subscript}}}"
    return (
        sp.Symbol(Rf"\phi{suffix}", real=True),
        sp.Symbol(Rf"\theta{suffix}", real=True),
    )


def _create_wigner_rotation_symbols(
    state_id: int, topology: TopologyTuple
) -> tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
    label = _get_topology_label(topology)
    α, β, γ = (
        sp.Symbol(Rf"{euler}_{{{state_id}}}^{{{label}}}", real=True)
        for euler in (R"\alpha", R"\beta", R"\gamma")
    )
    return α, β, γ


def _to_compact_string(value: NodeValue) -> str:
    if isinstance(value, tuple):
        return "".join(map(str, value))
    return str(value)


def _get_topology_label(topology: TopologyTuple | int) -> str:
    """Render a topology descriptor as a compact string for use in symbol names.

    >>> _get_topology_label((((1, 2), 3), 4))
    '((12)3)4'
    """
    if isinstance(topology, int):
        return str(topology)

    def render(node: TopologyTuple | int) -> str:
        if isinstance(node, int):
            return str(node)
        return "(" + "".join(render(child) for child in node) + ")"

    return "".join(render(child) for child in topology)


def _group_chains_by_topology(
    chains: Iterable[DecayChain],
) -> dict[TopologyTuple, list[DecayChain]]:
    groups: dict[TopologyTuple, list[DecayChain]] = defaultdict(list)
    for chain in chains:
        groups[to_topology(chain)].append(chain)
    return dict(groups)


@functools.cache
def _get_amplitude_base(topology: TopologyTuple) -> sp.IndexedBase:
    return sp.IndexedBase(Rf"A^{{{_get_topology_label(topology)}}}")


@functools.cache
def _get_helicity_coupling_base() -> sp.IndexedBase:
    return sp.IndexedBase(R"\mathcal{H}")


@functools.cache
def _get_ls_coupling_base() -> sp.IndexedBase:
    return sp.IndexedBase(R"\mathcal{H}^\mathrm{LS}")
