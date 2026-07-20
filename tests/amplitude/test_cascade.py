from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from ampform.amplitude.cascade import (
    CascadeAmplitudeBuilder,
    HelicityAngle,
    InvariantMass,
    WignerRotation,
    create_invariant_mass_symbol,
    to_topology,
)
from ampform.amplitude.dpd import DalitzPlotDecompositionBuilder
from ampform.decay import (
    Decay,
    DecayChain,
    IsobarNode,
    Particle,
    State,
    ThreeBodyDecay,
    ThreeBodyDecayChain,
    get_final_state_ids,
)
from ampform.dynamics.builder import DefinedExpression
from ampform.sympy import PoolSum


@pytest.fixture(scope="session")
def four_body_decay() -> Decay:
    """Lambda_c+ -> p K- pi+ pi0 with one double-pair and one cascade topology."""
    lambda_c = State(
        "Lambda_c+",
        R"\Lambda_c^+",
        spin=0.5,
        parity=+1,
        mass=2.28646,
        width=0.0,
        index=0,
    )
    proton = State("p", "p", spin=0.5, parity=+1, mass=0.93827, width=0.0, index=1)
    kaon = State("K-", "K^-", spin=0, parity=-1, mass=0.49368, width=0.0, index=2)
    pion = State("pi+", R"\pi^+", spin=0, parity=-1, mass=0.13957, width=0.0, index=3)
    pion0 = State("pi0", R"\pi^0", spin=0, parity=-1, mass=0.13498, width=0.0, index=4)
    delta = Particle(
        "Delta(1232)++",
        R"\Delta(1232)^{++}",
        spin=1.5,
        parity=+1,
        mass=1.232,
        width=0.117,
    )
    k_star = Particle(
        "K*(892)-", R"K^*(892)^-", spin=1, parity=-1, mass=0.892, width=0.0514
    )
    lambda_1520 = Particle(
        "Lambda(1520)",
        R"\Lambda(1520)",
        spin=1.5,
        parity=-1,
        mass=1.51945,
        width=0.0157,
    )
    sigma_star = Particle(
        "Sigma(1670)0", R"\Sigma(1670)^0", spin=1.5, parity=-1, mass=1.670, width=0.06
    )
    chain_a = DecayChain(
        decay=IsobarNode(
            parent=lambda_c,
            child1=IsobarNode(delta, proton, pion),
            child2=IsobarNode(k_star, kaon, pion0),
        )
    )
    chain_b = DecayChain(
        decay=IsobarNode(
            parent=lambda_c,
            child1=IsobarNode(sigma_star, IsobarNode(lambda_1520, proton, kaon), pion0),
            child2=pion,
        )
    )
    return Decay(
        states={s.index: s for s in [lambda_c, proton, kaon, pion, pion0]},
        chains=[chain_a, chain_b],
    )


@pytest.fixture(scope="session")
def spinless_four_body_decay() -> Decay:
    """X(1--) -> 4 pions, with a double-pair and a cascade topology."""
    x = State("X", "X", spin=1, parity=-1, mass=3.0, width=0.0, index=0)
    pi1 = State("pi+", R"\pi^+", spin=0, parity=-1, mass=0.13957, width=0.0, index=1)
    pi2 = State("pi-", R"\pi^-", spin=0, parity=-1, mass=0.13957, width=0.0, index=2)
    pi3 = State("pi+", R"\pi^+", spin=0, parity=-1, mass=0.13957, width=0.0, index=3)
    pi4 = State("pi-", R"\pi^-", spin=0, parity=-1, mass=0.13957, width=0.0, index=4)
    rho = Particle("rho(770)", R"\rho(770)", spin=1, parity=-1, mass=0.775, width=0.148)
    f0 = Particle("f(0)(980)", "f_0(980)", spin=0, parity=+1, mass=0.99, width=0.06)
    a1 = Particle("a(1)(1260)", "a_1(1260)", spin=1, parity=+1, mass=1.23, width=0.42)
    return Decay(
        states={s.index: s for s in [x, pi1, pi2, pi3, pi4]},
        chains=[
            DecayChain(
                decay=IsobarNode(
                    parent=x,
                    child1=IsobarNode(rho, pi1, pi2),
                    child2=IsobarNode(f0, pi3, pi4),
                )
            ),
            DecayChain(
                decay=IsobarNode(
                    parent=x,
                    child1=IsobarNode(a1, IsobarNode(rho, pi1, pi2), pi3),
                    child2=pi4,
                )
            ),
        ],
    )


class TestCascadeAmplitudeBuilder:
    @pytest.fixture(scope="session")
    def model(self, four_body_decay: Decay):
        builder = CascadeAmplitudeBuilder(four_body_decay)
        return builder.formulate(reference_topology=((1, 3), (2, 4)))

    def test_topology_descriptors(self, four_body_decay: Decay):
        topologies = [to_topology(chain) for chain in four_body_decay.chains]
        assert topologies == [((1, 3), (2, 4)), (((1, 2), 4), 3)]

    def test_intensity_structure(self, model):
        assert isinstance(model.intensity, PoolSum)
        helicity_ranges = {
            str(symbol): values for symbol, values in model.intensity.indices
        }
        assert helicity_ranges == {
            "lambda0": (-sp.Rational(1, 2), +sp.Rational(1, 2)),
            "lambda1": (-sp.Rational(1, 2), +sp.Rational(1, 2)),
            "lambda2": (0,),
            "lambda3": (0,),
            "lambda4": (0,),
        }

    def test_amplitudes_per_topology(self, model):
        bases = {str(symbol.base) for symbol in model.amplitudes}
        assert bases == {R"A^{(13)(24)}", R"A^{((12)4)3}"}
        n_helicity_combinations = 2 * 2 * 1 * 1 * 1
        assert len(model.amplitudes) == 2 * n_helicity_combinations
        assert all(expr != 0 for expr in model.amplitudes.values())

    def test_kinematic_variables_have_no_definitions(self, model):
        """All kinematic symbols map to decayangle pointers, not sympy expressions."""
        assert len(model.kinematic_variables) == 15
        for descriptor in model.kinematic_variables.values():
            assert isinstance(
                descriptor, (HelicityAngle, WignerRotation, InvariantMass)
            )
        wigner_angles = {
            str(symbol): descriptor
            for symbol, descriptor in model.kinematic_variables.items()
            if isinstance(descriptor, WignerRotation)
        }
        assert set(wigner_angles) == {
            R"\alpha_{1}^{((12)4)3}",
            R"\beta_{1}^{((12)4)3}",
            R"\gamma_{1}^{((12)4)3}",
        }
        for descriptor in wigner_angles.values():
            assert descriptor.reference_topology == ((1, 3), (2, 4))
            assert descriptor.topology == (((1, 2), 4), 3)
            assert descriptor.state_id == 1

    def test_helicity_couplings_are_parameters(self, model):
        couplings = sorted(
            (
                str(symbol)
                for symbol in model.parameter_defaults
                if str(symbol).startswith(R"\mathcal{H}")
            ),
        )
        assert len(couplings) == 17
        assert couplings[0] == R"\mathcal{H}[K^*(892)^- \to K^- \pi^0, 0, 0]"

    def test_full_expression_free_symbols(self, model):
        intensity = model.full_expression.xreplace(dict(model.parameter_defaults))
        free_symbols = {str(s) for s in intensity.doit().free_symbols}
        assert free_symbols == {str(s) for s in model.kinematic_variables}

    def test_dynamics_builder_receives_node(self, four_body_decay: Decay):
        builder = CascadeAmplitudeBuilder(four_body_decay)
        collected_nodes = []

        def formulate_dynamics(node: IsobarNode) -> DefinedExpression:
            collected_nodes.append(node)
            mass = create_invariant_mass_symbol(get_final_state_ids(node))
            return DefinedExpression(expression=1 / mass, parameters={})

        builder.dynamics_choices.register_builder("Lambda(1520)", formulate_dynamics)
        model = builder.formulate()
        assert len(collected_nodes) == 1
        assert collected_nodes[0].parent.name == "Lambda(1520)"
        mass_symbols = {
            str(symbol): descriptor
            for symbol, descriptor in model.kinematic_variables.items()
            if isinstance(descriptor, InvariantMass)
        }
        assert mass_symbols == {"m_12": InvariantMass(state_ids=(1, 2))}

    def test_unknown_resonance_raises(self, four_body_decay: Decay):
        builder = CascadeAmplitudeBuilder(four_body_decay)
        with pytest.raises(ValueError, match="does not contain a resonance"):
            builder.dynamics_choices.register_builder(
                "N(1440)", lambda node: DefinedExpression()
            )

    def test_invalid_reference_topology_raises(self, four_body_decay: Decay):
        builder = CascadeAmplitudeBuilder(four_body_decay)
        with pytest.raises(ValueError, match="does not appear in the decay"):
            builder.formulate(reference_topology=((1, 2), (3, 4)))

    def test_ls_basis_requires_ls_couplings(self, four_body_decay: Decay):
        builder = CascadeAmplitudeBuilder(four_body_decay, min_ls=False)
        with pytest.raises(ValueError, match="LS-couplings are missing"):
            builder.formulate()


class TestSpinlessFinalState:
    """A decay into spinless particles requires no spin alignment at all."""

    @pytest.fixture(scope="session")
    def models(self, spinless_four_body_decay: Decay):
        builder = CascadeAmplitudeBuilder(spinless_four_body_decay)
        return [
            builder.formulate(reference_topology=topology)
            for topology in (((1, 2), (3, 4)), (((1, 2), 3), 4))
        ]

    def test_no_wigner_rotations(self, models):
        for model in models:
            rotations = [
                descriptor
                for descriptor in model.kinematic_variables.values()
                if isinstance(descriptor, WignerRotation)
            ]
            assert rotations == []

    def test_intensity_is_reference_independent(self, models):
        """Without Wigner rotations, the two models are literally the same."""
        model_a, model_b = models
        assert set(model_a.kinematic_variables) == set(model_b.kinematic_variables)
        assert model_a.full_expression == model_b.full_expression


class TestDalitzPlotDecompositionEquivalence:
    r"""For a three-body decay, the cascade model reproduces the DPD model.

    Both builders implement the same helicity conventions, so the two intensities have
    to agree point by point over the phase space, for each choice of reference
    subsystem. The angles of the DPD model are computed from the Mandelstam variables,
    those of the cascade model with :code:`decayangle`.
    """

    @pytest.fixture(scope="session")
    def decay(self) -> ThreeBodyDecay:
        """Lambda_c+ -> p K- pi+ with a Lambda(1520) and a Delta(1232)++ chain."""
        lambda_c = State(
            "Lambda_c+",
            R"\Lambda_c^+",
            spin=0.5,
            parity=+1,
            mass=2.28646,
            width=0.0,
            index=0,
        )
        proton = State("p", "p", spin=0.5, parity=+1, mass=0.93827, width=0.0, index=1)
        kaon = State("K-", "K^-", spin=0, parity=-1, mass=0.49368, width=0.0, index=2)
        pion = State(
            "pi+", R"\pi^+", spin=0, parity=-1, mass=0.13957, width=0.0, index=3
        )
        lambda_1520 = Particle(
            "Lambda(1520)",
            R"\Lambda(1520)",
            spin=1.5,
            parity=-1,
            mass=1.51945,
            width=0.0157,
        )
        delta = Particle(
            "Delta(1232)++",
            R"\Delta(1232)^{++}",
            spin=1.5,
            parity=+1,
            mass=1.232,
            width=0.117,
        )
        return ThreeBodyDecay(
            states={s.index: s for s in [lambda_c, proton, kaon, pion]},
            chains=[
                ThreeBodyDecayChain(
                    decay=IsobarNode(
                        parent=lambda_c,
                        child1=IsobarNode(lambda_1520, proton, kaon),
                        child2=pion,
                    )
                ),
                ThreeBodyDecayChain(
                    decay=IsobarNode(
                        parent=lambda_c,
                        child1=IsobarNode(delta, proton, pion),
                        child2=kaon,
                    )
                ),
            ],
        )

    @pytest.fixture(scope="session")
    def momenta(self, decay: ThreeBodyDecay) -> dict[int, np.ndarray]:
        return generate_momenta(decay, n_events=50, seed=0)

    def test_generated_momenta_are_physical(self, decay, momenta):
        total = sum(momenta.values())
        np.testing.assert_allclose(total[:, :3], 0, atol=1e-14)
        np.testing.assert_allclose(total[:, 3], decay.initial_state.mass)
        for i, state in decay.final_state.items():
            mass_squared = _compute_invariant_mass_squared(momenta, (i,))
            np.testing.assert_allclose(np.sqrt(mass_squared), state.mass)

    @pytest.mark.parametrize(
        ("reference_subsystem", "reference_topology"),
        [(3, ((1, 2), 3)), (2, ((1, 3), 2))],
    )
    def test_intensities_are_identical(
        self,
        decay: ThreeBodyDecay,
        momenta: dict[int, np.ndarray],
        reference_subsystem: int,
        reference_topology: tuple,
    ):
        dpd_builder = DalitzPlotDecompositionBuilder(decay)
        for chain in decay.chains:
            dpd_builder.dynamics_choices.register_builder(chain, _formulate_chain_bw)
        dpd_model = dpd_builder.formulate(reference_subsystem=reference_subsystem)

        cascade_builder = CascadeAmplitudeBuilder(decay)
        for chain in decay.chains:
            cascade_builder.dynamics_choices.register_builder(
                chain.resonance.name, _formulate_node_bw
            )
        cascade_model = cascade_builder.formulate(reference_topology=reference_topology)

        dpd_intensities = _evaluate_dpd_model(dpd_model, momenta)
        cascade_intensities = _evaluate_cascade_model(cascade_model, momenta)
        assert dpd_intensities.min() > 0
        np.testing.assert_allclose(cascade_intensities, dpd_intensities, rtol=1e-8)


def _formulate_chain_bw(chain: ThreeBodyDecayChain) -> DefinedExpression:
    """Breit-Wigner over the Mandelstam variable of the spectator of a DPD chain."""
    sigma = sp.Symbol(f"sigma{chain.spectator.index}", nonnegative=True)
    return _formulate_breit_wigner(chain.resonance, sigma)


def _formulate_node_bw(node: IsobarNode) -> DefinedExpression:
    """Breit-Wigner over the invariant mass of the decay products of a cascade node."""
    mass = create_invariant_mass_symbol(get_final_state_ids(node))
    return _formulate_breit_wigner(node.parent, mass**2)


def _formulate_breit_wigner(
    resonance: Particle, mass_squared: sp.Expr
) -> DefinedExpression:
    m0 = sp.Symbol(Rf"m_{{{resonance.latex}}}", nonnegative=True)
    gamma0 = sp.Symbol(Rf"\Gamma_{{{resonance.latex}}}", nonnegative=True)
    return DefinedExpression(
        expression=1 / (m0**2 - mass_squared - sp.I * m0 * gamma0),
        parameters={m0: resonance.mass, gamma0: resonance.width},
    )


def _evaluate_dpd_model(model, momenta: dict[int, np.ndarray]) -> np.ndarray:
    expression = model.full_expression.xreplace(dict(model.parameter_defaults)).doit()
    sigma3, sigma3_expression = list(model.invariants.items())[2]
    angle_definitions = {
        symbol: definition
        .doit()
        .xreplace({sigma3: sigma3_expression})
        .xreplace(model.masses)
        for symbol, definition in model.variables.items()
    }
    expression = (
        expression
        .xreplace(angle_definitions)
        .xreplace({sigma3: sigma3_expression})
        .xreplace(model.masses)
    )
    sigma1, sigma2 = sp.symbols("sigma1 sigma2", nonnegative=True)
    func = sp.lambdify([sigma1, sigma2], expression, "numpy", cse=True)
    return np.real(
        func(
            _compute_invariant_mass_squared(momenta, (2, 3)),
            _compute_invariant_mass_squared(momenta, (3, 1)),
        )
    )


def _evaluate_cascade_model(model, momenta: dict[int, np.ndarray]) -> np.ndarray:
    data = compute_kinematic_variables(model, momenta)
    expression = model.full_expression.xreplace(dict(model.parameter_defaults)).doit()
    symbols = sorted(model.kinematic_variables, key=str)
    func = sp.lambdify(symbols, expression, "numpy", cse=True)
    return np.real(func(*(data[s] for s in symbols)))


def compute_kinematic_variables(
    model, momenta: dict[int, np.ndarray]
) -> dict[sp.Symbol, np.ndarray]:
    """Compute the kinematic variables of a cascade model with :code:`decayangle`.

    This mirrors the procedure that is described in the :doc:`/cascade` notebook and
    in the docstrings of `.HelicityAngle`, `.WignerRotation`, and
    `~ampform.amplitude.cascade.InvariantMass`.
    """
    decay_topology = pytest.importorskip("decayangle.decay_topology")
    data: dict[sp.Symbol, np.ndarray] = {}
    helicity_angles: dict[tuple, dict] = {}
    wigner_angles: dict[tuple, dict] = {}
    for symbol, descriptor in model.kinematic_variables.items():
        if isinstance(descriptor, HelicityAngle):
            angles = helicity_angles.get(descriptor.topology)
            if angles is None:
                topology = decay_topology.Topology(
                    0, decay_topology=descriptor.topology
                )
                angles = helicity_angles.setdefault(
                    descriptor.topology, topology.helicity_angles(momenta)
                )
            data[symbol] = getattr(angles[descriptor.node], descriptor.angle)
        elif isinstance(descriptor, WignerRotation):
            key = (descriptor.reference_topology, descriptor.topology)
            angles = wigner_angles.get(key)
            if angles is None:
                reference = decay_topology.Topology(
                    0, decay_topology=descriptor.reference_topology
                )
                topology = decay_topology.Topology(
                    0, decay_topology=descriptor.topology
                )
                angles = wigner_angles.setdefault(
                    key, reference.relative_wigner_angles(topology, momenta)
                )
            data[symbol] = getattr(angles[descriptor.state_id], descriptor.angle)
        elif isinstance(descriptor, InvariantMass):
            mass_squared = _compute_invariant_mass_squared(
                momenta, descriptor.state_ids
            )
            data[symbol] = np.sqrt(mass_squared)
    return data


def generate_momenta(decay: Decay, n_events: int, seed: int) -> dict[int, np.ndarray]:
    r"""Generate four-momenta :math:`(p_x, p_y, p_z, E)` in the initial-state frame.

    Momenta are generated by recursively splitting the topology of the first decay
    chain into two-body decays with a random orientation. The generated points cover
    the phase space, but they are not distributed over it: the tests only require valid
    kinematic configurations, not a correctly weighted sample.
    """
    rng = np.random.default_rng(seed)
    minimum_masses = {i: state.mass for i, state in decay.final_state.items()}
    mass = np.full(n_events, decay.initial_state.mass)
    return _decay_node(to_topology(decay.chains[0]), mass, minimum_masses, rng)


def _decay_node(
    node: tuple | int,
    mass: np.ndarray,
    minimum_masses: dict[int, float],
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    """Compute the final-state momenta of a node in the rest frame of that node."""
    if isinstance(node, int):
        return {node: _to_four_momentum(np.zeros_like(mass), 0, 1, mass)}
    child1, child2 = node
    maximum = mass - _minimum_mass(child2, minimum_masses)
    mass1 = _sample_mass(child1, maximum, minimum_masses, rng)
    mass2 = _sample_mass(child2, mass - mass1, minimum_masses, rng)
    q = _breakup_momentum(mass, mass1, mass2)
    phi = rng.uniform(-np.pi, np.pi, size=mass.shape)
    cos_theta = rng.uniform(-1, 1, size=mass.shape)
    momenta = {}
    for child, child_mass, momentum in [
        (child1, mass1, _to_four_momentum(+q, phi, cos_theta, mass1)),
        (child2, mass2, _to_four_momentum(-q, phi, cos_theta, mass2)),
    ]:
        for i, p in _decay_node(child, child_mass, minimum_masses, rng).items():
            momenta[i] = _boost(p, momentum, child_mass)
    return momenta


def _sample_mass(
    node: tuple | int,
    maximum: np.ndarray,
    minimum_masses: dict[int, float],
    rng: np.random.Generator,
) -> np.ndarray:
    minimum = _minimum_mass(node, minimum_masses)
    if isinstance(node, int):
        return np.full_like(maximum, minimum)
    return rng.uniform(minimum, maximum)


def _minimum_mass(node: tuple | int, minimum_masses: dict[int, float]) -> float:
    if isinstance(node, int):
        return minimum_masses[node]
    return sum(_minimum_mass(child, minimum_masses) for child in node)


def _breakup_momentum(m0, m1, m2):
    return np.sqrt((m0**2 - (m1 + m2) ** 2) * (m0**2 - (m1 - m2) ** 2)) / (2 * m0)


def _to_four_momentum(momentum, phi, cos_theta, mass) -> np.ndarray:
    sin_theta = np.sqrt(1 - np.asarray(cos_theta) ** 2)
    return np.stack(
        [
            momentum * sin_theta * np.cos(phi),
            momentum * sin_theta * np.sin(phi),
            momentum * cos_theta,
            np.sqrt(momentum**2 + mass**2),
        ],
        axis=-1,
    )


def _boost(p: np.ndarray, momentum: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Boost :code:`p` from the rest frame of a particle to the frame of that particle.

    The particle is given by its four-`momentum` in the target frame and its invariant
    `mass`.
    """
    overlap = np.sum(p[:, :3] * momentum[:, :3], axis=-1)
    energy = (p[:, 3] * momentum[:, 3] + overlap) / mass
    factor = (overlap / (momentum[:, 3] + mass) + p[:, 3]) / mass
    return np.concatenate(
        [p[:, :3] + factor[:, None] * momentum[:, :3], energy[:, None]],
        axis=-1,
    )


def _compute_invariant_mass_squared(
    momenta: dict[int, np.ndarray], state_ids: tuple[int, ...]
) -> np.ndarray:
    p = np.sum([momenta[i] for i in state_ids], axis=0)
    return p[..., 3] ** 2 - np.sum(p[..., :3] ** 2, axis=-1)
