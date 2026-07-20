from __future__ import annotations

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
from ampform.decay import (
    Decay,
    DecayChain,
    IsobarNode,
    Particle,
    State,
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
