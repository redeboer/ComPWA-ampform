from __future__ import annotations

import pytest
import sympy as sp
from qrules import ReactionInfo

from ampform import get_builder
from ampform.helicity import (
    HelicityAmplitudeBuilder,
    _generate_kinematic_variables,
    formulate_isobar_wigner_d,
    group_by_spin_projection,
)


class TestHelicityAmplitudeBuilder:
    @pytest.mark.parametrize("permutate_topologies", [False, True])
    @pytest.mark.parametrize("stable_final_state_ids", [None, (1, 2), (0, 1, 2)])
    def test_formulate(
        self,
        permutate_topologies,
        reaction: ReactionInfo,
        stable_final_state_ids,
    ):
        # pylint: disable=too-many-locals
        if reaction.formalism == "canonical-helicity":
            n_amplitudes = 16
            n_parameters = 4
        else:
            n_amplitudes = 8
            n_parameters = 2
        n_kinematic_variables = 9
        n_symbols = 4 + n_parameters
        if stable_final_state_ids is not None:
            n_parameters += len(stable_final_state_ids)
            n_kinematic_variables -= len(stable_final_state_ids)

        model_builder: HelicityAmplitudeBuilder = get_builder(reaction)
        model_builder.config.stable_final_state_ids = stable_final_state_ids
        if permutate_topologies:
            model_builder.adapter.permutate_registered_topologies()
            n_kinematic_variables += 10

        model = model_builder.formulate()
        assert len(model.parameter_defaults) == n_parameters
        assert len(model.components) == 4 + n_amplitudes
        assert len(model.expression.free_symbols) == n_symbols
        assert len(model.kinematic_variables) == n_kinematic_variables

        variables = set(model.kinematic_variables)
        paremeters = set(model.parameter_defaults)
        free_symbols = model.expression.free_symbols
        undefined_symbols = free_symbols - paremeters - variables
        assert not undefined_symbols

        final_state_masses = set(sp.symbols("m_(0:3)", nonnegative=True))
        stable_final_state_masses = set()
        if stable_final_state_ids is not None:
            stable_final_state_masses = {
                sp.Symbol(f"m_{i}", nonnegative=True) for i in stable_final_state_ids
            }
        unstable_final_state_masses = final_state_masses - stable_final_state_masses
        assert stable_final_state_masses <= paremeters
        assert unstable_final_state_masses <= variables

        no_dynamics: sp.Expr = model.expression.doit()
        no_dynamics = no_dynamics.subs(model.parameter_defaults)
        assert len(no_dynamics.free_symbols) == 1

        existing_theta = next(iter(no_dynamics.free_symbols))
        theta = sp.Symbol("theta", real=True)
        no_dynamics = no_dynamics.subs({existing_theta: theta})
        no_dynamics = no_dynamics.trigsimp()

        if reaction.formalism == "canonical-helicity":
            assert (
                no_dynamics
                == 0.8 * sp.sqrt(10) * sp.cos(theta) ** 2
                + 4.4 * sp.cos(theta) ** 2
                + 0.8 * sp.sqrt(10)
                + 4.4
            )
        else:
            assert no_dynamics == 8.0 - 4.0 * sp.sin(theta) ** 2

    def test_stable_final_state_ids(self, reaction: ReactionInfo):
        builder: HelicityAmplitudeBuilder = get_builder(reaction)
        assert builder.config.stable_final_state_ids is None
        builder.config.stable_final_state_ids = (1, 2)  # type: ignore[assignment]
        assert builder.config.stable_final_state_ids == {1, 2}

    def test_scalar_initial_state(self, reaction: ReactionInfo):
        builder: HelicityAmplitudeBuilder = get_builder(reaction)
        assert builder.config.scalar_initial_state_mass is False
        initial_state_mass = sp.Symbol("m_012", nonnegative=True)

        model = builder.formulate()
        assert initial_state_mass in model.kinematic_variables
        assert initial_state_mass not in model.parameter_defaults

        builder.config.scalar_initial_state_mass = True
        model = builder.formulate()
        assert initial_state_mass not in model.kinematic_variables
        assert initial_state_mass in model.parameter_defaults

    def test_use_helicity_couplings(self, reaction: ReactionInfo):
        # cspell:ignore coeff
        builder: HelicityAmplitudeBuilder = get_builder(reaction)
        builder.config.use_helicity_couplings = False
        coeff_model = builder.formulate()
        builder.config.use_helicity_couplings = True
        coupling_model = builder.formulate()

        coefficient_names = {p.name for p in coeff_model.parameter_defaults}
        coupling_names = {p.name for p in coupling_model.parameter_defaults}
        if reaction.formalism == "canonical-helicity":
            assert len(coefficient_names) == 4
            assert coefficient_names == {
                R"C_{J/\psi(1S) \xrightarrow[S=1]{L=0} f_{0}(980) \gamma;"
                R" f_{0}(980) \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
                R"C_{J/\psi(1S) \xrightarrow[S=1]{L=2} f_{0}(1500) \gamma;"
                R" f_{0}(1500) \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
                R"C_{J/\psi(1S) \xrightarrow[S=1]{L=0} f_{0}(1500) \gamma;"
                R" f_{0}(1500) \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
                R"C_{J/\psi(1S) \xrightarrow[S=1]{L=2} f_{0}(980) \gamma;"
                R" f_{0}(980) \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
            }
            assert len(coupling_names) == 6
            assert coupling_names == {
                R"H_{J/\psi(1S) \xrightarrow[S=1]{L=0} f_{0}(1500) \gamma}",
                R"H_{J/\psi(1S) \xrightarrow[S=1]{L=0} f_{0}(980) \gamma}",
                R"H_{J/\psi(1S) \xrightarrow[S=1]{L=2} f_{0}(1500) \gamma}",
                R"H_{J/\psi(1S) \xrightarrow[S=1]{L=2} f_{0}(980) \gamma}",
                R"H_{f_{0}(1500) \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
                R"H_{f_{0}(980) \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
            }
        else:
            assert len(coefficient_names) == 2
            assert coefficient_names == {
                R"C_{J/\psi(1S) \to {f_{0}(980)}_{0} \gamma_{+1}; f_{0}(980)"
                R" \to \pi^{0}_{0} \pi^{0}_{0}}",
                R"C_{J/\psi(1S) \to {f_{0}(1500)}_{0} \gamma_{+1};"
                R" f_{0}(1500) \to \pi^{0}_{0} \pi^{0}_{0}}",
            }
            assert len(coupling_names) == 6
            assert coupling_names == {
                R"H_{J/\psi(1S) \to {f_{0}(1500)}_{0} \gamma_{+1}}",
                R"H_{J/\psi(1S) \to {f_{0}(1500)}_{0} \gamma_{-1}}",
                R"H_{J/\psi(1S) \to {f_{0}(980)}_{0} \gamma_{+1}}",
                R"H_{J/\psi(1S) \to {f_{0}(980)}_{0} \gamma_{-1}}",
                R"H_{f_{0}(1500) \to \pi^{0}_{0} \pi^{0}_{0}}",
                R"H_{f_{0}(980) \to \pi^{0}_{0} \pi^{0}_{0}}",
            }


@pytest.mark.parametrize(
    ("node_id", "mass", "phi", "theta"),
    [
        (0, "m_012", "phi_0", "theta_0"),
        (1, "m_12", "phi_1^12", "theta_1^12"),
    ],
)
def test_generate_kinematic_variables(
    reaction: ReactionInfo,
    node_id: int,
    mass: str,
    phi: str,
    theta: str,
):
    for transition in reaction.transitions:
        variables = _generate_kinematic_variables(transition, node_id)
        assert variables[0].name == mass
        assert variables[1].name == phi
        assert variables[2].name == theta


@pytest.mark.parametrize(
    ("transition", "node_id", "expected"),
    [
        (0, 0, "WignerD(1, -1, -1, -phi_0, theta_0, 0)"),
        (0, 1, "WignerD(0, 0, 0, -phi_1^12, theta_1^12, 0)"),
        (1, 0, "WignerD(1, -1, 1, -phi_0, theta_0, 0)"),
        (1, 1, "WignerD(0, 0, 0, -phi_1^12, theta_1^12, 0)"),
        (2, 0, "WignerD(1, 1, -1, -phi_0, theta_0, 0)"),
        (2, 1, "WignerD(0, 0, 0, -phi_1^12, theta_1^12, 0)"),
    ],
)
def test_formulate_isobar_wigner_d(
    reaction: ReactionInfo, transition: int, node_id: int, expected: str
):
    if reaction.formalism == "canonical-helicity":
        transition *= 2
    transitions = [
        t for t in reaction.transitions if t.states[3].particle.name == "f(0)(980)"
    ]
    some_transition = transitions[transition]
    wigner_d = formulate_isobar_wigner_d(some_transition, node_id)
    assert str(wigner_d) == expected


def test_group_by_spin_projection(reaction: ReactionInfo):
    transition_groups = group_by_spin_projection(reaction.transitions)
    assert len(transition_groups) == 4
    for group in transition_groups:
        transition_iter = iter(group)
        first_transition = next(transition_iter)
        for transition in transition_iter:
            assert transition.initial_states == first_transition.initial_states
            assert transition.final_states == first_transition.final_states
