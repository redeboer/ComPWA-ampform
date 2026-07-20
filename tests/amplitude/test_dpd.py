# cspell:ignore pksigma
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import qrules
import sympy as sp

from ampform.adapter.qrules import normalize_state_ids, to_three_body_decay
from ampform.amplitude.dpd import (
    AmplitudeModel,
    DalitzPlotDecompositionBuilder,
    _get_best_reference_subsystems,
)
from ampform.decay import (
    IsobarNode,
    Particle,
    State,
    ThreeBodyDecay,
    ThreeBodyDecayChain,
)
from ampform.dynamics.builder import formulate_breit_wigner_with_form_factor
from ampform.sympy import PoolSum

if TYPE_CHECKING:
    from qrules.transition import ReactionInfo


class TestDalitzPlotDecompositionBuilder:
    @pytest.mark.parametrize("all_subsystems", [False, True])
    @pytest.mark.parametrize("min_ls", [False, True])
    def test_all_subsystems(
        self, jpsi2pksigma_reaction: ReactionInfo, all_subsystems: bool, min_ls: bool
    ):
        if jpsi2pksigma_reaction.formalism == "helicity" and not min_ls:
            pytest.skip("Helicity formalism with all LS not supported")
        transitions = normalize_state_ids(jpsi2pksigma_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=min_ls)
        builder = DalitzPlotDecompositionBuilder(
            decay, min_ls=min_ls, all_subsystems=all_subsystems
        )
        if jpsi2pksigma_reaction.formalism == "canonical-helicity":
            for chain in builder.decay.chains:
                builder.dynamics_choices.register_builder(
                    chain, formulate_breit_wigner_with_form_factor
                )
        if all_subsystems:
            with pytest.warns(
                UserWarning,
                match=r"Decay J/psi\(1S\) → 1: K0, 2: Sigma\+, 3: p~ only has subsystems 2, 3, not 1",
            ):
                model = builder.formulate(reference_subsystem=2)
        else:
            model = builder.formulate(reference_subsystem=2)
        expected_variables = {
            R"\zeta^0_{2(2)}",
            R"\zeta^0_{3(2)}",
            R"\zeta^2_{2(2)}",
            R"\zeta^2_{3(2)}",
            R"\zeta^3_{2(2)}",
            R"\zeta^3_{3(2)}",
            "theta_12",
            "theta_23",
            "theta_31",
        }
        if not all_subsystems:
            expected_variables.remove("theta_23")
        assert {str(s) for s in model.variables} == expected_variables

    @pytest.mark.parametrize("min_ls", [False, True])
    @pytest.mark.parametrize("use_coefficients", [False, True])
    def test_use_coefficients(
        self, jpsi2pksigma_reaction: ReactionInfo, min_ls: bool, use_coefficients: bool
    ):
        if jpsi2pksigma_reaction.formalism == "helicity" and not min_ls:
            pytest.skip("Helicity formalism with all LS not supported")
        transitions = normalize_state_ids(jpsi2pksigma_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=min_ls)
        builder = DalitzPlotDecompositionBuilder(decay, min_ls=min_ls)
        model = builder.formulate(
            reference_subsystem=2,
            use_coefficients=use_coefficients,
        )
        amplitudes = _get_physical_amplitudes(model)
        coupling_symbols = _collect_indexed_symbols(amplitudes)

        n_coupling_symbols = len(coupling_symbols)
        coupling_symbols_str = sorted(str(s) for s in coupling_symbols)
        # ----==== COEFFICIENTS ===--- #
        if use_coefficients:
            if min_ls:  # HELICITY BASIS
                assert n_coupling_symbols == 20
                assert coupling_symbols_str == [
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[-1/2, -1/2, 0, -1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[-1/2, -1/2, 0, 1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[-1/2, 1/2, 0, -1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[-1/2, 1/2, 0, 1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[-3/2, -1/2, 0, -1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[-3/2, -1/2, 0, 1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[1/2, -1/2, 0, -1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[1/2, -1/2, 0, 1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[1/2, 1/2, 0, -1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[1/2, 1/2, 0, 1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[3/2, 1/2, 0, -1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[3/2, 1/2, 0, 1/2]",
                    R"\mathcal{H}^\mathrm{\overline{\Sigma}(1660)^{-}}[-1/2, -1/2, -1/2, 0]",
                    R"\mathcal{H}^\mathrm{\overline{\Sigma}(1660)^{-}}[-1/2, -1/2, 1/2, 0]",
                    R"\mathcal{H}^\mathrm{\overline{\Sigma}(1660)^{-}}[-1/2, 1/2, -1/2, 0]",
                    R"\mathcal{H}^\mathrm{\overline{\Sigma}(1660)^{-}}[-1/2, 1/2, 1/2, 0]",
                    R"\mathcal{H}^\mathrm{\overline{\Sigma}(1660)^{-}}[1/2, -1/2, -1/2, 0]",
                    R"\mathcal{H}^\mathrm{\overline{\Sigma}(1660)^{-}}[1/2, -1/2, 1/2, 0]",
                    R"\mathcal{H}^\mathrm{\overline{\Sigma}(1660)^{-}}[1/2, 1/2, -1/2, 0]",
                    R"\mathcal{H}^\mathrm{\overline{\Sigma}(1660)^{-}}[1/2, 1/2, 1/2, 0]",
                ]
            else:  # CANONICAL BASIS
                assert n_coupling_symbols == 4
                assert coupling_symbols_str == [
                    R"\mathcal{H}^\mathrm{LS,N(1700)^{+}}[1, 1, 2, 1/2]",
                    R"\mathcal{H}^\mathrm{LS,N(1700)^{+}}[1, 2, 2, 1/2]",
                    R"\mathcal{H}^\mathrm{LS,\overline{\Sigma}(1660)^{-}}[0, 1, 1, 1/2]",
                    R"\mathcal{H}^\mathrm{LS,\overline{\Sigma}(1660)^{-}}[2, 1, 1, 1/2]",
                ]
        # ----==== COUPLING ===--- #
        else:
            n_products = len(_collect_products(amplitudes))
            if min_ls:  # HELICITY BASIS
                assert n_coupling_symbols == 14
                assert n_products == 20
                assert coupling_symbols_str == [
                    R"\mathcal{H}^\mathrm{decay}[N(1700)^{+}, 0, -1/2]",
                    R"\mathcal{H}^\mathrm{decay}[N(1700)^{+}, 0, 1/2]",
                    R"\mathcal{H}^\mathrm{decay}[\overline{\Sigma}(1660)^{-}, -1/2, 0]",
                    R"\mathcal{H}^\mathrm{decay}[\overline{\Sigma}(1660)^{-}, 1/2, 0]",
                    R"\mathcal{H}^\mathrm{production}[N(1700)^{+}, -1/2, -1/2]",
                    R"\mathcal{H}^\mathrm{production}[N(1700)^{+}, -1/2, 1/2]",
                    R"\mathcal{H}^\mathrm{production}[N(1700)^{+}, -3/2, -1/2]",
                    R"\mathcal{H}^\mathrm{production}[N(1700)^{+}, 1/2, -1/2]",
                    R"\mathcal{H}^\mathrm{production}[N(1700)^{+}, 1/2, 1/2]",
                    R"\mathcal{H}^\mathrm{production}[N(1700)^{+}, 3/2, 1/2]",
                    R"\mathcal{H}^\mathrm{production}[\overline{\Sigma}(1660)^{-}, -1/2, -1/2]",
                    R"\mathcal{H}^\mathrm{production}[\overline{\Sigma}(1660)^{-}, -1/2, 1/2]",
                    R"\mathcal{H}^\mathrm{production}[\overline{\Sigma}(1660)^{-}, 1/2, -1/2]",
                    R"\mathcal{H}^\mathrm{production}[\overline{\Sigma}(1660)^{-}, 1/2, 1/2]",
                ]
            else:  # CANONICAL BASIS
                assert n_coupling_symbols == 6
                assert n_products == 4
                assert coupling_symbols_str == [
                    R"\mathcal{H}^\mathrm{LS,decay}[N(1700)^{+}, 2, 1/2]",
                    R"\mathcal{H}^\mathrm{LS,decay}[\overline{\Sigma}(1660)^{-}, 1, 1/2]",
                    R"\mathcal{H}^\mathrm{LS,production}[N(1700)^{+}, 1, 1]",
                    R"\mathcal{H}^\mathrm{LS,production}[N(1700)^{+}, 1, 2]",
                    R"\mathcal{H}^\mathrm{LS,production}[\overline{\Sigma}(1660)^{-}, 0, 1]",
                    R"\mathcal{H}^\mathrm{LS,production}[\overline{\Sigma}(1660)^{-}, 2, 1]",
                ]

    @pytest.mark.parametrize("basis", ["canonical", "helicity"])
    @pytest.mark.parametrize("resonance", ["N(1675)", "Sigma(1775)"])
    def test_use_coefficients_combinations(self, basis: str, resonance: str):
        reaction = qrules.generate_transitions(
            initial_state=[("J/psi(1S)", [+1])],
            final_state=[("Sigma+", [+0.5]), "K0", ("p~", [+0.5])],
            allowed_interaction_types="strong",
            allowed_intermediate_particles=[resonance],
            formalism="canonical-helicity",
        )
        transitions = normalize_state_ids(reaction.transitions)
        min_ls = basis == "helicity"
        decay = to_three_body_decay(transitions, min_ls)
        builder = DalitzPlotDecompositionBuilder(decay, min_ls)
        # cspell:ignore coeff
        reference_subsystem = 1 if resonance.startswith("Sigma") else 3
        coupling_model = builder.formulate(reference_subsystem)
        coeff_model = builder.formulate(reference_subsystem, use_coefficients=True)
        coupling_amplitudes = _get_physical_amplitudes(coupling_model)
        coeff_amplitudes = _get_physical_amplitudes(coeff_model)

        couplings = _collect_indexed_symbols(coupling_amplitudes)
        coefficients = _collect_indexed_symbols(coeff_amplitudes)
        coupling_products = _collect_products(coupling_amplitudes)

        n_couplings = len(couplings)
        n_decay_couplings = len({s for s in couplings if "decay" in s.name})
        n_production_couplings = len({s for s in couplings if "production" in s.name})
        assert n_couplings == n_decay_couplings + n_production_couplings

        n_coupling_products = len(coupling_products)
        n_coefficients = len(coefficients)
        assert n_coefficients == n_coupling_products
        assert n_coefficients == n_decay_couplings * n_production_couplings


class TestPolarizedIntensity:
    @pytest.fixture(scope="session")
    def models(
        self, jpsi2pksigma_reaction: ReactionInfo
    ) -> tuple[AmplitudeModel, AmplitudeModel]:
        if jpsi2pksigma_reaction.formalism != "helicity":
            pytest.skip("Polarized intensity is tested in the helicity basis only")
        transitions = normalize_state_ids(jpsi2pksigma_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=True)
        builder = DalitzPlotDecompositionBuilder(decay, min_ls=True)
        unpolarized = builder.formulate(reference_subsystem=2)
        polarized = builder.formulate(reference_subsystem=2, polarized=True)
        return unpolarized, polarized

    def test_intensity_is_matrix_sandwich(
        self, models: tuple[AmplitudeModel, AmplitudeModel]
    ):
        _, model = models
        intensity = model.intensity
        assert isinstance(intensity, sp.Trace)
        matrix_symbols = {str(s): s for s in model.matrices}
        assert set(matrix_symbols) == {R"\rho", R"\overline{D}", "O"}
        for symbol, matrix in model.matrices.items():
            assert symbol.shape == (3, 3)
            assert matrix.shape == (3, 3)
        for name in (R"\rho", "O"):
            matrix = model.matrices[matrix_symbols[name]]
            for i in range(3):
                for j in range(i + 1, 3):
                    assert matrix[j, i] == sp.conjugate(matrix[i, j])

    def test_spin_density_defaults_are_unpolarized(
        self, models: tuple[AmplitudeModel, AmplitudeModel]
    ):
        _, model = models
        ρ = sp.IndexedBase(R"\rho")
        spin_projections = [-1, 0, +1]
        for i, m in enumerate(spin_projections):
            for j, m_prime in enumerate(spin_projections):
                if i > j:
                    assert ρ[m, m_prime] not in model.parameter_defaults
                elif i == j:
                    assert model.parameter_defaults[ρ[m, m_prime]] == 1 / 3
                else:
                    assert model.parameter_defaults[ρ[m, m_prime]] == 0

    @pytest.mark.slow
    def test_unpolarized_limit(self, models: tuple[AmplitudeModel, AmplitudeModel]):
        unpolarized, polarized = models
        funcs = [
            _lambdify_over_mandelstam_and_orientation(model)
            for model in (unpolarized, polarized)
        ]
        f_unpolarized, f_polarized = funcs
        σ1, σ2 = 4.6, 3.1  # inside phase space
        for angles in [(0.3, 0.7, -1.1), (2.0, 1.4, 0.5)]:
            intensity = f_polarized(σ1, σ2, *angles)
            assert intensity.imag == pytest.approx(0, abs=1e-12)
            n_initial_spin_states = 3
            assert n_initial_spin_states * intensity.real == pytest.approx(
                f_unpolarized(σ1, σ2, *angles).real
            )

    @pytest.mark.slow
    def test_longitudinally_polarized_intensity_depends_on_orientation(
        self, models: tuple[AmplitudeModel, AmplitudeModel]
    ):
        _, model = models
        ρ = sp.IndexedBase(R"\rho")
        polarization = {ρ[-1, -1]: 0.5, ρ[0, 0]: 0.0, ρ[+1, +1]: 0.5}
        func = _lambdify_over_mandelstam_and_orientation(model, polarization)
        σ1, σ2 = 4.6, 3.1
        intensities = [func(σ1, σ2, 0.3, θ, -1.1).real for θ in (0.0, 0.7, 1.5)]
        assert len({round(i, 7) for i in intensities}) == len(intensities)


class TestFinalStateDecays:
    """Test attaching a polarimeter decay Σ⁺ → p π⁰ to the final state."""

    @pytest.fixture(scope="session")
    def models(
        self, jpsi2pksigma_reaction: ReactionInfo
    ) -> tuple[AmplitudeModel, AmplitudeModel]:
        if jpsi2pksigma_reaction.formalism != "helicity":
            pytest.skip("Final-state decays are tested in the helicity basis only")
        transitions = normalize_state_ids(jpsi2pksigma_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=True)
        builder = DalitzPlotDecompositionBuilder(decay, min_ls=True)
        without_decay = builder.formulate(reference_subsystem=2)
        with_decay = builder.formulate(
            reference_subsystem=2,
            final_state_decays={2: _create_sigma_decay_products()},
        )
        return without_decay, with_decay

    def test_summation_indices_replaced_by_decay_products(
        self, models: tuple[AmplitudeModel, AmplitudeModel]
    ):
        _, model = models
        intensity = model.intensity
        assert isinstance(intensity, PoolSum)
        helicity_ranges = {str(symbol): values for symbol, values in intensity.indices}
        assert helicity_ranges == {
            "lambda0": (-1, 0, 1),
            "lambda1": (0,),
            "lambda3": (-sp.Rational(1, 2), +sp.Rational(1, 2)),
            R"\lambda_{p}": (-sp.Rational(1, 2), +sp.Rational(1, 2)),
            R"\lambda_{\pi^0}": (0,),
        }

    def test_decay_couplings_have_default_parameters(
        self, models: tuple[AmplitudeModel, AmplitudeModel]
    ):
        _, model = models
        couplings = _get_sigma_decay_couplings(model)
        assert len(couplings) == 2
        assert all(model.parameter_defaults[c] == 1 + 0j for c in couplings)

    def test_polarized_intensity_sums_over_decay_products(
        self, jpsi2pksigma_reaction: ReactionInfo
    ):
        if jpsi2pksigma_reaction.formalism != "helicity":
            pytest.skip("Final-state decays are tested in the helicity basis only")
        transitions = normalize_state_ids(jpsi2pksigma_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=True)
        builder = DalitzPlotDecompositionBuilder(decay, min_ls=True)
        model = builder.formulate(
            reference_subsystem=2,
            polarized=True,
            final_state_decays={2: _create_sigma_decay_products()},
        )
        matrix_element = next(
            expr
            for symbol, expr in model.amplitudes.items()
            if str(symbol).startswith("O")
        )
        assert isinstance(matrix_element, PoolSum)
        index_names = {str(symbol) for symbol, _ in matrix_element.indices}
        assert index_names == {
            "lambda1",
            "lambda3",
            R"\lambda_{p}",
            R"\lambda_{\pi^0}",
        }

    def test_impossible_spin_coupling_raises(self, jpsi2pksigma_reaction: ReactionInfo):
        if jpsi2pksigma_reaction.formalism != "helicity":
            pytest.skip("Final-state decays are tested in the helicity basis only")
        transitions = normalize_state_ids(jpsi2pksigma_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=True)
        builder = DalitzPlotDecompositionBuilder(decay, min_ls=True)
        proton, pion = _create_sigma_decay_products()
        with pytest.raises(ValueError, match="cannot be coupled"):
            builder.formulate(
                reference_subsystem=2,
                final_state_decays={1: (proton, pion)},  # K0 → p π0 is impossible
            )

    @pytest.mark.slow
    def test_intensity_integrates_to_two_pi_times_undecayed(
        self, models: tuple[AmplitudeModel, AmplitudeModel]
    ):
        """Check the normalization :math:`\\int I \\, d\\Omega_p = 2\\pi I_0`.

        The couplings for the weak decay Σ⁺ → p π⁰ are fixed from the decay-asymmetry
        parameter alpha through :math:`H_{\\pm 1/2} = \\sqrt{(1 \\pm \\alpha)/2}`, so that
        :math:`\\sum_{\\lambda_p} |H_{\\lambda_p}|^2 = 1`.
        """
        without_decay, with_decay = models
        asymmetric_couplings: dict[sp.Basic, complex] = {
            symbol: 0.7 * np.exp(0.9j) if ", 1/2," in str(symbol) else 1
            for symbol in with_decay.parameter_defaults
            if str(symbol).startswith(R"\mathcal{H}^\mathrm{production}")
        }
        α = -0.98
        h_minus, h_plus = _get_sigma_decay_couplings(with_decay)
        weak_couplings = {
            h_plus: np.sqrt((1 + α) / 2),
            h_minus: np.sqrt((1 - α) / 2),
        }
        f_undecayed = _lambdify_over_mandelstam_and_orientation(
            without_decay, asymmetric_couplings, orientation_symbols=()
        )
        f_decayed = _lambdify_over_mandelstam_and_orientation(
            with_decay,
            asymmetric_couplings | weak_couplings,
            orientation_symbols=(R"\phi_{p}", R"\theta_{p}"),
        )
        σ1, σ2 = 5.63, 2.7  # inside phase space
        intensity_undecayed = f_undecayed(σ1, σ2).real
        φ_values = np.linspace(0, 2 * np.pi, num=64, endpoint=False)
        cos_θ_values, weights = np.polynomial.legendre.leggauss(deg=20)
        θ_values = np.arccos(cos_θ_values)
        intensities = np.array([
            [f_decayed(σ1, σ2, φ, θ).real for θ in θ_values] for φ in φ_values
        ])
        # the integrand is a trigonometric polynomial in φ and, after φ-integration,
        # polynomial in cos(θ), so these quadratures are exact
        integral = (2 * np.pi / len(φ_values)) * np.sum(
            intensities.sum(axis=0) * weights
        )
        assert integral == pytest.approx(2 * np.pi * intensity_undecayed)


def _create_sigma_decay_products() -> tuple[Particle, Particle]:
    proton = Particle("p", latex="p", spin=0.5, parity=+1, mass=0.938, width=0.0)
    pion = Particle("pi0", latex=R"\pi^0", spin=0, parity=-1, mass=0.135, width=0.0)
    return proton, pion


def _get_sigma_decay_couplings(model: AmplitudeModel) -> list[sp.Basic]:
    return sorted(
        (
            symbol
            for symbol in model.parameter_defaults
            if str(symbol).startswith(R"\mathcal{H}^\mathrm{decay}[\Sigma^{+}")
        ),
        key=str,
    )


def _lambdify_over_mandelstam_and_orientation(
    model: AmplitudeModel,
    parameters: dict[sp.Basic, complex] | None = None,
    orientation_symbols: tuple[str, ...] = ("phi", "theta", "chi"),
):
    substitutions = dict(model.parameter_defaults)
    if parameters is not None:
        substitutions.update(parameters)
    expr = model.full_expression.xreplace(substitutions).doit()
    kinematic_definitions = {k: v.doit() for k, v in model.variables.items()}
    σ3, σ3_expr = list(model.invariants.items())[2]
    kinematic_definitions = {
        k: v.xreplace({σ3: σ3_expr}).xreplace(model.masses)
        for k, v in kinematic_definitions.items()
    }
    expr = expr.xreplace(kinematic_definitions).xreplace({σ3: σ3_expr})
    expr = expr.xreplace(model.masses)
    σ1, σ2 = sp.symbols("sigma1 sigma2", nonnegative=True)
    orientation = tuple(sp.Symbol(s, real=True) for s in orientation_symbols)
    func = sp.lambdify([σ1, σ2, *orientation], expr, "numpy")
    return lambda *args: complex(func(*args))


def test_no_zero_helicity_for_massless_states():
    """Massless final states are summed over helicities -1 and +1 only."""
    ψ = State("psi", latex=R"\psi", spin=1, parity=-1, mass=3.1, width=0.0, index=0)
    γ = State("gamma", latex=R"\gamma", spin=1, parity=-1, mass=0.0, width=0.0, index=1)
    π2 = State("pi0", latex=R"\pi^0", spin=0, parity=-1, mass=0.135, width=0.0, index=2)
    π3 = State("pi0", latex=R"\pi^0", spin=0, parity=-1, mass=0.135, width=0.0, index=3)
    f0 = Particle(
        "f(0)(980)", latex="f_0(980)", spin=0, parity=+1, mass=0.99, width=0.06
    )
    chain = ThreeBodyDecayChain(
        decay=IsobarNode(parent=ψ, child1=IsobarNode(f0, π2, π3), child2=γ)
    )
    decay = ThreeBodyDecay(
        states={state.index: state for state in [ψ, γ, π2, π3]},
        chains=[chain],
    )
    builder = DalitzPlotDecompositionBuilder(decay, min_ls=True)
    model = builder.formulate()
    intensity = model.intensity
    assert isinstance(intensity, PoolSum)
    helicity_ranges = {str(symbol): values for symbol, values in intensity.indices}
    assert helicity_ranges == {
        "lambda0": (-1, 0, 1),
        "lambda1": (-1, 1),
        "lambda2": (0,),
        "lambda3": (0,),
    }


def test_get_best_reference_subsystems(
    a2pipipi_reaction: ReactionInfo,
    xib2pkk_reaction: ReactionInfo,
    jpsi2pksigma_reaction: ReactionInfo,
):
    decay = to_three_body_decay(a2pipipi_reaction.transitions)
    assert _get_best_reference_subsystems(decay) == 1
    decay = to_three_body_decay(xib2pkk_reaction.transitions)
    assert _get_best_reference_subsystems(decay) == 2
    decay = to_three_body_decay(jpsi2pksigma_reaction.transitions)
    assert _get_best_reference_subsystems(decay) == 2


def _collect_indexed_symbols(amplitudes: list[sp.Expr]) -> set[sp.Indexed]:
    coupling_symbols: set[sp.Indexed] = set()
    for expr in amplitudes:
        symbols = {s for s in expr.free_symbols if isinstance(s, sp.Indexed)}
        coupling_symbols.update(symbols)
    return coupling_symbols


def _collect_products(amplitudes: list[sp.Expr]) -> list[tuple[sp.Indexed, sp.Indexed]]:
    products = set()
    for amp in amplitudes:
        for node in sp.postorder_traversal(amp):
            couplings = {s for s in node.free_symbols if isinstance(s, sp.Indexed)}
            if len(couplings) == 2:
                products.add(tuple(sorted(couplings, key=str)))
    return sorted(products, key=str)  # ty:ignore[invalid-return-type]


def _get_physical_amplitudes(model: AmplitudeModel) -> list[sp.Expr]:
    amplitudes = [expr.doit() for expr in model.amplitudes.values()]
    return [expr for expr in amplitudes if expr]
