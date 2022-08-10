from __future__ import annotations

import logging

import pytest
import qrules
import sympy as sp
from _pytest.logging import LogCaptureFixture

from ampform import get_builder
from ampform.helicity.model import HelicityModel, ParameterValue, ParameterValues


class TestHelicityModel:
    def test_parameter_defaults_item_types(
        self, amplitude_model: tuple[str, HelicityModel]
    ):
        _, model = amplitude_model
        for symbol, value in model.parameter_defaults.items():
            assert isinstance(symbol, sp.Symbol)
            assert isinstance(value, ParameterValue.__args__)  # type: ignore[attr-defined]

    def test_rename_symbols_no_renames(
        self, amplitude_model: tuple[str, HelicityModel]
    ):
        _, model = amplitude_model
        new_model = model.rename_symbols({})
        assert new_model == model

    def test_rename_parameters(self, amplitude_model: tuple[str, HelicityModel]):
        _, model = amplitude_model
        d1, d2 = sp.symbols("d_{f_{0}(980)} d_{f_{0}(1500)}", positive=True)
        assert {d1, d2} <= set(model.parameter_defaults)
        assert {d1, d2} <= model.expression.free_symbols

        new_d = sp.Symbol("d", positive=True)
        new_model = model.rename_symbols(
            {
                d1.name: new_d.name,
                d2.name: new_d.name,
            }
        )
        assert not {d1, d2} & new_model.expression.free_symbols
        assert not {d1, d2} & set(new_model.parameter_defaults)
        assert new_d in new_model.parameter_defaults
        assert new_d in new_model.expression.free_symbols
        assert (
            len(new_model.expression.free_symbols)
            == len(model.expression.free_symbols) - 1
        )
        assert len(new_model.parameter_defaults) == len(model.parameter_defaults) - 1
        assert model.expression.xreplace({d1: new_d, d2: new_d}) == new_model.expression

    def test_rename_variables(self, amplitude_model: tuple[str, HelicityModel]):
        _, model = amplitude_model
        old_symbol = sp.Symbol("m_12", nonnegative=True)
        assert old_symbol in model.kinematic_variables
        assert old_symbol in model.expression.free_symbols

        new_symbol = sp.Symbol("m_{f_0}", nonnegative=True)
        new_model = model.rename_symbols({old_symbol.name: new_symbol.name})
        assert old_symbol not in new_model.kinematic_variables
        assert old_symbol not in new_model.expression.free_symbols
        assert new_symbol in new_model.kinematic_variables
        assert new_symbol in new_model.expression.free_symbols
        assert (
            model.expression.xreplace({old_symbol: new_symbol}) == new_model.expression
        )

    def test_assumptions_after_rename(self, amplitude_model: tuple[str, HelicityModel]):
        # pylint: disable=protected-access
        _, model = amplitude_model
        old = "m_{f_{0}(980)}"
        new = "m"
        new_model = model.rename_symbols({old: new})
        assert (
            new_model.parameter_defaults._get_parameter(new).assumptions0
            == model.parameter_defaults._get_parameter(old).assumptions0
        )

    def test_rename_symbols_warnings(
        self,
        amplitude_model: tuple[str, HelicityModel],
        caplog: LogCaptureFixture,
    ):
        _, model = amplitude_model
        old_name = "non-existent"
        with caplog.at_level(logging.WARNING):
            new_model = model.rename_symbols({old_name: "new name"})
        assert caplog.records
        assert old_name in caplog.records[-1].msg
        assert new_model == model

    @pytest.mark.parametrize("formalism", ["canonical-helicity", "helicity"])
    def test_amplitudes(self, formalism: str):
        reaction = qrules.generate_transitions(
            initial_state=("J/psi(1S)", [-1, +1]),
            final_state=["K0", "Sigma+", "p~"],
            allowed_intermediate_particles=["Sigma(1660)~-"],
            allowed_interaction_types=["strong"],
            formalism=formalism,
        )
        assert len(reaction.get_intermediate_particles()) == 1

        builder = get_builder(reaction)
        helicity_combinations = {
            tuple(
                state.spin_projection
                for state_id, state in transition.states.items()
                if state_id not in transition.intermediate_states
            )
            for transition in reaction.transitions
        }
        assert len(helicity_combinations) == 8

        model = builder.formulate()
        assert len(model.amplitudes) == len(helicity_combinations)
        intensity_terms = model.intensity.evaluate().args
        assert len(intensity_terms) == len(helicity_combinations)


class TestParameterValues:
    @pytest.mark.parametrize("subs_method", ["subs", "xreplace"])
    def test_subs_xreplace(self, subs_method: str):
        a, b, x, y = sp.symbols("a b x y")
        expr: sp.Expr = a * x + b * y
        parameters = ParameterValues({a: 2, b: -3})
        if subs_method == "subs":
            expr = expr.subs(parameters)
        elif subs_method == "xreplace":
            expr = expr.xreplace(parameters)
        else:
            raise NotImplementedError
        assert expr == 2 * x - 3 * y
