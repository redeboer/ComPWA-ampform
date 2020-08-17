# pylint: disable=redefined-outer-name
import json
import logging
from os.path import dirname, realpath

import pytest

import yaml

from expertsystem import io
from expertsystem.amplitude.helicity_decay import HelicityAmplitudeGenerator
from expertsystem.ui import (
    InteractionTypes,
    StateTransitionManager,
)

logging.basicConfig(level=logging.ERROR)

SCRIPT_PATH = dirname(realpath(__file__))


@pytest.fixture(scope="module")
def amplitude_generator():
    stm = StateTransitionManager(
        initial_state=[("J/psi", [-1, 1])],
        final_state=["gamma", "pi0", "pi0"],
        allowed_intermediate_particles=["f0"],
    )
    stm.set_allowed_interaction_types(
        [InteractionTypes.Strong, InteractionTypes.EM]
    )
    graph_interaction_settings_groups = stm.prepare_graphs()
    solutions, _ = stm.find_solutions(graph_interaction_settings_groups)

    hel_amp_gen = HelicityAmplitudeGenerator()
    hel_amp_gen.generate(solutions)
    return hel_amp_gen


@pytest.fixture(scope="module")
def imported_dict(amplitude_generator: HelicityAmplitudeGenerator) -> dict:
    output_filename = "JPsiToGammaPi0Pi0_heli_recipe.yml"
    amplitude_generator.write_to_file(output_filename)
    with open(output_filename, "rb") as input_file:
        loaded_dict = yaml.load(input_file, Loader=yaml.FullLoader)
    return loaded_dict


@pytest.fixture(scope="module")
def expected_dict() -> dict:
    expected_recipe_file = f"{SCRIPT_PATH}/expected_recipe.yml"
    with open(expected_recipe_file, "rb") as input_file:
        expected_recipe_dict = yaml.load(input_file, Loader=yaml.FullLoader)
    return expected_recipe_dict


def equalize_dict(input_dict):
    output_dict = json.loads(json.dumps(input_dict, sort_keys=True))
    return output_dict


def test_recipe_validation(expected_dict):
    io.yaml.validation.amplitude_model(expected_dict)


def test_not_implemented_writer(amplitude_generator):
    with pytest.raises(NotImplementedError):
        amplitude_generator.write_to_file("JPsiToGammaPi0Pi0.csv")


def test_create_recipe_dict(amplitude_generator):
    recipe = (
        amplitude_generator._create_recipe_dict()  # pylint: disable=protected-access
    )
    assert len(recipe) == 3


def test_particle_section(imported_dict):
    particle_list = imported_dict.get("ParticleList", imported_dict)
    gamma = particle_list["gamma"]
    assert gamma["PID"] == 22
    assert gamma["Mass"] == 0.0
    gamma_qns = gamma["QuantumNumbers"]
    assert gamma_qns["Spin"] == 1
    assert gamma_qns["Charge"] == 0
    assert gamma_qns["Parity"] == -1
    assert gamma_qns["CParity"] == -1

    f0_980 = particle_list["f0(980)"]
    assert f0_980["Width"] == 0.06

    pi0_qns = particle_list["pi0"]["QuantumNumbers"]
    assert pi0_qns["IsoSpin"]["Value"] == 1
    assert pi0_qns["IsoSpin"]["Projection"] == 0


def test_kinematics_section(imported_dict):
    kinematics = imported_dict["Kinematics"]
    initial_state = kinematics["InitialState"]
    final_state = kinematics["FinalState"]
    assert kinematics["Type"] == "Helicity"
    assert len(initial_state) == 1
    assert initial_state[0]["Particle"] == "J/psi"
    assert len(final_state) == 3


def test_parameter_section(imported_dict):
    parameter_list = imported_dict["Parameters"]
    assert len(parameter_list) == 5
    for parameter in parameter_list:
        assert "Name" in parameter
        assert "Value" in parameter


def test_dynamics_section(imported_dict):
    dynamics = imported_dict["Dynamics"]
    assert len(dynamics) == 1

    j_psi = dynamics["J/psi"]
    assert j_psi["Type"] == "NonDynamic"
    assert j_psi["FormFactor"]["Type"] == "BlattWeisskopf"
    assert j_psi["FormFactor"]["MesonRadius"] == 1.0

    f0_980 = dynamics.get("f0(980)", None)
    if f0_980:
        assert f0_980["Type"] == "RelativisticBreitWigner"
        assert f0_980["FormFactor"]["Type"] == "BlattWeisskopf"
        assert f0_980["FormFactor"]["MesonRadius"] == {
            "Max": 2.0,
            "Min": 0.0,
            "Value": 1.0,
        }


def test_intensity_section(imported_dict):
    intensity = imported_dict["Intensity"]
    assert intensity["Class"] == "StrengthIntensity"
    intensity = intensity["Intensity"]
    assert intensity["Class"] == "NormalizedIntensity"

    intensity = intensity["Intensity"]
    assert intensity["Class"] == "IncoherentIntensity"
    assert len(intensity["Intensities"]) == 4


@pytest.mark.parametrize(
    "section", ["ParticleList", "Kinematics"],
)
def test_expected_recipe_shape(imported_dict, expected_dict, section):
    expected_section = equalize_dict(expected_dict[section])
    imported_section = equalize_dict(imported_dict[section])
    if isinstance(expected_section, dict):
        assert expected_section.keys() == imported_section.keys()
        imported_items = list(imported_section.values())
        expected_items = list(expected_section.values())
    else:
        expected_items = list(expected_section)
        imported_items = list(imported_section)
    is_parameter_section = False
    if section == "Parameters":
        is_parameter_section = True
        imported_items = imported_items[1:]
    assert len(imported_items) == len(expected_items)
    for imported, expected in zip(imported_items, expected_items):
        if is_parameter_section:
            imported["Name"] = imported["Name"].replace("_-1", "_1")
        assert imported == expected