"""Serialization from and to an XML recipe file."""

__all__ = [
    "load_particle_collection",
    "write",
]

import json

import xmltodict

from expertsystem.data import (
    ComplexEnergyState,
    Particle,
    ParticleCollection,
    Spin,
)

from . import _build
from . import _dump
from . import validation


def load_particle_collection(filename: str) -> ParticleCollection:
    with open(filename, "rb") as stream:
        definition = xmltodict.parse(stream)
    definition = definition.get("root", definition)
    json.loads(json.dumps(definition))  # remove OrderedDict
    return _build.build_particle_collection(definition)


def write(instance: object, filename: str) -> None:
    if isinstance(instance, ParticleCollection):
        output_dict = _dump.from_particle_collection(instance)
        entries = list(output_dict.values())
        output_dict = {"ParticleList": {"Particle": entries}}
        validation.particle_list(output_dict)
    else:
        raise NotImplementedError(
            f"No XML writer for class {instance.__class__.__name__}"
        )
    with open(filename, "w") as stream:
        xmlstring = xmltodict.unparse(
            {"root": output_dict}, pretty=True, indent="  "
        )
        stream.write(xmlstring)


def object_to_dict(instance: object) -> dict:
    if isinstance(instance, ParticleCollection):
        return _dump.from_particle_collection(instance)
    if isinstance(instance, (ComplexEnergyState, Particle)):
        return _dump.from_particle_state(instance)
    raise NotImplementedError


def dict_to_particle_collection(definition: dict) -> ParticleCollection:
    return _build.build_particle_collection(definition)


def dict_to_particle(definition: dict) -> Particle:
    return _build.build_particle(definition)


def dict_to_spin(definition: dict) -> Spin:
    return _build.build_spin(definition)