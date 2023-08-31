"""Collections of functions for working with QRules."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ampform.helicity import CanonicalAmplitudeBuilder, HelicityAmplitudeBuilder

if TYPE_CHECKING:
    from qrules import ReactionInfo


def get_builder(reaction: ReactionInfo) -> HelicityAmplitudeBuilder:
    """Get the correct `.HelicityAmplitudeBuilder`.

    For instance, get `.CanonicalAmplitudeBuilder` if the
    `~qrules.transition.ReactionInfo.formalism` is :code:`"canonical-helicity"`.
    """
    formalism = reaction.formalism
    if formalism is None:
        msg = f"ReactionInfo does not have a formalism type:\n{reaction}"
        raise ValueError(msg)
    if formalism == "helicity":
        amplitude_builder = HelicityAmplitudeBuilder(reaction)
    elif formalism in ["canonical-helicity", "canonical"]:
        amplitude_builder = CanonicalAmplitudeBuilder(reaction)
    else:
        msg = f'No amplitude generator for formalism type "{formalism}"'
        raise NotImplementedError(msg)
    return amplitude_builder
