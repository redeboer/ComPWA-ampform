# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% hideCode=true hideOutput=true hidePrompt=true jupyter={"source_hidden": true} tags=["remove-cell", "skip-execution"]
# WARNING: advised to install a specific version, e.g. ampform==0.1.2
# %pip install -q ampform[doc,viz] IPython

# %% hideCode=true hideOutput=true hidePrompt=true jupyter={"source_hidden": true} tags=["remove-cell"]
import os

STATIC_WEB_PAGE = {"EXECUTE_NB", "READTHEDOCS"}.intersection(os.environ)

# %% [markdown]
# ```{autolink-concat}
# ```

# %% [markdown]
# # Helicity versus canonical

# %% jupyter={"source_hidden": true} mystnb={"code_prompt_show": "Import Python libraries"} tags=["hide-cell"]
import logging

import graphviz
import matplotlib as mpl
import numpy as np
import qrules
import sympy as sp
from IPython.display import HTML, Math, display
from matplotlib import cm
from rich.table import Table

import ampform

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.ERROR)


# %% [markdown]
# In this notebook, we have a look at the decay
#
# $$D_1(2420)^0 \to a_1(1260)^+ K^- \to (K^+K^0)K^-$$
#
# in order to see the difference between a {class}`.HelicityModel` formulated in the **canonical** basis and one formulated in the **helicity** basis. To simplify things, we only look at spin projection $+1$ for $D_1(2420)^0$, because the intensities for each of the spin projections of $D_1(2420)^0$ are incoherent, no matter which spin formalism we choose.
#
# :::{tip}
#
# For more information about the helicity formalism, see {cite}`chungSpinFormalismsUpdated2014`, {cite}`Richman:1984gh`, and {cite}`kutschkeAngularDistributionCookbook1996`.
#
# :::
#
# First, we use {func}`qrules.generate_transitions` to generate a {class}`~qrules.transition.ReactionInfo` instance for both formalisms:

# %%
def generate_transitions(formalism: str):
    reaction = qrules.generate_transitions(
        initial_state=("D(1)(2420)0", [+1]),
        final_state=["K+", "K-", "K~0"],
        allowed_intermediate_particles=["a(1)(1260)+"],
        formalism=formalism,
    )
    builder = ampform.get_builder(reaction)
    return builder.formulate()


cano_model = generate_transitions("canonical-helicity")
heli_model = generate_transitions("helicity")

# %% [markdown]
# :::{margin}
#
# {ref}`usage/helicity/formalism:Coefficient names` shows how to generate different coefficient names.
#
# :::
#
# From {attr}`.components` and {attr}`.parameter_defaults`, we can see that the canonical formalism has a larger number of amplitudes.

# %% jupyter={"source_hidden": true} tags=["hide-input"]
table = Table(show_edge=False)
table.add_column("Formalism")
table.add_column("Coefficients", justify="right")
table.add_column("Amplitudes", justify="right")
table.add_row(
    "Canonical",
    str(len(cano_model.parameter_defaults)),
    str(len(cano_model.components) - 1),
)
table.add_row(
    "Helicity",
    str(len(heli_model.parameter_defaults)),
    str(len(heli_model.components) - 1),
)
table


# %% [markdown]
# The reason for this is that canonical basis distinguishes amplitudes over their $LS$-combinations. This becomes clear if we define $a$ to be the amplitude _without coefficient_ ($A = C a$), and consider what the full, coherent intensity looks like.
#
# If we write the full intensity as $I = \left|\sum_i A_i\right|^2$, then we have, in the case of the **canonical** basis:

# %% jupyter={"source_hidden": true} tags=["hide-input", "full-width"]
def extract_amplitude_substitutions(model, colorize=False):
    amplitude_to_symbol = {}
    amplitude_names = sorted(c for c in model.components if c.startswith("A"))
    n_colors = len(amplitude_names)
    color_map = cm.brg(np.linspace(0, 1, num=n_colors + 1)[:-1])
    color_iter = (mpl.colors.to_hex(color) for color in color_map)
    for name in amplitude_names:
        expr = model.components[name]
        for par in model.parameter_defaults:
            if par in expr.args:
                expr /= par
        name = "a" + name[1:]
        if colorize:
            color = next(color_iter)
            name = Rf"\color{{{color}}}{{{name}}}"
        symbol = sp.Symbol(name)
        amplitude_to_symbol[expr] = symbol
    return amplitude_to_symbol


cano_amplitude_to_symbol = extract_amplitude_substitutions(cano_model)
heli_amplitude_to_symbol = extract_amplitude_substitutions(heli_model)


def render_amplitude_summation(model, colorize=False):
    amplitude_to_symbol = extract_amplitude_substitutions(model, colorize)
    collected_expr = sp.collect(
        model.expression.subs(amplitude_to_symbol).args[0].args[0],
        tuple(model.parameter_defaults),
    )
    terms = collected_expr.args
    latex = ""
    latex += R"\begin{align}"
    latex += Rf"\sum_i A_i & = {sp.latex(terms[0])}\\"
    for term in terms[1:]:
        latex += Rf"& + {sp.latex(term)} \\"
    latex += R"\end{align}"
    return Math(latex)


render_amplitude_summation(cano_model, colorize=True)

# %% [markdown]
# In the **helicity** basis, the $LS$-combinations have been summed over already and we can only see an amplitude for each helicity:

# %% jupyter={"source_hidden": true} tags=["hide-input", "full-width"]
render_amplitude_summation(heli_model)


# %% [markdown]
# Amplitudes in the **canonical** basis are formulated with regard to their $LS$-couplings. As such, they contain additional [Clebsch-Gordan coefficients](https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients) that serve as _expansion coefficients_.

# %% jupyter={"source_hidden": true} tags=["hide-input", "full-width"]
def extract_amplitudes(model):
    return {
        expr: sp.Symbol(name)
        for name, expr in model.components.items()
        if name.startswith("A")
    }


cano_amplitudes = extract_amplitudes(cano_model)
heli_amplitudes = extract_amplitudes(heli_model)

expression, symbol = next(iter(cano_amplitude_to_symbol.items()))
display(symbol, Math(Rf"\quad = {sp.latex(expression)}"))

# %% [markdown]
# In the **helicity** basis, these Clebsch-Gordan coefficients and [Wigner-$D$ functions](https://en.wikipedia.org/wiki/Wigner_D-matrix) have been summed up, leaving only a Wigner-$D$ for each node in the decay chain (two in this case):

# %% jupyter={"source_hidden": true} tags=["hide-input"]
expression, symbol = next(iter(heli_amplitude_to_symbol.items()))
display(symbol, Math(Rf"\quad = {sp.latex(expression)}"))


# %% [markdown]
# See {func}`.formulate_isobar_wigner_d` and {func}`.formulate_isobar_cg_coefficients` for how these Wigner-$D$ functions and Clebsch-Gordan coefficients are computed for each node on a {class}`~qrules.topology.Transition`.
#
# We can see this also from the original {class}`~qrules.transition.ReactionInfo` objects. Let's select only the {attr}`~qrules.transition.ReactionInfo.transitions` where the $a_1(1260)^+$ resonance has spin projection $-1$ (taken to be helicity $-1$ in the helicity formalism). We then see just one {class}`~qrules.topology.Transition` in the helicity basis and three transitions in the canonical basis:

# %% jupyter={"source_hidden": true} tags=["hide-input"]
def render_selection(model):
    transitions = model.reaction_info.transitions
    selection = filter(lambda s: s.states[3].spin_projection == -1, transitions)
    dot = qrules.io.asdot(selection, render_node=True, render_final_state_id=False)
    return graphviz.Source(dot)


display(
    HTML("<b>Helicity</b> basis:"),
    render_selection(heli_model),
    HTML("<b>Canonical</b> basis:"),
    render_selection(cano_model),
)

# %% [markdown]
# ## Coefficient names

# %% [markdown]
# In the previous section, we saw that the {class}`.HelicityAmplitudeBuilder` by default generates coefficient names that only contain **helicities of the decay products**, while coefficients generated by the {class}`.CanonicalAmplitudeBuilder` contain only **$LS$-combinations**. It's possible to tweak this behavior with the {attr}`~.HelicityAmplitudeBuilder.naming` attribute. Here are two extreme examples, where we generate coefficient names that contain $LS$-combinations, the helicities of each parent state, and the helicity of each decay product, as well as a {class}`.HelicityModel` of which the coefficient names only contain information about the resonances:

# %%
reaction = qrules.generate_transitions(
    initial_state=("D(1)(2420)0", [+1]),
    final_state=["K+", "K-", "K~0"],
    allowed_intermediate_particles=["a(1)(1260)+"],
    formalism="canonical-helicity",
)
builder = ampform.get_builder(reaction)
builder.naming.insert_parent_helicities = True
builder.naming.insert_child_helicities = True
builder.naming.insert_ls_combinations = True
model = builder.formulate()

# %% jupyter={"source_hidden": true} tags=["hide-input"]
amplitudes = [c for c in model.components if c.startswith("A")]
assert len(model.parameter_defaults) == len(amplitudes)
sp.Matrix(model.parameter_defaults)

# %%
builder.naming.insert_parent_helicities = False
builder.naming.insert_child_helicities = False
builder.naming.insert_ls_combinations = False
model = builder.formulate()

# %% jupyter={"source_hidden": true} tags=["hide-input"]
assert len(model.parameter_defaults) == 1
display(*model.parameter_defaults)
