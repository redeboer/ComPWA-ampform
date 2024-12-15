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
# # Inspect model interactively

# %% [markdown]
# In this notebook, we illustrate how to interactively inspect a {class}`.HelicityModel`. The procedure should in fact work for any {class}`sympy.Expr <sympy.core.expr.Expr>`.

# %% [markdown]
# ## Create amplitude model

# %% [markdown]
# First, we create some {class}`.HelicityModel`. We could also have used {mod}`pickle` to {func}`~pickle.load` the {class}`.HelicityModel` that we created in {doc}`/usage/amplitude`, but the cell below allows running this notebook independently.

# %%
import qrules

from ampform import get_builder
from ampform.dynamics.builder import (
    create_non_dynamic_with_ff,
    create_relativistic_breit_wigner_with_ff,
)

reaction = qrules.generate_transitions(
    initial_state=("J/psi(1S)", [-1, +1]),
    final_state=["gamma", "pi0", "pi0"],
    allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
    allowed_interaction_types=["strong", "EM"],
    formalism="canonical-helicity",
)
model_builder = get_builder(reaction)
initial_state_particle = reaction.initial_state[-1]
model_builder.dynamics.assign(initial_state_particle, create_non_dynamic_with_ff)
for name in reaction.get_intermediate_particles().names:
    model_builder.dynamics.assign(name, create_relativistic_breit_wigner_with_ff)
model = model_builder.formulate()

# %% [markdown]
# In this case, {ref}`as we saw <usage/amplitude:Mathematical formula>`, the overall model contains just one intensity term $I = |\sum_i A_i|^2$, with $\sum_i A_i$ some coherent sum of amplitudes. We can extract $\sum_i A_i$ as follows:

# %%
import sympy as sp

amplitude = model.expression.args[0].args[0].args[0]
assert isinstance(amplitude, sp.Add)

# %% [markdown]
# Substitute some of the boring parameters with the provided {attr}`~.HelicityModel.parameter_defaults`:

# %%
substitutions = {
    symbol: value
    for symbol, value in model.parameter_defaults.items()
    if not symbol.name.startswith(R"\Gamma_") and not symbol.name.startswith("m_")
}
amplitude = amplitude.doit().subs(substitutions)

# %% [markdown]
# ## Lambdify

# %% [markdown]
# We now need to identify the {class}`~sympy.core.symbol.Symbol` over which the amplitude is to be plotted. The remaining symbols will be turned into slider parameters.

# %%
plot_symbol = sp.Symbol("m_12", nonnegative=True)
slider_symbols = sorted(amplitude.free_symbols, key=lambda s: s.name)
slider_symbols.remove(plot_symbol)
slider_symbols

# %% [markdown]
# Next, {func}`~sympy.utilities.lambdify.lambdify` the expression:

# %%
np_amplitude = sp.lambdify(
    (plot_symbol, *slider_symbols),
    amplitude,
    "numpy",
)

# %% [markdown]
# We also have to define some functions that formulate what we want to plot. A pure amplitude won't do, because we can only plot real values:

# %% [markdown]
# :::{margin}
#
# See {doc}`mpl_interactions:examples/plot` and {doc}`mpl_interactions:examples/scatter` for why these functions are constructed this way.
#
# :::

# %%
import numpy as np


def intensity(plot_variable, **kwargs):
    values = np_amplitude(plot_variable, **kwargs)
    return np.abs(values) ** 2


def argand(**kwargs):
    values = np_amplitude(plot_domain, **kwargs)
    argand = np.array([values.real, values.imag])
    return argand.T


# %% [markdown]
# ## Prepare sliders

# %% [markdown]
# :::{tip}
#
# This procedure has been extracted to the façade function {func}`symplot.prepare_sliders`.
#
# :::

# %% [markdown]
# We also need to define the domain over which to plot, as well sliders for each of the remaining symbols. The function {func}`.create_slider` helps creating an [ipywidgets slider](https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20List.html) for each {class}`~sympy.core.symbol.Symbol`:

# %% [markdown]
# :::{margin}
#
# {doc}`/usage/symplot` is not a published module, just a set of helper functions and classes provided for this documentation. Since the procedure sketched on this page is quite general, this module could be published as a separate interactive plotting package for {mod}`sympy` later on.
#
# :::

# %%
from symplot import create_slider

plot_domain = np.linspace(0.2, 2.5, num=400)
sliders_mapping = {symbol.name: create_slider(symbol) for symbol in slider_symbols}

# %% [markdown]
# If the name of a {class}`~sympy.core.symbol.Symbol` is not a valid name for a Python variable (see {meth}`str.isidentifier`), {func}`~sympy.utilities.lambdify.lambdify` 'dummifies' it, so it can be used as argument for the lambdified function. Since {func}`~mpl_interactions.pyplot.interactive_plot` works with these (dummified) arguments as identifiers for the sliders, we need some mapping between the {class}`~sympy.core.symbol.Symbol` and their dummified name. This can be constructed with the help of {func}`inspect.signature`:

# %%
import inspect

symbols_names = [s.name for s in (plot_symbol, *slider_symbols)]
arg_names = list(inspect.signature(np_amplitude).parameters)
arg_to_symbol = dict(zip(arg_names, symbols_names))

# %% [markdown]
# The class {class}`.SliderKwargs` comes in as a handy manager for this {obj}`dict` of sliders:

# %% tags=["full-width"]
from symplot import SliderKwargs

sliders = SliderKwargs(sliders_mapping, arg_to_symbol)
sliders

# %% [markdown]
# {class}`.SliderKwargs` also provides convenient methods for setting ranges and initial values for the sliders.

# %%
n_steps = 100
sliders.set_ranges({
    "m_{f_{0}(980)}": (0.3, 1.8, n_steps),
    "m_{f_{0}(1500)}": (0.3, 1.8, n_steps),
    R"\Gamma_{f_{0}(980)}": (0.01, 1, n_steps),
    R"\Gamma_{f_{0}(1500)}": (0.01, 1, n_steps),
    "m_1": (0.01, 1, n_steps),
    "m_2": (0.01, 1, n_steps),
    "phi_0": (0, 2 * np.pi, 40),
    "theta_0": (-np.pi, np.pi, 40),
})

# %% [markdown]
# The method {meth}`~.SliderKwargs.set_values` is designed in particular for {attr}`~.HelicityModel.parameter_defaults`, but also works well with both argument names (that may have been dummified) and the original symbol names:

# %%
import qrules

pdg = qrules.load_pdg()
sliders.set_values(model.parameter_defaults)
sliders.set_values(
    {  # symbol names
        "phi_0": 0,
        "theta_0": np.pi / 2,
    },
    # argument names
    m_1=pdg["pi0"].mass,
    m_2=pdg["pi0"].mass,
)

# %% jupyter={"source_hidden": true} tags=["remove-cell"]
if STATIC_WEB_PAGE:
    # Concatenate flipped domain for reverse animation
    domain = np.linspace(0.8, 2.2, 100)
    domain = np.concatenate((domain, np.flip(domain[1:])))
    sliders._sliders["m_{f_{0}(980)}"] = domain

# %% [markdown]
# ## Interactive Argand plot

# %% [markdown]
# Finally, we can use {doc}`mpl-interactions <mpl_interactions:index>` to plot the functions defined above with regard to the parameter values:

# %% [markdown]
# :::{margin}
#
# Interactive {mod}`~matplotlib.widgets` do not render well on web pages, so run this notebook in on Binder or locally in Jupyter Lab to see the full potential of {doc}`mpl-interactions <mpl_interactions:index>`!
#
# :::

# %% [markdown]
# ```{autolink-skip}
# ```

# %% jupyter={"source_hidden": true} tags=["remove-cell"]
# %matplotlib widget

# %% jupyter={"source_hidden": true} tags=["hide-input"]
# %config InlineBackend.figure_formats = ['svg']

import matplotlib.pyplot as plt
import mpl_interactions.ipyplot as iplt

# Create figure
fig, axes = plt.subplots(1, 2, figsize=0.9 * np.array((8, 3.8)), tight_layout=True)
fig.canvas.toolbar_visible = False
fig.canvas.header_visible = False
fig.canvas.footer_visible = False
fig.suptitle(R"$J/\psi \to \gamma f_0, f_0 \to \pi^0\pi^0$")
ax_intensity, ax_argand = axes
m_label = R"$m_{\pi^0\pi^0}$ (GeV)"
ax_intensity.set_xlabel(m_label)
ax_intensity.set_ylabel("$I = |A|^2$")
ax_argand.set_xlabel("Re($A$)")
ax_argand.set_ylabel("Im($A$)")

# Fill plots
controls = iplt.plot(
    plot_domain,
    intensity,
    **sliders,
    slider_formats=dict.fromkeys(arg_names, "{:.3f}"),
    ylim="auto",
    ax=ax_intensity,
)
iplt.scatter(
    argand,
    controls=controls,
    xlim="auto",
    ylim="auto",
    parametric=True,
    c=plot_domain,
    s=1,
    ax=ax_argand,
)
plt.colorbar(label=m_label, ax=ax_argand, aspect=30, pad=0.01)
plt.winter()

# %% [markdown]
# :::{margin}
#
# This figure is an animation over **just one of the parameters**. Run the notebook itself to play around with all parameters!
#
# :::

# %% jupyter={"source_hidden": true} tags=["remove-input"]
# Export for Read the Docs
if STATIC_WEB_PAGE:
    from IPython.display import Image, display

    output_path = "animation.gif"
    symbol_to_arg = dict(zip(symbols_names, arg_names))
    arg_name = symbol_to_arg["m_{f_{0}(980)}"]
    controls.save_animation(output_path, fig, arg_name, fps=25)
    with open(output_path, "rb") as f:
        display(Image(data=f.read(), format="png"))

# %% [markdown]
# :::{tip}
#
# See {doc}`/usage/dynamics/k-matrix` for why $\boldsymbol{K}$-matrix dynamics are better than simple Breit-Wigners when resonances are close to each other.
#
# :::

# %% [markdown]
# ```{toctree}
# :hidden:
# symplot
# ```
