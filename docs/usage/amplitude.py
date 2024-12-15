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
# # Formulate amplitude model

# %% jupyter={"source_hidden": true} mystnb={"code_prompt_show": "Import Python libraries"} tags=["hide-cell"]
# %config InlineBackend.figure_formats = ['svg']

import sympy as sp
from IPython.display import Math

from ampform.io import aslatex, improve_latex_rendering

improve_latex_rendering()

# %% [markdown]
# ## Generate transitions

# %% [markdown]
# In {doc}`qrules:usage/reaction`, we used {func}`~qrules.generate_transitions` to create a list of allowed {class}`~qrules.topology.Transition`s for a specific decay channel:

# %%
import qrules

reaction = qrules.generate_transitions(
    initial_state=("J/psi(1S)", [-1, +1]),
    final_state=["gamma", "pi0", "pi0"],
    allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
    allowed_interaction_types=["strong", "EM"],
    formalism="canonical-helicity",
)

# %% jupyter={"source_hidden": true} tags=["hide-input"]
import graphviz

dot = qrules.io.asdot(reaction, collapse_graphs=True)
graphviz.Source(dot)

# %% [markdown]
# ## Build model

# %% [markdown]
# We can now use the {class}`~qrules.transition.ReactionInfo` to formulate an amplitude model. The type of this amplitude model is dependent on the {attr}`~qrules.transition.ReactionInfo.formalism`. The function {func}`.get_builder` helps to get the correct amplitude builder class for this {attr}`~qrules.transition.ReactionInfo.formalism`:

# %%
from ampform import get_builder

model_builder = get_builder(reaction)
type(model_builder)

# %% [markdown]
# If we now use the {meth}`.HelicityAmplitudeBuilder.formulate` method of this builder, we get a {class}`.HelicityModel` without any dynamics:

# %%
model_no_dynamics = model_builder.formulate()

# %% [markdown]
# :::{seealso} {doc}`/usage/helicity/formalism`
#
# :::

# %% [markdown]
# ### Main expressions

# %% [markdown]
# A {class}`.HelicityModel` has a few attributes. The expression for the total intensity is given by {attr}`.intensity`:
#
# :::{margin}
#
# As can be seen in {ref}`usage/amplitude:Generate transitions`, the transition that this {class}`.HelicityModel` describes features only one decay topology. This means that the main {attr}`~.intensity` expression is relatively simple. In case there are more decay topologies, AmpForm {doc}`aligns spin </usage/helicity/spin-alignment>`, which makes the main intensity expression more complicated.
#
# :::

# %%
model_no_dynamics.intensity

# %% [markdown]
# This shows that the main intensity is an **incoherent** sum of the amplitude for each spin projection combination of the initial and final states. The expressions for each of these amplitudes are provided with the {attr}`~.amplitudes` attribute. This is an {class}`~collections.OrderedDict`, so we can inspect the first of these amplitudes as follows:

# %% tags=["full-width"]
(symbol, expression), *_ = model_no_dynamics.amplitudes.items()
Math(aslatex({symbol: expression}, terms_per_line=1))

# %% [markdown]
# The intensity and its amplitudes are recombined through the {attr}`~.HelicityModel.expression` attribute. This is just a {class}`sympy.Expr <sympy.core.expr.Expr>`, which we can pull apart by using its {attr}`~sympy.core.basic.Basic.args` (see {doc}`sympy:tutorials/intro-tutorial/manipulation`). Here's an example:

# %%
intensities = model_no_dynamics.expression.args
intensity_1 = intensities[0]
base, power = intensity_1.args
abs_arg = base.args[0]
amplitude_sum = abs_arg.args
some_amplitude = amplitude_sum[0]
some_amplitude

# %%
some_amplitude.doit()

# %% [markdown]
# ### Parameters and kinematic variables

# %% [markdown]
# As can be seen, the expression contains several {class}`~sympy.core.symbol.Symbol`s. Some of these represent (kinematic) **variables**, such as the helicity angles $\phi_0$ and $\theta_0$ (see {func}`.get_helicity_angle_symbols` for the meaning of their subscripts). Others will later on be interpreted **parameters** when fitting the model to data.
#
# The {class}`.HelicityModel` comes with expressions for these {attr}`~.HelicityModel.kinematic_variables`, so that it's possible to compute them from 4-momentum data.

# %%
kinematic_variables = set(model_no_dynamics.kinematic_variables)
kinematic_variables

# %%
theta = sp.Symbol("theta_0", real=True)
expr = model_no_dynamics.kinematic_variables[theta]
expr.doit()

# %% [markdown]
# :::{seealso}
# [Lorentz vectors](./kinematics.ipynb#lorentz-vectors)
# :::
#
# The remaining symbols in the {class}`.HelicityModel` are parameters. Each of these parameters comes with suggested parameter values ({attr}`~.HelicityModel.parameter_defaults`), that have been extracted from the {class}`~qrules.transition.ReactionInfo` object where possible:

# %%
Math(aslatex(model_no_dynamics.parameter_defaults))

# %% [markdown]
# #### Helicity couplings

# %% [markdown]
# If you prefer to characterize the strength of each partial _two-body_ decay, set {attr}`~.BuilderConfiguration.use_helicity_couplings` to {obj}`True` and formulate a model:

# %%
model_builder.config.use_helicity_couplings = True
model_with_couplings = model_builder.formulate()
Math(aslatex(model_with_couplings.parameter_defaults))

# %% jupyter={"source_hidden": true} tags=["remove-cell"]
model_builder.config.use_helicity_couplings = False

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# #### Scalar masses

# %% [markdown]
# By default, the {class}`.HelicityAmplitudeBuilder` creates {class}`sympy.Expr <sympy.core.expr.Expr>`s for each kinematic variable, including all 'invariant' final state masses ($m_0, m_1, \dots$). However, it often happens that certain particles in a final state are stable. In that case, you may want to substitute these symbols with _scalar_ values. This can be achieved by specifying which final state IDs are to be considered _stable_. Their corresponding mass symbols will then be considered parameters and a scalar suggested parameter value will be provided.

# %%
model_builder.config.stable_final_state_ids = [0, 1, 2]
model_stable_masses = model_builder.formulate()
Math(aslatex(model_stable_masses.parameter_defaults))

# %%
set(model_stable_masses.kinematic_variables)

# %% [markdown]
# We can reset this option as follows:

# %%
model_builder.config.stable_final_state_ids = None

# %% [markdown]
# Similarly, it can happen that the initial state mass is fixed (for instance when analyzing generated data or when applying a kinematic fit while reconstructing measured data). In that case, set {attr}`~.BuilderConfiguration.scalar_initial_state_mass` to {obj}`True`.

# %%
model_builder.config.scalar_initial_state_mass = True
model_stable_masses = model_builder.formulate()
Math(aslatex(model_stable_masses.parameter_defaults))

# %% [markdown]
# #### Extend kinematic variables

# %% [markdown]
# The {class}`.HelicityAmplitudeBuilder` by default only generates {attr}`.kinematic_variables` (helicity angles and invariant masses) for the topologies that are available in the {class}`~qrules.transition.ReactionInfo` object that it was created with. If you want to calculate more kinematic variables, you can use the method {meth}`.register_topology` of its helicity {attr}`.HelicityAmplitudeBuilder.adapter` to register more topologies and generate more kinematic variables. This is especially useful when generating data later on with [TensorWaves](https://tensorwaves.rtfd.io).
#
# To make this a bit easier, there is {meth}`.permutate_registered_topologies`, which is a small convenience function makes it possible to generate permutations of all {attr}`.registered_topologies` and register them as well. Note that initially, only one {class}`~qrules.topology.Topology` is registered in the {attr}`.HelicityAmplitudeBuilder.adapter`, namely the one for the decay $J/\psi \to \gamma f_0, f_0 \to \pi^0\pi^0$:

# %%
dot = qrules.io.asdot(model_builder.adapter.registered_topologies)
graphviz.Source(dot)

# %% [markdown]
# We now {meth}`.permutate_registered_topologies` to register permutations of this {class}`~qrules.topology.Topology`:

# %%
model_builder.adapter.permutate_registered_topologies()

# %% [markdown]
# There are now **three** {attr}`.registered_topologies`―one for each permutation:

# %%
len(model_builder.adapter.registered_topologies)

# %%
assert len(model_builder.adapter.registered_topologies) == 3
dot = qrules.io.asdot(model_builder.adapter.registered_topologies)
graphviz.Source(dot)

# %% [markdown]
# And if we {meth}`~.HelicityAmplitudeBuilder.formulate` a new {class}`.HelicityModel`, we see that it has many more {attr}`.kinematic_variables`:

# %%
set(model_builder.formulate().kinematic_variables)

# %% [markdown]
# :::{tip}
#
# To register even more topologies, use e.g. {func}`~qrules.topology.create_isobar_topologies` to generate other, non-isomorphic topologies that cannot be created with permutations. This is relevant for more than three final states.
#
# :::

# %% [markdown]
# ### Set dynamics

# %% [markdown]
# To set dynamics for specific resonances, use {meth}`.DynamicsSelector.assign` on the same {attr}`.HelicityAmplitudeBuilder.dynamics` attribute.  You can set the dynamics to be any kind of {class}`~sympy.core.expr.Expr`, as long as you keep track of which {class}`~sympy.core.symbol.Symbol` names you use (see {doc}`/usage/dynamics/custom`).
#
# AmpForm does provide a few common {mod}`.dynamics` functions, which can be constructed as {class}`~sympy.core.expr.Expr` with the correct {class}`~sympy.core.symbol.Symbol` names using {meth}`.DynamicsSelector.assign`. This function takes specific {mod}`.dynamics.builder` functions and classes, such as {class}`.RelativisticBreitWignerBuilder`, which can create {func}`.relativistic_breit_wigner` functions for specific resonances. Here's an example for a relativistic Breit-Wigner _with form factor_ for the intermediate resonances and use a Blatt-Weisskopf barrier factor for the production decay:

# %%
from ampform.dynamics import PhaseSpaceFactor  # optional
from ampform.dynamics.builder import RelativisticBreitWignerBuilder

bw_builder = RelativisticBreitWignerBuilder(
    energy_dependent_width=True,
    form_factor=True,
    phsp_factor=PhaseSpaceFactor,  # optional
)
for name in reaction.get_intermediate_particles().names:
    model_builder.dynamics.assign(name, bw_builder)

# %% [markdown]
# Note that this {class}`.RelativisticBreitWignerBuilder` can also be initialized with a different {class}`.PhaseSpaceFactorProtocol`. This allows us to insert different phase space factors (see {doc}`/usage/dynamics/analytic-continuation` and {func}`.create_analytic_breit_wigner`).

# %% [markdown]
# ```{seealso}
# {doc}`/usage/dynamics/custom`
# ```

# %% [markdown]
# And we use the reconfigured {class}`.HelicityAmplitudeBuilder` to generate another {class}`.HelicityModel`, this time with relativistic Breit-Wigner functions and form factors:

# %%
model = model_builder.formulate()

# %% tags=["hide-input", "full-width"]
(symbol, expression), *_ = model.amplitudes.items()
Math(aslatex({symbol: expression}, terms_per_line=1))

# %% [markdown]
# ## Export

# %% [markdown]
# There is no special export function to export an {class}`.HelicityModel`. However, we can just use the built-in {mod}`pickle` module to write the model to disk and load it back:

# %%
import pickle

with open("helicity_model.pickle", "wb") as stream:
    pickle.dump(model, stream)
with open("helicity_model.pickle", "rb") as stream:
    model = pickle.load(stream)

# %% [markdown]
# ## Cached expression 'unfolding'
#
# Amplitude model expressions can be extremely large. AmpForm can formulate such expressions relatively fast, but {mod}`sympy` has to 'unfold' these expressions with {meth}`~sympy.core.basic.Basic.doit`, which can take a long time. AmpForm provides a function that can cache the 'unfolded' expression to disk, so that the expression unfolding runs faster upon the next run.

# %%
from ampform.sympy import perform_cached_doit

full_expression = perform_cached_doit(model.expression)
sp.count_ops(full_expression)

# %% [markdown]
# See {func}`.perform_cached_doit` for some tips on how to improve performance.

# %% [markdown]
# ## Visualize

# %% [markdown]
# ### Mathematical formula

# %% [markdown]
# It's possible to view the complete amplitude model as an expression. This would, however, clog the screen here, so we instead just view the formula of one of its {attr}`~.HelicityModel.components`:

# %%
some_amplitude, *_ = model.components.values()
some_amplitude.doit()

# %% [markdown]
# ```{note}
# We use {meth}`~sympy.core.basic.Basic.doit` to evaluate the Wigner-$D$ ({meth}`Rotation.D <sympy.physics.quantum.spin.Rotation.D>`) and Clebsch-Gordan ({class}`~sympy.physics.quantum.cg.CG`) functions in the full expression.
# ```

# %% [markdown]
# The {attr}`.HelicityModel.parameter_defaults` attribute can be used to substitute all parameters with suggested values:

# %%
some_amplitude.doit().subs(model.parameter_defaults)

# %% [markdown]
# :::{tip}
#
# To view the full expression for the amplitude model without crashing Jupyter Lab, install [`jupyterlab-katex`](https://pypi.org/project/jupyterlab-katex).
#
# :::

# %% [markdown]
# ### Plotting

# %% [markdown]
# In this case ($J/\psi \to \gamma f_0, f_0 \to \pi^0\pi^0$) _without dynamics_, the total intensity is only dependent on the $\theta$ angle of $\gamma$ in the center of mass frame (see {func}`.get_helicity_angle_symbols`):

# %%
no_dynamics = model_no_dynamics.expression.doit()
no_dynamics_substituted = no_dynamics.subs(model.parameter_defaults)
no_dynamics_substituted

# %%
assert len(no_dynamics_substituted.free_symbols) == 1

# %% tags=["hide-input"]
theta = next(iter(no_dynamics_substituted.free_symbols))
sp.plot(
    no_dynamics_substituted,
    (theta, 0, sp.pi),
    axis_center=(0, 0),
    ylabel="$I$",
    ylim=(0, 16),
);

# %% [markdown]
# For this decay channel, the amplitude model is built up of four components:

# %%
no_dynamics.subs(zip(no_dynamics.args, sp.symbols("I_{:4}")))

# %% [markdown]
# This can be nicely visualized as follows:

# %% tags=["hide-input"]
plots = []
colors = ["red", "blue", "green", "purple"]

total = 0
for i, intensity in enumerate(no_dynamics.args):
    total += intensity.subs(model.parameter_defaults).doit()
    plots.append(
        sp.plot(
            total,
            (theta, 0, sp.pi),
            axis_center=(0, 0),
            ylabel="$I$",
            ylim=(0, 16),
            line_color=colors[i],
            show=False,
            label=f"$I_{i}$",
            legend=True,
        )
    )
for i in range(1, 4):
    plots[0].extend(plots[i])
plots[0].show()

# %% [markdown]
# ## Plot the model

# %% [markdown]
# ```{tip}
# See {doc}`/usage/interactive` for a much more didactic way to visualize the model!
# ```

# %% [markdown]
# In the model _with dynamics_, we have several free symbols, such as the mass and width of the resonances. For the fitting package these will be considered **parameters** that are to be optimized and (kinematic) **variables** that represent the data set. Examples of parameters are mass ($m_\text{particle}$) and width ($\Gamma_\text{particle}$) of the resonances and certain amplitude coefficients ($C$). Examples of kinematic variables are the helicity angles $\theta$ and $\phi$ and the invariant masses ($m_{ij...}$).

# %% tags=["full-width"]
sorted(model.expression.free_symbols, key=lambda s: s.name)

# %% [markdown]
# Let's say we want to plot the amplitude model with respect to $m_{3+4}$. We would have to substitute all other free symbols with some value.

# %% [markdown]
# First, we can use {attr}`.HelicityModel.parameter_defaults` to substitute the parameters with suggested values:

# %%
suggested_expression = model.expression.subs(model.parameter_defaults)
suggested_expression.free_symbols

# %% [markdown]
# Ideally, we would now 'integrate out' the helicity angles. Here, we however just set these angles to $0$, as computing the integral would take quite some time:

# %%
angle = 0
angle_substitutions = {
    s: angle
    for s in suggested_expression.free_symbols
    if s.name.startswith("phi") or s.name.startswith("theta")
}
evaluated_angle_intensity = suggested_expression.subs(angle_substitutions)
evaluated_angle_intensity.free_symbols

# %% [markdown]
# By now we are only left with the masses of the final state particles ($m_1$ and $m_2$), since they appear as symbols in the {func}`.relativistic_breit_wigner_with_ff`. Final state particles 3 and 4 are the $\pi^0$'s, so we can just substitute them with their masses. (Alternatively, see {ref}`usage/amplitude:Scalar masses`.)

# %%
from qrules import load_pdg

pi0 = load_pdg()["pi0"]
plotted_intensity = evaluated_angle_intensity.doit().subs(
    {
        sp.Symbol("m_1", nonnegative=True): pi0.mass,
        sp.Symbol("m_2", nonnegative=True): pi0.mass,
    },
    simultaneous=True,
)

# %% [markdown]
# ```{tip}
# Use {meth}`~sympy.core.basic.Basic.subs` with `simultaneous=True`, as that avoids a bug later on when plotting with {mod}`matplotlib.pyplot`.
# ```

# %% [markdown]
# That's it! Now we are only left with the invariant mass $m_{3+4}$ of the two pions:

# %%
assert len(plotted_intensity.free_symbols) == 1
m = next(iter(plotted_intensity.free_symbols))
m

# %% [markdown]
# ...and we can plot it with with {func}`sympy.plot <sympy.plotting.plot.plot>`:

# %%
sp.plot(
    plotted_intensity,
    (m, 2 * pi0.mass, 2.5),
    axis_center=(2 * pi0.mass, 0),
    xlabel=Rf"$m(\pi^{0}\pi^{0})$",
    ylabel="$I$",
    backend="matplotlib",
);


# %% [markdown]
# The expression itself looks like this (after some rounding of the {class}`float` values in this expression using {doc}`tree traversal <sympy:tutorials/intro-tutorial/manipulation>`):

# %% jupyter={"source_hidden": true} tags=["hide-input"]
def round_nested(expression: sp.Expr, n_decimals: int = 2) -> sp.Expr:
    for node in sp.preorder_traversal(expression):
        if node.free_symbols:
            continue
        if isinstance(node, (float, sp.Float)):
            expression = expression.subs(node, round(node, n_decimals))
        if isinstance(node, sp.Pow) and node.args[1] == 1 / 2:
            expression = expression.subs(node, round(node.n(), n_decimals))
    return expression


# %% tags=["hide-input"]
rounded = round_nested(plotted_intensity)
round_nested(rounded)
