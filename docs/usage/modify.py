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
# # Modify amplitude model

# %% jupyter={"source_hidden": true} mystnb={"code_prompt_show": "Import Python libraries"} tags=["hide-cell"]
import attrs
import graphviz
import qrules
import sympy as sp
from IPython.display import Math, display

from ampform import get_builder
from ampform.io import aslatex

# %% [markdown]
# Since a {attr}`.HelicityModel.expression` is simply a {class}`sympy.Expr <sympy.core.expr.Expr>`, it's relatively easy to modify it. The {class}`.HelicityModel` however also contains other attributes that need to be modified accordingly. In this notebook, we show how to do that for specific use cases using the following example decay:

# %% tags=["remove-output"]
result = qrules.generate_transitions(
    initial_state=("J/psi(1S)", [-1, +1]),
    final_state=["gamma", "pi0", "pi0"],
    allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
    allowed_interaction_types=["strong", "EM"],
    formalism="helicity",
)
model_builder = get_builder(result)
original_model = model_builder.formulate()

# %% jupyter={"source_hidden": true} tags=["hide-input"]
dot = qrules.io.asdot(result, collapse_graphs=True)
graphviz.Source(dot)

# %% [markdown]
# ## Couple parameters

# %% [markdown]
# We can couple parameters renaming them:

# %%
renames = {
    R"C_{J/\psi(1S) \to {f_{0}(980)}_{0} \gamma_{+1}; f_{0}(980) \to \pi^{0}_{0} \pi^{0}_{0}}": (
        "C"
    ),
    R"C_{J/\psi(1S) \to {f_{0}(1500)}_{0} \gamma_{+1}; f_{0}(1500) \to \pi^{0}_{0} \pi^{0}_{0}}": (
        "C"
    ),
}
new_model = original_model.rename_symbols(renames)

# %%
new_model.parameter_defaults

# %%
new_model.components[R"I_{J/\psi(1S)_{+1} \to \gamma_{+1} \pi^{0}_{0} \pi^{0}_{0}}"]

# %% [markdown]
# ## Parameter substitution

# %% [markdown]
# Let's say we want to express all coefficients as a product $Ce^{i\phi}$ of magnitude $C$  with phase $\phi$.

# %%
original_coefficients = [
    par for par in original_model.parameter_defaults if par.name.startswith("C")
]
original_coefficients

# %% [markdown]
# ```{margin}
# The attributes {attr}`~.HelicityModel.parameter_defaults` and {attr}`~.HelicityModel.components` are _mutable_ {obj}`dict`s, so these can be modified (even if not set as a whole). This is why we make a copy of them below.
# ```

# %% [markdown]
# There are two things to note now:
#
# 1. These parameters appear in {attr}`.HelicityModel.expression`, its {attr}`~.HelicityModel.parameter_defaults`, and its  {attr}`~.HelicityModel.components`, so both these attributes should be modified accordingly.
# 2. A {class}`.HelicityModel` is {doc}`immutable <attrs:how-does-it-work>`, so we cannot directly replace its attributes. Instead, we should create a new {class}`.HelicityModel` with substituted attributes using {func}`attrs.evolve`:

# %% [markdown]
# The following snippet shows how to do all this. It's shown in full, because it could well be you want to perform some completely different substitutions (can be any kinds of {meth}`~sympy.core.basic.Basic.subs`). The overall procedure is comparable, however.

# %%
new_intensity = original_model.intensity
new_amplitudes = dict(original_model.amplitudes)
new_parameter_defaults = dict(original_model.parameter_defaults)  # copy!
new_components = dict(original_model.components)  # copy!

for coefficient in original_coefficients:
    decay_description = coefficient.name[3:-1]
    magnitude = sp.Symbol(  # coefficient with same name, but real, not complex
        coefficient.name,
        nonnegative=True,
    )
    phase = sp.Symbol(
        Rf"\phi_{{{decay_description}}}",
        real=True,
    )
    replacement = magnitude * sp.exp(sp.I * phase)
    display(replacement)
    # replace parameter defaults
    del new_parameter_defaults[coefficient]
    new_parameter_defaults[magnitude] = 1.0
    new_parameter_defaults[phase] = 0.0
    # replace parameters in expression
    new_intensity = new_intensity.subs(coefficient, replacement, simultaneous=True)
    # replace parameters in each component
    new_amplitudes = {
        key: old_expression.subs(coefficient, replacement, simultaneous=True)
        for key, old_expression in new_amplitudes.items()
    }
    new_components = {
        key: old_expression.subs(coefficient, replacement, simultaneous=True)
        for key, old_expression in new_components.items()
    }

# create new model from the old
new_model = attrs.evolve(
    original_model,
    intensity=new_intensity,
    amplitudes=new_amplitudes,
    parameter_defaults=new_parameter_defaults,
    components=new_components,
)

# %% jupyter={"source_hidden": true} tags=["hide-cell"]
assert new_model != original_model

# %% [markdown]
# As can be seen, the {attr}`~.HelicityModel.parameter_defaults` have bene updated, as have the {attr}`~.HelicityModel.components`:

# %%
Math(aslatex(new_model.parameter_defaults))

# %% tags=["full-width"]
new_model.components[
    R"A_{J/\psi(1S)_{-1} \to {f_{0}(980)}_{0} \gamma_{-1}; {f_{0}(980)}_{0} \to \pi^{0}_{0} \pi^{0}_{0}}"
]

# %% [markdown]
# Also note that the new model reduces to the old once we replace the parameters with their suggested default values:

# %%
evaluated_expr = new_model.expression.subs(new_model.parameter_defaults).doit()
evaluated_expr

# %% jupyter={"source_hidden": true} tags=["hide-cell"]
assert (
    original_model.expression.subs(original_model.parameter_defaults).doit()
    == evaluated_expr
)
