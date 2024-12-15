# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + hideCode=true hideOutput=true hidePrompt=true jupyter={"source_hidden": true} tags=["remove-cell", "skip-execution"]
# WARNING: advised to install a specific version, e.g. ampform==0.1.2
# %pip install -q ampform[doc,viz] IPython

# + hideCode=true hideOutput=true hidePrompt=true jupyter={"source_hidden": true} tags=["remove-cell"]
import os

STATIC_WEB_PAGE = {"EXECUTE_NB", "READTHEDOCS"}.intersection(os.environ)
# -

# ```{autolink-concat}
# ```

# # Usage

# ## Overview

# ### Library of symbolic dynamics functions

# AmpForm offers a number of dynamics parametrization functions. The functions are expressed with {mod}`sympy`, so that they can easily be visualized, simplified, or modified:

# + tags=["remove-cell"]
import logging

from IPython.display import Math

from ampform.io import improve_latex_rendering

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.ERROR)
improve_latex_rendering()

# +
import sympy as sp

from ampform.dynamics import relativistic_breit_wigner

m, m0, w0 = sp.symbols("m m0 Gamma0")
relativistic_breit_wigner(s=m**2, mass0=m0, gamma0=w0)

# +
from ampform.dynamics import relativistic_breit_wigner_with_ff

m1, m2, L = sp.symbols("m1 m2 L")
relativistic_breit_wigner_with_ff(
    s=m**2,
    mass0=m0,
    gamma0=w0,
    m_a=m1,
    m_b=m2,
    angular_momentum=L,
    meson_radius=1,
)

# +
from ampform.dynamics.kmatrix import NonRelativisticKMatrix

n_poles = sp.Symbol("n_R", integer=True, positive=True)
NonRelativisticKMatrix.formulate(
    n_poles=n_poles,
    n_channels=1,
)[0, 0]
# -

matrix = NonRelativisticKMatrix.formulate(n_poles=1, n_channels=2)
matrix[0, 0].doit().simplify()

# More dynamics functions can be found in the {mod}`.dynamics` library, as well as on the {doc}`/usage/dynamics` page!

# ### Formulate amplitude models

# Together with [QRules](https://qrules.rtfd.io), AmpForm can automatically formulate amplitude models for generic, multi-body decays. These models can then be used as templates for faster computational back-ends with [TensorWaves](https://tensorwaves.rtfd.io). Here's an example:

# +
import qrules

reaction = qrules.generate_transitions(
    initial_state=("psi(4160)", [-1, +1]),
    final_state=["D-", "D0", "pi+"],
    allowed_intermediate_particles=["D*(2007)0"],
    formalism="helicity",
)

# +
import ampform
from ampform.dynamics.builder import create_relativistic_breit_wigner

builder = ampform.get_builder(reaction)
for particle in reaction.get_intermediate_particles():
    builder.dynamics.assign(particle.name, create_relativistic_breit_wigner)
model = builder.formulate()
model.intensity

# + jupyter={"source_hidden": true} tags=["hide-input", "full-width"]
from ampform.io import aslatex

(symbol, expr), *_ = model.amplitudes.items()
Math(aslatex({symbol: expr}, terms_per_line=1))
# -

# In case of multiple decay topologies, AmpForm also takes care of {doc}`spin alignment </usage/helicity/spin-alignment>` with {cite}`Marangotto:2019ucc`!

reaction = qrules.generate_transitions(
    initial_state="Lambda(c)+",
    final_state=["p", "K-", "pi+"],
    allowed_intermediate_particles=[
        "Lambda(1405)",
        "Delta(1232)++",
    ],
)

# + jupyter={"source_hidden": true} tags=["hide-input"]
import graphviz

dot = qrules.io.asdot(reaction, collapse_graphs=True)
graphviz.Source(dot)

# + tags=["full-width"]
builder = ampform.get_builder(reaction)
model = builder.formulate()
model.intensity
# -

# ## Advanced examples

# The following pages provide more advanced examples of how to use AmpForm. You can run each of them as Jupyter notebooks with the {fa}`rocket` launch button in the top-right corner.

# ```{toctree}
# ---
# maxdepth: 2
# ---
# usage/amplitude
# usage/modify
# usage/interactive
# usage/dynamics
# usage/helicity/formalism
# usage/helicity/spin-alignment
# usage/kinematics
# usage/sympy
# ```
