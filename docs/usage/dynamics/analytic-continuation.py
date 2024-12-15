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

# # Analytic continuation

# :::{note}
#
# Improvements to analytic continuation in AmpForm are currently being developed in {doc}`compwa-report:003/index` and {doc}`compwa-report:004/index`.
#
# :::

# Analytic continuation allows one to handle resonances just below threshold ($m_0 < m_a + m_b$  in Eq. {eq}`relativistic_breit_wigner_with_ff`). In practice, this entails using a specific function for $\rho$ in Eq. {eq}`EnergyDependentWidth`.

# ## Definitions

# Three usual choices for $\rho$ are the following:

# + jupyter={"source_hidden": true} mystnb={"code_prompt_show": "Import Python libraries"} tags=["hide-input"]
# %config InlineBackend.figure_formats = ['svg']

import warnings

import sympy as sp
from IPython.display import Math

from ampform.io import aslatex

warnings.filterwarnings("ignore")
# -

# ### 1) Break-up momentum

# The {func}`~sympy.functions.elementary.miscellaneous.sqrt` or {class}`.ComplexSqrt` of {class}`.BreakupMomentumSquared`:

# +
from ampform.dynamics import BreakupMomentumSquared

s, m_a, m_b = sp.symbols("s, m_a, m_b", nonnegative=True)
q_squared = BreakupMomentumSquared(s, m_a, m_b)
Math(aslatex({q_squared: q_squared.evaluate()}))
# -

# ### 2) 'Normal' phase space factor

# The 'normal' {class}`.PhaseSpaceFactor` (the denominator makes the difference to {eq}`EnergyDependentWidth`!):

# +
from ampform.dynamics import PhaseSpaceFactor

rho = PhaseSpaceFactor(s, m_a, m_b)
Math(aslatex({rho: rho.evaluate()}))
# -

# ### 3) 'Complex' phase space factor

# A {class}`.PhaseSpaceFactorComplex` that uses {class}`.ComplexSqrt`:

# +
from ampform.dynamics import PhaseSpaceFactorComplex

rho_c = PhaseSpaceFactorComplex(s, m_a, m_b)
Math(aslatex({rho_c: rho_c.evaluate()}))
# -

# ### 4) 'Analytic continuation' of the phase space factor

# The following 'case-by-case' **analytic continuation** for decay products with an _equal_ mass, {class}`.EqualMassPhaseSpaceFactor`:

# +
from ampform.dynamics import EqualMassPhaseSpaceFactor

rho_ac = EqualMassPhaseSpaceFactor(s, m_a, m_b)
Math(aslatex({rho_ac: rho_ac.evaluate()}))
# -

# with

# + jupyter={"source_hidden": true} tags=["hide-input"]
from ampform.dynamics import PhaseSpaceFactorAbs

rho_hat = PhaseSpaceFactorAbs(s, m_a, m_b)
Math(aslatex({rho_hat: rho_hat.evaluate()}))
# -

# (Mind the absolute value.)

# ### 5) Chew-Mandelstam for $S$-waves

# A {class}`.PhaseSpaceFactorSWave` that uses {func}`.chew_mandelstam_s_wave`:

# + tags=["full-width"]
from ampform.dynamics import PhaseSpaceFactorSWave

rho_cm = PhaseSpaceFactorSWave(s, m_a, m_b)
Math(aslatex({rho_cm: rho_cm.evaluate()}))
# -

# ## Visualization

# ```{autolink-skip}
# ```

# %matplotlib widget

# + jupyter={"source_hidden": true} tags=["hide-input"]
import matplotlib.pyplot as plt
import mpl_interactions.ipyplot as iplt
import numpy as np

import symplot

# +
from ampform.sympy.math import ComplexSqrt

m = sp.Symbol("m", nonnegative=True)
rho_c = PhaseSpaceFactorComplex(m**2, m_a, m_b)
rho_cm = PhaseSpaceFactorSWave(m**2, m_a, m_b)
rho_ac = EqualMassPhaseSpaceFactor(m**2, m_a, m_b)
np_rho_c, sliders = symplot.prepare_sliders(plot_symbol=m, expression=rho_c.doit())
np_rho_ac = sp.lambdify((m, m_a, m_b), rho_ac.doit())
np_rho_cm = sp.lambdify((m, m_a, m_b), rho_cm.doit())
np_breakup_momentum = sp.lambdify(
    (m, m_a, m_b),
    2 * ComplexSqrt(q_squared.subs(s, m**2).doit()),
)
# -

# {{ run_interactive }}

plot_domain = np.linspace(0, 3, 500)
sliders.set_ranges(
    m_a=(0, 2, 200),
    m_b=(0, 2, 200),
)
sliders.set_values(
    m_a=0.3,
    m_b=0.75,
)

# + jupyter={"source_hidden": true} tags=["remove-output", "hide-input"]
fig, axes = plt.subplots(
    ncols=2,
    nrows=2,
    figsize=[8, 8],
    sharex=True,
    sharey=True,
)
fig.canvas.toolbar_visible = False

(ax_q, ax_rho), (ax_rho_ac, ax_rho_cm) = axes
for ax in [ax_q, ax_rho, ax_rho_cm, ax_rho_ac]:
    ax.set_xlabel("$m$")
    ax.set_yticks([])
for ax in [ax_rho_cm, ax_rho_ac]:
    ax.set_yticks([])

ylim = (-0.1, 1.4)


def func_imag(func, *args, **kwargs):
    return lambda *args, **kwargs: func(*args, **kwargs).imag


def func_real(func, *args, **kwargs):
    return lambda *args, **kwargs: func(*args, **kwargs).real


q_math = ComplexSqrt(sp.Symbol("q^2")) / (8 * sp.pi)
ax_q.set_title(f"${sp.latex(q_math)}$")
controls = iplt.plot(
    plot_domain,
    func_real(np_breakup_momentum),
    label="real",
    **sliders,
    ylim=ylim,
    ax=ax_q,
    alpha=0.7,
)
iplt.plot(
    plot_domain,
    func_imag(np_breakup_momentum),
    label="imaginary",
    controls=controls,
    ylim=ylim,
    ax=ax_q,
    alpha=0.7,
)

ax_rho.set_title(f"${sp.latex(rho_c)}$")
iplt.plot(
    plot_domain,
    func_real(np_rho_c),
    label="real",
    controls=controls,
    ylim=ylim,
    ax=ax_rho,
    alpha=0.7,
)
iplt.plot(
    plot_domain,
    func_imag(np_rho_c),
    label="imaginary",
    controls=controls,
    ylim=ylim,
    ax=ax_rho,
    alpha=0.7,
)

ax_rho_ac.set_title(R"equal mass $\rho^\mathrm{eq}(m^2)$")
iplt.plot(
    plot_domain,
    func_real(np_rho_ac),
    label="real",
    controls=controls,
    ylim=ylim,
    ax=ax_rho_ac,
    alpha=0.7,
)
iplt.plot(
    plot_domain,
    func_imag(np_rho_ac),
    label="imaginary",
    controls=controls,
    ylim=ylim,
    ax=ax_rho_ac,
    alpha=0.7,
)

ax_rho_cm.set_title(R"Chew-Mandelstam $\rho^\mathrm{CM}(m^2)$")
iplt.plot(
    plot_domain,
    func_real(np_rho_cm),
    label="real",
    controls=controls,
    ylim=ylim,
    ax=ax_rho_cm,
    alpha=0.7,
)
iplt.plot(
    plot_domain,
    func_imag(np_rho_cm),
    label="imaginary",
    controls=controls,
    ylim=ylim,
    ax=ax_rho_cm,
    alpha=0.7,
)

fig.tight_layout()
plt.legend(loc="upper right")
plt.show()

# + jupyter={"source_hidden": true} tags=["remove-input", "full-width"]
if STATIC_WEB_PAGE:
    from IPython.display import SVG, display

    output_file = "analytic-continuation.svg"
    plt.savefig(output_file)
    display(SVG(output_file))
