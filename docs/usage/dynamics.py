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

# # Dynamics

# {{ run_interactive }}

# By default, the dynamic terms in an amplitude model are set to $1$ by the {class}`.HelicityAmplitudeBuilder`. The method {meth}`~.DynamicsSelector.assign` of the {attr}`~.HelicityAmplitudeBuilder.dynamics` attribute can then be used to set dynamics lineshapes for specific resonances. The {mod}`.dynamics.builder` module provides some tools to set standard lineshapes (see below), but it is also possible to set {doc}`custom dynamics </usage/dynamics/custom>`.
#
# The standard lineshapes provided by AmpForm are illustrated below. For more info, have a look at the following pages:
#
# ```{toctree}
# :maxdepth: 2
# dynamics/custom
# dynamics/analytic-continuation
# dynamics/k-matrix
# ```
#
# ```{autolink-skip}
# ```

# + jupyter={"source_hidden": true} mystnb={"code_prompt_show": "Import Python libraries"} tags=["hide-input"]
import logging
import warnings

import matplotlib.pyplot as plt
import mpl_interactions.ipyplot as iplt
import numpy as np
import sympy as sp
from IPython.display import Math, display

import symplot

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

warnings.filterwarnings("ignore")
# -

# ## Form factor

# AmpForm uses Blatt-Weisskopf functions $B_L$ as _barrier factors_ (also called _form factors_, see {class}`.BlattWeisskopfSquared` and **[TR-029](https://compwa.github.io/report/029)**):

# +
from ampform.dynamics.form_factor import BlattWeisskopfSquared

L = sp.Symbol("L", integer=True, nonnegative=True)
z = sp.Symbol("z", nonnegative=True, real=True)
bl2 = BlattWeisskopfSquared(z, L)

# + jupyter={"source_hidden": true} tags=["hide-input"]
from ampform.dynamics.form_factor import SphericalHankel1
from ampform.io import aslatex

ell = sp.Symbol(R"\ell", integer=True, nonnegative=True)
exprs = [bl2, SphericalHankel1(ell, z)]
Math(aslatex({e: e.doit(deep=False) for e in exprs}))

# + jupyter={"source_hidden": true} tags=["hide-input"]
# %config InlineBackend.figure_formats = ['svg']
# %matplotlib inline

bl2_func = sp.lambdify((z, L), bl2.doit())
x_values = np.linspace(0.0, 5.0, num=500)

fig, ax = plt.subplots()
ax.set_xlabel(f"${sp.latex(z)}$")
ax.set_ylabel(f"${sp.latex(bl2)}$")
ax.set_ylim(0, 10)

for i in range(5):
    y_values = bl2_func(x_values, L=i)
    ax.plot(x_values, y_values, color=f"C{i}", label=f"$L={i}$")
ax.legend()
fig.show()
# -

# The Blatt-Weisskopf form factor is used to 'dampen' the breakup-momentum at threshold and when going to infinity. A usual choice for $z$ is therefore $z=q^2d^2$ with $q^2$ the {class}`.BreakupMomentumSquared` and $d$ the impact parameter (also called meson radius). The {class}`.FormFactor` expression class can be used for this:

# + jupyter={"source_hidden": true}
from ampform.dynamics.form_factor import FormFactor

s, m1, m2, d = sp.symbols("s m1 m2 d", nonnegative=True)
ff2 = FormFactor(s, m1, m2, angular_momentum=L, meson_radius=d)

# + jupyter={"source_hidden": true} tags=["hide-input"]
from ampform.dynamics.form_factor import BreakupMomentumSquared

q2 = BreakupMomentumSquared(s, m1, m2)
exprs = [ff2, q2]
Math(aslatex({e: e.doit(deep=False) for e in exprs}))

# + tags=["remove-input"]
# %matplotlib widget

# + jupyter={"source_hidden": true} tags=["hide-input", "remove-output", "scroll-input"]
import ipywidgets as w

ff2_func = sp.lambdify((s, m1, m2, L, d), ff2.doit())
q2_func = sp.lambdify((s, m1, m2, L, d), q2.doit())

x = np.linspace(0.01, 4, 500)
sliders = dict(
    m1=w.FloatSlider(description="$m_1$", min=0, max=2, value=0.3),
    m2=w.FloatSlider(description="$m_2$", min=0, max=2, value=0.2),
    L=w.IntSlider(description="$L$", min=0, max=10, value=1),
    d=w.FloatSlider(description="$d$", min=0.1, max=5, value=1),
)

fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
ax.set_xlabel("$m$")
ax.axhline(0, c="black", linewidth=0.5)
LINES = None


def plot(m1, m2, L, d):
    global LINES
    s = x**2
    y_ff2 = ff2_func(s, m1, m2, L, d)
    y_q2 = q2_func(s, m1, m2, L, d)
    m_thr = m1 + m2
    left = x < m_thr
    right = x > m_thr
    if LINES is None:
        LINES = (
            ax.axvline(m_thr, c="black", ls="dotted", label="$m_1+m_2$"),
            ax.plot(x[left], y_ff2[left], c="C0", ls="dashed")[0],
            ax.plot(x[left], y_q2[left], c="C1", ls="dashed")[0],
            ax.plot(x[right], y_ff2[right], c="C0", label=f"${sp.latex(ff2)}$")[0],
            ax.plot(x[right], y_q2[right], c="C1", label=f"${sp.latex(q2)}$")[0],
        )
    else:
        LINES[0].set_xdata(m_thr)
        LINES[1].set_data(x[left], y_ff2[left])
        LINES[2].set_data(x[left], y_q2[left])
        LINES[3].set_data(x[right], y_ff2[right])
        LINES[4].set_data(x[right], y_q2[right])
    y_min = np.nanmin(y_q2)
    y_max = np.nanmax(y_q2[right])
    ax.set_ylim(y_min, y_max)
    fig.canvas.draw()


UI = w.VBox(list(sliders.values()))
OUTPUT = w.interactive_output(plot, controls=sliders)
ax.legend(loc="upper right")
display(UI, OUTPUT)

# + jupyter={"source_hidden": true} tags=["remove-input"]
if STATIC_WEB_PAGE:
    from IPython.display import SVG

    output_file = "blatt-weisskopf.svg"
    fig.savefig(output_file)
    display(SVG(output_file))
# -

# ## Relativistic Breit-Wigner

# AmpForm has two types of relativistic Breit-Wigner functions. Both are compared below ― for more info, see the links to the API.

# ### _Without_ form factor

# The 'normal' {func}`.relativistic_breit_wigner` looks as follows:

# +
from ampform.dynamics import relativistic_breit_wigner

m, m0, w0 = sp.symbols("m, m0, Gamma0", nonnegative=True)
rel_bw = relativistic_breit_wigner(s=m**2, mass0=m0, gamma0=w0)
rel_bw
# -

# ### _With_ form factor

# The relativistic Breit-Wigner can be adapted slightly, so that its amplitude goes to zero at threshold ($m_0 = m1 + m2$) and that it becomes normalizable. This is done with {ref}`form factors <usage/dynamics:Form factor>` and can be obtained with the function {func}`.relativistic_breit_wigner_with_ff`:

# +
from ampform.dynamics import PhaseSpaceFactorSWave, relativistic_breit_wigner_with_ff

rel_bw_with_ff = relativistic_breit_wigner_with_ff(
    s=s,
    mass0=m0,
    gamma0=w0,
    m_a=m1,
    m_b=m2,
    angular_momentum=L,
    meson_radius=1,
    phsp_factor=PhaseSpaceFactorSWave,
)
rel_bw_with_ff
# -

# Here, $\Gamma(m)$ is the {class}`.EnergyDependentWidth` (also called running width or mass-dependent width), defined as:

# + jupyter={"source_hidden": true}
from ampform.dynamics import EnergyDependentWidth

L = sp.Symbol("L", integer=True)
width = EnergyDependentWidth(
    s=s,
    mass0=m0,
    gamma0=w0,
    m_a=m1,
    m_b=m2,
    angular_momentum=L,
    meson_radius=1,
    phsp_factor=PhaseSpaceFactorSWave,
)
Math(aslatex({width: width.evaluate()}))
# -

# It is possible to choose different formulations for the phase space factor $\rho$, see {doc}`/usage/dynamics/analytic-continuation`.

# ### Analytic continuation

# The following shows the effect of {doc}`/usage/dynamics/analytic-continuation` a on relativistic Breit-Wigner:

# + jupyter={"source_hidden": true} tags=["hide-cell", "remove-output"]
from ampform.dynamics import PhaseSpaceFactorComplex

# Two types of relativistic Breit-Wigners
rel_bw_with_ff = relativistic_breit_wigner_with_ff(
    s=m**2,
    mass0=m0,
    gamma0=w0,
    m_a=m1,
    m_b=m2,
    angular_momentum=L,
    meson_radius=d,
    phsp_factor=PhaseSpaceFactorComplex,
)
rel_bw_with_ff_ac = relativistic_breit_wigner_with_ff(
    s=m**2,
    mass0=m0,
    gamma0=w0,
    m_a=m1,
    m_b=m2,
    angular_momentum=L,
    meson_radius=d,
    phsp_factor=PhaseSpaceFactorSWave,
)

# Lambdify
np_rel_bw_with_ff, sliders = symplot.prepare_sliders(
    plot_symbol=m,
    expression=rel_bw_with_ff.doit(),
)
np_rel_bw_with_ff_ac = sp.lambdify(
    args=(m, w0, L, d, m0, m1, m2),
    expr=rel_bw_with_ff_ac.doit(),
)
np_rel_bw = sp.lambdify(
    args=(m, w0, L, d, m0, m1, m2),
    expr=rel_bw.doit(),
)

# Set sliders
plot_domain = np.linspace(0, 4, num=500)
sliders.set_ranges(
    m0=(0, 4, 200),
    Gamma0=(0, 1, 100),
    L=(0, 10),
    m1=(0, 2, 200),
    m2=(0, 2, 200),
    d=(0, 5),
)
sliders.set_values(
    m0=1.5,
    Gamma0=0.6,
    L=0,
    m1=0.6,
    m2=0.6,
    d=1,
)

fig, axes = plt.subplots(
    nrows=2,
    figsize=(8, 6),
    sharex=True,
)
ax_ff, ax_ac = axes
ax_ac.set_xlabel("$m$")
for ax in axes:
    ax.axhline(0, c="gray", linewidth=0.5)

rho_c = PhaseSpaceFactorComplex(m**2, m1, m2)
ax_ff.set_title(f"'Complex' phase space factor: ${sp.latex(rho_c.evaluate())}$")
ax_ac.set_title("$S$-wave Chew-Mandelstam as phase space factor")

ylim = "auto"  # (-0.6, 1.2)
controls = iplt.plot(
    plot_domain,
    lambda *args, **kwargs: np_rel_bw_with_ff(*args, **kwargs).real,
    label="real",
    **sliders,
    ylim=ylim,
    ax=ax_ff,
)
iplt.plot(
    plot_domain,
    lambda *args, **kwargs: np_rel_bw_with_ff(*args, **kwargs).imag,
    label="imaginary",
    controls=controls,
    ylim=ylim,
    ax=ax_ff,
)
iplt.plot(
    plot_domain,
    lambda *args, **kwargs: np.abs(np_rel_bw_with_ff(*args, **kwargs)) ** 2,
    label="absolute",
    controls=controls,
    ylim=ylim,
    ax=ax_ff,
    c="black",
    linestyle="dotted",
)


iplt.plot(
    plot_domain,
    lambda *args, **kwargs: np_rel_bw_with_ff_ac(*args, **kwargs).real,
    label="real",
    controls=controls,
    ylim=ylim,
    ax=ax_ac,
)
iplt.plot(
    plot_domain,
    lambda *args, **kwargs: np_rel_bw_with_ff_ac(*args, **kwargs).imag,
    label="imaginary",
    controls=controls,
    ylim=ylim,
    ax=ax_ac,
)
iplt.plot(
    plot_domain,
    lambda *args, **kwargs: np.abs(np_rel_bw_with_ff_ac(*args, **kwargs)) ** 2,
    label="absolute",
    controls=controls,
    ylim=ylim,
    ax=ax_ac,
    c="black",
    linestyle="dotted",
)

for ax in axes:
    iplt.axvline(
        controls["m0"],
        ax=ax,
        c="red",
        label=f"${sp.latex(m0)}$",
        alpha=0.3,
    )
    iplt.axvline(
        lambda m1, m2, **kwargs: m1 + m2,
        controls=controls,
        ax=ax,
        c="black",
        alpha=0.3,
        label=f"${sp.latex(m1)} + {sp.latex(m2)}$",
    )
ax_ac.legend(loc="upper right")
fig.tight_layout()
plt.show()

# + jupyter={"source_hidden": true} tags=["remove-input", "full-width"]
if STATIC_WEB_PAGE:
    from IPython.display import SVG

    output_file = "relativistic-breit-wigner-with-form-factor.svg"
    fig.savefig(output_file)
    display(SVG(output_file))
