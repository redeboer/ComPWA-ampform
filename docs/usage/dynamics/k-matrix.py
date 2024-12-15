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
# # K-matrix

# %% [markdown]
# <!-- cspell:ignore amma -->

# %% [markdown]
# While {mod}`ampform` does not yet provide a generic way to formulate an amplitude model with $\boldsymbol{K}$-matrix dynamics, the (experimental) {mod}`.kmatrix` module makes it fairly simple to produce a symbolic expression for a parameterized $\boldsymbol{K}$-matrix with an arbitrary number of poles and channels and play around with it interactively. For more info on the $\boldsymbol{K}$-matrix, see the classic paper by Chung {cite}`Chung:1995dx`, {pdg-review}`2021; Resonances`, or this instructive presentation {cite}`meyerMatrixTutorial2008`.
#
# Section {ref}`usage/dynamics/k-matrix:Physics` summarizes {cite}`Chung:1995dx`, so that the {mod}`.kmatrix` module can reference to the equations. It also points out some subtleties and deviations.
#
# :::{note}
#
# The $\boldsymbol{K}$-matrix approach was originally worked worked out in {doc}`compwa-report:005/index`, {doc}`compwa-report:009/index`, and {doc}`compwa-report:010/index`. Those reports contained a few mistakes, which have been addressed here.
#
# :::
#
# ```{autolink-skip}
# ```

# %%
# %matplotlib widget

# %% jupyter={"source_hidden": true} mystnb={"code_prompt_show": "Import Python libraries"} tags=["hide-cell"]
import logging
import re
import warnings

import graphviz
import ipywidgets
import matplotlib.pyplot as plt
import mpl_interactions.ipyplot as iplt
import numpy as np
import sympy as sp
from IPython.display import Image, display
from matplotlib import cm
from mpl_interactions.controller import Controls

import symplot
from ampform.dynamics import PhaseSpaceFactor, kmatrix, relativistic_breit_wigner

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

# %% [markdown]
# ## Physics

# %% [markdown] tags=["scroll-input"]
# The $\boldsymbol{K}$-matrix formalism is used to describe coupled, two-body **formation processes** of the form $c_j d_j \to R \to a_i b_i$, with $i,j$ representing each separate channel and $R$ an intermediate state by which these channels are coupled.

# %% jupyter={"source_hidden": true} tags=["hide-input"]
dot = """
digraph {
    rankdir=LR;
    node [shape=point, width=0];
    edge [arrowhead=none];
    "Na" [shape=none, label="aᵢ"];
    "Nb" [shape=none, label="bᵢ"];
    "Nc" [shape=none, label="cⱼ"];
    "Nd" [shape=none, label="dⱼ"];
    { rank=same "Nc", "Nd" };
    { rank=same "Na", "Nb" };
    "Nc" -> "N0";
    "Nd" -> "N0";
    "N1" -> "Na";
    "N1" -> "Nb";
    "N0" -> "N1" [label="R"];
    "N0" [shape=none, label=""];
    "N1" [shape=none, label=""];
}
"""
graphviz.Source(dot)

# %% [markdown]
# A small adaptation allows us to describe a coupled, two-body **production process** of the form $R \to a_ib_i$ (see {ref}`usage/dynamics/k-matrix:Production processes`).
#
# In the following, $n$ denotes the number of channels and $n_R$ the number of poles. In the {mod}`.kmatrix` module, we use $0 \leq i,j < n$ and $1 \leq R \leq n_R$.

# %% [markdown]
# ### Partial wave expansion

# %% [markdown]
# In amplitude analysis, the main aim is to express the differential cross section $\frac{d\sigma}{d\Omega}$, that is, the intensity distribution in each spherical direction $\Omega=(\phi,\theta)$ as we can observe in experiments. This differential cross section can be expressed in terms of the **scattering amplitude** $A$:
#
# ```{margin}
# {cite}`Chung:1995dx` Eq. (1)
# ```
#
# $$
# \frac{d\sigma}{d\Omega} = \left|A(\Omega)\right|^2.
# $$ (differential cross section)
#
# We can now further express $A$ in terms of **partial wave amplitudes** by expanding it in terms of its angular momentum components:[^spin-formalisms]
#
# ```{margin}
# {cite}`Chung:1995dx` Eq. (2)
# ```
#
# $$
# A(\Omega) = \frac{1}{q_i}\sum_L\left(2L+1\right) T_L(s) {D^{*L}_{\lambda\mu}}\left(\phi,\theta,0\right)
# $$ (partial-wave-expansion)
#
# with $L$ the total angular momentum of the decay product pair, $\lambda=\lambda_a-\lambda_b$ and $\mu=\lambda_c-\lambda_d$ the helicity differences of the final and initial states, $D$ a [Wigner-$D$ function](https://en.wikipedia.org/wiki/Wigner_D-matrix), and $T_J$ an operator representing the partial wave amplitude.
#
# [^spin-formalisms]: Further subtleties arise when taking spin into account, especially for sequential decays. This is where {doc}`spin formalisms </usage/helicity/formalism>` come in.
#
# The above sketch is just with one channel in mind. The same holds true though for a number of channels $n$, with the only difference being that the $T$ operator becomes an $n\times n$ $\boldsymbol{T}$-matrix.

# %% [markdown]
# ### Transition operator

# %% [markdown]
# The important point is that we have now expressed $A$ in terms of an angular part (depending on $\Omega$) and a dynamical part $\boldsymbol{T}$ that depends on the {ref}`Mandelstam variable <pwa:introduction:Mandelstam variables>` $s$.
#
#
# The dynamical part $\boldsymbol{T}$ is usually called the **transition operator**. It describes the interacting part of the more general **scattering operator** $\boldsymbol{S}$, which describes the (complex) amplitude $\langle f|\boldsymbol{S}|i\rangle$  of an initial state $|i\rangle$ transitioning to a final state $|f\rangle$. The scattering operator describes both the non-interacting amplitude and the transition amplitude, so it relates to the transition operator as:
#
# ```{margin}
# {cite}`Chung:1995dx` Eq. (10)
# ```
#
# $$
# \boldsymbol{S} = \boldsymbol{I} + 2i\boldsymbol{T}
# $$ (S in terms of T)
#
# with $\boldsymbol{I}$ the identity operator. Just like in {cite}`Chung:1995dx`, we use a factor 2, while other authors choose $\boldsymbol{S} = \boldsymbol{I} + i\boldsymbol{T}$. In that case, one would have to multiply Eq. {eq}`partial-wave-expansion` by a factor $\frac{1}{2}$.

# %% [markdown]
# ### Ensuring unitarity

# %% [markdown]
# Knowing the origin of the $\boldsymbol{T}$-matrix, there is an important restriction that we need to comply with when we further formulate a {ref}`parametrization <usage/dynamics/k-matrix:Pole parametrization>`: **unitarity**. This means that $\boldsymbol{S}$ should conserve probability, namely $\boldsymbol{S}^\dagger\boldsymbol{S} = \boldsymbol{I}$. Luckily, there is a trick that makes this easier. If we express $\boldsymbol{S}$ in terms of an operator $\boldsymbol{K}$ by applying a [Cayley transformation](https://en.wikipedia.org/wiki/Cayley_transform):
#
# ```{margin}
# {cite}`Chung:1995dx` Eq. (20)
# ```
#
# $$
# \boldsymbol{S} = (\boldsymbol{I} + i\boldsymbol{K})(I - i\boldsymbol{K})^{-1},
# $$ (Cayley transformation)
#
# _unitarity is conserved if $\boldsymbol{K}$ is real_. With some matrix jumbling, we can derive that the $\boldsymbol{T}$-matrix can be expressed in terms of $\boldsymbol{K}$ as follows:
#
# ```{margin}
# {cite}`Chung:1995dx` Eq. (19);
# compare with {eq}`T-hat-in-terms-of-K-hat`
# ```
#
# $$
# \boldsymbol{T}
# = \boldsymbol{K} \left(\boldsymbol{I} - i\boldsymbol{K}\right)^{-1}
# = \left(\boldsymbol{I} - i\boldsymbol{K}\right)^{-1} \boldsymbol{K}.
# $$ (T-in-terms-of-K)

# %% [markdown]
# ### Lorentz-invariance

# %% [markdown]
# The description so far did not take Lorentz-invariance into account. For this, we first need to define a **two-body phase space matrix** $\boldsymbol{\rho}$:
#
# ```{margin}
# {cite}`Chung:1995dx` Eq. (36)
# ```
#
# $$
# \boldsymbol{\rho} = \begin{pmatrix}
# \rho_0 & \cdots & 0      \\
# \vdots & \ddots & \vdots \\
# 0      & \cdots & \rho_{n-1}
# \end{pmatrix}.
# $$ (rho matrix)
#
# with $\rho_i$ given by {eq}`PhaseSpaceFactor` in {class}`.PhaseSpaceFactor` for the final state masses $m_{a,i}, m_{b,i}$. The **Lorentz-invariant amplitude $\boldsymbol{\hat{T}}$** and corresponding Lorentz-invariant $\boldsymbol{\hat{K}}$-matrix can then be computed from $\boldsymbol{T}$ and $\boldsymbol{K}$ with:[^rho-dagger]
#
# ```{margin}
# {cite}`Chung:1995dx` Eqs. (34) and (47)
# ```
#
# $$
# \begin{eqnarray}
# \boldsymbol{T} & = & \sqrt{\boldsymbol{\rho}} \; \boldsymbol{\hat{T}} \sqrt{\boldsymbol{\rho}} \\
# \boldsymbol{K} & = & \sqrt{\boldsymbol{\rho}} \; \boldsymbol{\hat{K}} \sqrt{\boldsymbol{\rho}}.
# \end{eqnarray}
# $$ (K-hat-and-T-hat)
#
# [^rho-dagger]: An unpublished primer on the $\boldsymbol{K}$-matrix by Chung {cite}`chungPrimerKmatrixFormalism1995` uses a conjugate transpose of $\boldsymbol{\rho}$, e.g. $\boldsymbol{T} = \sqrt{\boldsymbol{\rho^\dagger}} \; \boldsymbol{\hat{T}} \sqrt{\boldsymbol{\rho}}$. This should not matter above threshold, where the phase space factor is real, but could have effects below threshold. This is where things become tricky: on the one hand, we need to ensure that $\boldsymbol{K}$ remains real (unitarity) and on the other, we need to take keep track of which imaginary square root we choose (**Riemann sheet**). The latter is often called the requirement of **analyticity**. This is currently being explored in {doc}`compwa-report:003/index` and {doc}`compwa-report:004/index`.
#
# With these definitions, we can deduce that:
#
# ```{margin}
# {cite}`Chung:1995dx` Eq. (51);
# compare with {eq}`T-in-terms-of-K`
# ```
#
# $$
# \boldsymbol{\hat{T}}
# = \boldsymbol{\hat{K}} (\boldsymbol{I} - i\boldsymbol{\rho}\boldsymbol{\hat{K}})^{-1}
# = (\boldsymbol{I} - i\boldsymbol{\hat{K}}\boldsymbol{\rho})^{-1} \boldsymbol{\hat{K}}.
# $$ (T-hat-in-terms-of-K-hat)

# %% [markdown]
# ### Production processes
#
# {ref}`As noted in the intro <usage/dynamics/k-matrix:Physics>`, the $\boldsymbol{K}$-matrix describes scattering processes of type $cd \to ab$. It can however be generalized to describe **production processes** of type $R \to ab$. Here, the amplitude is described by a **final state $F$-vector** of size $n$, so the question is how to express $F$ in terms of transition matrix $\boldsymbol{T}$.

# %% jupyter={"source_hidden": true} tags=["hide-input"]
dot = """
digraph {
    rankdir=LR;
    node [shape=point, width=0];
    edge [arrowhead=none];
    "a" [shape=none, label="aᵢ"];
    "b" [shape=none, label="bᵢ"];
    "R" [shape=none, label="R"];
    "N" [shape=none, label=""];
    "R" -> "N";
    "N" -> "a";
    "N" -> "b";
    { rank=same "Na", "Nb" };
}
"""
graphviz.Source(dot)

# %% [markdown] tags=["scroll-input"]
# One approach by {cite}`Aitchison:1972ay` is to transform $\boldsymbol{T}$ into $F$ (and its relativistic form $\hat{F}$) through the **production amplitude $P$-vector**:
#
# ```{margin}
# {cite}`Chung:1995dx` Eqs. (114) and (115)
# ```
#
# $$
# \begin{eqnarray}
# F & = & \left(\boldsymbol{I}-i\boldsymbol{K}\right)^{-1}P \\
# \hat{F} & = & \left(\boldsymbol{I}-i\boldsymbol{\hat{K}\boldsymbol{\rho}}\right)^{-1}\hat{P},
# \end{eqnarray}
# $$ (F-in-terms-of-P)
#
# where we can compute $\boldsymbol{\hat{K}}$ from {eq}`K-hat-and-T-hat`:
#
# $$
# \hat{\boldsymbol{K}} = \sqrt{\boldsymbol{\rho}^{-1}} \boldsymbol{K} \sqrt{\boldsymbol{\rho}^{-1}}.
# $$ (K-hat in terms of K)
#
# Another approach by {cite}`Cahn:1985wu` further approximates this by defining a **$Q$-vector**:
#
# ```{margin}
# {cite}`Chung:1995dx` Eq. (124)
# ```
#
# $$
# Q = \boldsymbol{K}^{-1}P \quad \mathrm{and} \quad
# \hat{Q} = \boldsymbol{\hat{K}}^{-1}\hat{P}
# $$ (Q-vector)
#
# that _is taken to be constant_ (just some 'fitting' parameters). The $F$-vector can then be expressed as:
#
# ```{margin}
# {cite}`Chung:1995dx` Eq. (125)
# ```
#
# $$
# F = \boldsymbol{T}Q
# \quad \mathrm{and} \quad
# \hat{F} = \boldsymbol{\hat{T}}\hat{Q}
# $$ (F in terms of Q)
#
# Note that for all these vectors, we have:
#
# ```{margin}
# {cite}`Chung:1995dx` Eqs. (116) and (124)
# ```
#
# $$
# F=\sqrt{\boldsymbol{\rho}}\hat{F},\quad
# P=\sqrt{\boldsymbol{\rho}}\hat{P},\quad\mathrm{and}\quad
# Q=\sqrt{\boldsymbol{\rho}^{-1}}\hat{Q}.
# $$ (invariant-vectors)

# %% [markdown]
# ### Pole parametrization

# %% [markdown]
# After all these matrix definitions, the final challenge is to choose a correct parametrization for the elements of $\boldsymbol{K}$ and $P$ that accurately describes the resonances we observe.[^pole-vs-resonance] There are several choices, but a common one is the following summation over the **poles** $R$:[^complex-conjugate-parametrization]
#
# [^complex-conjugate-parametrization]: Eqs. (51) and (52) in {cite}`chungPrimerKmatrixFormalism1995` take a complex conjugate of one of the residue functions and one of the phase space factors.
#
# ```{margin}
# {cite}`Chung:1995dx` Eqs. (73) and (74)
# ```
#
# $$
# \begin{eqnarray}
# K_{ij} &=& \sum_R\frac{g_{R,i}g_{R,j}}{m_R^2-s} + c_{ij} \\
# \hat{K}_{ij} &=& \sum_R \frac{g_{R,i}(s)g_{R,j}(s)}{\left(m_R^2-s\right)\sqrt{\rho_i\rho_j}} + \hat{c}_{ij}
# \end{eqnarray}
# $$ (K-matrix parametrization)
#
# with $c_{ij}, \hat{c}_{ij}$ some optional background characterization and $g_{R,i}$ the **residue functions**. The residue functions are often further expressed as:
#
# ```{margin}
# {cite}`Chung:1995dx` Eqs. (75-78)
# ```
#
# $$
# \begin{eqnarray}
# g_{R,i} &=& \gamma_{R,i}\sqrt{m_R\Gamma^0_{R,i}} \\
# g_{R,i}(s) &=& \gamma_{R,i}\sqrt{m_R\Gamma_{R,i}(s)}
# \end{eqnarray}
# $$ (residue-function)
#
# with $\gamma_{R,i}$ some _real_ constants and $\Gamma^0_{R,i}$ the **partial width** of each pole. In the Lorentz-invariant form, the fixed width $\Gamma^0$ is replaced by an "energy dependent" {class}`.EnergyDependentWidth` $\Gamma(s)$.[^phase-space-factor-normalization] The **width** for each pole can be computed as $\Gamma^0_R = \sum_i\Gamma^0_{R,i}$.
#
# [^phase-space-factor-normalization]: Unlike Eq. (77) in {cite}`Chung:1995dx`, AmpForm defines {class}`.EnergyDependentWidth` as in {pdg-review}`2021; Resonances; p.6`, Eq. (50.28). The difference is that the phase space factor denoted by $\rho_i$ in Eq. (77) in {cite}`Chung:1995dx` is divided by the phase space factor at the pole position $m_R$. So in AmpForm, the choice is $\rho_i \to \frac{\rho_i(s)}{\rho_i(m_R)}$.

# %% [markdown]
# The production vector $P$ is commonly parameterized as:[^damping-factor-P-parametrization]
#
# ```{margin}
# {cite}`Chung:1995dx` Eqs. (118-119) and (122)
# ```
#
# $$
# \begin{eqnarray}
# P_i &=& \sum_R \frac{\beta^0_R\,g_{R,i}(s)}{m_R^2-s} \\
# \hat{P}_i
# &=& \sum_R \frac{\beta^0_R\,g_{R,i}(s)}{\left(m_R^2-s\right)\sqrt{\rho_i}} \\
# &=& \sum_R \frac{
#     \beta^0_R\gamma_{R,i}m_R\Gamma^0_R B_{R,i}(q(s))
# }{m_R^2-s}
# \end{eqnarray}
# $$ (P-vector parametrization)
#
# with $B_{R,i}(q(s))$ the **centrifugal damping factor** (see {class}`.FormFactor` and {class}`.BlattWeisskopfSquared`) for channel $i$ and $\beta_R^0$ some (generally complex) constants that describe the production information of the decaying state $R$. Usually, these constants are rescaled just like the residue functions in {eq}`residue-function`:
#
# [^damping-factor-P-parametrization]: Just as with [^phase-space-factor-normalization], we have smuggled a bit in the last equation in order to be able to reproduce Equation (50.23) in {pdg-review}`2021; Resonances; p.9` in the case $n=1,n_R=1$, on which {func}`.relativistic_breit_wigner_with_ff` is based.
#
# ```{margin}
# {cite}`Chung:1995dx` Eq. (121)
# ```
#
# $$
# \beta^0_R = \beta_R\sqrt{m_R\Gamma^0_R}.
# $$ (beta functions)

# %% [markdown]
# ## Implementation

# %% [markdown]
# ### Non-relativistic K-matrix

# %% [markdown]
# A non-relativistic $\boldsymbol{K}$-matrix for an arbitrary number of channels and an arbitrary number of poles can be formulated with the {meth}`.NonRelativisticKMatrix.formulate` method:

# %%
n_poles = sp.Symbol("n_R", integer=True, positive=True)
k_matrix_nr = kmatrix.NonRelativisticKMatrix.formulate(n_poles=n_poles, n_channels=1)
k_matrix_nr[0, 0]

# %% [markdown]
# Notice how the $\boldsymbol{K}$-matrix reduces to a {func}`.relativistic_breit_wigner` in the case of one channel and one pole (but for a residue constant $\gamma$):

# %%
k_matrix_1r = kmatrix.NonRelativisticKMatrix.formulate(n_poles=1, n_channels=1)
k_matrix_1r[0, 0].doit().simplify()

# %% [markdown]
# Now let's investigate the effect of using a $\boldsymbol{K}$-matrix to describe **two poles** in one channel and see how it compares with the sum of two Breit-Wigner functions (two 'resonances'). Two Breit-Wigner 'poles' with the same parameters would look like this:

# %%
s, m1, m2, Gamma1, Gamma2 = sp.symbols("s m1 m2 Gamma1 Gamma2", nonnegative=True)
bw1 = relativistic_breit_wigner(s, m1, Gamma1)
bw2 = relativistic_breit_wigner(s, m2, Gamma2)
bw = bw1 + bw2
bw

# %% [markdown]
# while a $\boldsymbol{K}$-matrix parametrizes the two poles as:

# %%
k_matrix_2r = kmatrix.NonRelativisticKMatrix.formulate(n_poles=2, n_channels=1)
k_matrix = k_matrix_2r[0, 0].doit()

# %% jupyter={"source_hidden": true} tags=["hide-input"]
# reformulate terms
*rest, denominator, nominator = k_matrix.args
term1 = nominator.args[0] * denominator * sp.Mul(*rest)
term2 = nominator.args[1] * denominator * sp.Mul(*rest)
k_matrix = term1 + term2
k_matrix


# %% [markdown]
# To simplify things, we can set the residue constants $\gamma$ to one. Notice how the $\boldsymbol{K}$-matrix has introduced some coupling ('interference') between the two terms.

# %% jupyter={"source_hidden": true} tags=["hide-input"]
def remove_residue_constants(expression):
    expression = symplot.substitute_indexed_symbols(expression)
    residue_constants = filter(
        lambda s: re.match(r"^\\?gamma", s.name),
        expression.free_symbols,
    )
    return expression.xreplace(dict.fromkeys(residue_constants, 1))


display(
    remove_residue_constants(bw),
    remove_residue_constants(k_matrix),
)

# %% [markdown]
# Now, just like in {doc}`/usage/interactive`, we use {mod}`symplot` to visualize the difference between the two expressions. The important thing is that the Argand plot on the right shows that **the $\boldsymbol{K}$-matrix conserves unitarity**.
#
# Note that we have to call {func}`symplot.substitute_indexed_symbols` to turn the {class}`~sympy.tensor.indexed.Indexed` instances in this {obj}`~sympy.matrices.dense.Matrix` expression into {class}`~sympy.core.symbol.Symbol`s before calling this function. We also call {func}`symplot.rename_symbols` so that the residue $\gamma$'s get a name that does not have to be dummified by {func}`~sympy.utilities.lambdify.lambdify`.

# %% jupyter={"source_hidden": true} tags=["hide-input", "remove-output", "scroll-input"]
# Prepare expressions
m = sp.Symbol("m", nonnegative=True)
k_matrix = symplot.substitute_indexed_symbols(k_matrix)
rename_gammas = lambda s: re.sub(  # noqa: E731
    r"\\([Gg])amma_{([0-9]),0}", r"\1amma\2", s
)
gamma1, gamma2 = sp.symbols("gamma1 gamma2", nonnegative=True)
bw = symplot.rename_symbols(bw, rename_gammas)
k_matrix = symplot.rename_symbols(k_matrix, rename_gammas)
bw = bw.xreplace({s: m**2})
k_matrix = k_matrix.xreplace({s: m**2})

# Prepare sliders and domain
np_kmatrix, sliders = symplot.prepare_sliders(k_matrix, m)
np_bw = sp.lambdify((m, Gamma1, Gamma2, gamma1, gamma2, m1, m2), bw.doit())

m_min, m_max = 0, 3
domain_1d = np.linspace(m_min, m_max, 200)
domain_argand = np.linspace(m_min - 2, m_max + 2, 1_000)
sliders.set_ranges(
    m1=(0, 3, 100),
    m2=(0, 3, 100),
    Gamma1=(0, 2, 100),
    Gamma2=(0, 2, 100),
    gamma1=(0, 2),
    gamma2=(0, 2),
)
sliders.set_values(
    m1=1.1,
    m2=1.9,
    Gamma1=0.2,
    Gamma2=0.3,
    gamma1=1,
    gamma2=1,
)
if STATIC_WEB_PAGE:
    # Concatenate flipped domain for reverse animation
    domain = np.linspace(1.0, 2.7, 50)
    domain = np.concatenate((domain, np.flip(domain[1:])))
    sliders._sliders["m1"] = domain


def create_argand(func):
    def wrapped(**kwargs):
        values = func(domain_argand, **kwargs)
        argand = np.array([values.real, values.imag])
        return argand.T

    return wrapped


# Create figure
fig, axes = plt.subplots(
    ncols=2,
    figsize=1.2 * np.array((8, 3.8)),
    tight_layout=True,
)
ax_intensity, ax_argand = axes
m_label = "$m_{a+b}$"
ax_intensity.set_xlabel(m_label)
ax_intensity.set_ylabel("$|A|^2$")
ax_argand.set_xlabel("Re($A$)")
ax_argand.set_ylabel("Im($A$)")

# Plot intensity
controls = iplt.plot(
    domain_1d,
    lambda *args, **kwargs: np.abs(np_kmatrix(*args, **kwargs) ** 2),
    label="$K$-matrix",
    **sliders,
    ylim="auto",
    ax=ax_intensity,
)
iplt.plot(
    domain_1d,
    lambda *args, **kwargs: np.abs(np_bw(*args, **kwargs) ** 2),
    label="Breit-Wigner",
    controls=controls,
    ylim="auto",
    ax=ax_intensity,
)
plt.legend(loc="upper right")
iplt.axvline(controls["m1"], c="gray", linestyle="dotted")
iplt.axvline(controls["m2"], c="gray", linestyle="dotted")

# Argand plots
iplt.scatter(
    create_argand(np_kmatrix),
    label="$K$-matrix",
    controls=controls,
    parametric=True,
    s=1,
    ax=ax_argand,
)
iplt.scatter(
    create_argand(np_bw),
    label="Breit-Wigner",
    controls=controls,
    parametric=True,
    s=1,
    ax=ax_argand,
)
plt.legend(loc="upper right");

# %% [markdown]
# {{ run_interactive }}

# %% jupyter={"source_hidden": true} tags=["remove-input"]
if STATIC_WEB_PAGE:
    output_path = "non-relativistic-k-matrix.gif"
    ax_intensity.set_ylim([0, 2])
    ax_argand.set_xlim([-1, +1])
    ax_argand.set_ylim([0, 2])
    controls.save_animation(output_path, fig, "m1", fps=20)
    with open(output_path, "rb") as f:
        display(Image(data=f.read(), format="png"))

# %% [markdown]
# ### Relativistic K-matrix

# %% [markdown]
# Relativistic $\boldsymbol{K}$-matrices for an arbitrary number of channels and an arbitrary number of poles can be formulated with the {meth}`.RelativisticKMatrix.formulate` method:

# %%
L = sp.Symbol("L", integer=True, negative=False)
n_poles = sp.Symbol("n_R", integer=True, positive=True)
rel_k_matrix_nr = kmatrix.RelativisticKMatrix.formulate(
    n_poles=n_poles, n_channels=1, angular_momentum=L
)
rel_k_matrix_nr[0, 0]

# %% [markdown]
# Again, as in {ref}`usage/dynamics/k-matrix:Non-relativistic K-matrix`, the $\boldsymbol{K}$-matrix reduces to something of a {func}`.relativistic_breit_wigner`. This time, the width has been replaced by a {class}`.EnergyDependentWidth` and some {class}`.PhaseSpaceFactor`s have been inserted that take care of the decay into two decay products:

# %%
rel_k_matrix_1r = kmatrix.RelativisticKMatrix.formulate(
    n_poles=1, n_channels=1, angular_momentum=L
)
symplot.partial_doit(rel_k_matrix_1r[0, 0], sp.Sum).simplify(doit=False)

# %% [markdown]
# Note that another difference with {func}`.relativistic_breit_wigner_with_ff` is an additional phase space factor in the denominator. That one disappears in {ref}`usage/dynamics/k-matrix:P-vector`.
#
# The $\boldsymbol{K}$-matrix with two poles becomes (neglecting the $\sqrt{\rho_0(s)}$):

# %%
rel_k_matrix_2r = kmatrix.RelativisticKMatrix.formulate(
    n_poles=2, n_channels=1, angular_momentum=L
)
rel_k_matrix_2r = symplot.partial_doit(rel_k_matrix_2r[0, 0], sp.Sum)

# %% jupyter={"source_hidden": true} tags=["hide-input", "full-width"]
rel_k_matrix_2r = symplot.substitute_indexed_symbols(rel_k_matrix_2r)
s, m_a, m_b = sp.symbols("s, m_a0, m_b0", nonnegative=True)
rho = PhaseSpaceFactor(s, m_a, m_b)
rel_k_matrix_2r = rel_k_matrix_2r.xreplace({
    sp.sqrt(rho): 1,
    sp.conjugate(sp.sqrt(rho)): 1,
})
*rest, denominator, nominator = rel_k_matrix_2r.args
term1 = nominator.args[0] * denominator * sp.Mul(*rest)
term2 = nominator.args[1] * denominator * sp.Mul(*rest)
rel_k_matrix_2r = term1 + term2
rel_k_matrix_2r

# %% [markdown]
# This again shows the interference introduced by the $\boldsymbol{K}$-matrix, when compared with a sum of two Breit-Wigner functions.

# %% [markdown]
# ### P-vector

# %% [markdown]
# For one channel and an arbitrary number of poles $n_R$, the $F$-vector gets the following form:

# %%
n_poles = sp.Symbol("n_R", integer=True, positive=True)
kmatrix.NonRelativisticPVector.formulate(n_poles=n_poles, n_channels=1)[0]

# %% [markdown]
# The {class}`.RelativisticPVector` looks like:

# %%
kmatrix.RelativisticPVector.formulate(
    n_poles=n_poles, n_channels=1, angular_momentum=L
)[0]

# %% [markdown]
# As in {ref}`usage/dynamics/k-matrix:Non-relativistic K-matrix`, if we take $n_R=1$, the $F$-vector reduces to a Breit-Wigner function, but now with an additional factor $\beta$.

# %%
f_vector_1r = kmatrix.NonRelativisticPVector.formulate(n_poles=1, n_channels=1)
symplot.partial_doit(f_vector_1r[0], sp.Sum)

# %% [markdown]
# And when we neglect the phase space factors $\sqrt{\rho_0(s)}$, the {class}`.RelativisticPVector` reduces to a {func}`.relativistic_breit_wigner_with_ff`!

# %%
rel_f_vector_1r = kmatrix.RelativisticPVector.formulate(
    n_poles=1, n_channels=1, angular_momentum=L
)
rel_f_vector_1r = symplot.partial_doit(rel_f_vector_1r[0], sp.Sum)

# %% jupyter={"source_hidden": true} tags=["hide-input", "full-width"]
rel_f_vector_1r = symplot.substitute_indexed_symbols(rel_f_vector_1r)
s, m_a, m_b = sp.symbols("s, m_a0, m_b0", nonnegative=True)
rho = PhaseSpaceFactor(s, m_a, m_b)
rel_f_vector_1r.xreplace({
    sp.sqrt(rho): 1,
    sp.conjugate(sp.sqrt(rho)): 1,
}).simplify(doit=False)

# %% [markdown]
# Note that the $F$-vector approach introduces additional $\beta$-coefficients. These can constants can be complex and can introduce phase differences form the production process.

# %%
f_vector_2r = kmatrix.NonRelativisticPVector.formulate(n_poles=2, n_channels=1)
f_vector = f_vector_2r[0].doit()

# %% jupyter={"source_hidden": true} tags=["hide-input"]
*rest, denominator, nominator = f_vector.args
term1 = nominator.args[0] * denominator * sp.Mul(*rest)
term2 = nominator.args[1] * denominator * sp.Mul(*rest)
f_vector = term1 + term2
f_vector

# %% [markdown]
# Now again let's compare the compare this with a sum of two {func}`.relativistic_breit_wigner`s, now with the two additional $\beta$-constants.

# %% jupyter={"source_hidden": true}
beta1, beta2 = sp.symbols("beta1 beta2", nonnegative=True)
bw_with_phases = beta1 * bw1 + beta2 * bw2
display(
    bw_with_phases,
    remove_residue_constants(f_vector),
)

# %% [markdown]
# {{ run_interactive }}

# %% jupyter={"source_hidden": true} tags=["hide-input", "remove-output", "scroll-input"]
# Prepare expressions
f_vector = symplot.substitute_indexed_symbols(f_vector)
rename_gammas = lambda s: re.sub(  # noqa: E731
    r"\\([Gg])amma_{([0-9]),0}", r"\1amma\2", s
)
c1, c2, phi1, phi2 = sp.symbols("c1 c2 phi1 phi2", real=True)
bw_with_phases = symplot.rename_symbols(bw_with_phases, rename_gammas)
f_vector = symplot.rename_symbols(f_vector, rename_gammas)
substitutions = {
    s: m**2,
    beta1: c1 * sp.exp(sp.I * phi1),
    beta2: c2 * sp.exp(sp.I * phi2),
}
bw_with_phases = bw_with_phases.xreplace(substitutions)
f_vector = f_vector.xreplace(substitutions)

# Prepare sliders and domain
np_f_vector, sliders = symplot.prepare_sliders(f_vector, m)
np_bw = sp.lambdify(
    (m, Gamma1, Gamma2, c1, c2, gamma1, gamma2, m1, m2, phi1, phi2),
    bw_with_phases.doit(),
)

# Set plot domain
x_min, x_max = 0, 3
y_min, y_max = -0.5, +0.5
plot_domain = np.linspace(x_min, x_max, num=150)
plot_domain_argand = np.linspace(x_min - 2, x_max + 2, num=400)
x_values = np.linspace(x_min, x_max, num=160)
y_values = np.linspace(y_min, y_max, num=80)
X, Y = np.meshgrid(x_values, y_values)
plot_domain_complex = X + Y * 1j

# Set slider values and ranges
sliders.set_ranges(
    c1=(0, 2, 100),
    c2=(0, 2, 100),
    phi1=(0, np.pi, np.pi / 20),
    phi2=(0, np.pi, np.pi / 20),
    m1=(0, 3, 300),
    m2=(0, 3, 300),
    Gamma1=(-2, 2, 100),
    Gamma2=(-2, 2, 100),
    gamma1=(-1, +1, 100),
    gamma2=(-1, +1, 100),
)
sliders.set_values(
    c1=1,
    c2=1,
    m1=1.4,
    m2=1.7,
    Gamma1=0.2,
    Gamma2=0.3,
    gamma1=1 / np.sqrt(2),
    gamma2=1 / np.sqrt(2),
)


def create_argand(func):
    def wrapped(**kwargs):
        values = func(plot_domain_argand, **kwargs)
        argand = np.array([values.real, values.imag])
        return argand.T

    return wrapped


# Create figure
fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(10, 9),
    gridspec_kw=dict(
        width_ratios=[2.5, 1],
        wspace=0.08,
        left=0.05,
        right=0.99,
        top=0.99,
        bottom=0.05,
    ),
)
fig.canvas.toolbar_visible = False
fig.canvas.header_visible = False
fig.canvas.footer_visible = False
(ax_1d, ax_argand), (ax_2d, ax_empty1), (ax_2d_bw, ax_empty2) = axes
ax_empty1.axis("off")
ax_empty2.axis("off")
for ax in axes.flatten():
    ax.set_yticks([])
ax_argand.set_xticks([])
ax_1d.axes.get_xaxis().set_visible(False)
ax_2d.axes.get_xaxis().set_visible(False)
ax_1d.sharex(ax_2d_bw)
ax_2d.sharex(ax_2d_bw)
ax_2d_bw.set_xlabel("Re $m$")
for ax in (ax_2d, ax_2d_bw):
    ax.set_ylabel("Im $m$")
ax_1d.set_ylabel("$|A|^2$")
ax_argand.set_xlabel("Re($A$)")
ax_argand.set_ylabel("Im($A$)")

for ax in (ax_2d, ax_2d_bw):
    ax.axhline(0, linewidth=0.5, c="black", linestyle="dotted")

# 1D intensity plot
controls = Controls(**sliders)
controls = iplt.plot(
    plot_domain,
    lambda *args, **kwargs: np.abs(np_f_vector(*args, **kwargs) ** 2),
    label="$P$-vector",
    controls=controls,
    ylim="auto",
    ax=ax_1d,
)
iplt.plot(
    plot_domain,
    lambda *args, **kwargs: np.abs(np_bw(*args, **kwargs) ** 2),
    label="Breit-Wigner",
    controls=controls,
    ylim="auto",
    ax=ax_1d,
)
ax_1d.legend(loc="upper right")
mass_line_style = dict(
    c="red",
    alpha=0.3,
)
for name in controls.params:
    if not re.match(r"^m[0-9]+$", name):
        continue
    iplt.axvline(controls[name], ax=ax_1d, **mass_line_style)

# Argand plot
iplt.scatter(
    create_argand(np_f_vector),
    label="$P$-vector",
    controls=controls,
    parametric=True,
    s=1,
    ax=ax_argand,
    xlim="auto",
    ylim="auto",
)
iplt.scatter(
    create_argand(np_bw),
    label="Breit-Wigner",
    controls=controls,
    parametric=True,
    s=1,
    ax=ax_argand,
    xlim="auto",
    ylim="auto",
)

# 3D plot
color_mesh = None
color_mesh_bw = None
pole_indicators = []
pole_indicators_bw = []


def plot3(*, z_cutoff, complex_rendering, **kwargs):
    global color_mesh, color_mesh_bw
    Z = np_f_vector(plot_domain_complex, **kwargs)
    Z_bw = np_bw(plot_domain_complex, **kwargs)
    if complex_rendering == "imag":
        projection = np.imag
        ax_title = "Im $A$"
    elif complex_rendering == "real":
        projection = np.real
        ax_title = "Re $A$"
    elif complex_rendering == "abs":
        projection = np.vectorize(lambda z: np.abs(z) ** 2)
        ax_title = "$|A|$"
    else:
        raise NotImplementedError
    ax_2d.set_title(ax_title + " ($P$-vector)")
    ax_2d_bw.set_title(ax_title + " (Breit-Wigner)")

    if color_mesh is None:
        color_mesh = ax_2d.pcolormesh(X, Y, projection(Z), cmap=cm.coolwarm)
    else:
        color_mesh.set_array(projection(Z))
    color_mesh.set_clim(vmin=-z_cutoff, vmax=+z_cutoff)

    if color_mesh_bw is None:
        color_mesh_bw = ax_2d_bw.pcolormesh(X, Y, projection(Z_bw), cmap=cm.coolwarm)
    else:
        color_mesh_bw.set_array(projection(Z_bw))
    color_mesh_bw.set_clim(vmin=-z_cutoff, vmax=+z_cutoff)

    if pole_indicators:
        for R, (line, text) in enumerate(pole_indicators, 1):
            mass = kwargs[f"m{R}"]
            line.set_xdata(mass)
            text.set_x(mass + (x_max - x_min) * 0.008)
    else:
        for R in range(1, 2 + 1):
            mass = kwargs[f"m{R}"]
            line = ax_2d.axvline(mass, **mass_line_style)
            text = ax_2d.text(
                x=mass + (x_max - x_min) * 0.008,
                y=0.95 * y_min,
                s=f"$m_{R}$",
                c="red",
            )
            pole_indicators.append((line, text))

    if pole_indicators_bw:
        for R, (line, text) in enumerate(pole_indicators_bw, 1):
            mass = kwargs[f"m{R}"]
            line.set_xdata(mass)
            text.set_x(mass + (x_max - x_min) * 0.008)
    else:
        for R in range(1, 2 + 1):
            mass = kwargs[f"m{R}"]
            line = ax_2d_bw.axvline(mass, **mass_line_style)
            text = ax_2d_bw.text(
                x=mass + (x_max - x_min) * 0.008,
                y=0.95 * y_min,
                s=f"$m_{R}$",
                c="red",
            )
            pole_indicators_bw.append((line, text))


# Create switch for imag/real/abs
name = "complex_rendering"
sliders._sliders[name] = ipywidgets.RadioButtons(
    options=["imag", "real", "abs"],
    description=R"\(s\)-plane plot",
)
sliders._arg_to_symbol[name] = name

# Create cut-off slider for z-direction
name = "z_cutoff"
sliders._sliders[name] = ipywidgets.FloatSlider(
    value=2,
    min=0,
    max=+5,
    step=0.1,
    description=R"\(z\)-cutoff",
)
sliders._arg_to_symbol[name] = name

# Create GUI
sliders_copy = dict(sliders)
slider_groups = []
for R in range(1, 2 + 1):
    vertical_slider_group = [
        sliders_copy.pop(f"c{R}"),
        sliders_copy.pop(f"phi{R}"),
        sliders_copy.pop(f"m{R}"),
        sliders_copy.pop(f"Gamma{R}"),
        sliders_copy.pop(f"gamma{R}"),
    ]
    slider_groups.append(vertical_slider_group)
slider_pairs = np.array(slider_groups).T
h_boxes = [ipywidgets.HBox(tuple(pair)) for pair in slider_pairs]
remaining_sliders = sorted(
    sliders_copy.values(),
    key=lambda s: (str(type(s)), s.description),
)
ui = ipywidgets.VBox(h_boxes + remaining_sliders)
output = ipywidgets.interactive_output(plot3, controls=sliders)
display(ui, output)

# %% jupyter={"source_hidden": true} tags=["remove-input", "full-width"]
if STATIC_WEB_PAGE:
    output_path = "p-vector-comparison.png"
    plt.savefig(output_path, dpi=150)
    display(Image(output_path))

# %% [markdown]
# ## Interactive visualization

# %% [markdown]
# All $\boldsymbol{K}$-matrices can be inspected interactively for arbitrary poles and channels with the following applet:

# %% [markdown]
# [^pole-vs-resonance]: See {pdg-review}`2021; Resonances`, Section 50.1, for a discussion about what poles and resonances are. See also the intro to Section 5 in {cite}`Chung:1995dx`.

# %% jupyter={"source_hidden": true} tags=["hide-cell", "scroll-input"]
if STATIC_WEB_PAGE:
    L = 0


def plot(
    kmatrix_type: kmatrix.TMatrix,
    n_channels: int,
    n_poles: int,
    angular_momentum=0,
    phsp_factor=PhaseSpaceFactor,
    substitute_sqrt_rho: bool = False,
) -> None:
    # Convert to Symbol: symplot cannot handle IndexedBase
    i, j = sp.symbols("i, j", integer=True, negative=False)
    j = i
    expr = kmatrix_type.formulate(
        n_poles=n_poles,
        n_channels=n_channels,
        angular_momentum=angular_momentum,
        phsp_factor=phsp_factor,
    ).doit()[i, j]
    expr = symplot.substitute_indexed_symbols(expr)
    if substitute_sqrt_rho:

        def rho_i(i):
            return phsp_factor(*sp.symbols(f"s m_a{i} m_b{i}", nonnegative=True)).doit()

        expr = expr.xreplace({
            sp.sqrt(rho_i(i)): 1 for i in range(n_channels)
        }).xreplace({sp.conjugate(sp.sqrt(rho_i(i))): 1 for i in range(n_channels)})
    expr = expr.xreplace({s: m**2})
    expr = symplot.substitute_indexed_symbols(expr)
    np_expr, sliders = symplot.prepare_sliders(expr, m)

    # Set plot domain
    x_min, x_max = 1e-3, 3
    y_min, y_max = -0.5, +0.5

    plot_domain = np.linspace(x_min, x_max, num=500, dtype=np.complex128)
    x_values = np.linspace(x_min, x_max, num=160)
    y_values = np.linspace(y_min, y_max, num=80)
    X, Y = np.meshgrid(x_values, y_values)
    plot_domain_complex = X + Y * 1j

    # Set slider values and ranges
    m0_values = np.linspace(x_min, x_max, num=n_poles + 2)
    m0_values = m0_values[1:-1]
    if "L" in sliders:
        sliders.set_ranges(L=(0, 10))
    sliders.set_ranges(i=(0, n_channels - 1))
    for R in range(1, n_poles + 1):
        for i in range(n_channels):
            if kmatrix_type in {
                kmatrix.RelativisticKMatrix,
                kmatrix.RelativisticPVector,
            }:
                sliders.set_ranges({
                    f"m{R}": (0, 3, 100),
                    Rf"\Gamma_{{{R},{i}}}": (-1.5, +1.5, 100),
                    Rf"\gamma_{{{R},{i}}}": (-1, 1, 100),
                    f"m_a{i}": (0, 1, 0.01),
                    f"m_b{i}": (0, 1, 0.01),
                })
                sliders.set_values({
                    f"m{R}": m0_values[R - 1],
                    Rf"\Gamma_{{{R},{i}}}": 0.35 + R * 0.2 - i * 0.3,
                    Rf"\gamma_{{{R},{i}}}": 1 / np.sqrt(n_channels * n_poles),
                    f"m_a{i}": (i + 1) * 0.25,
                    f"m_b{i}": (i + 1) * 0.25,
                })
            if kmatrix_type in {
                kmatrix.NonRelativisticPVector,
                kmatrix.NonRelativisticKMatrix,
            }:
                sliders.set_ranges({
                    f"m{R}": (0, 3, 100),
                    Rf"\Gamma_{{{R},{i}}}": (-1, 1, 100),
                    Rf"\gamma_{{{R},{i}}}": (-1, 1, 100),
                })
                sliders.set_values({
                    f"m{R}": m0_values[R - 1],
                    Rf"\Gamma_{{{R},{i}}}": (R + 1) * 0.1,
                    Rf"\gamma_{{{R},{i}}}": 1 / np.sqrt(n_channels * n_poles),
                })
            if kmatrix_type in {
                kmatrix.NonRelativisticPVector,
                kmatrix.RelativisticPVector,
            }:
                sliders.set_ranges({f"beta{R}": (-1, 1, 100)})
                sliders.set_values({f"beta{R}": 1})

    # Create interactive plots
    fig, axes = plt.subplots(
        nrows=2,
        figsize=(8, 6),
        sharex=True,
        tight_layout=True,
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    if kmatrix_type in {
        kmatrix.NonRelativisticKMatrix,
        kmatrix.RelativisticKMatrix,
    }:
        fig.suptitle(
            Rf"${n_channels} \times {n_channels}$ $K$-matrix"
            f" with {n_poles} resonances"
        )
    elif kmatrix_type in {
        kmatrix.NonRelativisticPVector,
        kmatrix.RelativisticPVector,
    }:
        fig.suptitle(f"$P$-vector for {n_channels} channels and {n_poles} resonances")

    for ax in axes:
        ax.set_xlim(x_min, x_max)
    ax_2d, ax_3d = axes
    ax_2d.set_ylabel("$|T|^{2}$")
    ax_2d.set_yticks([])
    ax_3d.set_xlabel("Re $m$")
    ax_3d.set_ylabel("Im $m$")

    ax_3d.axhline(0, linewidth=0.5, c="black", linestyle="dotted")

    # 2D plot
    def plot(channel: int):
        def wrapped(*args, **kwargs) -> sp.Expr:
            kwargs["i"] = channel
            return np.abs(np_expr(*args, **kwargs)) ** 2

        return wrapped

    controls = Controls(**sliders)
    for i in range(n_channels):
        iplt.plot(
            plot_domain,
            plot(i),
            ax=axes[0],
            controls=controls,
            ylim="auto",
            label=f"channel {i}",
        )
    if n_channels > 1:
        axes[0].legend(loc="upper right")
    mass_line_style = dict(
        c="red",
        alpha=0.3,
    )
    for name in controls.params:
        if not re.match(r"^m[0-9]+$", name):
            continue
        iplt.axvline(controls[name], ax=axes[0], **mass_line_style)

    # 3D plot
    color_mesh = None
    pole_indicators = []
    threshold_indicators = []

    def plot3(*, z_cutoff, complex_rendering, **kwargs):
        nonlocal color_mesh
        Z = np_expr(plot_domain_complex, **kwargs)
        if complex_rendering == "imag":
            Z_values = Z.imag
            ax_title = "Im $T$"
        elif complex_rendering == "real":
            Z_values = Z.real
            ax_title = "Re $T$"
        elif complex_rendering == "abs":
            Z_values = np.abs(Z)
            ax_title = "$|T|$"
        else:
            raise NotImplementedError

        if n_channels == 1:
            axes[-1].set_title(ax_title)
        else:
            i = kwargs["i"]
            axes[-1].set_title(f"{ax_title}, channel {i}")

        if color_mesh is None:
            color_mesh = ax_3d.pcolormesh(X, Y, Z_values, cmap=cm.coolwarm)
        else:
            color_mesh.set_array(Z_values)
        color_mesh.set_clim(vmin=-z_cutoff, vmax=+z_cutoff)

        if pole_indicators:
            for R, (line, text) in enumerate(pole_indicators, 1):
                mass = kwargs[f"m{R}"]
                line.set_xdata(mass)
                text.set_x(mass + (x_max - x_min) * 0.008)
        else:
            for R in range(1, n_poles + 1):
                mass = kwargs[f"m{R}"]
                line = ax_3d.axvline(mass, **mass_line_style)
                text = ax_3d.text(
                    x=mass + (x_max - x_min) * 0.008,
                    y=0.95 * y_min,
                    s=f"$m_{R}$",
                    c="red",
                )
                pole_indicators.append((line, text))

        if kmatrix_type is kmatrix.RelativisticKMatrix:
            x_offset = (x_max - x_min) * 0.015
            if threshold_indicators:
                for i, (line_thr, line_diff, text_thr, text_diff) in enumerate(
                    threshold_indicators
                ):
                    m_a = kwargs[f"m_a{i}"]
                    m_b = kwargs[f"m_b{i}"]
                    s_thr = m_a + m_b
                    m_diff = abs(m_a - m_b)
                    line_thr.set_xdata(s_thr)
                    line_diff.set_xdata(m_diff)
                    text_thr.set_x(s_thr)
                    text_diff.set_x(m_diff - x_offset)
            else:
                colors = cm.plasma(np.linspace(0, 1, n_channels))
                for i, color in enumerate(colors):
                    m_a = kwargs[f"m_a{i}"]
                    m_b = kwargs[f"m_b{i}"]
                    s_thr = m_a + m_b
                    m_diff = abs(m_a - m_b)
                    line_thr = ax.axvline(s_thr, c=color, linestyle="dotted")
                    line_diff = ax.axvline(m_diff, c=color, linestyle="dashed")
                    text_thr = ax.text(
                        x=s_thr,
                        y=0.95 * y_min,
                        s=f"$m_{{a{i}}}+m_{{b{i}}}$",
                        c=color,
                        rotation=-90,
                    )
                    text_diff = ax.text(
                        x=m_diff - x_offset,
                        y=0.95 * y_min,
                        s=f"$m_{{a{i}}}-m_{{b{i}}}$",
                        c=color,
                        rotation=+90,
                    )
                    threshold_indicators.append((
                        line_thr,
                        line_diff,
                        text_thr,
                        text_diff,
                    ))
            for i, (_, line_diff, _, text_diff) in enumerate(threshold_indicators):
                m_a = kwargs[f"m_a{i}"]
                m_b = kwargs[f"m_b{i}"]
                s_thr = m_a + m_b
                m_diff = abs(m_a - m_b)
                if m_diff > x_offset + 0.01 and s_thr - abs(m_diff) > x_offset:
                    line_diff.set_alpha(0.5)
                    text_diff.set_alpha(0.5)
                else:
                    line_diff.set_alpha(0)
                    text_diff.set_alpha(0)

    # Create switch for imag/real/abs
    name = "complex_rendering"
    sliders._sliders[name] = ipywidgets.RadioButtons(
        options=["imag", "real", "abs"],
        description=R"\(s\)-plane plot",
    )
    sliders._arg_to_symbol[name] = name

    # Create cut-off slider for z-direction
    name = "z_cutoff"
    sliders._sliders[name] = ipywidgets.FloatSlider(
        value=1,
        min=+0.01,
        max=+5,
        step=0.01,
        description=R"\(z\)-cutoff",
    )
    sliders._arg_to_symbol[name] = name

    # Link sliders
    if kmatrix_type is kmatrix.RelativisticKMatrix:
        for i in range(n_channels):
            ipywidgets.dlink(
                (sliders[f"m_a{i}"], "value"),
                (sliders[f"m_b{i}"], "value"),
            )

    # Create GUI
    sliders_copy = dict(sliders)
    h_boxes = []
    for R in range(1, n_poles + 1):
        buttons = [sliders_copy.pop(f"m{R}")]
        if n_channels == 1:
            buttons += [
                sliders_copy.pop(sliders.symbol_to_arg[Rf"\Gamma_{{{R},0}}"]),
                sliders_copy.pop(sliders.symbol_to_arg[Rf"\gamma_{{{R},0}}"]),
            ]
        h_box = ipywidgets.HBox(buttons)
        h_boxes.append(h_box)
    remaining_sliders = sorted(
        sliders_copy.values(), key=lambda s: (str(type(s)), s.description)
    )
    if n_channels == 1:
        remaining_sliders.remove(sliders["i"])
    ui = ipywidgets.VBox(h_boxes + remaining_sliders)
    output = ipywidgets.interactive_output(plot3, controls=sliders)
    display(ui, output)


# %% [markdown]
# {{ run_interactive }}

# %% tags=["remove-output"]
plot(
    kmatrix.RelativisticKMatrix,
    n_poles=2,
    n_channels=1,
    angular_momentum=L,
    phsp_factor=PhaseSpaceFactor,
    substitute_sqrt_rho=False,
)

# %% jupyter={"source_hidden": true} tags=["full-width", "remove-input"]
if STATIC_WEB_PAGE:
    output_path = "k-matrix.png"
    plt.savefig(output_path, dpi=150)
    display(Image(output_path))
