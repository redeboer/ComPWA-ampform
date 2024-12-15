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
# # SymPy helper functions

# %% [markdown]
# The {mod}`ampform.sympy` module contains a few classes that make it easier to construct larger expressions that consist of several mathematical definitions.

# %% [markdown]
# ## Unevaluated expressions

# %% [markdown]
# The {func}`.unevaluated` decorator makes it easier to write classes that represent a mathematical function definition. It makes a class that derives from {class}`sp.Expr <sympy.core.expr.Expr>` behave more like a {func}`~.dataclasses.dataclass` (see [PEP&nbsp;861](https://peps.python.org/pep-0681)). All you have to do is:
#
# 1. Specify the arguments the function requires.
# 2. Specify how to render the 'unevaluated' or 'folded' form of the expression with a `_latex_repr_` string or method.
# 3. Specify how to unfold the expression using an `evaluate()` method.
#
# In the example below, we define a phase space factor $\rho^\text{CM}$ using the Chew-Mandelstam function (see PDG Resonances section, [Eq.&nbsp;(50.44)](https://pdg.lbl.gov/2023/reviews/rpp2023-rev-resonances.pdf#page=15)). For this, you need to define a break-up momentum $q$ as well.

# %%
import sympy as sp

from ampform.sympy import unevaluated


@unevaluated(real=False)
class BreakupMomentum(sp.Expr):
    s: sp.Symbol
    m1: sp.Symbol
    m2: sp.Symbol
    _latex_repr_ = R"q\left({s}\right)"  # not an f-string!

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        return sp.sqrt((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2) / (s * 4))


# %%
from sympy.printing.latex import LatexPrinter


@unevaluated(real=False)
class PhspFactorSWave(sp.Expr):
    s: sp.Symbol
    m1: sp.Symbol
    m2: sp.Symbol

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        q = BreakupMomentum(s, m1, m2)
        cm = (
            (2 * q / sp.sqrt(s))
            * sp.log((m1**2 + m2**2 - s + 2 * sp.sqrt(s) * q) / (2 * m1 * m2))
            - (m1**2 - m2**2) * (1 / s - 1 / (m1 + m2) ** 2) * sp.log(m1 / m2)
        ) / (16 * sp.pi**2)
        return 16 * sp.pi * sp.I * cm

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.s)
        s, *_ = map(printer._print, self.args)  # or via args
        return Rf"\rho^\text{{CM}}\left({s}\right)"  # f-string here!


# %% [markdown]
# :::{note}
# For illustrative purposes, the phase space factor defines `_latex_repr_()` [as a printer method](https://docs.sympy.org/latest/modules/printing.html#example-of-custom-printing-method). It is recommended to do so only if rendering the expression class as $\LaTeX$ requires more logics. The disadvantage of defining `_latex_repr_()` as a method is that it requires more boilerplate code, such as explicitly converting the symbolic {attr}`~sympy.core.basic.Basic.args` of the expression class first. In this phase space factor, defining `_latex_repr_` as a {class}`str` would have been just fine.
# :::

# %% [markdown]
# As can be seen, the LaTeX rendering of these classes makes them ideal for mathematically defining and building up larger amplitude models:

# %% tags=["hide-input"]
from IPython.display import Math

from ampform.io import aslatex

s, m1, m2 = sp.symbols("s m1 m2")
q_expr = BreakupMomentum(s, m1, m2)
rho_expr = PhspFactorSWave(s, m1, m2)
Math(aslatex({e: e.evaluate() for e in [rho_expr, q_expr]}))

# %% [markdown]
# Class variables and default arguments to instance arguments are also supported. They can either be indicated with {class}`typing.ClassVar` or by not providing a type hint:

# %%
from __future__ import annotations

from typing import Any, ClassVar


@unevaluated
class FunkyPower(sp.Expr):
    x: Any
    m: int = 1
    default_return: ClassVar[sp.Expr | None] = None
    class_name = "my name"
    _latex_repr_ = R"f_{{{m}}}\left({x}\right)"

    def evaluate(self) -> sp.Expr | None:
        if self.default_return is None:
            return self.x**self.m
        return self.default_return


x = sp.Symbol("x")
exprs = (
    FunkyPower(x),
    FunkyPower(x, 2),
    FunkyPower(x, m=3),
)
Math(aslatex({e: e.doit() for e in exprs}))

# %%
FunkyPower.default_return = sp.Rational(0.5)
Math(aslatex({e: e.doit() for e in exprs}))

# %% [markdown]
# By default, instance attributes are converted ['sympified'](https://docs.sympy.org/latest/modules/core.html#module-sympy.core.sympify). To avoid this behavior, use the {func}`.argument` function.

# %%
from typing import Callable

from ampform.sympy import argument


class Transformation:
    def __init__(self, power: int) -> None:
        self.power = power

    def __call__(self, x: sp.Basic, y: sp.Basic) -> sp.Expr:
        return x + y**self.power


@unevaluated
class MyExpr(sp.Expr):
    x: Any
    y: Any
    functor: Callable = argument(sympify=False)

    def evaluate(self) -> sp.Expr:
        return self.functor(self.x, self.y)


# %% [markdown]
# Notice how the `functor` attribute has not been sympified (there is no SymPy equivalent for a callable object), but the `functor` can be called in the `evaluate()`/`doit()` method.

# %%
a, b, k = sp.symbols("a b k")
expr = MyExpr(a, y=b, functor=Transformation(power=k))
assert expr.x is a
assert expr.y is b
assert not isinstance(expr.functor, sp.Basic)
Math(aslatex({expr: expr.doit()}))

# %% [markdown]
# :::{tip}
# An example where this is used, is in the {class}`.EnergyDependentWidth` class, where we do not want to sympify the {attr}`~.EnergyDependentWidth.phsp_factor` protocol.
# :::

# %% [markdown]
# ## Numerical integrals

# %% [markdown]
# In hadron physics and high-energy physics, it often happens that models contain integrals that do not have an analytical solution.. They can arise in theoretical models, complex scattering problems, or in the analysis of experimental data. In such cases, we need to resort to numerical integrations.
#
# SymPy provides the [`sympy.Integral`](https://docs.sympy.org/latest/modules/integrals/integrals.html#sympy.integrals.integrals.Integral) class, but this does not give us control over whether or not we want to avoid integrating the class analytically. An example of such an analytically unsolvable integral is shown below. Note that the integral does not evaluate despite the `doit()` call.

# %%
import sympy as sp

x, a, b = sp.symbols("x a b")
p = sp.Symbol("p", positive=True)
integral_expr = sp.Integral(sp.exp(x) / (x**p + 1), (x, a, b))
integral_expr.doit()

# %% [markdown]
# For amplitude models that contain such integrals that should not be solved analytically, AmpForm provides the {class}`.UnevaluatableIntegral` class. It functions in the same way as [`sympy.Integral`](https://docs.sympy.org/latest/modules/integrals/integrals.html#sympy.integrals.integrals.Integral), but prevents the class from evaluating at all, even if the integral can be solved analytically.

# %%
from ampform.sympy import UnevaluatableIntegral

UnevaluatableIntegral(x**p, (x, a, b)).doit()

# %%
sp.Integral(x**p, (x, a, b)).doit()

# %% [markdown]
# This allows {class}`.UnevaluatableIntegral` to serve as a placeholder in expression trees that we call `doit` on when lambdifying to a numerical function. The resulting numerical function takes **complex-valued** and **multidimensional arrays** as function arguments.
#
# In the following, we see an example where the parameter $p$ inside the integral gets an array as input.

# %%
integral_expr = UnevaluatableIntegral(sp.exp(x) / (x**p + 1), (x, a, b))
integral_func = sp.lambdify(args=[p, a, b], expr=integral_expr)

# %%
import numpy as np

a_val = 1.2
b_val = 3.6
p_array = np.array([0.4, 0.6, 0.8])

areas = integral_func(p_array, a_val, b_val)
areas

# %% jupyter={"source_hidden": true} tags=["hide-input", "scroll-input"]
# %config InlineBackend.figure_formats = ['svg']

import matplotlib.pyplot as plt

x_area = np.linspace(a_val, b_val, num=100)
x_line = np.linspace(0, 4, num=100)

fig, ax = plt.subplots()
ax.set_xlabel("$x$")
ax.set_ylabel("$x^p$")

for i, p_val in enumerate(p_array):
    ax.plot(x_line, x_line**p_val, label=f"$p={p_val}$", c=f"C{i}")
    ax.fill_between(x_area, x_area**p_val, alpha=(0.7 - i * 0.2), color="C0")

ax.text(
    x=(a_val + b_val) / 2,
    y=((a_val ** p_array[0] + b_val ** p_array[0]) / 2) * 0.5,
    s="Area",
    horizontalalignment="center",
    verticalalignment="center",
)
text_kwargs = dict(ha="center", textcoords="offset points", xytext=(0, -15))
ax.annotate("a", (a_val, 0.08), **text_kwargs)
ax.annotate("b", (b_val, 0.08), **text_kwargs)

ax.legend()
plt.show()

# %% [markdown]
# The arrays can be complex-valued as well. This is particularly useful when calculating dispersion integrals (see **[TR-003](https://compwa.github.io/report/003#general-dispersion-integral)**).

# %%
integral_func(
    p=np.array([1.5 - 8.6j, -4.6 + 5.5j]),
    a=a_val,
    b=b_val,
)

# %% [markdown]
# ## Summations

# %% [markdown]
# The {class}`.PoolSum` class makes it possible to write sums over non-integer ranges. This is for instance useful when summing over allowed helicities. Here are some examples:

# %%
from ampform.sympy import PoolSum

i, j, m, n = sp.symbols("i j m n")
expr = PoolSum(i**m + j**n, (i, (-1, 0, +1)), (j, (2, 4, 5)))
Math(aslatex({expr: expr.doit()}))

# %%
import numpy as np

A = sp.IndexedBase("A")
λ, μ = sp.symbols("lambda mu")
to_range = lambda a, b: tuple(sp.Rational(i) for i in np.arange(a, b + 0.5))
expr = abs(PoolSum(A[λ, μ], (λ, to_range(-0.5, +0.5)), (μ, to_range(-1, +1)))) ** 2
Math(aslatex({expr: expr.doit()}))
