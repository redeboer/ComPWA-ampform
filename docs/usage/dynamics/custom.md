---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python hideCode=true hideOutput=true hidePrompt=true jupyter={"source_hidden": true} tags=["remove-cell", "skip-execution"]
# WARNING: advised to install a specific version, e.g. ampform==0.1.2
%pip install -q ampform[doc,viz] IPython
```

```python hideCode=true hideOutput=true hidePrompt=true jupyter={"source_hidden": true} tags=["remove-cell"]
import os

STATIC_WEB_PAGE = {"EXECUTE_NB", "READTHEDOCS"}.intersection(os.environ)
```

```{autolink-concat}
```


# Custom dynamics

```python jupyter={"source_hidden": true} mystnb={"code_prompt_show": "Import Python libraries"} tags=["hide-cell"]
%config InlineBackend.figure_formats = ['svg']

from __future__ import annotations

import graphviz
import qrules
import sympy as sp
from IPython.display import Math

from ampform.io import aslatex
```

We start by generating allowed transitions for a simple decay channel, just like in {doc}`/usage/amplitude`:

```python tags=["remove-output"]
reaction = qrules.generate_transitions(
    initial_state=("J/psi(1S)", [+1]),
    final_state=[("gamma", [+1]), "pi0", "pi0"],
    allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
    allowed_interaction_types=["strong", "EM"],
    formalism="helicity",
)
```

```python tags=["hide-input"]
dot = qrules.io.asdot(reaction, collapse_graphs=True)
graphviz.Source(dot)
```

Next, create a {class}`.HelicityAmplitudeBuilder` using {func}`.get_builder`:

```python
from ampform import get_builder

model_builder = get_builder(reaction)
```

In {doc}`/usage/amplitude`, we used {meth}`.DynamicsSelector.assign` with some standard lineshape builders from the {mod}`.builder` module. These builders have a signature that follows the {class}`.ResonanceDynamicsBuilder` {class}`~typing.Protocol`:

```python
import inspect

from ampform.dynamics.builder import (
    ResonanceDynamicsBuilder,
    create_relativistic_breit_wigner,
)

print(inspect.getsource(ResonanceDynamicsBuilder))
print(inspect.getsource(create_relativistic_breit_wigner))
```

A function that behaves like a {class}`.ResonanceDynamicsBuilder` should return a {class}`tuple` of some {class}`~sympy.core.expr.Expr` (which formulates your lineshape) and a {class}`dict` of {class}`~sympy.core.symbol.Symbol`s to some suggested initial values. This signature is required so the builder knows how to extract the correct symbol names and their suggested initial values from a {class}`~qrules.topology.Transition`.


The {class}`~sympy.core.expr.Expr` you use for the lineshape can be anything. Here, we use a Gaussian function and wrap it in a function. As you can see, this function stands on its own, independent of {mod}`ampform`:

```python
def my_dynamics(x: sp.Symbol, mu: sp.Symbol, sigma: sp.Symbol) -> sp.Expr:
    return sp.exp(-((x - mu) ** 2) / sigma**2 / 2) / (sigma * sp.sqrt(2 * sp.pi))
```

```python
x, mu, sigma = sp.symbols("x mu sigma")
sp.plot(my_dynamics(x, 0, 1), (x, -3, 3), axis_center=(0, 0))
my_dynamics(x, mu, sigma)
```

We can now follow the example of the {func}`.create_relativistic_breit_wigner` to create a builder for this custom lineshape:

```python
from qrules.particle import Particle

from ampform.dynamics.builder import TwoBodyKinematicVariableSet


def create_my_dynamics(
    resonance: Particle, variable_pool: TwoBodyKinematicVariableSet
) -> tuple[sp.Expr, dict[sp.Symbol, float]]:
    res_mass = sp.Symbol(f"m_{resonance.name}")
    res_width = sp.Symbol(f"sigma_{resonance.name}")
    expression = my_dynamics(
        x=variable_pool.incoming_state_mass,
        mu=res_mass,
        sigma=res_width,
    )
    parameter_defaults = {
        res_mass: resonance.mass,
        res_width: resonance.width,
    }
    return expression, parameter_defaults
```

Now, just like in {ref}`usage/amplitude:Set dynamics`, it's simply a matter of plugging this builder into {meth}`.DynamicsSelector.assign` and we can {meth}`~.HelicityAmplitudeBuilder.formulate` a model with this custom lineshape:

```python
for name in reaction.get_intermediate_particles().names:
    model_builder.dynamics.assign(name, create_my_dynamics)
model = model_builder.formulate()
```

As can be seen, the {attr}`.HelicityModel.parameter_defaults` section has been updated with the some additional parameters for the custom parameter and there corresponding suggested initial values:

```python
Math(aslatex(model.parameter_defaults))
```

Let's quickly have a look what this lineshape looks like. First, check which {class}`~sympy.core.symbol.Symbol`s remain once we replace the parameters with their suggested initial values. These are the kinematic variables of the model:

```python
expr = model.expression.doit().subs(model.parameter_defaults)
free_symbols = tuple(sorted(expr.free_symbols, key=lambda s: s.name))
free_symbols
```

To create an invariant mass distribution, we should integrate out the $\theta$ angle. This can be done with {func}`~sympy.integrals.integrals.integrate`:

```python
m, theta = free_symbols
integrated_expr = sp.integrate(
    expr,
    (theta, 0, sp.pi),
    meijerg=True,
    conds="piecewise",
    risch=None,
    heurisch=None,
    manual=None,
)
Math(aslatex(integrated_expr.n(1), terms_per_line=1))
```

Finally, here is the resulting expression as a function of the invariant mass, with **custom dynamics**!

```python
x1, x2 = 0.6, 1.9
sp.plot(integrated_expr, (m, x1, x2), axis_center=(x1, 0));
```
