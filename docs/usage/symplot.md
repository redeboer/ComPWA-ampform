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


# symplot


```{eval-rst}
.. automodule:: symplot
```


## Examples


The following examples show how to work with {func}`.prepare_sliders` and the resulting {class}`.SliderKwargs`. For more explanation about what happens behind the scenes, see {doc}`interactive`.


### Exponential wave


Construct a mathematical expression with {mod}`sympy`:

```python
import sympy as sp

n = sp.Symbol("n", integer=True)
x, a = sp.symbols("x, a")
expression = sp.sin(n * x) * sp.exp(-a * x)
expression
```

Create sliders with {func}`.prepare_sliders`, set their ranges and (optionally) provide some initial values:

```python
from symplot import prepare_sliders

np_expression, sliders = prepare_sliders(expression, plot_symbol=x)
sliders.set_ranges(
    n=(0, 10),
    a=(-1, 1, 200),
)
sliders.set_values(n=6, a=0.3)
```

```python jupyter={"source_hidden": true} tags=["remove-cell"]
if STATIC_WEB_PAGE:
    import numpy as np

    # Concatenate flipped domain for reverse animation
    domain = np.linspace(-1, 1, 50)
    domain = np.concatenate((domain, np.flip(domain[1:])))
    sliders._sliders["a"] = domain
```

Now use {doc}`mpl-interactions <mpl_interactions:index>` to plot the {doc}`lambdified <sympy:modules/utilities/lambdify>` expression. Note how the {class}`SliderKwargs` are unpacked as keyword arguments:

```{autolink-skip}
```

```python
%matplotlib widget
```

```python tags=["remove-output"]
%config InlineBackend.figure_formats = ['svg']

import matplotlib.pyplot as plt
import mpl_interactions.ipyplot as iplt
import numpy as np

plot_domain = np.linspace(0, 10, 1_000)
fig, ax = plt.subplots(figsize=(7, 4))
controls = iplt.plot(
    plot_domain,
    np_expression,
    **sliders,
    ylim="auto",
)
ax.set_xlabel("$x$")
ax.set_ylabel(f"${sp.latex(expression)}$");
```

{{ run_interactive }}

```python jupyter={"source_hidden": true} tags=["remove-input"]
# Export for Read the Docs
if STATIC_WEB_PAGE:
    from IPython.display import Image, display

    output_path = "exponential-wave.gif"
    ax.set_yticks([])
    iplt.title("$n = {n}, a = {a:.2f}$", controls=controls)
    controls.save_animation(output_path, fig, "a", fps=25)
    with open(output_path, "rb") as f:
        display(Image(data=f.read(), format="png"))
```

### Range slider


See {doc}`mpl_interactions:examples/range-sliders`.

```python
np_expression, sliders = prepare_sliders(expression, plot_symbol=x)
sliders.set_values(n=6, a=0.3)
sliders.set_ranges(
    n=(0, 10),
    a=(-1, 1, 200),
)


def x_domain(x_range, **kwargs):
    min_, max_ = x_range
    return np.linspace(min_, max_, 1_000)


def f(x, **kwargs):
    del kwargs["x_range"]
    return np_expression(x, **kwargs)


fig, ax = plt.subplots()
controls = iplt.plot(
    x_domain,
    f,
    x_range=("r", 0, 10),
    **sliders,
    xlim="auto",
    ylim="auto",
)
```
