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


# Kinematics

```python jupyter={"source_hidden": true} mystnb={"code_prompt_show": "Import Python libraries"} tags=["hide-cell"]
import sympy as sp
from IPython.display import Math

from ampform.io import aslatex
```

## Lorentz vectors


AmpForm provides classes for formulating symbolic expressions for boosting and rotating Lorentz vectors. Usually, when building an amplitude model, you don't have to use these classes, but sometimes you want to boost some four-momenta yourself (for instance to boost into the center-of-mass frame of your experiment. Here, we boost a four-momentum $q$ from the lab frame of the BESIII detector into the center-of-mass frame $p$ of the $e^-e^+$&nbsp;collision. Symbolically, this looks like this:

```python
from ampform.kinematics.lorentz import (
    ArrayMultiplication,
    ArraySize,
    BoostZMatrix,
    Energy,
    FourMomentumSymbol,
    three_momentum_norm,
)

p = FourMomentumSymbol("p", shape=[])
q = FourMomentumSymbol("q", shape=[])
beta = three_momentum_norm(p) / Energy(p)
Bz = BoostZMatrix(beta, n_events=ArraySize(beta))
Bz_expr = ArrayMultiplication(Bz, q)
Bz_expr
```

We now use SymPy to create a numerical function. (Of course, you can use [TensorWaves](https://tensorwaves.rtfd.io) instead to use other numerical backends.)

```python
Bz_func = sp.lambdify([p, q], Bz_expr.doit(), cse=True)
```

```python jupyter={"source_hidden": true} mystnb={"code_prompt_show": "Show generated NumPy code"} tags=["hide-cell"]
import inspect

from black import FileMode, format_str

src = inspect.getsource(Bz_func)
src = format_str(src, mode=FileMode())
print(src)
```

Finally, plugging in some numbers that represent data, we get the $q$ in the rest frame of $p$:

```python
import numpy as np

pz_array = np.array([[3.0971, 0, 0, 30e-3]])  # J/psi in BESIII lab frame
q_array = np.array([
    [2.4, 0.3, -1.5, 0.02],
    [3.4, -0.045, 0.6, 1.1],
    # list of measured four-momenta q in lab frame
])
Bz_func(pz_array, q_array)
```

:::{admonition} Four-vector array format
Lambdified expressions that involve Lorentz vector computations, expect the format $p = \left(E, p_x, p_y, p_z\right)$. In addition, the shape of input arrays should be `(n, 4)` with `n` the number of events.
:::

As a cross-check, notice how boosting the original boost momentum into its own rest frame, results in $B_z(p) p = \left(m_{J/\psi}, 0, 0, 0\right)$:

```python
Bz_func(pz_array, pz_array)
```

Note that in this case, boost vector $p$ was in the $z$&nbsp;direction, so we were able to just boost with {class}`.BoostZMatrix`. In the more general case, we can use:

```python
from ampform.kinematics.lorentz import BoostMatrix

B = BoostMatrix(p)
B_expr = ArrayMultiplication(B, q)
B_expr
```

```python
B_func = sp.lambdify([p, q], B_expr.doit(), cse=True)
px_array = np.array([[3.0971, 30e-3, 0, 0]])  # x direction!
B_func(px_array, q_array)
```

And again,  $B(p) p = \left(m_{J/\psi}, 0, 0, 0\right)$:

```python
B_func(px_array, px_array)
```

## Phase space


:::{margin}
This notebook originates from {doc}`compwa-report:017/index`.
:::

Kinematics for a three-body decay $0 \to 123$ can be fully described by two **Mandelstam variables** $\sigma_1, \sigma_2$, because the third variable $\sigma_3$ can be expressed in terms $\sigma_1, \sigma_2$, the mass $m_0$ of the initial state, and the masses $m_1, m_2, m_3$ of the final state. As can be seen, the roles of $\sigma_1, \sigma_2, \sigma_3$ are interchangeable.

```{margin}
See Eq. (1.2) in {cite}`Byckling:1971vca`
```

```python jupyter={"source_hidden": true} tags=["hide-input"]
from ampform.kinematics.phasespace import compute_third_mandelstam

m0, m1, m2, m3 = sp.symbols("m:4")
s1, s2, s3 = sp.symbols("sigma1:4")
s3_expr = compute_third_mandelstam(s1, s2, m0, m1, m2, m3)

latex = aslatex({s3: s3_expr})
Math(latex)
```

<!-- #region -->
The phase space is defined by the closed area that satisfies the condition $\phi(\sigma_1,\sigma_2) \leq 0$, where $\phi$ is a **Kibble function**:


```{margin}
See §V.2 in {cite}`Byckling:1971vca`
```
<!-- #endregion -->

```python jupyter={"source_hidden": true} tags=["hide-input"]
from ampform.kinematics.phasespace import Kibble

kibble = Kibble(s1, s2, s3, m0, m1, m2, m3)

latex = aslatex({kibble: kibble.evaluate()})
Math(latex)
```

and $\lambda$ is the **Källén function**:

```python jupyter={"source_hidden": true} tags=["hide-input"]
from ampform.kinematics.phasespace import Kallen

x, y, z = sp.symbols("x:z")
kallen = Kallen(x, y, z)

latex = aslatex({kallen: kallen.evaluate()})
Math(latex)
```

Any distribution over the phase space can now be defined using a two-dimensional grid over a Mandelstam pair $\sigma_1,\sigma_2$ of choice, with the condition $\phi(\sigma_1,\sigma_2)<0$ selecting the values that are physically allowed.

```python jupyter={"source_hidden": true} tags=["hide-input"]
from ampform.kinematics.phasespace import is_within_phasespace

is_within_phasespace(s1, s2, m0, m1, m2, m3)
```

See {doc}`compwa-report:017/index` for an interactive visualization of the phase space region and an analytic expression for the phase space boundary.
