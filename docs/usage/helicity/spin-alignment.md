---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: "1.3"
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

# Spin alignment

```python jupyter={"source_hidden": true} mystnb={"code_prompt_show": "Import Python libraries"} tags=["hide-cell"]
import logging

import graphviz
import qrules
import sympy as sp
from IPython.display import Math

import ampform
from ampform.helicity import HelicityModel
from ampform.io import aslatex, improve_latex_rendering

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.ERROR)
improve_latex_rendering()
```

As described in {doc}`compwa-report:015/index`, the {doc}`'standard' helicity formalism </usage/helicity/formalism>` is not suited for state transitions that have different decay topologies. For this reason, the {class}`.HelicityAmplitudeBuilder` can insert a number of Wigner-$D$ functions into the amplitude model to 'align' the final state spins of underlying {class}`~qrules.topology.Topology` instances in the full decay.

Imagine we have the following the decay:

```python
reaction = qrules.generate_transitions(
    initial_state=("J/psi(1S)", [-1, +1]),
    final_state=["K0", "Sigma+", "p~"],
    allowed_intermediate_particles=["Sigma(1660)", "N(1650)"],
    allowed_interaction_types=["strong"],
    formalism="helicity",
)
```

```python jupyter={"source_hidden": true} tags=["hide-input"]
src = qrules.io.asdot(
    reaction,
    collapse_graphs=True,
    render_initial_state_id=True,
)
graphviz.Source(src)
```

This decay has **two** different decay topologies, that is, it has resonances in **two different sub-systems**. By default, the {class}`.HelicityAmplitudeBuilder` does not take these differing decay topologies into account and falls back aligning the amplitudes with {class}`.NoAlignment`. Explicitly:

```python
from ampform.helicity.align import NoAlignment

builder = ampform.get_builder(reaction)
builder.config.spin_alignment = NoAlignment()
non_aligned_model = builder.formulate()
non_aligned_model.intensity
```

The symbols for the amplitudes are defined through {attr}`.HelicityModel.amplitudes`:

```python jupyter={"source_hidden": true} tags=["full-width", "hide-input"]
def render_amplitudes(model: HelicityModel) -> Math:
    selected_amplitudes = {
        symbol: expr
        for i, (symbol, expr) in enumerate(model.amplitudes.items())
        if i % 5 == 0
    }
    src = aslatex(selected_amplitudes, terms_per_line=1)
    src = src.replace(R"\end{array}", R"\dots \\ \end{array}")
    return Math(src)


render_amplitudes(non_aligned_model)
```

## Dalitz-Plot Decomposition

One way of aligning the spins of each sub-system, is Dalitz-Plot Decomposition (DPD) {cite}`Marangotto:2019ucc`. DPD **can only be used for three-body decays**, but results in a quite condense amplitude model expression.

We can select DPD alignment as follows:

:::{warning}
The {class}`.DalitzPlotDecomposition` is not yet fully functional for reactions with a polarized initial or final state. In this example, the sums inside the incoherent sum should also include $\lambda_0=0$.
:::

```python tags=["full-width"]
from ampform.helicity.align.dpd import DalitzPlotDecomposition, relabel_edge_ids

reaction_123 = relabel_edge_ids(reaction)
builder_123 = ampform.get_builder(reaction_123)
builder_123.config.spin_alignment = DalitzPlotDecomposition(reference_subsystem=1)
builder_123.config.scalar_initial_state_mass = True
builder_123.config.stable_final_state_ids = [1, 2, 3]
dpd_model = builder_123.formulate()
dpd_model.intensity.cleanup()
```

:::{warning}
The {class}`.DalitzPlotDecomposition` formalism uses different indices for the initial and final state, so relabel the reaction with {func}`.relabel_edge_ids` first.
:::

```python jupyter={"source_hidden": true} tags=["hide-input"]
src = qrules.io.asdot(
    reaction_123,
    collapse_graphs=True,
    render_initial_state_id=True,
)
graphviz.Source(src)
```

This method introduces several new angles that are defined through the {attr}`~.HelicityModel.kinematic_variables`:

```python jupyter={"source_hidden": true} tags=["hide-input"]
dpd_angles = {
    k: v for k, v in dpd_model.kinematic_variables.items() if "zeta" in str(k)
}
src = aslatex(dpd_angles)
Math(src)
```

Note that the amplitudes are the same as those in the non-aligned model:

:::{warning}
This behavior is a bug that will be fixed through [ComPWA/ampform#318](https://github.com/ComPWA/ampform/issues/318).
:::

```python jupyter={"source_hidden": true} tags=["full-width", "hide-input"]
render_amplitudes(dpd_model)
```

## Axis-angle method

The second spin alignment method is the 'axis-angle method' {cite}`Marangotto:2019ucc`. This method results in much larger expressions and is therefore much less efficient, but theoretically it **can handle $n$-body final states**. It can be selected as follows:

```python
from ampform.helicity.align.axisangle import AxisAngleAlignment

builder.config.spin_alignment = AxisAngleAlignment()
axisangle_model = builder.formulate()
```

```python jupyter={"source_hidden": true} tags=["full-width", "hide-input"]
Math(aslatex({"I": axisangle_model.intensity.evaluate()}, terms_per_line=1))
```

This method of alignment introduces several **Wigner rotation angles** to the {attr}`.HelicityModel.kinematic_variables`. An example:

```python jupyter={"source_hidden": true} tags=["full-width", "hide-input"]
alpha = sp.Symbol("alpha_0^01", real=True)
Math(aslatex({alpha: axisangle_model.kinematic_variables[alpha]}))
```

For more information about these angles, see {ref}`compwa-report:015/index:Compute Wigner rotation angles` in {doc}`TR-015 <compwa-report:015/index>`.
