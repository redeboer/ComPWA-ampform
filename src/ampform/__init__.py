"""Build amplitude models with different PWA formalisms.

AmpForm formalizes formalisms from :doc:`Partial Wave Analysis <pwa:index>`. It provides
tools to convert `~qrules.transition.StateTransition` solutions that the `.qrules`
package found into an `.HelicityModel`. The output `.HelicityModel` can then be used by
external fitter packages to generate a data set (toy Monte Carlo) for this specific
reaction process, or to optimize ('fit') its parameters so that they resemble the data
set as good as possible.
"""
