"""Converged references of the suite cases.

Each suite case directory holds one ``reference.json`` (``suite-reference/1``):
the converged values a run is scored against, their bound, the plausibility
ceiling per order, the DOF pin per mesh and order, and provenance.

- :mod:`benchmarks.references.document`: the file's fields, checks and writer.
- :mod:`benchmarks.references.closed`: closed forms evaluated at 40 digits with
  the standard library's ``decimal`` (``lcavity``, ``bragg``) and the no-reference
  file of ``r10M_void``.
- :mod:`benchmarks.references.microstrip`: the microstrip ladder (rung case files
  under ``microstrip/``, campaign stages, Richardson extrapolation, bound).

Standard library only.
"""
