r"""Broadband models and time responses of network parameters.

Two views of S beyond the frequencies it was solved at:

- a rational model, :math:`S(f) = D + \sum_n R_n / (2\pi i f - p_n)`, fitted
  to S by vector fitting, made passive, and written as a SPICE subcircuit or
  a Verilog-A module;
- time responses: the impulse and step responses of S, from which the
  reflected step (TDR), the transmitted step (TDT) and the impedance profile
  are read.

Both start from S on the frequencies of a result (`gfem.NetworkParameters`).
"""

from __future__ import annotations

__all__: list[str] = []
