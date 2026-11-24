"""The port mode section kernel (placeholder, filled in below)."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(eq=False)
class FormPair:
    r"""A form contracted with a pair of vectors, with a weight.

    :math:`\mathrm{d}L = \operatorname{Re}\,w\,\ell^T\,\mathrm{d}F\,r` for a
    bilinear form of the port section (rows the test side, ``left``) and
    :math:`\operatorname{Re}\,w\,\mathrm{d}F\cdot r` for a linear functional.

    Attributes:
        form: ``"power_g"``, ``"power_t"``, ``"gauge_n"``, ``"gauge_t"``.
        left: The test-side vector on the free DOFs, ``None`` for a
            functional.
        right: The trial-side vector on the free DOFs.
        weight: The complex weight.
    """

    form: str
    left: np.ndarray | None
    right: np.ndarray
    weight: complex
