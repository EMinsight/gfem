"""The diffraction values are pytrees whose static parts hash and compare by content.

Two equal-shaped objects with several frequencies meet inside ``jax.jit``'s
cache and ``jax.tree.map``; a NumPy array or a dict among the static fields
used to make JAX's metadata comparison raise.
"""

import copy

import jax
import jax.numpy as jnp
import numpy as np

from gfem import diffraction as dif
from gfem._diffraction.emissivity import Emissivity
from gfem._diffraction.outputs import Efficiencies, Jones, PowerBalance

FREQ = np.array([2.0e14, 3.0e14])


def _aperture():
    x = np.arange(4) * 1e-7
    values = (np.arange(2 * 2 * 4) + 1.0).reshape(2, 2, 4) * (1 + 0.5j)
    return dif.ApertureField(values, FREQ.copy(), x)


def _cell_response():
    from unit import test_cell_response as tcr

    return tcr._response()


def _jones():
    values = (np.arange(8) + 1.0).reshape(2, 2, 2) * (1 + 0.5j)
    return Jones(
        values=values,
        frequency=FREQ.copy(),
        outgoing=("z_max", (0, 0)),
        incident=("z_max", (0, 0)),
        direction=(1, -1),
    )


def _efficiencies():
    rows = (("z_max", (0, 0), "TM"), ("z_max", (0, 0), "TE"))
    return Efficiencies(
        frequency=FREQ.copy(),
        values=jnp.asarray([[0.2, 0.3], [0.4, 0.1]]),
        rows=rows,
        incidence=("z_max", (0, 0), "TE"),
    )


def _balance():
    r = jnp.asarray([0.2, 0.3])
    return PowerBalance(
        frequency=FREQ.copy(),
        reflectance=r,
        transmittance=r,
        coupled=r * 0,
        absorptance=1 - 2 * r,
        incidence=("z_max", (0, 0), "TE"),
    )


def _emissivity():
    return Emissivity(
        values=jnp.asarray([[0.2, 0.3], [0.4, 0.1]]),
        frequency=FREQ.copy(),
        theta=(0.1, 0.2),
        phi=(0.0, 0.0),
        surface="z_max",
        polarization="TE",
    )


def _total(obj):
    return sum(jnp.abs(leaf).sum() for leaf in jax.tree.leaves(obj))


def test_equal_shaped_objects_share_one_jit_and_map_together():
    for make in (
        _aperture,
        _cell_response,
        _jones,
        _efficiencies,
        _balance,
        _emissivity,
    ):
        first = make()
        # a second object with equal metadata held in distinct arrays
        second = jax.tree.map(lambda a: 2 * a, copy.deepcopy(first))
        assert type(second) is type(first)
        f = jax.jit(_total)
        assert np.isclose(f(first), _total(first))
        assert np.isclose(f(second), _total(second))
        summed = jax.tree.map(lambda a, b: a + b, first, second)
        assert np.isclose(_total(summed), 3 * _total(first))
        assert jax.tree.structure(first) == jax.tree.structure(copy.deepcopy(first))
