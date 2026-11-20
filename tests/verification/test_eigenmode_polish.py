"""The forward polishes every mode in mode-field reach.

A refined mode's error is an admixture of the other modes, which its
backward error hardly sees: participation from the default
``eigen_tolerance`` depends on the target (where the shift sat) by up to
1e-4. Newton corrections on the live factors take it to round-off, so two
targets agree, and a plain solve and a differentiated one polish the same
modes (the differentiated one keeps its guard mode refined anyway).

Fixture J: the two-layer line (ramped ``eps_r``) closed by a 1 nH junction
port, two modes near 3.18 and 9.91 GHz.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from gradient_cases import LAYERS, _mesh, ramp

import gfem


@pytest.fixture(scope="module")
def junction_line(tmp_path_factory):
    mesh = _mesh(
        tmp_path_factory.mktemp("j"),
        "j.msh",
        sm.strip_line(10.0, 2.0, 1.0, (10, 2, 1), layers=LAYERS),
    )
    return gfem.Model(
        mesh,
        materials={
            r: gfem.MaterialField(eps_r=ramp(mesh, r, 3.5, 4.5))
            for r in ("sub", "cover")
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[gfem.LumpedPort("jj", "port1", "+z", inductance=1e-9)],
    )


def _solve(model, target, **options):
    return gfem.solve(
        model,
        gfem.Eigenmode(target=target, n_modes=2),
        options=gfem.SolverOptions(**options),
    )


# unpolished, the default tolerance happens to give 2.5e-10 on this small
# line; 1e-6 gave 5e-6
@pytest.mark.parametrize("tolerance", [1e-10, 1e-6])
def test_participation_from_two_targets_agrees(junction_line, tolerance):
    low = _solve(junction_line, 3e9, eigen_tolerance=tolerance)
    high = _solve(junction_line, 9e9, eigen_tolerance=tolerance)
    order = np.argsort(np.real(np.asarray(high.omega)))
    np.testing.assert_allclose(
        np.real(np.asarray(high.omega))[order],
        np.real(np.asarray(low.omega)),
        rtol=1e-10,
    )
    p_low = np.asarray(low.participation["jj"])
    p_high = np.asarray(high.participation["jj"])[order]
    assert np.all(p_low > 0.1)
    np.testing.assert_allclose(p_high, p_low, rtol=1e-8)
    assert np.all(np.asarray(low.error) <= tolerance)
    assert np.all(np.asarray(high.error) <= tolerance)


def test_polished_solve_and_value_and_grad_agree(junction_line):
    plain = _solve(junction_line, 3e9)

    def objective(scale):
        model = junction_line.replace(
            materials={
                r: gfem.MaterialField(eps_r=scale * m.eps_r)
                for r, m in junction_line.materials.items()
            }
        )
        result = _solve(model, 3e9)
        value = jnp.real(result.omega[0]) / 1e10
        return value, (result.participation["jj"], result.error)

    (value, (participation, error)), grad = jax.value_and_grad(objective, has_aux=True)(
        jnp.asarray(1.0)
    )
    assert np.isfinite(grad)
    np.testing.assert_allclose(value, np.real(plain.omega[0]) / 1e10, rtol=1e-13)
    np.testing.assert_allclose(
        participation, plain.participation["jj"], rtol=1e-10, atol=0
    )
    tolerance = gfem.SolverOptions().eigen_tolerance
    assert np.all(np.asarray(error) <= tolerance)
    assert np.all(np.asarray(plain.error) <= tolerance)
