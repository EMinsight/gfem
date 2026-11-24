"""Length-scale constraints on a layer grid, and the robust triple's filter radius."""

import itertools
import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

import gfem
from gfem import design
from gfem.errors import NonDifferentiableError

MM = 1e-3
R = 0.1 * MM  # filter radius, and the rule length of the sign tests
SQRT3 = math.sqrt(3)


def _plate(tmp_path_factory, name, side, cells):
    import structured_meshes as sm

    path = sm.write_msh22(
        tmp_path_factory.mktemp(name) / f"{name}.msh",
        sm.plate(side[0], side[1], (("design", 0.05),), (cells[0], cells[1], 1)),
    )
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


@pytest.fixture(scope="module")
def square(tmp_path_factory):
    """An 8R x 8R single-layer plate, cells of R/8."""
    return _plate(tmp_path_factory, "square", (0.8, 0.8), (64, 64))


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """A 3R x 2.5R plate, cells of R/4."""
    return _plate(tmp_path_factory, "small", (0.3, 0.25), (12, 10))


def _edges(grid, axis):
    """Pixel edges along x (axis 1 of the array) or y (axis 0)."""
    n = grid.shape[axis]
    return grid.origin[1 - axis] + np.arange(n + 1) * grid.pitch


def _bar(grid, width, centre):
    """A drawn bar along y of the given width (m) at x = centre, area-weighted."""
    edges = _edges(grid, 1)
    lo, hi = centre - width / 2, centre + width / 2
    cover = np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo)
    frac = np.clip(cover, 0, None) / grid.pitch
    return np.broadcast_to(frac[None, :], grid.shape).copy()


def _disk(grid, diameter, centre, sub=8):
    """A drawn disk, pixels weighted by the area inside it (sub x sub samples)."""
    o = (np.arange(sub) + 0.5) / sub
    xs = (_edges(grid, 1)[:-1, None] + o * grid.pitch).ravel()
    ys = (_edges(grid, 0)[:-1, None] + o * grid.pitch).ravel()
    inside = (xs[None, :] - centre[0]) ** 2 + (ys[:, None] - centre[1]) ** 2
    inside = inside <= (diameter / 2) ** 2
    return inside.reshape(grid.shape[0], sub, grid.shape[1], sub).mean(axis=(1, 3))


def _drawn_for_printed(width, radius):
    """The drawn bar width whose filtered line prints ``width`` at eta = 1/2.

    Arrieta, Romano and Johnson 2025, Eq. 43 (1D Helmholtz filter).
    """
    return radius / SQRT3 * math.log(2 * math.cosh(SQRT3 * width / radius))


def _printed_width(profile, x, eta):
    """Width of the region above eta of a 1D profile, edges by linear interpolation."""
    above = np.flatnonzero(profile > eta)
    if not len(above):
        return 0.0
    i, j = above[0], above[-1]

    def cross(a, b):
        t = (eta - profile[a]) / (profile[b] - profile[a])
        return x[a] + t * (x[b] - x[a])

    return cross(j + 1, j) - cross(i - 1, i)


def _centre(grid):
    xmin, ymin, xmax, ymax = grid.bounds
    return (xmin + xmax) / 2, (ymin + ymax) / 2


# --- closed forms -------------------------------------------------------------------


def test_closed_forms_at_half_threshold_and_width_of_the_radius(square):
    grid = design.LayerGrid(square, "design", pitch=R / 8)
    blur = design.helmholtz_filter(grid, radius=R)
    rules = design.length_scale_constraints(blur, min_width=R, min_spacing=R)
    # Arrieta, Romano and Johnson 2025 for the PDE filter at w = R: 0.8284, 1.029
    assert rules.eta_solid == pytest.approx(0.8284, abs=1e-4)
    assert rules.eta_void == pytest.approx(1 - 0.8284, abs=1e-4)
    assert rules.decay == pytest.approx(10 * R**2)
    assert (rules.radius, rules.eta) == (R, 0.5)
    assert (rules.min_width, rules.min_spacing) == (R, R)
    # epsilon is Eq. 55 with gamma = 1.029, L = sqrt(area) on the sampled area
    area = grid.sampled.sum() * grid.pitch**2
    scale = R / math.sqrt(area) * 3 * math.sqrt(math.pi) / (128 * 10**2.5)
    for eps in (rules.epsilon_solid, rules.epsilon_void):
        assert (scale / eps) ** (1 / 3) == pytest.approx(1.029, abs=1e-3)


def test_separate_rules_give_separate_thresholds(square):
    grid = design.LayerGrid(square, "design", pitch=R / 8)
    blur = design.helmholtz_filter(grid, radius=R)
    rules = design.length_scale_constraints(blur, min_width=R, min_spacing=0.5 * R)
    wider = design.length_scale_constraints(
        blur, min_width=1.2 * R, min_spacing=0.5 * R
    )
    # a wider line must keep a higher filtered peak; the spacing rule is unchanged
    assert wider.eta_solid > rules.eta_solid
    assert wider.eta_void == rules.eta_void
    # a narrower spacing lets the trough sit higher, closer to eta
    assert rules.eta_void > 1 - rules.eta_solid


# --- constraint sign on printed features ---------------------------------------------


@pytest.mark.parametrize("pitch", [R / 4, R / 8])
@pytest.mark.parametrize(("printed", "held"), [(0.9, False), (1.1, True)])
def test_bar_and_slot_keep_the_rule_from_the_printed_width_on(
    square, pitch, printed, held
):
    grid = design.LayerGrid(square, "design", pitch=pitch)
    blur = design.helmholtz_filter(grid, radius=R)
    rules = design.length_scale_constraints(blur, min_width=R, min_spacing=R)
    drawn = _bar(
        grid, _drawn_for_printed(printed * R, R), _centre(grid)[0] + 0.37 * pitch
    )
    line = rules(blur(drawn))
    slot = rules(blur(1 - drawn))
    assert isinstance(line, design.LengthScaleValues)
    assert (float(line.solid) <= 0) == held
    assert (float(slot.void) <= 0) == held
    # the wide void beside the line and the wide solid beside the slot are fine
    assert float(line.void) <= 0
    assert float(slot.solid) <= 0


@pytest.mark.parametrize(
    ("drawn", "printed_below", "held"), [(1.0, 0.8, False), (1.6, None, True)]
)
def test_a_disk_and_a_hole_keep_the_rule_from_a_wider_printed_diameter(
    square, drawn, printed_below, held
):
    grid = design.LayerGrid(square, "design", pitch=R / 8)
    blur = design.helmholtz_filter(grid, radius=R)
    rules = design.length_scale_constraints(blur, min_width=R, min_spacing=R)
    cx, cy = _centre(grid)
    disk = _disk(grid, drawn * R, (cx + 0.3 * grid.pitch, cy - 0.2 * grid.pitch))
    filtered = np.asarray(blur(disk))
    row = int(np.argmin(abs(grid.y - cy)))
    printed = _printed_width(filtered[row], grid.x, 0.5) / R
    if held:
        assert printed > 1.4  # about 1.5 R printed
    else:
        assert 0.6 < printed < printed_below  # about 0.75 R printed
    assert (float(rules(filtered).solid) <= 0) == held
    assert (float(rules(1 - filtered).void) <= 0) == held


def test_an_empty_or_full_design_holds_both_rules(square):
    grid = design.LayerGrid(square, "design", pitch=R / 4)
    blur = design.helmholtz_filter(grid, radius=R)
    rules = design.length_scale_constraints(blur, min_width=R, min_spacing=R)
    for value in (0.0, 1.0):
        values = rules(blur(jnp.full(grid.shape, value)))
        np.testing.assert_allclose(jnp.stack(values), [-1.0, -1.0], atol=1e-9)


# --- gradients, transforms, seams ----------------------------------------------------


def _layout(grid, seed=0):
    """A drawn bar and a grey blob with noise: a layout-like design."""
    rng = np.random.default_rng(seed)
    cx, cy = _centre(grid)
    rho = _bar(grid, 0.6 * R, cx - 0.4 * R) + 0.6 * _disk(
        grid, 0.9 * R, (cx + 0.6 * R, cy)
    )
    return jnp.asarray(np.clip(rho + 0.05 * rng.standard_normal(grid.shape), 0, 1))


@pytest.mark.parametrize("periodic", [(False, False), (True, False), (True, True)])
def test_constraint_gradients(small, periodic):
    grid = design.LayerGrid(small, "design", pitch=R / 4, periodic=periodic)
    blur = design.helmholtz_filter(grid, radius=R)
    rules = design.length_scale_constraints(blur, min_width=R, min_spacing=R)

    @jax.jit
    def f(rho):
        return jnp.stack(rules(blur(rho)))

    rho = _layout(grid)
    assert np.all(np.isfinite(f(rho)))
    check_grads(
        f, (rho,), order=1, modes=["fwd", "rev"], eps=1e-6, atol=1e-5, rtol=1e-5
    )
    check_grads(f, (rho,), order=2, modes=["rev"], eps=1e-6, atol=1e-4, rtol=1e-3)


def test_constraints_average_over_sampled_pixels_only(tmp_path_factory):
    coarse = _plate(tmp_path_factory, "coarse", (0.3, 0.3), (3, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gfem.UnsampledPixelsWarning)
        grid = design.LayerGrid(coarse, "design", pitch=R / 4)
    assert not grid.sampled.all()
    blur = design.helmholtz_filter(grid, radius=R)
    rules = design.length_scale_constraints(blur, min_width=R, min_spacing=R)
    area = grid.sampled.sum() * grid.pitch**2
    scale = R / math.sqrt(area) * 3 * math.sqrt(math.pi) / (128 * 10**2.5)
    assert (scale / rules.epsilon_solid) ** (1 / 3) == pytest.approx(1.029, abs=1e-3)

    @jax.jit
    def f(rho):
        return jnp.stack(rules(blur(rho)))

    rho = _layout(grid)
    check_grads(
        f, (rho,), order=1, modes=["fwd", "rev"], eps=1e-6, atol=1e-5, rtol=1e-5
    )


def test_constraints_are_invariant_under_a_shift_across_a_seam(square):
    grid = design.LayerGrid(square, "design", pitch=R / 8, periodic=(True, True))
    blur = design.helmholtz_filter(grid, radius=R)
    rules = design.length_scale_constraints(blur, min_width=R, min_spacing=R)
    xmin = grid.bounds[0]
    # a thin line crossing the x seam, and a disk
    drawn = _bar(grid, _drawn_for_printed(0.9 * R, R), xmin + 0.1 * R)
    drawn = drawn + _bar(grid, _drawn_for_printed(0.9 * R, R), xmin + 8 * R + 0.1 * R)
    drawn = np.clip(drawn, 0, 1) + _disk(grid, 1.4 * R, _centre(grid))
    base = jnp.stack(rules(blur(drawn)))
    assert float(base[0]) > 0
    for shift, axis in ((17, 1), (23, 0)):
        moved = jnp.stack(rules(blur(np.roll(drawn, shift, axis=axis))))
        np.testing.assert_allclose(moved, base, rtol=1e-10)


def test_a_line_cut_by_a_wall_is_seen_with_its_mirror_image(square):
    grid = design.LayerGrid(square, "design", pitch=R / 8)
    blur = design.helmholtz_filter(grid, radius=R)
    rules = design.length_scale_constraints(blur, min_width=R, min_spacing=R)
    # half of a line that prints 1.1 R, against the left wall: with its mirror
    # image the wall line is wide enough
    xmin = grid.bounds[0]
    drawn = _drawn_for_printed(1.1 * R, R)
    assert float(rules(blur(_bar(grid, drawn, xmin))).solid) <= 0
    # the same half line away from the wall is too narrow
    inside = _bar(grid, drawn / 2, _centre(grid)[0])
    assert float(rules(blur(inside)).solid) > 0


def test_constraints_jit_vmap_and_stack(square):
    grid = design.LayerGrid(square, "design", pitch=R / 4)
    blur = design.helmholtz_filter(grid, radius=R)
    rules = design.length_scale_constraints(blur, min_width=R, min_spacing=R)
    cx = _centre(grid)[0]
    batch = jnp.stack(
        [_bar(grid, _drawn_for_printed(w * R, R), cx) for w in (0.9, 1.1)]
    )
    values = rules(blur(batch[0]))
    stacked = jnp.stack(values)
    assert stacked.shape == (2,)
    np.testing.assert_allclose(
        jax.jit(lambda r: jnp.stack(rules(blur(r))))(batch[0]), stacked
    )
    # the constraints close over a jitted function as a static value
    np.testing.assert_allclose(
        jax.jit(lambda c, r: jnp.stack(c(blur(r))))(rules, batch[0]), stacked
    )
    many = jax.vmap(lambda r: jnp.stack(rules(blur(r))))(batch)
    assert many.shape == (2, 2)
    assert float(many[0, 0]) > 0 >= float(many[1, 0])


def test_the_constraints_describe_themselves(square):
    grid = design.LayerGrid(square, "design", pitch=R / 4)
    blur = design.helmholtz_filter(grid, radius=R)
    rules = design.length_scale_constraints(blur, min_width=R, min_spacing=0.8 * R)
    text = repr(rules)
    assert "LengthScaleConstraints" in text
    assert "min_width=0.0001 m" in text
    assert "min_spacing=8e-05 m" in text


# --- refusals -----------------------------------------------------------------------


def test_the_names_live_in_gfem_design():
    import gfem._length_scale as private

    for name in (
        "LengthScaleConstraints",
        "LengthScaleValues",
        "length_scale_constraints",
        "radius_for_length_scale",
    ):
        assert name in design.__all__
        assert getattr(design, name) is getattr(private, name)


def test_both_rules_are_required(square):
    grid = design.LayerGrid(square, "design", pitch=R / 4)
    blur = design.helmholtz_filter(grid, radius=R)
    with pytest.raises(TypeError, match="min_spacing"):
        design.length_scale_constraints(blur, min_width=R)
    with pytest.raises(TypeError, match="min_width"):
        design.length_scale_constraints(blur, min_spacing=R)


def test_a_free_form_filter_is_refused(tmp_path_factory):
    mesh = _plate(tmp_path_factory, "free", (0.3, 0.3), (3, 3))
    blur = design.helmholtz_filter(mesh, "design", radius=R)
    with pytest.raises(NotImplementedError, match="LayerGrid"):
        design.length_scale_constraints(blur, min_width=R, min_spacing=R)
    with pytest.raises(TypeError, match="helmholtz_filter"):
        design.length_scale_constraints(R, min_width=R, min_spacing=R)


def test_rules_out_of_the_validated_range_are_refused_with_a_remedy(square):
    grid = design.LayerGrid(square, "design", pitch=R / 4)
    blur = design.helmholtz_filter(grid, radius=R)
    design.length_scale_constraints(blur, min_width=1.25 * R, min_spacing=1.25 * R)
    with pytest.raises(ValueError, match=r"min_width.*radius to at least 0\.00012"):
        design.length_scale_constraints(blur, min_width=1.5 * R, min_spacing=R)
    with pytest.raises(ValueError, match=r"min_spacing.*radius to at least 0\.00016"):
        design.length_scale_constraints(blur, min_width=R, min_spacing=2 * R)
    for bad in (0.0, -R, math.inf, math.nan):
        with pytest.raises(ValueError, match="min_width"):
            design.length_scale_constraints(blur, min_width=bad, min_spacing=R)
    with pytest.raises(TypeError, match="min_spacing"):
        design.length_scale_constraints(blur, min_width=R, min_spacing=True)
    with pytest.raises(ValueError, match="eta"):
        design.length_scale_constraints(blur, min_width=R, min_spacing=R, eta=1.0)


def test_lengths_that_look_like_micrometres_get_a_metres_hint(square):
    grid = design.LayerGrid(square, "design", pitch=R / 4)
    blur = design.helmholtz_filter(grid, radius=R)
    with pytest.raises(ValueError, match="lengths are in metres"):
        design.length_scale_constraints(blur, min_width=0.1, min_spacing=R)


def test_a_pitch_coarser_than_a_third_of_the_radius_is_refused(square):
    grid = design.LayerGrid(square, "design", pitch=R / 2)
    blur = design.helmholtz_filter(grid, radius=R)
    with pytest.raises(ValueError, match=r"pitch.*R/3|pitch.*radius / 3"):
        design.length_scale_constraints(blur, min_width=R, min_spacing=R)
    fine = design.LayerGrid(square, "design", pitch=R / 3)
    design.length_scale_constraints(
        design.helmholtz_filter(fine, radius=R), min_width=R, min_spacing=R
    )


def test_a_traced_rule_is_refused(square):
    grid = design.LayerGrid(square, "design", pitch=R / 4)
    blur = design.helmholtz_filter(grid, radius=R)

    def build(width):
        design.length_scale_constraints(blur, min_width=width, min_spacing=R)
        return width

    with pytest.raises(NonDifferentiableError):
        jax.grad(build)(R)


def test_the_call_checks_the_density(square):
    grid = design.LayerGrid(square, "design", pitch=R / 4)
    blur = design.helmholtz_filter(grid, radius=R)
    rules = design.length_scale_constraints(blur, min_width=R, min_spacing=R)
    with pytest.raises(ValueError, match="shape"):
        rules(jnp.zeros((3, 4)))
    with pytest.raises(ValueError, match="real"):
        rules(jnp.zeros(grid.shape, complex))


# --- radius for the robust triple ----------------------------------------------------


@pytest.mark.parametrize(
    ("delta", "ratio"),
    [
        (0.05, 3.708),
        (0.10, 2.499),
        (0.15, 1.934),
        (0.20, 1.577),
        (0.25, 1.315),
        (0.30, 1.105),
    ],
)
def test_radius_for_length_scale_values(delta, ratio):
    length = 150e-9
    radius = design.radius_for_length_scale(length, delta=delta)
    assert isinstance(radius, float)
    assert radius / length == pytest.approx(ratio, abs=1e-3)


def test_radius_for_length_scale_shrinks_as_the_tolerance_grows():
    radii = [
        design.radius_for_length_scale(1e-7, delta=d) for d in (0.05, 0.1, 0.2, 0.3)
    ]
    assert all(a > b for a, b in itertools.pairwise(radii))


@pytest.mark.parametrize("eta", [0.5, 0.4])
def test_radius_for_length_scale_round_trip(tmp_path_factory, eta):
    """At the returned radius, the bar whose eroded design just vanishes prints
    the length in the nominal design (and the slot whose dilated design just
    closes prints at least it)."""
    length, delta = 0.05 * MM, 0.15
    radius = design.radius_for_length_scale(length, delta=delta, eta=eta)
    side = 12 * radius / MM
    mesh = _plate(tmp_path_factory, "strip", (side, 2 * side / 480), (480, 2))
    grid = design.LayerGrid(mesh, "design", pitch=12 * radius / 480)
    blur = jax.jit(design.helmholtz_filter(grid, radius=radius))
    cx = _centre(grid)[0]

    def peak(width, solid):
        rho = _bar(grid, width, cx)
        out = np.asarray(blur(rho if solid else 1 - rho))[0]
        return out.max() if solid else out.min()

    def vanishing(solid, level):
        lo, hi = 1e-3 * radius, 3 * radius
        for _ in range(60):
            mid = (lo + hi) / 2
            if (peak(mid, solid) > level) == solid:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    printed = {}
    for solid, level in ((True, eta + delta), (False, eta - delta)):
        rho = _bar(grid, vanishing(solid, level), cx)
        profile = np.asarray(blur(rho if solid else 1 - rho))[0]
        printed[solid] = _printed_width(
            profile if solid else 1 - profile, grid.x, eta if solid else 1 - eta
        )
    assert min(printed.values()) == pytest.approx(length, rel=0.01)
    assert max(printed.values()) >= length * 0.99


def test_radius_for_length_scale_refusals():
    with pytest.raises(ValueError, match="delta"):
        design.radius_for_length_scale(1e-7, delta=0.0)
    with pytest.raises(ValueError, match="delta"):
        design.radius_for_length_scale(1e-7, delta=0.5)
    with pytest.raises(ValueError, match="delta"):
        design.radius_for_length_scale(1e-7, delta=0.25, eta=0.2)
    with pytest.raises(ValueError, match="length"):
        design.radius_for_length_scale(-1e-7, delta=0.1)
    with pytest.raises(TypeError, match="length"):
        design.radius_for_length_scale(True, delta=0.1)
    with pytest.raises(NonDifferentiableError):
        jax.grad(lambda d: design.radius_for_length_scale(1e-7, delta=d))(0.1)


def test_a_cross_section_grid_takes_the_pixel_array_not_its_layer_density():
    from pathlib import Path

    data = Path(__file__).parents[2] / "docs" / "tutorials" / "data"
    mesh = gfem.Mesh.from_gmsh(data / "line2d.msh", length_unit="mm")
    grid = design.LayerGrid(mesh, "design", pitch=1 * MM)
    blur = design.helmholtz_filter(grid, radius=3 * MM)
    rules = design.length_scale_constraints(blur, min_width=3 * MM, min_spacing=3 * MM)
    full = jnp.stack(rules(blur(jnp.ones(grid.shape))))
    np.testing.assert_allclose(full, [-1.0, -1.0], atol=1e-9)
    with pytest.raises(TypeError, match="pixels"):
        rules(grid.to_elements(jnp.ones(grid.shape)))
