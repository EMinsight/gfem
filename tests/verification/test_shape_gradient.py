"""Shape gradients: objectives differentiated with respect to node positions.

The node-coordinate cells of the verification matrix (electrostatic,
driven on lumped and wave ports, uniform and adaptive, eigenmode; both
solvers) are in test_gradient_matrix.py, on the designs of
`gradient_cases`. Here: shape gradients under jit and vmap, of a kept-field
functional, of a raw node displacement (no morph), and the raise when a
port's nodes move (a lumped port's; a wave port's face is differentiated).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from finite_differences import assert_gradient_matches, unit_directions
from gradient_cases import (
    WEIGHTS,
    capacitor,
    interface_move,
    shape_line,
    wave_line,
    wave_mesh,
)

import gfem
from gfem.shape import boundary_morph

STEP = 2e-6  # metres along a unit direction in t (layers are 0.5 mm)


@pytest.fixture(scope="module")
def capacitor_model(tmp_path_factory):
    return capacitor(tmp_path_factory.mktemp("plate"))


ELECTROSTATIC = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")


def test_shape_gradient_under_jit_and_vmap(capacitor_model):
    morph = boundary_morph(capacitor_model, "oxide__nitride", grows="oxide")
    options = gfem.SolverOptions(order=2, linear_solver="direct")

    def objective(t):
        model = capacitor_model.replace(mesh=morph(t))
        return (
            gfem.solve(model, ELECTROSTATIC, options=options).capacitance[0, 0] * 1e12
        )

    rng = np.random.default_rng(6)
    batch = jnp.asarray(rng.uniform(-1e-5, 1e-5, (2, morph.n_params)))
    loop = np.stack([np.asarray(jax.grad(objective)(t)) for t in batch])
    np.testing.assert_allclose(
        jax.jit(jax.grad(objective))(batch[0]), loop[0], rtol=1e-9
    )
    np.testing.assert_allclose(jax.vmap(jax.grad(objective))(batch), loop, rtol=1e-9)


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    return shape_line(tmp_path_factory.mktemp("line"))


GRID = tuple(float(f) for f in np.linspace(10e9, 30e9, 11))


def test_driven_kept_field_shape_gradient_matches_fd(line):
    # a field functional: one adjoint solve per frequency, then the same
    # pass; the layer interface moves along z, node by node (the side walls are
    # absorbing on every element of this coarse line, so a boundary_morph,
    # which holds the elements touching the truncation, would move nothing)
    problem = gfem.Driven(frequency=GRID[5], excite="p1", keep_fields=True)
    options = gfem.SolverOptions(order=2, linear_solver="direct")
    n, moved = interface_move(line)

    def objective(t):
        result = gfem.solve(line.replace(mesh=moved(t)), problem, options=options)
        e = result.e_field(GRID[5]).coefficients
        weights = np.random.default_rng(7).uniform(0.5, 1.5, e.shape[0])
        return jnp.sum(weights * jnp.abs(e) ** 2) * 1e-6

    t = jnp.zeros(n)
    rng = np.random.default_rng(3)
    assert_gradient_matches(objective, t, unit_directions(rng, t.shape, 2), step=STEP)


def test_node_gradient_of_a_raw_displacement_matches_fd(line):
    # every node but the port surfaces' moves (no morph): the gradient
    # with respect to model.mesh.nodes itself
    problem = gfem.Driven(frequency=GRID[5])
    options = gfem.SolverOptions(order=2, linear_solver="direct")
    mesh = line.mesh
    x0 = jnp.asarray(mesh.nodes)

    def objective(x):
        moved = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(mesh), [x])
        s = gfem.solve(line.replace(mesh=moved), problem, options=options).s.values
        return jnp.sum(WEIGHTS * jnp.abs(s[0]) ** 2)

    ports = np.isin(
        mesh.triangle_tags, [mesh.region(s).tag for s in ("port1", "port2")]
    )
    held = np.unique(mesh.triangles[ports])
    directions = []
    for d in unit_directions(np.random.default_rng(5), x0.shape, 2):
        d[held] = 0.0
        directions.append(d / np.linalg.norm(d))
    assert_gradient_matches(objective, x0, directions, step=STEP)


def test_moved_port_nodes_raise_port_geometry_error(line):
    problem = gfem.Driven(frequency=GRID[5])
    mesh = line.mesh
    node = int(mesh.triangles[mesh.triangle_tags == mesh.region("port1").tag][0, 0])

    def objective(dx):
        x = jnp.asarray(mesh.nodes).at[node, 0].add(dx)
        moved = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(mesh), [x])
        s = gfem.solve(line.replace(mesh=moved), problem).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    # the port's own geometry is frozen in shape gradients: moving it raises
    with pytest.raises(jax.errors.JaxRuntimeError, match=r"PortGeometryError.*port1"):
        jax.grad(objective)(1e-6)


# --- wave port faces ---------------------------------------------------------------


def test_moved_wave_port_face_nodes_are_differentiated(tmp_path):
    # the port's face nodes move in its plane: the modes follow, and S
    # differentiates in every node coordinate
    line = wave_line(wave_mesh(tmp_path), "lossless")
    problem = gfem.Driven(frequency=12e9)
    mesh = line.mesh
    face = np.unique(mesh.triangles[mesh.triangle_tags == mesh.region("port2").tag])
    x0 = jnp.asarray(mesh.nodes)

    def objective(t):
        x = x0.at[face, 1].add(t)
        moved = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(mesh), [x])
        s = gfem.solve(line.replace(mesh=moved), problem).s.values
        return jnp.sum(WEIGHTS * jnp.abs(s[0]) ** 2)

    t = jnp.zeros(len(face))
    rng = np.random.default_rng(4)
    assert_gradient_matches(objective, t, unit_directions(rng, t.shape, 2), step=STEP)


def test_moved_lumped_port_nodes_still_raise(tmp_path):
    line = wave_line(wave_mesh(tmp_path), "mixed")
    problem = gfem.Driven(frequency=12e9)
    mesh = line.mesh
    node = int(mesh.triangles[mesh.triangle_tags == mesh.region("port2").tag][0, 0])

    def objective(dy):
        x = jnp.asarray(mesh.nodes).at[node, 1].add(dy)
        moved = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(mesh), [x])
        s = gfem.solve(line.replace(mesh=moved), problem).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    with pytest.raises(jax.errors.JaxRuntimeError, match=r"PortGeometryError.*p2"):
        jax.grad(objective)(1e-6)


# the face-node split (WR-90 and a lossy slab guide, both faces move in plane)

DATA = Path(__file__).parents[1] / "unit" / "data"
SPLIT_OPTIONS = gfem.SolverOptions(
    order=2, linear_solver="direct", port_mode_tolerance=1e-12, retain_memory=0
)


class _Split:
    """S21 of a guide whose face nodes move, with the 3D and the sections apart.

    ``s21(nodes, sections)`` solves on ``nodes`` with the port sections cut from
    ``sections`` (the same by default): the difference with one frozen is
    the 3D part (sections frozen) or the section part (3D frozen).
    """

    def __init__(self, model, frequency):
        self.model, self.frequency = model, frequency
        self.x0 = np.asarray(model.mesh.nodes, float)
        self.sections = None

    def with_nodes(self, nodes, model=None):
        model = self.model if model is None else model
        mesh = model.mesh
        return model.replace(
            mesh=jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(mesh), [nodes]
            )
        )

    def s21(self, nodes, sections=None):
        self.sections = sections
        try:
            result = gfem.solve(
                self.with_nodes(nodes),
                gfem.Driven(self.frequency),
                options=SPLIT_OPTIONS,
            )
            return complex(np.asarray(result.s.values)[0, 1, 0])
        finally:
            self.sections = None

    def hook(self, monkeypatch):
        from gfem._engine import port_term, wave_ports

        original = wave_ports.port_section

        def section(model, port, stretch=None):
            if self.sections is not None:
                model = self.with_nodes(self.sections, model)
            return original(model, port, stretch)

        for module in (wave_ports, port_term):
            monkeypatch.setattr(module, "port_section", section)

    def total_gradient(self, direction, monkeypatch):
        """The analytic gradient along ``direction`` and its port pass part ``B``.

        ``B`` is the port pass's node cotangent without the Robin term; the rest
        (``A``) is the volume pass plus the Robin term.
        """
        from gfem._engine.driven import DrivenRule

        reports = []
        original = DrivenRule._add_port_nodes

        def spy(rule, found):
            reports.append(found.report)
            return original(rule, found)

        monkeypatch.setattr(DrivenRule, "_add_port_nodes", spy)
        x0 = jnp.asarray(self.x0)

        out = []
        for which in (0, 1):
            reports.clear()

            def f(s, which=which):
                result = gfem.solve(
                    self.with_nodes(x0 + s * jnp.asarray(direction)),
                    gfem.Driven(self.frequency),
                    options=SPLIT_OPTIONS,
                )
                v = result.s.values[0, 1, 0]
                return jnp.real(v) if which == 0 else jnp.imag(v)

            total = float(jax.grad(f)(0.0))
            rest = float(np.sum(reports[-1]["rest"] * direction))
            out.append((total, rest))
        return complex(out[0][0], out[1][0]), complex(out[0][1], out[1][1])


def _central4(f, h):
    return (8 * (f(h) - f(-h)) - (f(2 * h) - f(-2 * h))) / (12 * h)


def _check_split(split, direction, monkeypatch, h, *, pinned=None):
    split.hook(monkeypatch)
    x0 = split.x0
    d = np.asarray(direction)
    total, rest = split.total_gradient(d, monkeypatch)
    fd = _central4(lambda s: split.s21(x0 + s * d), h)
    a_fd = _central4(lambda s: split.s21(x0 + s * d, x0), h)
    b_fd = _central4(lambda s: split.s21(x0, x0 + s * d), h)
    assert abs(total - fd) <= 1e-6 * abs(fd), (total, fd)
    # the parts are O(1) against a sum that is 4e-4 of either: each is gated
    assert abs((total - rest) - a_fd) <= 1e-6 * abs(a_fd), (total - rest, a_fd)
    assert abs(rest - b_fd) <= 1e-6 * abs(b_fd), (rest, b_fd)
    assert abs(a_fd + b_fd - fd) <= 1e-6 * abs(fd)
    if pinned is not None:
        assert abs(fd - pinned) <= 1e-8, (fd, pinned)


def test_the_face_node_gradient_splits_into_the_3d_and_the_section_parts_wr90(
    monkeypatch,
):
    mesh = gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"air": gfem.Material()},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    split = _Split(model, 10e9)
    stretch = np.zeros_like(split.x0)
    stretch[:, 0] = split.x0[:, 0]  # x -> x (1 + s): both faces widen in plane
    _check_split(
        split,
        stretch,
        monkeypatch,
        1e-4,
        pinned=-0.58178209605 - 0.41674585372j,
    )


def test_the_face_node_gradient_splits_on_a_lossy_slab_guide(monkeypatch):
    mesh = gfem.Mesh.from_gmsh(DATA / "slab_waveguide.msh", length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={
            "slab": gfem.Material(eps_r=4.0, tan_delta=0.05),
            "air": gfem.Material(),
        },
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    split = _Split(model, 12e9)
    interface = np.zeros_like(split.x0)
    interface[np.isclose(split.x0[:, 0], 8e-3, atol=1e-12), 0] = 1.0  # both faces too
    _check_split(split, interface, monkeypatch, 1e-7)
