"""Sources and their measurements across ranks: one process's values.

The dielectric cube of `free_space_cases.scatterer_box` (an air box in a
perfectly matched layer, a lumped port on the cube's +x face), driven by a
plane wave and an electric dipole, on `RANKS` ranks against pytest's one
process:

- each rank assembles its own part of a plane wave's load, and the dipole's
  load sits on the one rank holding its element, here not rank 0;
- the collective measurements of the source columns (S, the port waves, the
  emitted power, the Purcell factor, the kept fields and their scattered and
  incident parts at points) equal one process's to `GATE`;
- the moment interpolant of an incident field is rank-local: a field's
  coefficients on each rank, read back at points, are one process's to
  `EXACT_GATE`, with and without an element set whose edge crosses the
  partition;
- the Purcell factor's gradient by the cube's permittivity is one process's;
- on a finer box, whose truncation layer is background, the cross sections of
  the plane wave, `gfem.power_flow` (the corrected one of the wave's total
  field, the plain one of the port's field), the far fields of both columns
  (amplitudes and radiated power) and their couplings into a wide Gaussian
  beam and into a plane wave given as an `gfem.IncidentField` (interpolated
  on each rank's piece) are one process's to `GATE`;
- a lossy particle on a film over a substrate, lit from the top and from
  the substrate past the critical angle: each rank evaluates the stack's
  field on its own piece, and the kept fields, their parts, the cross
  sections and the corrected power flow are one process's to `GATE`;
- ranks given different backgrounds stop at the entry check, naming it.
"""

import warnings

import free_space_cases as fs
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ranks import on_ranks

import gfem
from gfem import _mpi, _petsc

#: NGSolve's rank 0 holds no element: two more ranks share a partition face.
RANKS = 3
#: Relative to the largest one-process value: direct solves of one system
#: split over ranks (the measurements' gate).
GATE = 1e-10
#: The interpolant involves no solve: round-off of the same arithmetic.
EXACT_GATE = 1e-12
#: Relative to the largest one-process entry of the gradient.
GRADIENT_GATE = 1e-9
F = fs.F0 / 2
GRID = (F, 1.1 * F)
WAVE = gfem.PlaneWave("w", theta=0.4, phi=0.3, polarization="TM")
MOMENT = (0.0, 1e-28, 0.3e-28)
#: Dipole positions in the air, off every face of the Kuhn grid; the first
#: one whose element lies on one rank alone, not rank 0, is used.
CANDIDATES = (
    fs.DIPOLE_POSITION,
    (0.87e-6, 0.59e-6, 0.13e-6),
    (0.13e-6, 0.59e-6, 0.13e-6),
    (0.87e-6, 0.41e-6, 0.87e-6),
    (0.59e-6, 0.13e-6, 0.87e-6),
    (0.41e-6, 0.87e-6, 0.13e-6),
    (0.13e-6, 0.13e-6, 0.41e-6),
    (0.87e-6, 0.87e-6, 0.59e-6),
)

needs_stack = pytest.mark.skipif(not _petsc.has_mpi(), reason="needs the MPI stack")


def _options():
    return gfem.SolverOptions(order=2, linear_solver="direct", threads=1)


def _world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def _points(mesh, n=12, seed=3):
    """Centroids of tets outside the layer, off every face."""
    tets = np.concatenate([mesh.elements("air"), mesh.elements("scat")])
    chosen = np.random.default_rng(seed).choice(tets, n, replace=False)
    return np.asarray(mesh.nodes)[np.asarray(mesh.tetrahedra)[chosen]].mean(axis=1)


def _holders(model, position, comm):
    """The ranks whose local mesh holds the dipole's element."""
    from gfem._engine.cache import discretization
    from gfem._engine.probe import Layout, locate

    layout = Layout(("air", "pml", "scat"), (), "hcurl", 2)
    tet = int(locate(model.mesh, layout, np.array([position]), None).tetrahedra[0])
    built = discretization(model.mesh, (), comm).built
    held = bool(np.isin(tet, np.asarray(built.cell_order)))
    return [r for r, h in enumerate(comm.allgather(held)) if h]


def _off_root(model, comm):
    """The first candidate whose element one rank alone holds, not rank 0."""
    for position in CANDIDATES:
        holders = _holders(model, position, comm)
        if len(holders) == 1 and holders[0] != 0:
            return position, holders
    raise AssertionError("no candidate position lies on one rank alone")


def _measurements(model, position):
    """Every collective value of the two source columns, on this process."""
    dipole = gfem.ElectricDipole("d", position=position, moment=MOMENT)
    problem = gfem.Driven(GRID, sources=(WAVE, dipole), keep_fields=True)
    result = gfem.solve(model, problem, options=_options())
    points = _points(model.mesh)
    fields = {
        f"{label} {part} {k}": np.asarray(
            result.e_field(f, label, part=part).at(points)
        )
        for k, f in enumerate(GRID)
        for label, parts in (
            ("w", ("total", "scattered", "incident")),
            ("d", ("total",)),
        )
        for part in parts
    }
    return {
        "s": np.asarray(result.s.values),
        "port waves": np.asarray(result.port_waves),
        "emitted power": np.asarray(result.emitted_power["d"]),
        "purcell factor": np.asarray(gfem.purcell_factor(result, "d", model=model)),
        **fields,
    }


def _on_ranks_measurements(path):
    comm = _world()
    model = fs.scatterer_model(path)
    position, holders = _off_root(model, comm)
    return {"position": position, "holders": holders} | _measurements(model, position)


@pytest.fixture
def box(tmp_path):
    """The scatterer box, written once, before the ranks read it."""
    return str(fs.scatterer_box(tmp_path / "box.msh"))


def _relative(a, b):
    return np.abs(np.asarray(a) - b).max() / np.abs(b).max()


@needs_stack
def test_source_measurements_across_ranks_are_one_process_s(box):
    got, *_ = on_ranks(_on_ranks_measurements, box, ranks=RANKS, limit=300)
    # the dipole's load is assembled on one rank alone, not rank 0
    assert len(got["holders"]) == 1
    assert got["holders"][0] != 0
    one = _measurements(fs.scatterer_model(box), got["position"])
    for name, value in one.items():
        error = _relative(got[name], value)
        print(f"ranks={RANKS} {name} error={error:.3g}")  # noqa: T201
        assert error < GATE, f"{name} off by {error:.3g} relative"


# --- the moment interpolant -----------------------------------------------------------


def _interpolated(path, masked, comm=None):
    """A plane wave's moment interpolant as a field, read at points."""
    from gfem._engine import fields as engine_fields
    from gfem._engine.moments import field_interpolant
    from gfem._incident import values

    mesh = gfem.Mesh.from_gmsh(path)
    support, order = ("air", "pml", "scat"), 2
    # the elements below a slanted plane: its edge crosses any partition
    centroids = np.asarray(mesh.nodes)[np.asarray(mesh.tetrahedra)].mean(axis=1)
    elements = centroids @ np.array([1.0, 0.6, 0.3]) < 1.0e-6 if masked else None
    interpolant = field_interpolant(
        mesh, support, (), order, elements=elements, comm=comm
    )
    coefficients = interpolant(
        values(WAVE, gfem.Background([gfem.Material()]), interpolant.points, F)
    )
    width, ndof = engine_fields.dof_shape(mesh, support, (), "hcurl", order, comm=comm)
    piece = np.zeros(width, complex)
    piece[:ndof] = coefficients
    field = gfem.Field(
        coefficients=jnp.asarray(piece),
        mesh=mesh,
        quantity="E",
        unit="V/m",
        space="hcurl",
        order=order,
        support=support,
        omega=jnp.asarray(2 * np.pi * F),
        _comm=_mpi.static_comm(comm),
    )
    return np.asarray(field.at(_points(mesh)))


def _on_ranks_interpolated(path, masked):
    return _interpolated(path, masked, _world())


@pytest.mark.parametrize("masked", [False, True], ids=["domain", "element set"])
def test_the_moment_interpolant_is_one_process_s_across_ranks(box, masked):
    got, *_ = on_ranks(_on_ranks_interpolated, box, masked, ranks=RANKS, limit=120)
    one = _interpolated(box, masked)
    error = _relative(got, one)
    print(f"ranks={RANKS} masked={masked} interpolant error={error:.3g}")  # noqa: T201
    assert error < EXACT_GATE, f"off by {error:.3g} relative"


# --- the Purcell factor's gradient ----------------------------------------------------


def _purcell_gradient(path, position):
    model = fs.scatterer_model(path, field=True)
    dipole = gfem.ElectricDipole("d", position=position, moment=MOMENT)
    problem = gfem.Driven(F, excite=(), sources=dipole)

    def purcell(eps):
        moved = model.with_material("scat", gfem.MaterialField(eps_r=eps))
        result = gfem.solve(moved, problem, options=_options())
        return gfem.purcell_factor(result, "d", model=moved)[0]

    eps = jnp.asarray(model.materials["scat"].eps_r)
    value, grad = jax.value_and_grad(purcell)(eps)
    return np.asarray(value), np.asarray(grad)


def _on_ranks_purcell_gradient(path):
    position, holders = _off_root(fs.scatterer_model(path), _world())
    return position, holders, *_purcell_gradient(path, position)


@needs_stack
def test_the_purcell_gradient_across_ranks_is_one_process_s(box):
    (position, holders, value, grad), *_ = on_ranks(
        _on_ranks_purcell_gradient, box, ranks=RANKS, limit=600
    )
    assert len(holders) == 1
    assert holders[0] != 0
    value_one, grad_one = _purcell_gradient(box, position)
    assert np.abs(grad_one).max() > 0
    assert _relative(value, value_one) < GATE
    error = _relative(grad, grad_one)
    print(f"ranks={RANKS} Purcell gradient error={error:.3g}")  # noqa: T201
    assert error < GRADIENT_GATE, f"the gradient off by {error:.3g} relative"


# --- the power balance ----------------------------------------------------------------


def _inner(centroids):
    """The elements inside the truncation layer of the finer box: cube and port."""
    return ((centroids > 0.125e-6) & (centroids < 0.875e-6)).all(axis=1)


#: Beams coming down onto the box: wider than it (a clipped beam warns).
BEAM = gfem.GaussianBeam(
    "beam", waist=3e-6, focus=(0.5e-6, 0.5e-6, 1e-6), theta=np.pi, polarization="TM"
)


def _down_wave(points, frequency):
    """A plane wave coming down along -z, along y, 1 V/m."""
    k = 2 * np.pi * frequency / 299792458.0
    e = np.zeros((len(points), 3), complex)
    e[:, 1] = np.exp(1j * k * points[:, 2])
    return e


#: Far-field directions, rad: a few off the axes.
THETA = np.array([0.3, 1.2, 2.5])
PHI = np.array([0.1, 2.0, 4.0])


def _balance(path):
    """The cross sections, the power out of `_inner` and the far fields."""
    model = fs.scatterer_model(path)
    problem = gfem.Driven(GRID, sources=WAVE, keep_fields=True)
    result = gfem.solve(model, problem, options=_options())
    xs = gfem.cross_sections(result, "w", model=model)
    out = {
        "extinction": np.asarray(xs.extinction),
        "absorption": np.asarray(xs.absorption),
        "scattering": np.asarray(xs.scattering),
    }
    for k, f in enumerate(GRID):
        for label in ("w", "p1"):
            field = result.e_field(f, label)
            power = gfem.power_flow(field, _inner, model=model)
            out[f"power_flow {label} {k}"] = np.asarray(power)
    for label in ("w", "p1"):
        ff = gfem.far_field(result, label, THETA, PHI, model=model)
        out[f"far_field {label} e_theta"] = np.asarray(ff.e_theta)
        out[f"far_field {label} e_phi"] = np.asarray(ff.e_phi)
        out[f"far_field {label} radiated power"] = np.asarray(ff.radiated_power)
        for beam in (BEAM, gfem.IncidentField("down", _down_wave)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # clipped by the box
                c = gfem.beam_coupling(result, label, beam, model=model)
            out[f"beam_coupling {label} {beam.name}"] = np.asarray(c)
    return out


@pytest.fixture
def fine_box(tmp_path):
    """The scatterer box at 8 cells per um, a two-cell layer."""
    return str(fs.scatterer_box(tmp_path / "fine.msh", cells=8, thickness=0.25))


@needs_stack
def test_balance_and_far_field_across_ranks_are_one_process_s(fine_box):
    got, *_ = on_ranks(_balance, fine_box, ranks=RANKS, limit=600)
    one = _balance(fine_box)
    assert np.abs(one["absorption"]).min() > 0
    for name, value in one.items():
        error = _relative(got[name], value)
        print(f"ranks={RANKS} {name} error={error:.3g}")  # noqa: T201
        assert error < GATE, f"{name} off by {error:.3g} relative"


# --- a layered background -----------------------------------------------------------

SUB = gfem.Material(eps_r=2.25)
FILM = gfem.Material(eps_r=4.0)
AIR = gfem.Material()
#: Substrate below z = 1/4, film to z = 1/2, air above (um): grid planes.
STACK = gfem.Background([SUB, FILM, AIR], interfaces=[0.25e-6, 0.5e-6])
#: From the top at 40 degrees, and from the substrate at 50 degrees, past the
#: substrate/air critical angle: the particle sits in the evanescent tail.
LAYERED_WAVES = (
    gfem.PlaneWave("top", theta=np.pi - np.radians(40), polarization="TE"),
    gfem.PlaneWave("tir", theta=np.radians(50), phi=0.3, polarization="TM"),
)


def _stack_layer(z):
    return "sub" if z < 0.25 else "film" if z < 0.5 else "air"


def _layered_volume(c):
    """A particle on the film, the stack around it, a layer split at its planes."""
    if not ((c > 0) & (c < 1)).all():
        return f"pml_{_stack_layer(c[2])}"
    if (c[:2] > 0.25).all() and (c[:2] < 0.75).all() and 0.5 < c[2] < 0.75:
        return "particle"
    return _stack_layer(c[2])


def _layered_model(path):
    materials = {"particle": gfem.Material(eps_r=4.0, tan_delta=0.25)}
    for name, material in (("sub", SUB), ("film", FILM), ("air", AIR)):
        materials[name] = materials[f"pml_{name}"] = material
    return gfem.Model(
        gfem.Mesh.from_gmsh(path),
        materials=materials,
        pml=gfem.PML(["pml_sub", "pml_film", "pml_air"]),
        background=STACK,
    )


def _layered(path):
    """The kept fields at points, the cross sections and the absorbed power."""
    model = _layered_model(path)
    problem = gfem.Driven(F, sources=LAYERED_WAVES, keep_fields=True)
    result = gfem.solve(model, problem, options=_options())
    mesh = model.mesh
    tets = np.concatenate([mesh.elements(n) for n in ("sub", "film", "particle")])
    chosen = np.random.default_rng(5).choice(tets, 12, replace=False)
    points = np.asarray(mesh.nodes)[np.asarray(mesh.tetrahedra)[chosen]].mean(axis=1)
    out = {}
    for wave in LAYERED_WAVES:
        for part in ("total", "scattered", "incident"):
            field = result.e_field(F, wave.name, part=part)
            out[f"{wave.name} {part}"] = np.asarray(field.at(points))
        xs = gfem.cross_sections(result, wave.name, model=model)
        for name in ("extinction", "absorption", "scattering"):
            out[f"{wave.name} {name}"] = np.asarray(getattr(xs, name))
        total = result.e_field(F, wave.name)
        absorbed = -gfem.power_flow(total, "particle", model=model)
        out[f"{wave.name} absorbed"] = np.asarray(absorbed)
    return out


@pytest.fixture
def layered_box(tmp_path):
    """The stacked box at 4 cells per um, a one-cell layer split at the planes."""
    import pml_cases as pc

    path = pc.box(
        tmp_path / "layered.msh",
        ("-x", "+x", "-y", "+y", "-z", "+z"),
        thickness=0.25,
        cells=4,
        volume=_layered_volume,
    )
    return str(path)


@needs_stack
def test_a_layered_plane_wave_across_ranks_is_one_process_s(layered_box):
    got, *_ = on_ranks(_layered, layered_box, ranks=RANKS, limit=600)
    one = _layered(layered_box)
    assert (
        min(abs(float(np.ravel(one[f"{w.name} absorption"])[0])) for w in LAYERED_WAVES)
        > 0
    )
    for name, value in one.items():
        error = _relative(got[name], value)
        print(f"ranks={RANKS} layered {name} error={error:.3g}")  # noqa: T201
        assert error < GATE, f"{name} off by {error:.3g} relative"


# --- the entry check ------------------------------------------------------------------


def _different_backgrounds(path):
    comm = _world()
    model = fs.scatterer_model(path)
    model = model.replace(background=gfem.Material(eps_r=1.0 + 0.5 * comm.rank))
    try:
        gfem.solve(model, gfem.Driven(F, excite=(), sources=WAVE), options=_options())
    except _mpi.AgreedError as err:
        return str(err)
    raise AssertionError("ranks with different backgrounds solved")


def test_ranks_with_different_backgrounds_are_told_at_the_entry(box):
    messages = on_ranks(_different_backgrounds, box, ranks=RANKS, limit=120)
    assert all("the background" in m for m in messages), messages
