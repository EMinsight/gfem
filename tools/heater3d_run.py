"""Run the 3D spiral heater and write its run record.

Usage: python tools/heater3d_run.py STAGE --out DIR [--profile full|smoke]
[--record DIR]

A TiN heater over a racetrack double spiral of 220 nm silicon waveguide
(tools/heater_geometry.py), meshed with gmsh (OCC kernel, graded
tetrahedra, the heater one element thick, the cores meshed) and solved at
order 2 with the direct solver: the DC resistance, the steady temperature
per watt, the thermo-optic phase of the whole route and of every pass, a
2D cross-section of the same stack, a step response at the P_pi power, and
a three-level mesh ladder. The docs page ``docs/tutorials/heater3d.ipynb``
reads the record and solves nothing; it refuses a record whose script
hashes differ from these files', so editing one forces a rerun.
``just heater3d-data`` runs, in ``uv run --with gmsh==4.15.2 --group docs``::

    python tools/heater3d_run.py all --out WORK
    python tools/heater3d_run.py merge --out WORK --record docs/tutorials/data

Stages, each in its own process, saving to ``DIR/stages``:

- ``main``: the converged mesh (heater 0.2 um, cores 0.3 um): resistance,
  steady rise per watt, route and pass phases, power balance, temperature
  maps, the 2D section, and the step response at P_pi (about 3 GiB).
- ``ladder``: the same steady solve on two finer meshes (about 0.8 M and
  1.85 M DOF; the finest needs about 14 GiB), each level its own process.
- ``merge``: the run record, ``heater3d.json`` (provenance, settings,
  observations) and ``heater3d.npz`` (every solver output), into ``--record``
  (default ``DIR``). It refuses a record over 3 MB, or one whose stages ran
  at another commit or with other scripts, or with ``src/`` or ``tools/``
  uncommitted. Stages may run on different machines: each keeps its own.

``--profile smoke`` meshes at a few um and steps briefly: a complete record
in minutes, for development only.

The commit comes from ``git`` in a checkout, else from ``GFEM_RUN_COMMIT``
(a ``git archive`` export); the host label from ``GFEM_RUN_HOST``.
"""

import argparse
import dataclasses
import io
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import bend3d_run as provenance
import heater_geometry as hg
import numpy as np
from scipy.constants import c

import gfem

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("tools/heater3d_run.py", "tools/heater_geometry.py")

um = 1e-6
WAVELENGTH = 1.55 * um
FREQUENCY = c / WAVELENGTH
SPIRAL = hg.Spiral()
T_SINK = 293.15  # K, the substrate's underside

# --- the stack ---------------------------------------------------------------

#: z of each layer (um): the core's underside is z = 0.
SUBSTRATE = (-13.0, -3.0)
BOX = (-3.0, 0.0)
CLADDING = (0.0, 3.0)
MARGIN = 15.0  # um, oxide and substrate around the footprint
#: Material data (SI), from gsim's validation cases: thermal conductivity,
#: specific heat, mass density, thermo-optic coefficient, conductivity.
SILICON = {
    "thermal_conductivity": 148.0,
    "specific_heat": 711.0,
    "mass_density": 2330.0,
}
FILM = {**SILICON, "thermal_conductivity": 90.0}  # the 220 nm film
OXIDE = {"thermal_conductivity": 1.38, "specific_heat": 709.0, "mass_density": 2203.0}
TIN = {
    "thermal_conductivity": 28.0,
    "specific_heat": 598.0,
    "mass_density": 5240.0,
    "conductivity": 2.3e6,
}
N_SI, N_OX = 3.4777, 1.444
DNDT_SI, DNDT_OX = 1.86e-4, 1e-5


def materials():
    """The thermal model's materials by region (the 3D device)."""
    return {
        "substrate": gfem.Material(**SILICON),
        "box": gfem.Material(**OXIDE),
        "cladding": gfem.Material(**OXIDE),
        "core": gfem.Material(**FILM),
        "heater": gfem.Material(**TIN),
        "pad": gfem.Material(**TIN),
    }


def model(mesh):
    """The device: every region conducts heat, the substrate's underside sinks it.

    Every other outer face is insulated. The heater and pads conduct
    current; the DC terminals are the pads' top faces.
    """
    return gfem.Model(
        mesh,
        materials={k: v for k, v in materials().items() if k in mesh.volumes},
        boundaries=[gfem.FixedTemperature("sink", temperature=T_SINK)],
    )


def window_model(mesh):
    """The optical window: one isolated core in oxide, a wave port on its section."""
    return gfem.Model(
        mesh,
        materials={
            "core": gfem.Material(eps_r=N_SI**2, thermo_optic_coefficient=DNDT_SI),
            "cladding": gfem.Material(eps_r=N_OX**2, thermo_optic_coefficient=DNDT_OX),
        },
        ports=[gfem.WavePort("core", "front")],
    )


def routes(spiral=SPIRAL):
    """The phase routes (m): the whole spiral and each pass, at the core's underside."""
    line = hg.centreline(spiral, max_step=0.1)
    out = {"spiral": np.c_[line, np.zeros(len(line))] * um}
    for name, p in hg.passes(spiral).items():
        out[name] = np.array([[*p.start, 0.0], [*p.end, 0.0]]) * um
    return out


def phase_route(window, rise, spiral=SPIRAL):
    """`gfem.route_phase` of every route, at 1.55 um, up = +z."""
    return gfem.route_phase(
        window, rise, routes=routes(spiral), port="core", frequency=FREQUENCY
    )


# --- profiles ----------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Profile:
    """The size of a run.

    Attributes:
        levels: The ladder's meshes, (heater size, core size) in um; the
            first is ``main``'s.
        far: The largest element (um), far from the heater and cores.
        on: The step response's heating time (s), from t = 0.
        off: Its cooling time (s), after the switch-off.
        steps: The time steps over ``on + off``.
        map_step: The temperature maps' sample spacing (um).
    """

    levels: tuple[tuple[float, float], ...]
    far: float
    on: float
    off: float
    steps: int
    map_step: float


PROFILES = {
    "full": Profile(
        levels=((0.2, 0.3), (0.15, 0.2), (0.1, 0.15)),
        far=4.0,
        on=100e-6,
        off=100e-6,
        steps=400,
        map_step=0.2,
    ),
    "smoke": Profile(
        levels=((1.0, 1.0), (0.8, 0.8), (0.7, 0.7)),
        far=6.0,
        on=100e-6,
        off=100e-6,
        steps=200,
        map_step=1.0,
    ),
}
LADDER = ("medium", "fine")
STAGES = ("main", *LADDER)
SOLVER = gfem.SolverOptions(order=2, linear_solver="direct")

# --- meshes ------------------------------------------------------------------


def footprint(spiral=SPIRAL):
    """``(x0, y0, x1, y1)`` um of the core, heater and pads together."""
    parts = [
        hg.core_polygon(spiral),
        hg.heater_polygon(spiral),
        *hg.pad_polygons(spiral),
    ]
    stacked = np.vstack(parts)
    return (*stacked.min(0), *stacked.max(0))


def _gmsh(threads):
    import gmsh

    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", threads)
    return gmsh


def _threads():
    affinity = getattr(os, "sched_getaffinity", None)
    return len(affinity(0)) if affinity else os.cpu_count() or 1


def mesh_device(path, hh, hc, far, spiral=SPIRAL):
    """Write the device's tetrahedra (gmsh 2.2, um) to ``path``; tet count.

    Boxes for the substrate, buried oxide and cladding over the footprint
    plus `MARGIN`; prisms of the heater, pads (``heater_thickness`` thick)
    and core (``core_thickness``) from their footprints; fragmented and
    named. Size: ``hh`` at the heater's surfaces growing to ``far`` over 8
    um, ``hc`` at the core's growing to ``far`` over 4 um (their minimum).
    """
    gmsh = _gmsh(_threads())
    occ = gmsh.model.occ
    x0, y0, x1, y1 = footprint(spiral)
    X0, Y0, X1, Y1 = x0 - MARGIN, y0 - MARGIN, x1 + MARGIN, y1 + MARGIN

    def layer(z):
        return occ.addBox(X0, Y0, z[0], X1 - X0, Y1 - Y0, z[1] - z[0])

    def prism(points, z0, thickness):
        tags = [occ.addPoint(x, y, z0) for x, y in points]
        lines = [
            occ.addLine(tags[i], tags[(i + 1) % len(tags)]) for i in range(len(tags))
        ]
        face = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        return next(e for d, e in occ.extrude([(2, face)], 0, 0, thickness) if d == 3)

    sub, box, clad = layer(SUBSTRATE), layer(BOX), layer(CLADDING)
    zh, th = spiral.heater_z, spiral.heater_thickness
    heater = prism(hg.heater_polygon(spiral), zh, th)
    pads = [prism(p, zh, th) for p in hg.pad_polygons(spiral)]
    core = prism(hg.core_polygon(spiral), 0.0, spiral.core_thickness)
    objects = [sub, box, clad, heater, *pads, core]
    _, mapping = occ.fragment(
        [(3, sub), (3, box), (3, clad)], [(3, t) for t in objects[3:]]
    )
    occ.synchronize()
    pieces = {t: [e for _, e in m] for t, m in zip(objects, mapping, strict=True)}
    owned = (
        set(pieces[heater]) | {e for p in pads for e in pieces[p]} | set(pieces[core])
    )
    groups = {
        "substrate": pieces[sub],
        "box": [e for e in pieces[box] if e not in owned],
        "cladding": [e for e in pieces[clad] if e not in owned],
        "core": pieces[core],
        "heater": pieces[heater],
        "pad": [e for p in pads for e in pieces[p]],
    }
    for tag, (name, entities) in enumerate(groups.items(), start=1):
        gmsh.model.addPhysicalGroup(3, entities, tag, name)

    def faces(entities):
        return [
            f
            for _, f in gmsh.model.getBoundary(
                [(3, e) for e in entities], oriented=False
            )
        ]

    def flat(entities, z):
        return [
            f
            for f in faces(entities)
            if np.allclose(
                np.array(gmsh.model.getBoundingBox(2, f))[[2, 5]], z, atol=1e-6
            )
        ]

    gmsh.model.addPhysicalGroup(2, flat(groups["substrate"], SUBSTRATE[0]), 101, "sink")
    for k, p in enumerate(pads):
        gmsh.model.addPhysicalGroup(
            2, flat(pieces[p], zh + th), 102 + k, f"contact_{k}"
        )
    field = gmsh.model.mesh.field
    sizes = []
    for entities, size, reach in (
        (groups["heater"] + groups["pad"], hh, 8.0),
        (groups["core"], hc, 4.0),
    ):
        distance = field.add("Distance")
        field.setNumbers(distance, "SurfacesList", faces(entities))
        field.setNumber(distance, "Sampling", 20)
        threshold = field.add("Threshold")
        field.setNumber(threshold, "InField", distance)
        for key, value in (
            ("SizeMin", size),
            ("SizeMax", far),
            ("DistMin", 0.0),
            ("DistMax", reach),
        ):
            field.setNumber(threshold, key, value)
        sizes.append(threshold)
    smallest = field.add("Min")
    field.setNumbers(smallest, "FieldsList", sizes)
    field.setAsBackgroundMesh(smallest)
    for key in (
        "MeshSizeExtendFromBoundary",
        "MeshSizeFromPoints",
        "MeshSizeFromCurvature",
    ):
        gmsh.option.setNumber(f"Mesh.{key}", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write(str(path))
    _, tets, _ = gmsh.model.mesh.getElements(3)
    gmsh.finalize()
    return int(sum(len(t) for t in tets))


def mesh_section(path, rectangles, sink=None, size=0.04, far=1.0):
    """Write a 2D cross-section (triangles, gmsh 2.2, um) to ``path``.

    ``rectangles``: ``(x0, y0, x1, y1, name, size or None)``, later ones
    cut out of earlier ones. ``sink``: the y of a bottom edge named
    ``sink``. Size ``size`` at the small rectangles' edges, growing to
    ``far``.
    """
    gmsh = _gmsh(1)
    occ = gmsh.model.occ
    tags = [
        occ.addRectangle(x0, y0, 0, x1 - x0, y1 - y0)
        for x0, y0, x1, y1, *_ in rectangles
    ]
    _, mapping = occ.fragment([(2, tags[0])], [(2, t) for t in tags[1:]])
    occ.synchronize()
    taken, groups = set(), {}
    for (*_, name, _), pieces in reversed(list(zip(rectangles, mapping, strict=True))):
        own = [e for _, e in pieces if e not in taken]
        taken |= set(own)
        groups.setdefault(name, []).extend(own)
    for tag, (name, entities) in enumerate(groups.items(), start=1):
        gmsh.model.addPhysicalGroup(2, entities, tag, name)
    if sink is not None:
        edges = [
            e
            for _, e in gmsh.model.getEntities(1)
            if np.allclose(np.array(gmsh.model.getBoundingBox(1, e))[[1, 4]], sink)
        ]
        gmsh.model.addPhysicalGroup(1, edges, 101, "sink")
    small = [
        e
        for rect, pieces in zip(rectangles, mapping, strict=True)
        if rect[5] is not None
        for _, e in pieces
    ]
    edges = [
        e for _, e in gmsh.model.getBoundary([(2, s) for s in small], oriented=False)
    ]
    field = gmsh.model.mesh.field
    distance = field.add("Distance")
    field.setNumbers(distance, "CurvesList", edges)
    threshold = field.add("Threshold")
    field.setNumber(threshold, "InField", distance)
    for key, value in (
        ("SizeMin", size),
        ("SizeMax", far),
        ("DistMin", 0.0),
        ("DistMax", 3.0),
    ):
        field.setNumber(threshold, key, value)
    field.setAsBackgroundMesh(threshold)
    for key in (
        "MeshSizeExtendFromBoundary",
        "MeshSizeFromPoints",
        "MeshSizeFromCurvature",
    ):
        gmsh.option.setNumber(f"Mesh.{key}", 0)
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(path))
    gmsh.finalize()


#: The optical window (um): the core in oxide, 3.5 um wide, 1 um below the
#: core and 1 um above it, the core's underside at y = 0 (route-local).
WINDOW_HALF, WINDOW_BELOW, WINDOW_ABOVE = 1.75, 1.0, 1.22


def window_rectangles(x=0.0):
    """The window's rectangles with its core centred at section x ``x`` (um)."""
    w, t = SPIRAL.core_width / 2, SPIRAL.core_thickness
    return [
        (
            x - WINDOW_HALF,
            -WINDOW_BELOW,
            x + WINDOW_HALF,
            WINDOW_ABOVE,
            "cladding",
            None,
        ),
        (x - w, 0.0, x + w, t, "core", 0.04),
    ]


#: The 2D section's cut: x = 15 um, the middle of the straights.
SECTION_X = SPIRAL.straight / 2


def section_rectangles(spiral=SPIRAL):
    """The stack across the straights at x = `SECTION_X` (section x = device y).

    Every pass's core, the two heater straights, and the device's lateral
    extent and layers.
    """
    _, y0, _, y1 = footprint(spiral)
    lo, hi = y0 - MARGIN, y1 + MARGIN
    w, t = spiral.core_width / 2, spiral.core_thickness
    half, zh = spiral.heater_width / 2, spiral.heater_z
    rects = [
        (lo, SUBSTRATE[0], hi, SUBSTRATE[1], "substrate", None),
        (lo, BOX[0], hi, BOX[1], "box", None),
        (lo, CLADDING[0], hi, CLADDING[1], "cladding", None),
    ]
    rects += [
        (p.y - w, 0.0, p.y + w, t, "core", 0.04) for p in hg.passes(spiral).values()
    ]
    rects += [
        (y - half, zh, y + half, zh + spiral.heater_thickness, "heater", 0.05)
        for y in hg.heater_straights(spiral)
    ]
    return rects


# --- solves ------------------------------------------------------------------


class Clock:
    """Wall times of named steps."""

    def __init__(self):
        """Start with no steps."""
        self.seconds = {}

    def __call__(self, name, function, *args, **kwargs):
        """``function(*args, **kwargs)``, its wall time recorded as ``name``."""
        tic = time.perf_counter()
        out = function(*args, **kwargs)
        self.seconds[name] = round(time.perf_counter() - tic, 2)
        return out


def device(work, hh, hc, far, clock):
    """``(model, dc, steady, route, mesh info)`` of the device per watt on one mesh."""
    path = Path(work) / "meshes" / f"heater3d_h{hh:g}_c{hc:g}.msh"
    path.parent.mkdir(parents=True, exist_ok=True)
    tets = clock("mesh_s", mesh_device, path, hh, hc, far)
    mesh = clock("read_s", gfem.Mesh.from_gmsh, path, length_unit="um")
    window_path = Path(work) / "meshes" / "window.msh"
    mesh_section(window_path, window_rectangles())
    window = window_model(gfem.Mesh.from_gmsh_cross_section(window_path))
    thermal = model(mesh)
    terminals = gfem.DCConduction(terminals={"pad": "contact_0"}, ground="contact_1")
    dc = clock("dc_s", gfem.solve, thermal, terminals, options=SOLVER)
    heat = gfem.joule_heating(dc, power=1.0)
    steady = clock(
        "steady_s", gfem.solve, thermal, gfem.SteadyThermal(heat), options=SOLVER
    )
    route = clock("route_s", phase_route, window, steady.rise)
    info = {
        "heater_um": hh,
        "core_um": hc,
        "far_um": far,
        "tets": tets,
        "dof": int(steady.temperature.coefficients.shape[0]),
        "dc_dof": int(dc.potential["pad"].coefficients.shape[0]),
        "window_triangles": int(
            np.sum(window.mesh.triangle_tags == window.mesh.region("front").tag)
        ),
    }
    return thermal, dc, steady, route, info


def level_arrays(prefix, dc, steady, route):
    """The numbers every mesh level records, per watt, under ``prefix``."""
    names = list(hg.passes(SPIRAL))
    return {
        f"{prefix}_resistance": np.float64(dc.resistance[0, 0]),
        f"{prefix}_power": np.float64(steady.power),
        f"{prefix}_route_phase": np.float64(route.phase["spiral"]),
        f"{prefix}_pass_phase": np.array([float(route.phase[n]) for n in names]),
        f"{prefix}_heater_rise": np.float64(gfem.region_mean(steady.rise, "heater")),
        f"{prefix}_core_rise": np.float64(gfem.region_mean(steady.rise, "core")),
    }


def grid(step, z, box):
    """Sample points (m) of a plane z (um) over ``box`` (um), and its axes (um)."""
    x0, y0, x1, y1 = box
    xs = np.arange(np.floor(x0), np.ceil(x1) + step / 2, step)
    ys = np.arange(np.floor(y0), np.ceil(y1) + step / 2, step)
    X, Y = np.meshgrid(xs, ys)  # rows y
    return np.c_[X.ravel(), Y.ravel(), np.full(X.size, z)] * um, xs, ys


def sample(field, points, chunk=200_000):
    """``field.at`` in chunks, as float64."""
    return np.concatenate(
        [
            np.asarray(field.at(points[i : i + chunk]))
            for i in range(0, len(points), chunk)
        ]
    ).reshape(-1)


def main_stage(stage):
    """The converged mesh: R, rise, phases, maps, 2D section, step response."""
    profile, clock = stage.profile, Clock()
    hh, hc = profile.levels[0]
    thermal, dc, steady, route, info = device(stage.out, hh, hc, profile.far, clock)
    stage.meshes["main"] = info
    arrays = level_arrays("main", dc, steady, route)
    names = list(hg.passes(SPIRAL))
    arrays["pass_names"] = np.array(names)
    arrays["pass_y"] = np.array([hg.passes(SPIRAL)[n].y for n in names])
    arrays["spiral_arc_length"] = np.asarray(route.arc_length["spiral"])
    arrays["spiral_phase_per_length"] = np.asarray(route.phase_per_length["spiral"])
    arrays["centre_pass_arc_length"] = np.asarray(route.arc_length["top_1"])
    arrays["centre_pass_phase_per_length"] = np.asarray(route.phase_per_length["top_1"])
    arrays["core_polygon"] = hg.core_polygon(SPIRAL)
    arrays["heater_polygon"] = hg.heater_polygon(SPIRAL)
    arrays["pad_polygons"] = np.array(hg.pad_polygons(SPIRAL))
    arrays["n_eff"] = np.float64(route.n_eff)
    arrays["route_length"] = np.float64(route.length["spiral"])
    # Power balance: no source below the buried oxide and insulated sides,
    # so the heat crossing each plane of the substrate is the same F, and the
    # substrate's mean rise is F d / (2 k A) (d its thickness, A its area).
    x0, y0, x1, y1 = footprint()
    area = (x1 - x0 + 2 * MARGIN) * (y1 - y0 + 2 * MARGIN) * um**2
    depth = (SUBSTRATE[1] - SUBSTRATE[0]) * um
    mean = float(gfem.region_mean(steady.rise, "substrate"))
    arrays["sink_power"] = np.float64(
        2 * SILICON["thermal_conductivity"] * area * mean / depth
    )
    # maps per watt: the core's mid-plane, the heater's mid-plane, a cut
    box = (x0 - 5, y0 - 5, x1 + 5, y1 + 5)
    step = profile.map_step
    points, xs, ys = grid(step, SPIRAL.core_thickness / 2, box)
    arrays["map_x"], arrays["map_y"] = xs, ys
    arrays["map_core"] = (
        clock("maps_s", sample, steady.rise, points)
        .reshape(len(ys), len(xs))
        .astype(np.float32)
    )
    heater_z = SPIRAL.heater_z + SPIRAL.heater_thickness / 2
    points[:, 2] = heater_z * um
    arrays["map_heater"] = (
        sample(steady.rise, points).reshape(len(ys), len(xs)).astype(np.float32)
    )
    cut_y = np.arange(np.floor(y0) - 5, np.ceil(y1) + 5 + 1e-9, step / 2)
    cut_z = np.arange(SUBSTRATE[0], CLADDING[1] + 1e-9, step / 2)
    Y, Z = np.meshgrid(cut_y, cut_z)  # rows z
    cut = np.c_[np.full(Y.size, SECTION_X), Y.ravel(), Z.ravel()] * um
    arrays["cut_y"], arrays["cut_z"] = cut_y, cut_z
    arrays["map_cut"] = sample(steady.rise, cut).reshape(Y.shape).astype(np.float32)
    # the 2D section across the straights, at the 3D straight's power per length
    arrays |= clock("section_s", section_check, stage.out, dc)
    # the step response: heated at P_pi for `on`, then cooled for `off`
    p_pi = np.pi * float(steady.power) / float(route.phase["spiral"])
    arrays |= clock("transient_s", step_response, thermal, route, dc, p_pi, profile)
    stage.timings = clock.seconds
    return arrays


def section_check(work, dc):
    """The centre pass's phase per length on a 2D section of the same stack.

    The heater straights run at the 3D device's power per length at 1 W:
    the current 1/sqrt(R) through a strip of sheet resistance 1/(sigma t)
    and width w, I^2 / (sigma t w) W/m each.
    """
    path = Path(work) / "meshes" / "section.msh"
    mesh_section(path, section_rectangles(), sink=SUBSTRATE[0], far=2.0)
    section = gfem.Mesh.from_gmsh_cross_section(path)
    current2 = 1.0 / float(dc.resistance[0, 0])
    strip = (
        TIN["conductivity"] * SPIRAL.heater_thickness * um * SPIRAL.heater_width * um
    )
    per_length = current2 / strip
    heat = gfem.HeatSource.uniform("heater", power_per_length=2 * per_length)
    flat = gfem.solve(model(section), gfem.SteadyThermal(heat), options=SOLVER)
    centre = hg.heater_straights(SPIRAL)[0]
    window_path = Path(work) / "meshes" / "section_window.msh"
    mesh_section(window_path, window_rectangles(centre))
    window = window_model(gfem.Mesh.from_gmsh_cross_section(window_path))
    shift = gfem.thermo_optic_shift(window, flat.rise, port="core", frequency=FREQUENCY)
    return {
        "section_power_per_length": np.float64(per_length),
        "section_phase_per_length": np.float64(shift.phase_per_length),
    }


def step_response(thermal, route, dc, p_pi, profile):
    """Phase traces of a step at ``p_pi`` W: on at 0, off at ``profile.on``.

    The phase traces are ``route``'s (`gfem.RoutePhase.trace`): the same
    weights and stations as the steady phase.
    """
    traces = {
        "spiral": route.trace("spiral"),
        "centre_pass": route.trace("top_1"),
        "outer_pass": route.trace("top_7"),
        "heater": gfem.Trace.mean("heater"),
    }
    problem = gfem.TransientThermal(
        gfem.joule_heating(dc, power=p_pi),
        duration=profile.on + profile.off,
        steps=profile.steps,
        waveform=[(0.0, 1.0), (profile.on, 0.0)],
        traces=traces,
    )
    result = gfem.solve(thermal, problem, options=SOLVER)
    t = np.asarray(result.time)
    phase = np.asarray(result.traces["spiral"])
    on, off = t <= profile.on * (1 + 1e-9), t >= profile.on * (1 - 1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("error", gfem.UnsettledTraceWarning)
        times = {
            "rise_time_on": gfem.rise_time(t[on], phase[on], switch_times=(0.0,)),
            "time_constant_on": gfem.time_constant(
                t[on], phase[on], switch_times=(0.0,)
            ),
            "rise_time_off": gfem.rise_time(
                t[off], phase[off], switch_times=(profile.on,)
            ),
            "time_constant_off": gfem.time_constant(
                t[off], phase[off], switch_times=(profile.on,)
            ),
        }
    return {
        "time": t,
        "p_pi": np.float64(p_pi),
        **{f"trace_{k}": np.asarray(v) for k, v in result.traces.items()},
        **{k: np.float64(v) for k, v in times.items()},
    }


def level_stage(stage):
    """One finer mesh of the ladder: the same steady solve per watt."""
    profile, clock = stage.profile, Clock()
    index = 1 + LADDER.index(stage.name)
    hh, hc = profile.levels[index]
    _, dc, steady, route, info = device(stage.out, hh, hc, profile.far, clock)
    stage.meshes[stage.name] = info
    stage.timings = clock.seconds
    return level_arrays(stage.name, dc, steady, route)


RUNS = {"main": main_stage, "medium": level_stage, "fine": level_stage}


class Stage:
    """One stage's run: its work directory, profile, meshes and timings."""

    def __init__(self, name, out, profile):
        """Stage ``name`` in ``out`` with ``PROFILES[profile]``."""
        self.name, self.out, self.profile = name, Path(out), PROFILES[profile]
        self.meshes, self.timings = {}, {}


# --- provenance and the record ------------------------------------------------


def _stage_files(out, stage):
    folder = Path(out) / "stages"
    return folder / f"{stage}.json", folder / f"{stage}.npz"


def load_stage(out, stage):
    """``(info, arrays)`` a stage saved in the work directory ``out``."""
    info_path, npz_path = _stage_files(out, stage)
    if not info_path.exists():
        raise FileNotFoundError(
            f"no {stage} outputs in {out}: run "
            f"`python tools/heater3d_run.py {stage} --out {out}` first."
        )
    with np.load(npz_path) as npz:
        return json.loads(info_path.read_text()), {k: npz[k] for k in npz.files}


def sha256s():
    """The sha256 of the scripts, by path."""
    import hashlib

    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SCRIPTS}


def run_stage(name, out, profile):
    """Run one stage and save its arrays and record in ``out/stages``."""
    memory, start, tic = provenance.PeakMemory(), provenance._utc(), time.perf_counter()  # noqa: SLF001
    (commit, dirty), hashes = provenance.run_commit(ROOT), sha256s()
    info_path, npz_path = _stage_files(out, name)
    info_path.unlink(missing_ok=True)
    stage = Stage(name, out, profile)
    arrays = RUNS[name](stage)
    info = {
        "stage": name,
        "profile": profile,
        "command": " ".join(["python", "tools/heater3d_run.py", *sys.argv[1:]]),
        "start_utc": start,
        "end_utc": provenance._utc(),  # noqa: SLF001
        "commit": commit,
        "dirty": dirty,
        "sha256": hashes,
        "environment": provenance.environment(),
        "meshes": stage.meshes,
        "observations": {
            "wall_s": round(time.perf_counter() - tic, 1),
            "peak_rss_bytes": memory.peak_bytes,
            "steps_s": stage.timings,
        },
    }
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    with npz_path.open("wb") as f:
        np.savez_compressed(f, **arrays)
    info_path.write_text(json.dumps(info, indent=1) + "\n")
    seen = info["observations"]
    print(
        f"{name}: {seen['wall_s']} s, peak RSS {seen['peak_rss_bytes'] / 2**30:.2f} GiB"
    )


RECORD = ("heater3d.json", "heater3d.npz")
#: The cap on the run record, both files together (bytes).
MAX_RECORD_BYTES = 3_000_000
RECORD_KEYS = ("profile", "provenance", "settings", "observations")
PROVENANCE_KEYS = ("commit", "sha256", "stages")
STAGE_KEYS = ("start_utc", "end_utc", "command", "environment")
SETTINGS_KEYS = (
    "wavelength_um",
    "sink_temperature_k",
    "geometry",
    "stack_um",
    "materials",
    "window",
    "meshes",
    "order",
    "linear_solver",
    "step_response",
    "maps",
    "section",
)
OBSERVATION_KEYS = ("wall_s", "peak_rss_bytes", "steps_s")
LEVEL_KEYS = (
    "resistance",
    "power",
    "route_phase",
    "pass_phase",
    "heater_rise",
    "core_rise",
)
NPZ_KEYS = (
    *(f"{level}_{key}" for level in STAGES for key in LEVEL_KEYS),
    *("pass_names", "pass_y", "spiral_arc_length", "spiral_phase_per_length"),
    *("centre_pass_arc_length", "centre_pass_phase_per_length"),
    *("core_polygon", "heater_polygon", "pad_polygons"),
    *("n_eff", "route_length", "sink_power"),
    *("map_x", "map_y", "map_core", "map_heater", "cut_y", "cut_z", "map_cut"),
    *("section_power_per_length", "section_phase_per_length"),
    *("time", "p_pi", "trace_spiral", "trace_centre_pass", "trace_outer_pass"),
    *("trace_heater", "rise_time_on", "time_constant_on"),
    *("rise_time_off", "time_constant_off"),
)


def settings(profile, infos):
    """The record's settings: geometry, stack, materials, meshes, solves."""
    spiral = dataclasses.asdict(SPIRAL)
    top, bottom = hg.heater_straights(SPIRAL)
    return {
        "wavelength_um": WAVELENGTH / um,
        "sink_temperature_k": T_SINK,
        "geometry": {
            **spiral,
            "route_length_um": hg.route_length(SPIRAL),
            "heater_straights_y_um": [top, bottom],
            "footprint_um": [float(v) for v in footprint()],
            "margin_um": MARGIN,
        },
        "stack_um": {"substrate": SUBSTRATE, "box": BOX, "cladding": CLADDING},
        "materials": {
            "silicon substrate": SILICON,
            "silicon core": {**FILM, "thermo_optic_coefficient": DNDT_SI, "n": N_SI},
            "oxide": {**OXIDE, "thermo_optic_coefficient": DNDT_OX, "n": N_OX},
            "TiN heater and pads": TIN,
        },
        "window": {
            "half_width_um": WINDOW_HALF,
            "below_um": WINDOW_BELOW,
            "above_um": WINDOW_ABOVE,
            "station_spacing_um": 0.5,
        },
        "meshes": {
            stage: {**entry, "stage": stage}
            for stage, info in infos.items()
            for entry in info["meshes"].values()
        },
        "order": SOLVER.order,
        "linear_solver": SOLVER.linear_solver,
        "step_response": {
            "on_s": profile.on,
            "off_s": profile.off,
            "steps": profile.steps,
            "drive": "joule heating at P_pi from t = 0 to on, then off",
        },
        "maps": {
            "step_um": profile.map_step,
            "core_z_um": SPIRAL.core_thickness / 2,
            "heater_z_um": SPIRAL.heater_z + SPIRAL.heater_thickness / 2,
            "cut_x_um": SECTION_X,
            "quantity": "temperature rise per watt (K/W)",
        },
        "section": {"x_um": SECTION_X, "pass": "top_1", "heaters": [top, bottom]},
    }


def _same_keys(what, got, wanted):
    if set(got) != set(wanted):
        raise ValueError(
            f"{what}: missing {sorted(set(wanted) - set(got))}, "
            f"unexpected {sorted(set(got) - set(wanted))}"
        )


def write_record(record, arrays, dest):
    """Write ``heater3d.json`` and ``heater3d.npz`` into ``dest``, after checks.

    Raises:
        ValueError: A key is missing or unexpected, or the two files would
            pass `MAX_RECORD_BYTES` together (nothing is written then).
    """
    _same_keys("record", record, RECORD_KEYS)
    _same_keys("provenance", record["provenance"], PROVENANCE_KEYS)
    _same_keys("stages", record["provenance"]["stages"], STAGES)
    for stage, entry in record["provenance"]["stages"].items():
        _same_keys(f"provenance of {stage}", entry, STAGE_KEYS)
    _same_keys("settings", record["settings"], SETTINGS_KEYS)
    _same_keys("observations", record["observations"], STAGES)
    for stage, seen in record["observations"].items():
        _same_keys(f"observations of {stage}", seen, OBSERVATION_KEYS)
    _same_keys("arrays", arrays, NPZ_KEYS)
    text = (json.dumps(record, indent=1) + "\n").encode()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payloads = dict(zip(RECORD, (text, buffer.getvalue()), strict=True))
    total = sum(len(p) for p in payloads.values())
    if total > MAX_RECORD_BYTES:
        raise ValueError(
            f"the run record would be {total} bytes, over {MAX_RECORD_BYTES}"
        )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (dest / name).write_bytes(payload)


def mismatches(info, commit, hashes, profile):
    """How a stage's ``info`` differs from the run `merge` writes: [] if not."""
    return [
        f"{what} {got!r}, not {wanted!r}"
        for what, got, wanted in (
            ("commit", info["commit"], commit),
            ("uncommitted", info["dirty"], []),
            ("scripts", info["sha256"], hashes),
            ("profile", info["profile"], profile),
        )
        if got != wanted
    ]


def merge(out, dest):
    """Write the run record from the stages saved in ``out``, into ``dest``.

    Raises:
        RuntimeError: ``src/`` or ``tools/`` is uncommitted, or a stage ran
            at another commit, with other scripts or another profile than
            ``main``.
    """
    loaded = {stage: load_stage(out, stage) for stage in STAGES}
    infos = {stage: info for stage, (info, _) in loaded.items()}
    commit, hashes = provenance.clean_commit(ROOT), sha256s()
    profile = infos["main"]["profile"]
    for stage, info in infos.items():
        if problems := mismatches(info, commit, hashes, profile):
            raise RuntimeError(f"rerun {stage}: it ran with {'; '.join(problems)}")
    arrays = {}
    for _, stage_arrays in loaded.values():
        arrays |= stage_arrays
    record = {
        "profile": profile,
        "provenance": {
            "commit": commit,
            "sha256": hashes,
            "stages": {
                stage: {k: info[k] for k in STAGE_KEYS} for stage, info in infos.items()
            },
        },
        "settings": settings(PROFILES[profile], infos),
        "observations": {stage: info["observations"] for stage, info in infos.items()},
    }
    write_record(record, arrays, dest)
    for name in RECORD:
        print(f"{Path(dest) / name}: {(Path(dest) / name).stat().st_size} bytes")


def main(argv=None):
    """Run a stage, ``ladder``, the merge, or ``all`` (one process per stage)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=(*STAGES, "ladder", "merge", "all"))
    parser.add_argument("--out", type=Path, required=True, help="the work directory")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="full")
    parser.add_argument("--record", type=Path, help="merge: the record's directory")
    args = parser.parse_args(argv)
    groups = {"ladder": LADDER, "all": (*STAGES, "merge")}
    if args.stage in groups:
        record = ["--record", str(args.record)] if args.record else []
        for stage in groups[args.stage]:
            command = [sys.executable, __file__, stage, "--out", str(args.out)]
            subprocess.run([*command, "--profile", args.profile, *record], check=True)
    elif args.stage == "merge":
        merge(args.out, args.record or args.out)
    else:
        run_stage(args.stage, args.out, args.profile)


if __name__ == "__main__":
    main()
