"""The gfem worker: one suite run in the stack's gfem env (ADR 0406).

``python -m benchmarks.adapters.gfem_worker run.json`` (on every rank under srun)
reads the run the gfem adapter rendered, times the mesh read, logs ``gfem.timing``
records to ``phases.rank<k>.jsonl``, builds the model from the case
(:func:`model_from_case`), solves, and writes ``outcome.json`` on rank 0: status,
quantities, each rank's mesh read, span, ``VmHWM`` and NGSolve thread count. After
the timed span it writes ``job.json``, a gfem job bundle that ``python -m gfem run``
replays. ``--version`` prints gfem's and its libraries' versions as JSON.

At one rank the worker is the DOF guard: the ``discretize`` record is checked against
the pinned count and a mismatch stops the solve there. Above one rank the adapter
checks the summed count after the run.

The one module of ``benchmarks/`` that imports gfem (ADR 0400).
"""

from __future__ import annotations

import json
import logging
import math
import socket
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ngsolve

import gfem
from benchmarks.harness.case import UNITS, Case, case_from_dict

OUTCOME_JSON = "outcome.json"
JOB_JSON = "job.json"
PHASE_FILE = "phases.rank{rank}.jsonl"
_RECORD_KEYS = (
    "phase",
    "seconds",
    "problem",
    "frequency",
    "rank",
    "ranks",
    "ndof",
    "solver",
    "action",
    "nbytes",
    "iterations",
)


class DofMismatchError(RuntimeError):
    """The discretized space differs from the case's pinned DOF count."""

    def __init__(self, ndof: int, pinned: int) -> None:
        """``ndof`` found, ``pinned`` expected."""
        super().__init__(f"{ndof} != pinned {pinned}")
        self.ndof = ndof
        self.pinned = pinned


class PhaseFile(logging.Handler):
    """Appends each ``gfem.timing`` record as one JSON line; the one-rank DOF guard."""

    def __init__(self, path: Path, pinned_ndof: int | None = None) -> None:
        """Write to ``path``; check ``discretize`` records against ``pinned_ndof``."""
        super().__init__(logging.INFO)
        self.stream = path.open("w")
        self.pinned_ndof = pinned_ndof

    def emit(self, record: logging.LogRecord) -> None:
        """Write the record, then stop the solve if its DOF count is off the pin."""
        line = {k: getattr(record, k) for k in _RECORD_KEYS if hasattr(record, k)}
        self.stream.write(json.dumps(line, default=list) + "\n")
        self.stream.flush()
        ndof = line.get("ndof")
        pinned = self.pinned_ndof
        if line.get("phase") != "discretize" or pinned is None or ndof is None:
            return
        if int(ndof) != pinned:
            raise DofMismatchError(int(ndof), pinned)

    def close(self) -> None:
        """Close the file."""
        self.stream.close()
        super().close()


# --- case -> gfem -----------------------------------------------------------------


def _region(name: str | None, what: str) -> str:
    if name is None:
        raise ValueError(f"{what} names no region")
    return name


def _film_thickness(mesh: gfem.Mesh, region: str, thickness: float) -> float:
    """The gfem ``Conductor.thickness`` of a case ``thickness`` on ``region``.

    A case thickness is the film every face of the surface sees, Palace's
    meaning with ``External`` false (its default). gfem counts the thickness
    of a mesh-exterior face :data:`gfem.conventions.EXTERIOR_THICKNESS_FACTOR`
    times (a voided conductor's shell), so it is divided there; interior faces
    (a shell around an excluded volume) keep it. A surface with both is refused:
    one gfem thickness cannot carry both.
    """
    on = mesh.triangle_tags == mesh.region(region).tag
    exterior = (mesh.triangle_neighbours[on] < 0).any(axis=1)
    if exterior.all():
        return thickness / gfem.conventions.EXTERIOR_THICKNESS_FACTOR
    if exterior.any():
        raise ValueError(
            f"conductor {region!r} has mesh-exterior and interior faces; "
            "its thickness is not mapped for gfem"
        )
    return thickness


def _boundary(
    kind: str,
    region: str | None,
    settings: dict[str, Any],
    unit: float,
    mesh: gfem.Mesh,
):
    """One gfem boundary of a case ``[[boundaries]]`` entry."""
    if kind == "periodic":
        return gfem.Periodic((settings["donor"], settings["receiver"]))
    surface = _region(region, f"a {kind} boundary")
    if kind == "pec":
        return gfem.PEC(surface)
    if kind == "conductor":
        thickness = settings.get("thickness")
        return gfem.Conductor(
            surface,
            float(settings["conductivity"]),
            None
            if thickness is None
            else _film_thickness(mesh, surface, float(thickness) * unit),
        )
    if kind == "absorbing":
        order: Any = int(settings.get("order", 1))
        return gfem.Absorbing(surface, order=order)
    raise ValueError(f"boundary kind {kind!r} is not mapped for gfem")


def model_from_case(case: Case, mesh: gfem.Mesh) -> gfem.Model:
    """The gfem model of a case: a one-to-one map, the case being name-keyed.

    Lengths in the case are in the mesh's unit and become metres here. A
    conductor's thickness is each face's film (:func:`_film_thickness`).
    """
    unit = UNITS[case.mesh.length_unit]
    materials = {}
    for m in case.materials:
        values: dict[str, Any] = {
            "eps_r": m.eps_r,
            "mu_r": m.mu_r,
            "tan_delta": m.tan_delta,
            "conductivity": m.conductivity,
            "axes": m.axes,
        }
        materials[m.region] = gfem.Material(
            **{k: v for k, v in values.items() if v is not None}
        )
    boundaries = [
        _boundary(b.kind, b.region, b.settings, unit, mesh) for b in case.boundaries
    ]
    ports = []
    for p in case.ports:
        if p.kind != "lumped":
            raise ValueError(f"port kind {p.kind!r} is not mapped for gfem")
        surfaces = tuple(e.region for e in p.elements)
        directions: Any = tuple(e.direction for e in p.elements)
        one = len(surfaces) == 1
        ports.append(
            gfem.LumpedPort(
                p.name,
                surfaces[0] if one else surfaces,
                directions[0] if one else directions,
                impedance=p.settings.get("impedance_ohm"),
                inductance=p.settings.get("inductance_h"),
                capacitance=p.settings.get("capacitance_f"),
            )
        )
    excluded = tuple(gfem.enclosed_by(*surfaces) for surfaces in case.excluded)
    return gfem.Model(
        mesh,
        materials=materials,
        boundaries=boundaries,
        ports=ports,
        excluded=excluded,
    )


def problem_from_case(case: Case) -> gfem.Driven | gfem.Eigenmode:
    """The gfem problem of a case (no fields kept)."""
    problem = case.problem
    if problem.kind == "eigen":
        if problem.target_hz is None or problem.n_modes is None:
            raise ValueError("an eigen case needs target_hz and n_modes")
        return gfem.Eigenmode(
            target=problem.target_hz,
            n_modes=problem.n_modes,
            wave_vector=problem.wave_vector,
        )
    sweep: Any = problem.sweep
    return gfem.Driven(
        frequency=list(problem.frequencies()),
        excite=problem.excite,
        sweep=sweep,
        wave_vector=problem.wave_vector,
    )


def _finite(value: float) -> float | None:
    """``value``, or ``None`` for an infinite Q (a lossless mode): JSON has no inf."""
    return value if math.isfinite(value) else None


def quantities(case: Case, result: Any) -> dict[str, Any]:
    """Neutral quantities: S ``"to,from"`` -> ``[re, im]`` per frequency, or modes.

    Modes: ``f_hz`` as ``[re, im]`` and ``q`` (``None`` when infinite), in gfem's
    order (ascending real part from the target).
    """
    if case.problem.kind == "eigen":
        f = [complex(z) for z in result.frequency]
        return {
            "f_hz": [[z.real, z.imag] for z in f],
            "q": [_finite(float(q)) for q in result.q],
        }
    s = result.s
    ports, excited = list(s.ports), list(s.excited)
    out: dict[str, list[list[float]]] = {}
    for to, frm in case.quantities.s:
        i, j = ports.index(to), excited.index(frm)
        out[f"{to},{frm}"] = [
            [complex(z).real, complex(z).imag] for z in (v[i][j] for v in s.values)
        ]
    return {"frequencies_hz": [float(f) for f in s.frequency], "s": out}


# --- the run ----------------------------------------------------------------------


def versions() -> dict[str, str]:
    """The versions of gfem and of the libraries it runs on."""
    out = {"gfem": gfem.__version__, "ngsolve": ngsolve.__version__}
    for name in ("petsc4py", "mpi4py", "numpy", "jax"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        out[name] = str(getattr(module, "__version__", "?"))
    return out


def vmhwm_kib() -> int | None:
    """This process's peak resident set (``VmHWM``), KiB."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


@dataclass
class Solved:
    """What the timed span left: the report and, on success, the job's parts."""

    report: dict[str, Any] = field(default_factory=lambda: {"status": "ok"})
    mesh_read_s: float | None = None
    job: gfem.Job | None = None


def solve(spec: dict[str, Any], case: Case, comm: Any) -> Solved:
    """The timed span: read the mesh, build the model, solve, take the quantities."""
    out = Solved()
    start = time.perf_counter()
    stage = "mesh"
    try:
        mesh = gfem.Mesh.from_gmsh(
            spec["mesh"], length_unit=case.mesh.length_unit, comm=comm
        )
        out.mesh_read_s = time.perf_counter() - start
        stage = "model"
        model = model_from_case(case, mesh)
        problem = problem_from_case(case)
        options = gfem.SolverOptions(**spec["options"])
        stage = "solve"
        result = gfem.solve(model, problem, options=options, comm=comm)
        out.report["converged"] = "pass"
        stage = "quantities"
        out.report["quantities"] = quantities(case, result)
        out.job = gfem.Job(model, problem, options)
    except DofMismatchError as err:
        out.report.update(
            status="failed",
            failure_class="solver",
            reason=f"dof_mismatch: gfem {gfem.__version__} {err}",
        )
    except gfem.LinearSolverConvergenceError as err:
        out.report.update(
            status="failed",
            failure_class="solver",
            converged="fail",
            reason=f"not_converged: {str(err).splitlines()[0]}",
        )
    except Exception as err:  # noqa: BLE001 (every failure becomes the run's outcome)
        traceback.print_exc()
        out.report.update(
            status="failed",
            failure_class="solver" if stage == "solve" else "harness",
            reason=f"{stage}: {type(err).__name__}: {str(err).splitlines()[0]}",
        )
    return out


def run(path: Path) -> int:
    """Run ``run.json``; returns the process exit code (0 = ok)."""
    spec = json.loads(path.read_text())
    workdir = path.resolve().parent
    ranks = int(spec["layout"]["ranks"])
    comm: Any = None
    rank = 0
    if ranks > 1:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.rank
    handler = PhaseFile(
        workdir / PHASE_FILE.format(rank=rank),
        spec.get("pinned_ndof") if ranks == 1 else None,
    )
    logger = logging.getLogger("gfem.timing")
    saved = (logger.level, logger.propagate)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    case = case_from_dict(spec["case"])
    start = time.perf_counter()
    solved = solve(spec, case, comm)
    mine = {
        "rank": rank,
        "mesh_read_s": solved.mesh_read_s,
        "span_s": time.perf_counter() - start,
        "vmhwm_kib": vmhwm_kib(),
        "host": socket.gethostname(),
    }
    logger.removeHandler(handler)
    logger.setLevel(saved[0])
    logger.propagate = saved[1]
    handler.close()
    # GetNumThreads is 1 outside a TaskManager; inside one it is the count gfem set
    with ngsolve.TaskManager():
        threads = ngsolve.GetNumThreads()
    gathered = [(mine, threads)]
    if comm is not None:
        gathered = comm.gather(gathered[0], root=0)
    report = solved.report
    if rank == 0:
        report["ranks"] = [m for m, _ in gathered]
        report["threads"] = [t for _, t in gathered]
        report["versions"] = versions()
        (workdir / OUTCOME_JSON).write_text(json.dumps(report, indent=1) + "\n")
        if solved.job is not None:
            try:
                solved.job.to_json(workdir / JOB_JSON)
            except Exception:  # noqa: BLE001 (the bundle is a convenience)
                traceback.print_exc()
    return 0 if report["status"] == "ok" else 1


def main(argv: Sequence[str] | None = None) -> int:
    """``gfem_worker RUN.json`` or ``gfem_worker --version``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--version"]:
        print(json.dumps(versions(), sort_keys=True))
        return 0
    if len(args) != 1:
        print("usage: python -m benchmarks.adapters.gfem_worker RUN.json | --version")
        return 2
    return run(Path(args[0]))


if __name__ == "__main__":
    sys.exit(main())
