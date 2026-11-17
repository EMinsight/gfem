"""The solver adapter seam of the benchmark suite (ADR 0406).

A **solver adapter** is a pure module per code that renders one run of a suite case
into input files and command lines, and parses the outputs back into a neutral
:class:`Outcome`. It never starts a process: the harness sources the stack's env
script, runs the :class:`Launch` it gets from :meth:`Adapter.prepare`, measures it,
and hands the :class:`Measured` result to :meth:`Adapter.collect`. Accuracy is the
harness's scorer's job, never an adapter's.

Plan time asks :meth:`Adapter.supports` (``None`` or the reason a cell is refused)
and :meth:`Adapter.default_layout`. A third solver is one module plus one line in
:data:`ADAPTERS`.

Every launch is one srun step with one placement rule for every code and rank count
(ADR 0412): :func:`srun_argv`. At one rank a stack may say ``launch_r1 = "taskset"``;
the command then carries the bare program and the harness pins it with ``taskset``.

Standard library only (ADR 0400).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from benchmarks.harness.case import Case
from benchmarks.stacks import Resolved

CLASSES = ("direct", "iterative")
"""Solver kind classes; the report pairs codes by class."""
OUTCOME_STATUSES = ("ok", "failed", "refused")
"""An adapter's statuses; the harness sets every other record status itself."""
LAUNCHES = ("srun", "taskset")
_LAYOUT = re.compile(r"^([1-9]\d*)x([1-9]\d*)$")


class RunRefusedError(Exception):
    """A run an adapter refuses to prepare or collect; its record is ``refused``."""


@dataclass(frozen=True)
class Layout:
    """Ranks x threads per rank of one run."""

    ranks: int
    threads: int

    def __post_init__(self) -> None:
        if self.ranks < 1 or self.threads < 1:
            raise ValueError(f"layout {self.ranks}x{self.threads} is not positive")

    @classmethod
    def parse(cls, text: str) -> Layout:
        """``"4x4"`` -> ``Layout(4, 4)``."""
        match = _LAYOUT.match(text)
        if match is None:
            raise ValueError(f"layout {text!r} is not RxT")
        return cls(int(match[1]), int(match[2]))

    @property
    def cores(self) -> int:
        """Cores the layout uses."""
        return self.ranks * self.threads

    def __str__(self) -> str:
        return f"{self.ranks}x{self.threads}"


@dataclass(frozen=True)
class Kind:
    """A solver kind: its class and the settings the adapter renders for it."""

    klass: str
    """``direct`` or ``iterative``."""
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.klass not in CLASSES:
            raise ValueError(f"kind class {self.klass!r} not in {CLASSES}")


@dataclass(frozen=True)
class Run:
    """One planned run, as the harness hands it to an adapter."""

    case: Case
    solver: str
    """``code:kind``."""
    tolerance: float | None
    """Iterative kinds' run tolerance; ``None`` for direct kinds."""
    layout: Layout
    core_budget: int
    repeat: int
    stack: Resolved
    workdir: Path
    """Fresh per run: inputs are written and outputs read here."""
    mesh: Path
    """The staged, verified mesh (``{data.dir}/meshes/<sha256>.msh``)."""
    pinned_ndof: int | None = None
    """The case's pinned DOF count at its order (DOF guard), if known."""
    wall_cap_s: float | None = None
    mem_cap_gib: float | None = None

    @property
    def code(self) -> str:
        """The solver's code (``gfem``, ``palace``)."""
        return self.solver.partition(":")[0]

    @property
    def kind(self) -> str:
        """The solver's kind (``direct``, ``mumps``...)."""
        return self.solver.partition(":")[2]


@dataclass(frozen=True)
class Command:
    """One process an adapter asks the harness to run, in the run's workdir."""

    argv: tuple[str, ...]
    """What runs: the srun line and the program, or the bare program (taskset)."""
    program: tuple[str, ...]
    """The program without its launcher, for a local runner to wrap."""
    env: Mapping[str, str] = field(default_factory=dict)
    """Adapter-specific variables only; threads and modules come from the stack."""
    phase: str = "run"
    """Where the harness bills the command's time (``mesh`` for a conversion)."""
    stdout: str = "stdout.txt"
    """File in the workdir that receives stdout and stderr."""
    launch: str = "srun"
    """``srun`` (``argv`` holds the line) or ``taskset`` (the harness pins it)."""
    watch: Callable[[str], int | None] | None = field(default=None, compare=False)
    """DOF guard: a stdout line -> the DOF count it reports, else ``None``."""

    def __post_init__(self) -> None:
        if self.launch not in LAUNCHES:
            raise ValueError(f"launch {self.launch!r} not in {LAUNCHES}")


@dataclass(frozen=True)
class Launch:
    """The commands of one run (or of a version query), in order."""

    commands: tuple[Command, ...]


@dataclass(frozen=True)
class CommandResult:
    """What the harness observed of one command."""

    returncode: int | None
    """``None`` when the harness killed it."""
    stdout: Path
    wall_s: float | None = None


@dataclass(frozen=True)
class Measured:
    """The harness's measurements of a run, handed to :meth:`Adapter.collect`."""

    results: tuple[CommandResult, ...]
    version: str | None = None
    """stdout of the job's version query (:meth:`Adapter.version`)."""


@dataclass
class Outcome:
    """What a run produced, in the suite's neutral terms (ADR 0406)."""

    status: str
    """``ok``, ``failed`` or ``refused``."""
    reason: str | None = None
    failure_class: str | None = None
    """``solver`` or ``harness`` for a failure."""
    quantities: dict[str, Any] = field(default_factory=dict)
    """Driven: ``frequencies_hz`` and ``s`` (``"to,from"`` -> ``[re, im]`` per
    frequency); eigen: ``f_hz`` (``[re, im]`` per mode), ``q``."""
    ndof: int | None = None
    phases: dict[str, float | None] = field(default_factory=dict)
    """Record bucket name (``setup_s``...) -> seconds, headline rank reduction."""
    rank0: dict[str, float | None] = field(default_factory=dict)
    rank_reduction: str | None = None
    work_span_s: float | None = None
    """The code's own span of the run (startup = step wall minus this)."""
    native: dict[str, Any] = field(default_factory=dict)
    iterations: list[int] = field(default_factory=list)
    setups: int | None = None
    solves: int | None = None
    self_peak_mem_gib: float | None = None
    solver_nbytes: float | None = None
    ranks_seen: int | None = None
    threads_seen: int | None = None
    structural: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    """Workdir-relative files to keep with the record."""
    checks: dict[str, str] = field(default_factory=dict)
    """Record checks the adapter can decide (``inputs_match``, ``converged``...)."""

    def __post_init__(self) -> None:
        if self.status not in OUTCOME_STATUSES:
            raise ValueError(f"outcome status {self.status!r} not {OUTCOME_STATUSES}")


class Adapter(Protocol):
    """A solver adapter: pure, never starts a process."""

    @property
    def code(self) -> str:
        """The code's name (``gfem``, ``palace``)."""
        ...

    @property
    def kinds(self) -> Mapping[str, Kind]:
        """Kind name -> class and settings."""
        ...

    def supports(self, case: Case, kind: str, layout: Layout) -> str | None:
        """``None``, or why this cell is refused."""
        ...

    def default_layout(self, case: Case, kind: str, budget: int) -> Layout:
        """The code's default layout at ``budget`` cores."""
        ...

    def version(self, stack: Resolved) -> Launch:
        """The provenance query the harness runs once per job."""
        ...

    def prepare(self, run: Run) -> Launch:
        """Write the run's inputs into its workdir; raises :class:`RunRefusedError`."""
        ...

    def collect(self, run: Run, measured: Measured) -> Outcome:
        """Parse the run's outputs."""
        ...


def srun_argv(layout: Layout, mpi: str) -> tuple[str, ...]:
    """The one srun line of every suite launch (ADR 0412)."""
    return (
        "srun",
        "--exact",
        "-n",
        str(layout.ranks),
        "-c",
        str(layout.threads),
        "--cpu-bind=cores",
        "--distribution=block:block",
        "--hint=nomultithread",
        "--ear=off",
        f"--mpi={mpi}",
    )


def launched(
    program: tuple[str, ...], layout: Layout, mpi: str, launch_r1: str = "srun"
) -> tuple[tuple[str, ...], str]:
    """``(argv, launch)`` of ``program``: under srun, or bare for taskset at R = 1."""
    if layout.ranks == 1 and launch_r1 == "taskset":
        return program, "taskset"
    return (*srun_argv(layout, mpi), *program), "srun"


ADAPTERS: dict[str, Adapter] = {}
"""Solver adapters by code; each adapter module adds its line (gfem, palace)."""
