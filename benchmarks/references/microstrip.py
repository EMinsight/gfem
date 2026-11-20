r"""The microstrip converged reference: a split ladder extrapolated in h.

The rungs are sibling case files of ``microstrip`` (same physics, ``family =
"microstrip"``) under ``benchmarks/references/microstrip/``: L0 is the pinned mesh,
L1 and L2 its uniform splits (``tools/split_mesh.py``, byte-deterministic), external
meshes in the mesh stores. They run only in the reference campaigns
(``benchmarks/campaigns/reference-microstrip-<stage>.toml``):

- ``pilot``: gfem direct on L0 p1, p2, p3, L1 p2, p3. Its rate estimate against
  L1 p3 decides whether L2 is worth running (a rate below 1 is pre-asymptotic).
- ``l2``: gfem BDDC at 1e-11 on L2 p2 and on L1 p2 (the algebraic check); no
  same-code direct run is added (``same_code_direct = false``).
- ``palace-<version>``: Palace direct on L0 p2 and L1 p2, once per Palace version:
  the definition offset :math:`\delta` of its lumped ports, never averaged in.

``python -m benchmarks reference microstrip --from RUNDIR..`` extrapolates the
fetched records, per frequency and S entry (complex), at order 2:

- rate :math:`\alpha = \log_2(|S_0 - S_1| / |S_1 - S_2|)`, the maximum over entries
  per frequency, clipped to [1, 4];
- value :math:`S_\mathrm{ref} = S_2 + (S_2 - S_1) / (2^\alpha - 1)`;
- bound, the largest of the rate sensitivity
  :math:`|S_\mathrm{ref}(\alpha) - S_\mathrm{ref}(\alpha \pm 0.5)|`, the distance to
  L1 p3, and the BDDC-vs-direct difference on L1 p2.

Same-mesh error per order: :math:`\max |S_{L0,p} - S_\mathrm{ref}|`; the
plausibility ceiling is ten times it. With L2 missing, the command prints the pilot's
rate estimate and writes nothing. Standard library only.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.adapters import ADAPTERS
from benchmarks.harness import config, records
from benchmarks.harness.case import load_case
from benchmarks.references import document as refdoc

CASE = "microstrip"
RUNGS_DIR = Path(__file__).resolve().parent / "microstrip"
PREFIX = f"{CASE}_"
RUNGS = tuple(
    f"{PREFIX}{r}" for r in ("L0_p1", "L0_p2", "L0_p3", "L1_p2", "L1_p3", "L2_p2")
)
CAMPAIGNS_DIR = config.PACKAGE / "campaigns"
STAGE_PREFIX = "reference-microstrip-"
RATE_CLIP = (1.0, 4.0)
RATE_SENSITIVITY = 0.5
OVERKILL_TOLERANCE = 1e-11
"""The iterative rungs' tolerance (BDDC, ``l2`` stage)."""
TARGET_BOUND = 1e-4
"""Absolute target of the bound (and at most a tenth of the smallest floor)."""

Matrix = list[list[complex]]
"""S per frequency, per entry (``"to,from"`` in the case's order)."""


class LadderError(config.ConfigError):
    """Ladder records that cannot give a reference."""


def stages() -> list[str]:
    """The committed reference-campaign stages."""
    return sorted(
        p.stem.removeprefix(STAGE_PREFIX)
        for p in CAMPAIGNS_DIR.glob(f"{STAGE_PREFIX}*.toml")
    )


def campaign_file(stage: str) -> Path:
    """The campaign file of ``stage`` (``pilot``, ``l2``, ``palace-<version>``)."""
    path = CAMPAIGNS_DIR / f"{STAGE_PREFIX}{stage}.toml"
    if not path.is_file():
        raise config.ConfigError(
            f"no reference stage {stage!r} (stages: {', '.join(stages())})"
        )
    return path


def as_complex(pairs: Iterable[Sequence[float]]) -> list[complex]:
    """``[[re, im], ..]`` as complex numbers."""
    return [complex(re, im) for re, im in pairs]


# --- the extrapolation -------------------------------------------------------------


def _rate(coarse: float, fine: float) -> float:
    if fine == 0.0:
        return math.inf
    if coarse == 0.0:
        return -math.inf
    return math.log2(coarse / fine)


def _clip(alpha: float) -> float:
    return min(max(alpha, RATE_CLIP[0]), RATE_CLIP[1])


def _richardson(s1: complex, s2: complex, alpha: float) -> complex:
    return s2 + (s2 - s1) / (2.0**alpha - 1.0)


@dataclass
class Extrapolation:
    """Rate, value and bound per frequency."""

    raw_alpha: list[float]
    alpha: list[float]
    values: Matrix
    terms: dict[str, list[float]]
    """``rate``, ``finer_order``, ``algebraic``: each bound term per frequency."""
    bound_per_frequency: list[float]
    bound: float


def extrapolate(
    l0: Matrix,
    l1: Matrix,
    l2: Matrix,
    *,
    l1p3: Matrix,
    l1_iter: Matrix | None = None,
    l1_direct: Matrix | None = None,
) -> Extrapolation:
    """Richardson over the three p2 levels, with the three bound terms."""
    raw, alpha, values = [], [], []
    terms: dict[str, list[float]] = {"rate": [], "finer_order": [], "algebraic": []}
    for f, (r0, r1, r2) in enumerate(zip(l0, l1, l2, strict=True)):
        a_raw = max(
            _rate(abs(a - b), abs(b - c)) for a, b, c in zip(r0, r1, r2, strict=True)
        )
        a = _clip(a_raw)
        ref = [_richardson(b, c, a) for b, c in zip(r1, r2, strict=True)]
        raw.append(a_raw)
        alpha.append(a)
        values.append(ref)
        terms["rate"].append(
            max(
                abs(_richardson(b, c, a + s) - v)
                for b, c, v in zip(r1, r2, ref, strict=True)
                for s in (-RATE_SENSITIVITY, RATE_SENSITIVITY)
            )
        )
        terms["finer_order"].append(
            max(abs(v - p) for v, p in zip(ref, l1p3[f], strict=True))
        )
        terms["algebraic"].append(
            max(abs(i - d) for i, d in zip(l1_iter[f], l1_direct[f], strict=True))
            if l1_iter is not None and l1_direct is not None
            else 0.0
        )
    per_f = [max(t[f] for t in terms.values()) for f in range(len(values))]
    return Extrapolation(raw, alpha, values, terms, per_f, max(per_f))


def pilot_rate(l0: Matrix, l1: Matrix, finer: Matrix) -> list[float]:
    r"""The pilot's rate estimate per frequency, errors taken against ``finer``.

    :math:`\max_e \log_2(|S_0 - S_f| / |S_1 - S_f|)` with :math:`S_f` the L1 p3
    solve, unclipped; L1 p3's own error makes it a lower estimate.
    """
    return [
        max(_rate(abs(a - p), abs(b - p)) for a, b, p in zip(r0, r1, rf, strict=True))
        for r0, r1, rf in zip(l0, l1, finer, strict=True)
    ]


def definition_offset(d0: Matrix, d1: Matrix, alpha: Sequence[float]) -> dict:
    r"""The part of a cross-code difference that does not shrink with h.

    :math:`\delta = d_1 + (d_1 - d_0) / (2^\alpha - 1)` per entry, with
    :math:`d_L = S_\mathrm{other} - S_\mathrm{gfem}` on level L at p2.
    """
    per_f = [
        max(abs(_richardson(a, b, alpha[f])) for a, b in zip(r0, r1, strict=True))
        for f, (r0, r1) in enumerate(zip(d0, d1, strict=True))
    ]
    return {
        "delta": max(per_f),
        "delta_per_frequency": per_f,
        "d_L0": max(abs(v) for row in d0 for v in row),
        "d_L1": max(abs(v) for row in d1 for v in row),
    }


def _difference(a: Matrix, b: Matrix) -> Matrix:
    return [
        [x - y for x, y in zip(ra, rb, strict=True)]
        for ra, rb in zip(a, b, strict=True)
    ]


def _max_abs(a: Matrix, b: Matrix) -> float:
    return max(abs(v) for row in _difference(a, b) for v in row)


# --- records -> ladder ---------------------------------------------------------------

DIRECT, ITERATIVE = "direct", "iterative"


@dataclass
class _Rung:
    s: Matrix
    mesh_sha256: str | None
    ndof: int | None
    record: str


@dataclass
class Ladder:
    """The ladder's solves, by (rung, ``direct``/``iterative``), and Palace's."""

    frequencies: tuple[float, ...]
    entries: tuple[str, ...]
    gfem: dict[tuple[str, str], _Rung] = field(default_factory=dict)
    palace: dict[str, dict[str, _Rung]] = field(default_factory=dict)
    """Palace version -> rung -> solve."""

    @classmethod
    def from_records(
        cls,
        rows: Iterable[Mapping[str, Any]],
        *,
        frequencies: Sequence[float],
        entries: Sequence[str],
        palace_versions: Mapping[str, str] | None = None,
    ) -> Ladder:
        """Index the ``ok`` records (JSON) of the reference campaigns.

        Repeats of a solve: the first by name. gfem BDDC rows count only at the
        overkill tolerance or tighter.
        """
        ladder = cls(tuple(frequencies), tuple(entries))
        versions = palace_versions or {}
        for row in sorted(rows, key=lambda r: (r["campaign"], r["name"])):
            run = row["run"]
            rung = str(run["case"]).removeprefix(PREFIX)
            if row.get("status") != "ok" or f"{PREFIX}{rung}" not in RUNGS:
                continue
            code, _, kind = str(run["solver"]).partition(":")
            solve = _Rung(
                ladder._matrix(row),
                row.get("case", {}).get("mesh_sha256"),
                row.get("case", {}).get("ndof"),
                f"{row['campaign']}/{row['name']}",
            )
            if code == "palace" and kind == ADAPTERS["palace"].direct_kind:
                version = versions.get(row["campaign"], "unknown")
                ladder.palace.setdefault(version, {}).setdefault(rung, solve)
            elif code == "gfem" and kind == ADAPTERS["gfem"].direct_kind:
                ladder.gfem.setdefault((rung, DIRECT), solve)
            elif code == "gfem" and (run.get("tolerance") or 1.0) <= OVERKILL_TOLERANCE:
                ladder.gfem.setdefault((rung, ITERATIVE), solve)
        return ladder

    def _matrix(self, row: Mapping[str, Any]) -> Matrix:
        outputs = row.get("outputs") or {}
        have = list(outputs.get("frequencies_hz") or [])
        s = outputs.get("s") or {}
        out = []
        for f in self.frequencies:
            if f not in have:
                raise LadderError(f"{row['name']}: no output at {f} Hz")
            i = have.index(f)
            try:
                out.append([complex(*s[e][i]) for e in self.entries])
            except KeyError as error:
                raise LadderError(f"{row['name']}: no S entry {error}") from error
        return out

    def _get(self, rung: str, kind: str = DIRECT) -> _Rung | None:
        return self.gfem.get((rung, kind))

    def _need(self, rung: str, kind: str = DIRECT) -> _Rung:
        found = self._get(rung, kind)
        if found is None:
            raise LadderError(f"no ok gfem {kind} record of {PREFIX}{rung}")
        return found

    @property
    def complete(self) -> bool:
        """Every solve the extrapolation and its bound need is there."""
        wanted = [("L0_p2", DIRECT), ("L1_p3", DIRECT), ("L2_p2", ITERATIVE)]
        l1 = self._get("L1_p2") or self._get("L1_p2", ITERATIVE)
        return l1 is not None and all(self._get(*w) for w in wanted)

    def pilot_rate(self) -> list[float]:
        """The pilot's rate estimate per frequency (:func:`pilot_rate`)."""
        return pilot_rate(
            self._need("L0_p2").s, self._need("L1_p2").s, self._need("L1_p3").s
        )

    def extrapolate(self) -> Extrapolation:
        """The reference from L0, L1 (direct, else BDDC) and L2 (BDDC) at p2."""
        if not self.complete:
            raise LadderError("the ladder needs L0 p2, L1 p2, L1 p3 and L2 p2")
        direct = self._get("L1_p2")
        iterative = self._get("L1_p2", ITERATIVE)
        return extrapolate(
            self._need("L0_p2").s,
            (direct or self._need("L1_p2", ITERATIVE)).s,
            self._need("L2_p2", ITERATIVE).s,
            l1p3=self._need("L1_p3").s,
            l1_iter=iterative.s if iterative else None,
            l1_direct=direct.s if direct else None,
        )

    def same_mesh_error(self, values: Matrix) -> dict[str, float]:
        """The gfem direct runs' distance to the reference on L0, per order."""
        out = {}
        for order in (1, 2, 3):
            found = self._get(f"L0_p{order}")
            if found is not None:
                out[str(order)] = _max_abs(found.s, values)
        return out

    def cross_checks(self, alpha: Sequence[float]) -> list[dict[str, Any]]:
        """The definition offset per Palace version with L0 and L1 at p2."""
        gfem0, gfem1 = self._get("L0_p2"), self._get("L1_p2")
        out = []
        for version, solves in sorted(self.palace.items()):
            p0, p1 = solves.get("L0_p2"), solves.get("L1_p2")
            if p0 is None or p1 is None or gfem0 is None or gfem1 is None:
                continue
            offset = definition_offset(
                _difference(p0.s, gfem0.s), _difference(p1.s, gfem1.s), alpha
            )
            out.append({"code": "palace", "version": version, **offset})
        return out

    def meshes(self) -> dict[str, dict[str, Any]]:
        """Mesh sha256 -> level and DOF per order (direct rows first)."""
        out: dict[str, dict[str, Any]] = {}
        rows = sorted(self.gfem.items(), key=lambda kv: (kv[0][1] != DIRECT, kv[0][0]))
        for (rung, _), solve in rows:
            if solve.mesh_sha256 is None or solve.ndof is None:
                continue
            level, order = int(rung[1]), rung.split("_p")[1]
            mesh = out.setdefault(solve.mesh_sha256, {"level": level, "dof": {}})
            mesh["dof"].setdefault(order, solve.ndof)
        for mesh in out.values():
            mesh["dof"] = dict(sorted(mesh["dof"].items()))
        return dict(sorted(out.items(), key=lambda kv: kv[1]["level"]))

    def records(self) -> list[str]:
        """``<campaign>/<record name>`` of every solve used."""
        found = [r.record for r in self.gfem.values()]
        found += [r.record for v in self.palace.values() for r in v.values()]
        return sorted(found)


# --- the command -------------------------------------------------------------------


def _load_rundirs(rundirs: Sequence[Path]) -> tuple[list[dict], dict[str, str], dict]:
    rows: list[dict] = []
    versions: dict[str, str] = {}
    facts: dict[str, Any] = {
        "cost": 0.0,
        "commits": set(),
        "ladder": set(),
        "campaigns": [],
    }
    for rundir in rundirs:
        manifest = records.load_campaign(rundir / "campaign.json")
        facts["campaigns"].append(manifest.id)
        commit = manifest.pins.gfem.get("commit")
        facts["commits"].add(commit)
        facts["cost"] += sum(j.billed_core_h or 0.0 for j in manifest.jobs)
        if "palace" in manifest.pins.solvers:
            versions[manifest.id] = manifest.pins.solvers["palace"]
        elif commit:
            facts["ladder"].add((manifest.pins.gfem.get("commit_utc") or "", commit))
        rows += [json.loads(p.read_text()) for p in records.record_paths(rundir)]
    return rows, versions, facts


def _pairs(values: Matrix, entries: Sequence[str]) -> dict[str, list[list[float]]]:
    return {
        e: [[row[i].real, row[i].imag] for row in values] for i, e in enumerate(entries)
    }


def _finite(values: Iterable[float]) -> list[float | None]:
    return [v if math.isfinite(v) else None for v in values]


def write_from(rundirs: Sequence[Path], case_dir: Path) -> int:
    """Extrapolate the ladder records into ``case_dir/reference.json``.

    With L2 missing, print the pilot's rate estimate and the go/no-go and write
    nothing.
    """
    case = load_case(case_dir)
    entries = tuple(f"{t},{f}" for t, f in case.quantities.s)
    frequencies = case.problem.frequencies()
    rows, versions, facts = _load_rundirs(rundirs)
    ladder = Ladder.from_records(
        rows, frequencies=frequencies, entries=entries, palace_versions=versions
    )
    if not ladder.complete:
        estimate = ladder.pilot_rate()
        go = min(estimate) >= RATE_CLIP[0]
        print("pilot rate estimate per frequency (vs L1 p3): ", end="")
        print(
            ", ".join(
                f"{f / 1e9:g} GHz {a:.2f}"
                for f, a in zip(frequencies, estimate, strict=True)
            )
        )
        print("go: plan the l2 stage" if go else "stop: pre-asymptotic, no L2")
        print("no reference written (needs L2 p2)")
        return 0
    ext = ladder.extrapolate()
    meshes = ladder.meshes()
    if case.mesh.sha256 not in meshes or meshes[case.mesh.sha256]["level"] != 0:
        raise LadderError(f"no L0 record on {case.name}'s pinned mesh")
    commits = sorted(c for c in facts["commits"] if c)
    # named after the newest gfem ladder commit: a rerun of the ladder that reuses
    # older cross-check campaigns still gets a new id
    ladder_commit = max(facts["ladder"])[1] if facts["ladder"] else None
    doc = refdoc.skeleton(
        case,
        commit=ladder_commit or (commits[0] if commits else "0000000"),
        now=dt.datetime.now(dt.UTC),
    )
    floors = ladder.same_mesh_error(ext.values)
    smallest = min(floors.values()) if floors else math.inf
    met = ext.bound <= TARGET_BOUND and ext.bound <= smallest / 10
    doc.update(
        meshes=meshes,
        method="richardson",
        quantities={
            "s": {
                "frequencies_hz": list(frequencies),
                "values": _pairs(ext.values, entries),
                "bound": ext.bound,
                "bound_per_frequency": ext.bound_per_frequency,
                "bound_kind": "absolute",
                "metric": "max_abs_dS",
            }
        },
        ceiling={k: 10 * v for k, v in floors.items()},
        same_mesh_error=floors,
        cross_checks=ladder.cross_checks(ext.alpha),
        inputs={"meshes": list(meshes)},
        runs=f"bench-data:references/{doc['id']}/",
        cost_core_h=round(facts["cost"], 3),
        target={"bound": TARGET_BOUND, "met": met},
    )
    doc["provenance"].update(
        script="benchmarks/references/microstrip.py",
        gfem_commits=commits,
        campaigns=sorted(facts["campaigns"]),
        records=ladder.records(),
        rate={
            "raw": _finite(ext.raw_alpha),
            "alpha": ext.alpha,
            "clip": list(RATE_CLIP),
            "terms": ext.terms,
        },
    )
    path = case_dir / refdoc.FILE
    changed = refdoc.write(path, doc)
    print(f"{path}: {'written' if changed else 'unchanged'} ({doc['id']})")
    print(f"bound {ext.bound:.3g}; same-mesh error per order {floors}")
    if not met:
        print(
            f"unresolved: the bound misses {TARGET_BOUND:g} or a tenth of the "
            "smallest same-mesh error; runs below 10 x bound are scored unresolved"
        )
    return 0
