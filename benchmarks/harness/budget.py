r"""The allocation's core budget: what ``submit`` may still spend.

Spend so far = the site's ``[budget] prior_core_h`` (jobs the CLI did not submit:
one-off runs, the stack check, test gates; updated by hand) + billed core-hours of every
job listed in a ``submitted.tsv`` under the site's output directory (and the run
directory being submitted), from sacct: ``CPUTimeRAW`` / 2 / 3600, the formula that
matches ``bsc_acct`` until it catches up. A job not finished yet (pending, running...)
counts at :math:`\text{time limit} \times \text{nodes} \times 112` core-hours until
it finishes. ``submit`` refuses when spend + the plan's estimate passes the ceiling
(``[budget] ceiling_core_h``, or ``--ceiling``).

Standard library only.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from benchmarks.harness.config import ConfigError, Runner
from benchmarks.harness.matrix import NODE_CORES

SUBMITTED = "submitted.tsv"
SUBMITTED_HEADER = "job\tslurm_id\tsubmitted_utc\tqos\ttime_limit\test_core_h"
UNFINISHED = (
    "PENDING",
    "RUNNING",
    "REQUEUED",
    "RESIZING",
    "SUSPENDED",
    "CONFIGURING",
    "COMPLETING",
)
"""sacct states of a job that may still bill: counted at its time limit."""


class BudgetError(ConfigError):
    """A submission that would pass the allocation's core-hour ceiling."""


def billed_core_h(cputime_raw: int) -> float:
    """Billed core-hours of a job: ``CPUTimeRAW`` / 2 / 3600."""
    return cputime_raw / 2 / 3600


def submitted_ids(path: Path) -> list[str]:
    """Slurm job ids of one ``submitted.tsv``."""
    ids = []
    for line in path.read_text().splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) > 1 and fields[1].strip():
            ids.append(fields[1].strip())
    return ids


def submitted_files(output_dir: Path, extra: Iterable[Path] = ()) -> list[Path]:
    """Every ``submitted.tsv`` under ``output_dir``, plus ``extra`` run directories'."""
    found = set(output_dir.rglob(SUBMITTED)) if output_dir.is_dir() else set()
    found |= {d / SUBMITTED for d in extra if (d / SUBMITTED).is_file()}
    return sorted(found)


def sacct_jobs(
    ids: Iterable[str],
    run: Runner = subprocess.run,
) -> dict[str, dict[str, str]]:
    """Sacct facts of each job id (allocation lines only)."""
    ids = sorted(set(ids))
    if not ids:
        return {}
    done = run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(ids),
            "--format=JobID,State,ElapsedRaw,CPUTimeRAW,NodeList,QOS,TimelimitRaw,NNodes",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise BudgetError(f"sacct failed ({done.returncode}): {done.stderr.strip()}")
    out = {}
    for line in done.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 6 or not fields[0].strip():
            continue
        out[fields[0].strip()] = {
            "state": fields[1].split()[0] if fields[1].strip() else "",
            "elapsed_raw": fields[2],
            "cputime_raw": fields[3],
            "nodelist": fields[4],
            "qos": fields[5],
            "timelimit_raw": fields[6] if len(fields) > 6 else "",
            "nnodes": fields[7] if len(fields) > 7 else "",
        }
    return out


@dataclass(frozen=True)
class Spend:
    """Core-hours spent on the allocation so far."""

    prior_core_h: float
    jobs_core_h: float
    jobs: int

    @property
    def total(self) -> float:
        """Prior plus the CLI's jobs."""
        return self.prior_core_h + self.jobs_core_h


def job_core_h(facts: Mapping[str, str]) -> float:
    r"""Core-hours a job counts for: billed once finished, else its time limit.

    An unfinished job counts
    :math:`\mathrm{TimelimitRaw} / 60 \times \mathrm{NNodes} \times 112`
    (``TimelimitRaw`` in minutes).
    """
    if facts.get("state") in UNFINISHED:
        minutes, nodes = facts.get("timelimit_raw", ""), facts.get("nnodes", "")
        if minutes.isdigit() and nodes.isdigit():
            return int(minutes) / 60 * int(nodes) * NODE_CORES
    raw = facts.get("cputime_raw") or ""
    return billed_core_h(int(raw)) if raw.isdigit() else 0.0


def spend(
    prior_core_h: float,
    files: Iterable[Path],
    run: Runner = subprocess.run,
) -> Spend:
    """The allocation's spend: ``prior_core_h`` plus sacct over the listed jobs.

    Pending and running jobs count at their time limit (:func:`job_core_h`).
    """
    ids = sorted({i for path in files for i in submitted_ids(path)})
    facts: Mapping[str, Mapping[str, str]] = sacct_jobs(ids, run) if ids else {}
    core_h = sum(job_core_h(f) for f in facts.values())
    return Spend(prior_core_h, core_h, len(ids))


def check(spent: Spend, estimate_core_h: float, ceiling_core_h: float) -> None:
    """Refuse when ``spent`` plus ``estimate_core_h`` passes ``ceiling_core_h``."""
    total = spent.total + estimate_core_h
    if total > ceiling_core_h:
        raise BudgetError(
            f"budget: spend {spent.total:.1f} core-h (prior {spent.prior_core_h:.1f} "
            f"+ {spent.jobs} jobs {spent.jobs_core_h:.1f}) + estimate "
            f"{estimate_core_h:.1f} = {total:.1f} > ceiling {ceiling_core_h:.0f}; "
            "raising the ceiling needs the user"
        )
