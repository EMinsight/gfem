"""Command line of the benchmark suite: ``python -m benchmarks <subcommand>``.

Dispatch only; each subcommand's work lives in ``benchmarks.harness``. Subcommands
not built yet exit with status 2 and say so.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from benchmarks.harness import case, config

NOT_YET = 2


def _not_yet(args: argparse.Namespace) -> int:
    print(f"benchmarks {args.command}: not yet implemented", file=sys.stderr)
    return NOT_YET


def _print_axes() -> None:
    cases = config.known_cases()
    print("case:   " + (", ".join(cases) if cases else "(no committed case yet)"))
    print("solver: " + ", ".join(config.SOLVERS))
    print("stack:  " + ", ".join(config.STACKS))
    print("layout: " + ", ".join(config.NAMED_LAYOUTS) + ", RxT (ranks x threads)")


def _plan(args: argparse.Namespace) -> int:
    if args.list:
        _print_axes()
        return 0
    if args.campaign is None or args.out is None:
        args.parser.error("plan needs CAMPAIGN and --out RUNDIR (or --list)")
    try:
        config.load_campaign(args.campaign)
        config.load_site(args.site)
    except config.ConfigError as error:
        print(f"benchmarks plan: {error}", file=sys.stderr)
        return NOT_YET
    return _not_yet(args)


Handler = Callable[[argparse.Namespace], int]
Adder = Callable[[str, str, Handler], argparse.ArgumentParser]


def _add_plan(add: Adder) -> None:
    p = add("plan", "expand a campaign file into a run directory", _plan)
    p.add_argument("campaign", type=Path, nargs="?", metavar="CAMPAIGN")
    p.add_argument("--case", help="comma-separated suite cases")
    p.add_argument("--solver", help="comma-separated code:kind")
    p.add_argument("--stack", help="comma-separated software stacks")
    p.add_argument("--layout", help="comma-separated default, best or RxT")
    p.add_argument("--repeats", type=int)
    p.add_argument("--site", type=Path, help="site file (else $GFEM_BENCH_SITE)")
    p.add_argument("--out", type=Path, metavar="RUNDIR")
    p.add_argument("--list", action="store_true", help="print the axis values")


def _add_execution(add: Adder) -> None:
    p = add("submit", "submit a planned run directory's jobs (login node)", _not_yet)
    p.add_argument("rundir", type=Path, metavar="RUNDIR")
    p.add_argument("--qos")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ceiling", type=float, help="core-hour ceiling (else site)")
    p.add_argument("--site", type=Path)

    p = add("run", "run a planned run directory without Slurm", _not_yet)
    p.add_argument("rundir", type=Path, metavar="RUNDIR")
    p.add_argument("--job", metavar="J")
    p.add_argument("--local", action="store_true", required=True)

    p = add("step", "run one step line of a job (inside the job)", _not_yet)
    p.add_argument("rundir", type=Path, metavar="RUNDIR")
    p.add_argument("job", metavar="JOB")
    p.add_argument("line", type=int, metavar="LINE")

    p = add("status", "show the state of a run directory", _not_yet)
    p.add_argument("rundir", type=Path, metavar="RUNDIR")

    p = add("close", "stamp unfinished runs and check campaign gates", _not_yet)
    p.add_argument("rundir", type=Path, metavar="RUNDIR")


def _add_data(add: Adder) -> None:
    p = add("reference", "compute a suite case's converged reference", _not_yet)
    p.add_argument("case", metavar="CASE")
    p.add_argument("--site", type=Path)

    p = add("ship", "ship a committed revision to the cluster", _not_yet)
    p.add_argument("rev", metavar="REV")
    p.add_argument("--to", required=True, metavar="HOST:ROOT")

    p = add("fetch", "fetch a run directory from the cluster", _not_yet)
    p.add_argument("--from", dest="source", required=True, metavar="HOST:ROOT/RUNDIR")
    p.add_argument("--to", required=True, type=Path, metavar="DIR")
    p.add_argument("--records-only", action="store_true")

    p = add("report", "write a campaign report from records", _not_yet)
    p.add_argument("inputs", nargs="+", metavar="RECORDS")
    p.add_argument("--out", required=True, type=Path, metavar="DIR")
    p.add_argument("--allow-mixed", action="store_true")
    p.add_argument("--rescore", action="append", default=[], metavar="CASE=ID")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--check", type=Path, metavar="DIR")


def _stores(args: argparse.Namespace) -> tuple[Path, ...]:
    """Mesh stores of the site file, when one is given or set in the environment."""
    if args.site is None and not os.environ.get(config.SITE_ENV):
        return ()
    return tuple(Path(s) for s in config.load_site(args.site).data_stores)


def _report(pinned: case.Pinned, case_dir: Path) -> None:
    where = "committed" if pinned.committed else "external"
    print(f"{case_dir.name}: sha256 {pinned.sha256}, {pinned.bytes} bytes")
    print(f"mesh {where}: {pinned.path}")


def _case_pin(args: argparse.Namespace) -> int:
    case_dir = args.cases / args.name
    try:
        pinned = case.pin(case_dir, args.mesh, args.unit, _stores(args))
    except (case.CaseError, config.ConfigError) as error:
        print(f"benchmarks case pin: {error}", file=sys.stderr)
        return NOT_YET
    _report(pinned, case_dir)
    return 0


def _case_import(args: argparse.Namespace) -> int:
    case_dir = args.cases / args.name
    frequencies = None
    if args.frequencies_ghz is not None:
        frequencies = [float(f) * 1e9 for f in args.frequencies_ghz.split(",")]
    try:
        pinned = case.import_case(
            args.config, args.mesh, case_dir, _stores(args), frequencies
        )
    except (case.CaseError, config.ConfigError) as error:
        print(f"benchmarks case import: {error}", file=sys.stderr)
        return NOT_YET
    _report(pinned, case_dir)
    return 0


def _add_case(sub: argparse._SubParsersAction) -> None:
    group = sub.add_parser("case", help="import or pin suite cases")
    case_sub = group.add_subparsers(dest="case_command", required=True)
    p = case_sub.add_parser("import", help="import a solver config as a case file")
    p.set_defaults(run=_case_import, parser=p, command="case import")
    p.add_argument("config", type=Path, metavar="CONFIG.json")
    p.add_argument("mesh", type=Path, metavar="MESH")
    p.add_argument("--name", required=True)
    p.add_argument(
        "--frequencies-ghz",
        metavar="F1,F2,..",
        help="driven frequencies (replace the config's)",
    )
    _case_paths(p)
    p = case_sub.add_parser("pin", help="pin a case's mesh")
    p.set_defaults(run=_case_pin, parser=p, command="case pin")
    p.add_argument("name", metavar="NAME")
    p.add_argument("mesh", type=Path, metavar="MESH")
    p.add_argument("--unit", required=True, choices=tuple(case.UNITS), metavar="U")
    _case_paths(p)


def _case_paths(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--cases",
        type=Path,
        default=config.CASES_DIR,
        metavar="DIR",
        help="directory of case directories (default: benchmarks/cases)",
    )
    p.add_argument("--site", type=Path, help="site file naming the mesh stores")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Benchmark suite: plan, run and report campaigns.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="SUBCOMMAND")

    def add(name: str, help_: str, run: Handler) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(run=run, parser=p)
        return p

    _add_plan(add)
    _add_execution(add)
    _add_data(add)
    _add_case(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the subcommand; return the exit status."""
    args = _parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
