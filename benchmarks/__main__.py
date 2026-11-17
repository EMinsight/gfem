"""Command line of the benchmark suite: ``python -m benchmarks <subcommand>``.

Dispatch only; each subcommand's work lives in ``benchmarks.harness`` (``report`` in
``benchmarks.report``).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from benchmarks import report
from benchmarks.harness import case, config, matrix, records, ship, slurm
from benchmarks.references import closed, microstrip
from benchmarks.references import document as refdoc

NOT_YET = 2


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
        campaign = matrix.override(
            config.load_campaign(args.campaign),
            cases=_axis(args.case),
            solvers=_axis(args.solver),
            stacks=_axis(args.stack),
            layouts=_axis(args.layout),
            repeats=args.repeats,
        )
        site = config.load_site(args.site)
        commit, commit_utc = matrix.source_commit()
        plan = matrix.expand(campaign, site, commit=commit)
        manifest = matrix.write(plan, args.out, site=site, commit_utc=commit_utc)
        slurm.copy_site(site, args.out)
    except (config.ConfigError, records.RecordError) as error:
        print(f"benchmarks plan: {error}", file=sys.stderr)
        return NOT_YET
    print(matrix.summary(plan, site, manifest))
    print(f"run directory: {args.out}")
    return 0


def _reference(args: argparse.Namespace) -> int:
    name = args.case
    if name == microstrip.CASE:
        if args.plan is not None:
            if args.out is None:
                args.parser.error("--plan needs --out RUNDIR")
            try:
                args.campaign = microstrip.campaign_file(args.plan)
            except config.ConfigError as error:
                print(f"benchmarks reference: {error}", file=sys.stderr)
                return NOT_YET
            args.list = False
            args.case = args.solver = args.stack = args.layout = args.repeats = None
            return _plan(args)
        if args.sources:
            return _guarded(
                "reference",
                lambda: microstrip.write_from(args.sources, args.cases / name),
            )
        print(
            "benchmarks reference: microstrip is a ladder: --plan STAGE --out RUNDIR, "
            "then --from RUNDIR..",
            file=sys.stderr,
        )
        return NOT_YET
    try:
        loaded = case.load_case(args.cases / name)
        commit = matrix.source_commit()[0]
        doc = closed.document(loaded, commit=commit)
        path = args.cases / name / refdoc.FILE
        changed = refdoc.write(path, doc)
    except (case.CaseError, ValueError, matrix.PlanError) as error:
        print(f"benchmarks reference: {error}", file=sys.stderr)
        return NOT_YET
    print(f"{path}: {'written' if changed else 'unchanged'} ({doc['id']})")
    return 0


def _axis(value: str | None) -> tuple[str, ...] | None:
    return None if value is None else tuple(v for v in value.split(",") if v)


def _rundir(args: argparse.Namespace) -> slurm.RunDir:
    return slurm.load_rundir(args.rundir, getattr(args, "site", None))


def _guarded(name: str, body: Callable[[], int]) -> int:
    try:
        return body()
    except (config.ConfigError, records.RecordError) as error:
        print(f"benchmarks {name}: {error}", file=sys.stderr)
        return NOT_YET


def _submit(args: argparse.Namespace) -> int:
    return _guarded(
        "submit",
        lambda: slurm.submit(
            _rundir(args),
            jobs=args.job,
            qos=args.qos,
            dry_run=args.dry_run,
            ceiling=args.ceiling,
        ),
    )


def _run(args: argparse.Namespace) -> int:
    return _guarded("run", lambda: slurm.run_local(_rundir(args), args.job))


def _step(args: argparse.Namespace) -> int:
    return _guarded(
        "step",
        lambda: slurm.step(
            _rundir(args), args.job, args.line, slurm.Context.from_env(local=False)
        ),
    )


def _status(args: argparse.Namespace) -> int:
    return _guarded("status", lambda: slurm.status(_rundir(args)))


def _close(args: argparse.Namespace) -> int:
    return _guarded("close", lambda: slurm.close(_rundir(args)))


def _ship(args: argparse.Namespace) -> int:
    def body() -> int:
        site = config.load_site(args.site) if args.site else None
        return ship.ship(
            args.rev,
            ship.Target.parse(args.to),
            mesh_store=site.ship_mesh_store if site else None,
            stores=tuple(args.store),
            dry_run=args.dry_run,
        )

    return _guarded("ship", body)


def _fetch(args: argparse.Namespace) -> int:
    return records.fetch(args.source, args.to, records_only=args.records_only)


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
    p = add("submit", "submit a planned run directory's jobs (login node)", _submit)
    p.add_argument("rundir", type=Path, metavar="RUNDIR")
    p.add_argument("--qos", help="QOS for every job (else by time limit)")
    p.add_argument("--job", action="append", metavar="J", help="only job J (repeat)")
    p.add_argument("--dry-run", action="store_true", help="print the sbatch lines")
    p.add_argument("--ceiling", type=float, help="core-hour ceiling (else site)")
    p.add_argument("--site", type=Path, help="site file (else RUNDIR/site.toml)")

    p = add("run", "run a planned run directory without Slurm", _run)
    p.add_argument("rundir", type=Path, metavar="RUNDIR")
    p.add_argument("--job", metavar="J")
    p.add_argument("--local", action="store_true", required=True)

    p = add("step", "run one step line of a job (inside the job)", _step)
    p.add_argument("rundir", type=Path, metavar="RUNDIR")
    p.add_argument("job", metavar="JOB")
    p.add_argument("line", type=int, metavar="LINE")

    p = add("status", "show the state of a run directory", _status)
    p.add_argument("rundir", type=Path, metavar="RUNDIR")

    p = add("close", "stamp unfinished runs and check campaign gates", _close)
    p.add_argument("rundir", type=Path, metavar="RUNDIR")


def _add_data(add: Adder) -> None:
    p = add("reference", "compute a suite case's converged reference", _reference)
    p.add_argument("case", metavar="CASE")
    p.add_argument("--site", type=Path, help="site file (--plan only)")
    p.add_argument(
        "--plan",
        metavar="STAGE",
        help="microstrip: plan a ladder stage (pilot, l2, palace-<version>)",
    )
    p.add_argument("--out", type=Path, metavar="RUNDIR", help="with --plan")
    p.add_argument(
        "--from",
        dest="sources",
        nargs="+",
        type=Path,
        metavar="RUNDIR",
        help="microstrip: extrapolate the fetched ladder records",
    )
    p.add_argument(
        "--cases",
        type=Path,
        default=config.CASES_DIR,
        metavar="DIR",
        help="directory of case directories (default: benchmarks/cases)",
    )

    p = add("ship", "ship a committed revision to the cluster", _ship)
    p.add_argument("rev", metavar="REV")
    p.add_argument("--to", required=True, metavar="HOST:ROOT")
    p.add_argument("--site", type=Path, help="target site file ([ship] mesh_store)")
    p.add_argument(
        "--store",
        action="append",
        default=[],
        type=Path,
        metavar="DIR",
        help="local mesh store holding external meshes (repeat)",
    )
    p.add_argument("--dry-run", action="store_true", help="print, send nothing")

    p = add("fetch", "fetch a run directory from the cluster", _fetch)
    p.add_argument("--from", dest="source", required=True, metavar="HOST:ROOT/RUNDIR")
    p.add_argument("--to", required=True, type=Path, metavar="DIR")
    p.add_argument(
        "--records-only",
        action="store_true",
        help="only the records-store set (manifest, records, per-job files)",
    )

    p = add("report", "write a campaign report from records", _report_cmd)
    p.add_argument("inputs", nargs="*", metavar="CAMPAIGN", help="campaign id or dir")
    p.add_argument("--out", type=Path, metavar="DIR")
    p.add_argument("--allow-mixed", action="store_true", help="read a mixed campaign")
    p.add_argument(
        "--rescore",
        action="append",
        default=[],
        metavar="CASE=ID",
        help="also score CASE against the archived reference ID",
    )
    p.add_argument("--no-figures", action="store_true", help="tables and JSON only")
    p.add_argument(
        "--check", type=Path, metavar="DIR", help="regenerate DIR's report and compare"
    )


def _report_cmd(args: argparse.Namespace) -> int:
    if args.check is not None:
        if args.inputs or args.out is not None:
            args.parser.error("--check DIR takes no CAMPAIGN and no --out")
        try:
            same, messages = report.check(args.check)
        except records.RecordError as error:
            print(f"benchmarks report --check: {error}", file=sys.stderr)
            return 1
        for line in messages:
            print(line)
        print(f"{args.check}: {'unchanged' if same else 'DRIFT'}")
        return 0 if same else 1
    if not args.inputs or args.out is None:
        args.parser.error("report needs CAMPAIGN.. and --out DIR (or --check DIR)")
    rescore = {}
    for item in args.rescore:
        case_name, sep, rid = item.partition("=")
        if not sep or not case_name or not rid:
            args.parser.error(f"--rescore {item!r} is not CASE=ID")
        rescore[case_name] = rid
    flags = {
        "allow_mixed": args.allow_mixed,
        "rescore": rescore,
        "no_figures": args.no_figures,
    }
    try:
        dirs = report.load.resolve(args.inputs, args.out)
        rid = report.generate(dirs, args.out, flags)
    except records.RecordError as error:
        print(f"benchmarks report: {error}", file=sys.stderr)
        return 1
    print(f"report {rid}: {args.out}")
    return 0


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
