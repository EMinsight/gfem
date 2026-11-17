# Benchmark suite

## Records store

Suite records and campaign manifests are kept on `bench-data`, an orphan branch of
the gfem repository, never in main history (ADR 0402). Create its worktree once,
next to the gfem checkout:

```sh
git worktree add --orphan -b bench-data ../gfem-bench-data
```

After a campaign is closed on the cluster, copy its store set into the worktree:

```sh
python -m benchmarks fetch --from HOST:ROOT/campaigns/<id> \
    --to ../gfem-bench-data/campaigns/<id> --records-only
```

`--records-only` writes exactly `campaign.json`, `records/<case>/*.json` (superseded
records left out) and `jobs/<jobid>/{node.txt, pip-freeze.txt, machine.json}`; logs
stay in the run directory and in the full local copy (`fetch` without
`--records-only`). It exits 1 if a fetched file still names a home, scratch or
`/gpfs` path. Commit by hand in the worktree; the CLI never commits or pushes, and
pushing `bench-data` is a separate decision.
