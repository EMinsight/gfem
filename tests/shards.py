"""Balanced test shards from recorded durations.

``tests/durations.json`` maps ``<junit classname>::<name>`` to seconds (the keys
pytest writes in junit xml). `assign` deals tests longest first, each to the
least-loaded shard, so every shard job computes the same partition from the same
selection and file.
"""

import json
import re
import statistics
from pathlib import Path

DEFAULT_SECONDS = 1.0


def junit_key(nodeid: str) -> str:
    """The junit-xml ``classname::name`` of a node id.

    Follows pytest's junit naming: the path loses ``.py`` and ``/`` becomes ``.``,
    class names join the module with ``.``, and the last part (function with its
    parameter id) is the name.
    """
    names = nodeid.split("::")
    names[0] = re.sub(r"\.py$", "", names[0].replace("/", "."))
    return ".".join(names[:-1]) + "::" + names[-1]


def load(path: Path | str) -> dict[str, float]:
    """The recorded seconds by junit key; empty when the file is missing."""
    path = Path(path)
    if not path.exists():
        return {}
    return {key: float(value) for key, value in json.loads(path.read_text()).items()}


def assign(costs: dict[str, float], n: int) -> dict[str, int]:
    """Shard of each test: longest first onto the least-loaded shard.

    Tests are taken in the order ``(-cost, nodeid)`` and a tie between shards goes
    to the lowest index, so the result depends only on ``costs`` and ``n``.
    """
    load_by_shard = [0.0] * n
    owner: dict[str, int] = {}
    for nodeid in sorted(costs, key=lambda i: (-costs[i], i)):
        shard = min(range(n), key=lambda s: (load_by_shard[s], s))
        load_by_shard[shard] += costs[nodeid]
        owner[nodeid] = shard
    return owner


def costs_of(nodeids: list[str], known: dict[str, float]) -> dict[str, float]:
    """Cost of each node id: its recorded seconds, else the file's median.

    With no recorded duration at all, an unknown test costs one second.
    """
    default = statistics.median(known.values()) if known else DEFAULT_SECONDS
    return {nodeid: known.get(junit_key(nodeid), default) for nodeid in nodeids}
