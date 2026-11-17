"""Generators of the benchmark suite's helper-built meshes (provenance only).

    python tools/bench_meshes.py lcavity OUT_DIR

writes ``OUT_DIR/lcav16.msh``; pin it with
``python -m benchmarks case pin lcavity OUT_DIR/lcav16.msh --unit mm``. The pinned
file is authoritative (ADR 0405): no harness step, campaign or test re-runs a
generator. A human runs one only to author a new case or a ladder level, which gets
its own pin. Imports ``tools/structured_meshes`` only, never ``tests/``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import structured_meshes as sm

#: Cubic cells per millimetre of the L-cavity mesh (``l_cavity(2)`` split three
#: times: 246 416 DOF at order 2).
LCAVITY_CELLS_PER_MM = 16
#: Cavity height in millimetres.
LCAVITY_HEIGHT_MM = 0.5


def lcavity(out: Path) -> Path:
    r"""Write the L-cavity mesh into ``out`` and return its path.

    The L section :math:`[-1, 1]^2 \setminus [0, 1] \times [-1, 0]` (mm), extruded
    to 0.5 mm, in cubic cells; one volume ``air`` and its PEC ``wall``.
    """
    n = LCAVITY_CELLS_PER_MM
    xy = np.linspace(-1, 1, 2 * n + 1)
    z = np.linspace(0, LCAVITY_HEIGHT_MM, round(n * LCAVITY_HEIGHT_MM) + 1)
    mesh = sm.carve(
        (xy, xy.copy(), z),
        lambda c: None if c[0] > 0 and c[1] < 0 else "air",
        lambda _: "wall",
    )
    return sm.write_msh22(out / f"lcav{n}.msh", mesh)


GENERATORS = {"lcavity": lcavity}


def main() -> None:
    """Write the named case's mesh into OUT and print its sha256."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("case", choices=sorted(GENERATORS))
    parser.add_argument("out", type=Path, metavar="OUT_DIR")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = GENERATORS[args.case](args.out)
    print(f"{path}: sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
