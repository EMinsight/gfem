"""A structured triangle mesh of a symmetric slab crossed by line ports, no gmsh."""

from pathlib import Path

import numpy as np

NAMES = [
    (2, 1, "core"),
    (2, 2, "clad"),
    (1, 3, "outer"),
    (1, 4, "port1"),
    (1, 5, "port2"),
]


def write_slab(
    path: Path,
    thick: float,
    clear: float,
    *,
    h_core: float,
    h_clad: float,
    length: float = 0.2,
) -> Path:
    """A slab of thickness ``thick`` ("core") in "clad" to ``clear`` beyond each face.

    Cells are at most ``h_core`` in the core and ``h_clad`` outside; lengths in um.
    Line "port1" is at $x = 0$, "port2" at $x$ = ``length``, "outer" runs along
    $y = \\pm Y$. A gmsh msh 2.2 ASCII file.
    """
    half = thick / 2
    n_core = max(1, int(np.ceil(half / h_core - 1e-9)))
    n_clad = max(1, int(np.ceil(clear / h_clad - 1e-9)))
    up = np.concatenate(
        [np.linspace(0, half, n_core + 1), half + np.linspace(0, clear, n_clad + 1)[1:]]
    )
    ys = np.concatenate([-up[::-1], up[1:]])
    xs = np.array([0.0, length / 2, length])
    ny = len(ys)

    def node(i, j):
        return 1 + i * ny + j

    nodes = [(node(i, j), x, y) for i, x in enumerate(xs) for j, y in enumerate(ys)]
    elements = []
    for i in range(len(xs) - 1):
        for j in range(ny - 1):
            tag = 1 if abs(0.5 * (ys[j] + ys[j + 1])) < half else 2
            a, b, c, d = node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)
            elements += [(2, tag, (a, b, c)), (2, tag, (a, c, d))]
    for j in range(ny - 1):
        elements.append((1, 4, (node(0, j), node(0, j + 1))))
        elements.append((1, 5, (node(len(xs) - 1, j + 1), node(len(xs) - 1, j))))
    for i in range(len(xs) - 1):
        elements.append((1, 3, (node(i + 1, 0), node(i, 0))))
        elements.append((1, 3, (node(i, ny - 1), node(i + 1, ny - 1))))
    text = [
        "$MeshFormat",
        "2.2 0 8",
        "$EndMeshFormat",
        "$PhysicalNames",
        str(len(NAMES)),
    ]
    text += [f'{dim} {tag} "{name}"' for dim, tag, name in NAMES]
    text += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    text += [f"{k} {float(x)!r} {float(y)!r} 0" for k, x, y in nodes]
    text += ["$EndNodes", "$Elements", str(len(elements))]
    for e, (dim, tag, conn) in enumerate(elements, start=1):
        kind = 2 if dim == 2 else 1
        text.append(f"{e} {kind} 2 {tag} {tag} " + " ".join(map(str, conn)))
    text += ["$EndElements", ""]
    path.write_text("\n".join(text))
    return path
