"""A three-square strip of triangles, written as a gmsh file: a small 2D mesh."""

from pathlib import Path

# Three unit squares side by side, areas "a" | "b" | "c", each two triangles.
# Line "ab" runs up x = 1 (+y), "cb" down x = 2 (-y); "outer" goes round
# anticlockwise, except its segment on x = 0, which runs up (+y).
NODES = [(x, y) for y in (0, 1) for x in (0, 1, 2, 3)]  # gmsh nodes 1-8
TRIANGLES = {
    1: [(1, 2, 6), (1, 6, 5)],
    2: [(2, 3, 7), (2, 7, 6)],
    3: [(3, 4, 8), (3, 8, 7)],
}
LINES = {
    11: [(2, 6)],
    12: [(7, 3)],
    13: [(1, 2), (2, 3), (3, 4), (4, 8), (8, 7), (7, 6), (6, 5), (1, 5)],
}
NAMES = {(2, 1): "a", (2, 2): "b", (2, 3): "c", (1, 11): "ab", (1, 12): "cb"}
NAMES[1, 13] = "outer"


def write_strip(path: Path) -> Path:
    """The three-square strip as a gmsh msh 2.2 ASCII file, in um."""
    elements = [
        f"2 2 {tag} {tag} {' '.join(map(str, t))}"
        for tag, tris in TRIANGLES.items()
        for t in tris
    ]
    elements += [
        f"1 2 {tag} {tag} {' '.join(map(str, line))}"
        for tag, lines in LINES.items()
        for line in lines
    ]
    text = [
        "$MeshFormat",
        "2.2 0 8",
        "$EndMeshFormat",
        "$PhysicalNames",
        str(len(NAMES)),
        *(f'{dim} {tag} "{name}"' for (dim, tag), name in NAMES.items()),
        "$EndPhysicalNames",
        "$Nodes",
        str(len(NODES)),
        *(f"{i} {x} {y} 0" for i, (x, y) in enumerate(NODES, start=1)),
        "$EndNodes",
        "$Elements",
        str(len(elements)),
        *(f"{i} {e}" for i, e in enumerate(elements, start=1)),
        "$EndElements",
        "",
    ]
    path.write_text("\n".join(text))
    return path
