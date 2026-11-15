"""Driven cases around the refusals of the S estimate (ADRs 0134, 0139, 0140).

Each builder takes the directory to write its mesh to and its refinement
level; every mesh is structured (`structured_meshes`, no gmsh), in mm.

- Plate line, `plate_line_model`: a parallel-plate line 8 x 2 x 1 mm whose
  plates are the y walls "plate1" and "plate2", so its TEM field E_y is
  tangential to the z walls "side"; 200 ohm lumped ports along +y, at
  `PLATE_LINE_FREQUENCY`. (4n, n, 2n) cells. The plates and the side walls
  take any boundary: `Conductor` plates (admitted), `Impedance` or
  `Absorbing` side walls (refused).
- Split line, `split_line_model`: a parallel-plate line 10 x 2 x 1 mm (PEC
  plates z = 0, 1) in two materials "a" and "b", with lumped ports p1
  (50 ohm) and p2 (30 ohm) along +z, or wave ports. "beside" splits it at
  y = 1 (the interface parallel to E and meeting each port face), "across"
  at z = 0.5 (the interface crossing E). n cells along x and c across
  (2c in y and z for "beside", c in y and per layer for "across").
- Contrast line, `contrast_line_model`: a line 3 x 0.5 x 1 mm (PEC plates
  z = 0, 1) split "beside" at y = 0.25 into "a" and "b", with a "core"
  block (1 < x < 2, z < 0.5) that makes S11 nonzero; cells 6k x 2k x 2k,
  isotropic (h = 0.5 / k mm). Wave ports, or lumped ports p1 (50 ohm) and
  p2 (30 ohm) along +z. References `CONTRAST_LINE_S`: order 4 on the k = 2
  mesh, for the k = 1 mesh.
- Metal floor, `metal_floor_model`: a parallel-plate guide 1 x 0.3 x 0.6 mm
  (PEC plates z = 0, 0.6) whose lower 0.3 mm hold `DRUDE_METAL` under air,
  200 ohm lumped ports along +z; (10n, 1, 6n) cells. "wedge" puts the metal
  only at 0.3 < x < 0.7 mm (two 90 degree metal edges in air, full-height
  ports); "flat" runs it along the whole guide and puts each port on
  z > 0.4 mm of its end face (no metal edge, no port on the metal).
  References `METAL_FLOOR_S`: order 4 on the same mesh.
"""

from pathlib import Path

import numpy as np
import structured_meshes as sm

import gfem

#: The plate line's frequency, Hz.
PLATE_LINE_FREQUENCY = 10e9

#: A Drude metal, plasma frequency 300 GHz, damping 5 GHz.
DRUDE_METAL = gfem.PermittivityLaw(
    terms=[gfem.Drude(plasma_frequency=300e9, damping=5e9)]
)
#: Frequencies where `DRUDE_METAL` has Re eps -29.7, -19.9 and -1.5.
METAL_DEEP = 53881590608.03247
METAL_EDGE = 65465367070.797714
METAL_PLASMONIC = 300e9 / np.sqrt(2.5)

#: S of the metal floor at `METAL_DEEP`, (geometry, n) -> {(to, from): S},
#: order 4 on the same mesh (direct). Its own error, |S3 - S4|, is 0.24 to
#: 0.28 of |S2 - S4| (the lumped port edges; an eps_r 2 floor shows the same).
METAL_FLOOR_S = {
    ("flat", 1): {
        ("p1", "p1"): 0.9266001034516802 - 0.35557391764073437j,
        ("p2", "p1"): 0.026107458871137745 + 0.06171008064900523j,
        ("p2", "p2"): 0.9267226174737748 - 0.35522285771355794j,
    },
    ("flat", 2): {
        ("p1", "p1"): 0.9290473304245812 - 0.348894765893061j,
        ("p2", "p1"): 0.024832692521190188 + 0.05955488226714988j,
        ("p2", "p2"): 0.929118013557952 - 0.34868469719989315j,
    },
    ("wedge", 1): {
        ("p1", "p1"): 0.700820192383554 + 0.03136016652949589j,
        ("p2", "p1"): 0.026684880675584374 - 0.7035456191620519j,
        ("p2", "p2"): 0.7008198269579506 + 0.03136011400682114j,
    },
    ("wedge", 2): {
        ("p1", "p1"): 0.7011469981727683 + 0.03168759910542668j,
        ("p2", "p1"): 0.02702772487681145 - 0.7032339177628041j,
        ("p2", "p2"): 0.7011468409950194 + 0.03168767326912959j,
    },
}

_TOL = 1e-9


def _model(directory: Path, name: str, cells: tuple, **kwargs) -> gfem.Model:
    path = sm.write_msh22(Path(directory) / f"{name}.msh", cells)
    return gfem.Model(gfem.Mesh.from_gmsh(path, length_unit="mm"), **kwargs)


def _lumped(axis: str, r1: float, r2: float) -> list:
    return [
        gfem.LumpedPort("p1", "port1", axis, resistance=r1),
        gfem.LumpedPort("p2", "port2", axis, resistance=r2),
    ]


def _waves() -> list:
    return [gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")]


def plate_line_model(
    directory: Path,
    n: int,
    *,
    plates: tuple = (gfem.PEC("plate1"), gfem.PEC("plate2")),
    side: object = None,
) -> gfem.Model:
    """The plate line: ``plates``, and ``side`` on the z walls (natural if None)."""
    boundaries = [*plates] + ([side] if side is not None else [])
    return _model(
        directory,
        f"plate{n}",
        sm.sheet_line(8.0, 2.0, 1.0, (4 * n, n, n)),
        materials={"lower": gfem.Material(), "upper": gfem.Material()},
        boundaries=boundaries,
        ports=_lumped("+y", 200.0, 200.0),
    )


def _split_cells(geometry: str, n: int, c: int) -> tuple:
    length, width, height = 10.0, 2.0, 1.0
    if geometry == "across":
        layers = (("a", height / 2), ("b", height / 2))
        return sm.strip_line(length, width, height, (n, c, c), layers=layers)
    edges = (
        np.linspace(0, length, n + 1),
        np.linspace(0, width, 2 * c + 1),
        np.linspace(0, height, 2 * c + 1),
    )

    def volume(p):
        return "a" if p[1] < width / 2 else "b"

    def surface(p):
        if abs(p[0]) < _TOL:
            return "port1"
        if abs(p[0] - length) < _TOL:
            return "port2"
        if abs(p[2]) < _TOL:
            return "bottom"
        if abs(p[2] - height) < _TOL:
            return "top"
        return ""

    return sm.structured(edges, volume, surface)


def split_line_model(
    directory: Path,
    geometry: str,
    halves: tuple[gfem.Material, gfem.Material],
    n: int,
    c: int = 1,
    *,
    ports: str = "lumped",
) -> gfem.Model:
    """The split line, "a" and "b" of ``halves``.

    ``geometry`` is "beside" or "across", ``ports`` "lumped" or "wave".
    """
    return _model(
        directory,
        f"split-{geometry}{n}-{c}",
        _split_cells(geometry, n, c),
        materials=dict(zip(("a", "b"), halves, strict=True)),
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=_lumped("+z", 50.0, 30.0) if ports == "lumped" else _waves(),
    )


#: The contrast line's materials, contrast -> (a, b, core).
CONTRAST_LINE_MATERIALS = {
    "mu": (
        gfem.Material(mu_r=2.5),
        gfem.Material(),
        gfem.Material(eps_r=3.0, mu_r=2.5),
    ),
    "sigma": (
        gfem.Material(eps_r=3.0, conductivity=0.5),
        gfem.Material(eps_r=3.0),
        gfem.Material(eps_r=6.0),
    ),
}
#: The contrast line's frequencies, Hz.
CONTRAST_LINE_FREQUENCIES = (10e9, 30e9)


def contrast_line_model(
    directory: Path, contrast: str, k: int, *, ports: str
) -> gfem.Model:
    """The contrast line, ``contrast`` "mu" or "sigma", ``ports`` "wave" or "lumped"."""
    length, width, height = 3.0, 0.5, 1.0
    edges = (
        np.linspace(0, length, 6 * k + 1),
        np.linspace(0, width, 2 * k + 1),
        np.linspace(0, height, 2 * k + 1),
    )

    def volume(p):
        if 1.0 < p[0] < 2.0 and p[2] < height / 2:
            return "core"
        return "a" if p[1] < width / 2 else "b"

    def surface(p):
        if abs(p[0]) < _TOL:
            return "port1"
        if abs(p[0] - length) < _TOL:
            return "port2"
        if abs(p[2]) < _TOL:
            return "bottom"
        if abs(p[2] - height) < _TOL:
            return "top"
        return ""

    a, b, core = CONTRAST_LINE_MATERIALS[contrast]
    return _model(
        directory,
        f"contrast{k}",
        sm.structured(edges, volume, surface),
        materials={"a": a, "b": b, "core": core},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=_lumped("+z", 50.0, 30.0) if ports == "lumped" else _waves(),
    )


def metal_floor_model(
    directory: Path, geometry: str, n: int, floor: object = DRUDE_METAL
) -> gfem.Model:
    """The metal floor, ``geometry`` "wedge" or "flat"; ``floor`` its permittivity."""
    length, width, metal, air = 1.0, 0.3, 0.3, 0.3
    edges = (
        np.linspace(0, length, 10 * n + 1),
        np.linspace(0, width, 2),
        np.concatenate(
            [
                np.linspace(0, metal, 3 * n + 1),
                metal + np.linspace(0, air, 3 * n + 1)[1:],
            ]
        ),
    )
    block = (0.3, 0.7) if geometry == "wedge" else (0.0, length)
    port_from = metal + 0.1 if geometry == "flat" else 0.0

    def volume(p):
        return "metal" if p[2] < metal and block[0] < p[0] < block[1] else "air"

    def surface(p):
        for x, name in ((0.0, "port1"), (length, "port2")):
            if abs(p[0] - x) < _TOL:
                return name if p[2] > port_from else "end"
        if abs(p[2]) < _TOL:
            return "bottom"
        if abs(p[2] - metal - air) < _TOL:
            return "top"
        return ""

    return _model(
        directory,
        f"metal-{geometry}{n}",
        sm.structured(edges, volume, surface),
        materials={"metal": gfem.Material(eps_r=floor), "air": gfem.Material()},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=_lumped("+z", 200.0, 200.0),
    )
