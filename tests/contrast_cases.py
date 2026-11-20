"""Driven cases around the refusals of the S estimate.

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
  (2c in y and z for "beside", c in y and per layer for "across"). The y
  walls "side" are natural (PMC). References `SPLIT_LINE_S`: order 4 on the
  (2n, 2c) mesh.
- Contrast line, `contrast_line_model`: a line 3 x 0.5 x 1 mm (PEC plates
  z = 0, 1) split "beside" at y = 0.25 into "a" and "b", with a "core"
  block (1 < x < 2, z < 0.5) that makes S11 nonzero; cells 6k x 2k x 2k,
  isotropic (h = 0.5 / k mm). Wave ports, or lumped ports p1 (50 ohm) and
  p2 (30 ohm) along +z. Its y walls are left unnamed, as measured: the
  recovery does not constrain them. References `CONTRAST_LINE_S`:
  order 4 on the k = 2 mesh, for the k = 1 mesh; |S4(1) - S4(2)| is at most
  0.09 of the k = 1 error.
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
METAL_PLASMONIC = 189736659610.10275  # 300 GHz / sqrt(2.5)

#: S of the metal floor, (geometry, n, frequency) -> {(to, from): S}, order 4
#: on the same mesh (direct). At `METAL_DEEP` its own error, |S3 - S4|, is
#: 0.24 to 0.28 of |S2 - S4| (the lumped port edges; an eps_r 2 floor shows
#: the same); at `METAL_PLASMONIC` the wedge does not converge.
METAL_FLOOR_S = {
    ("flat", 1, METAL_DEEP): {
        ("p1", "p1"): 0.9266001034516802 - 0.35557391764073437j,
        ("p2", "p1"): 0.026107458871137745 + 0.06171008064900523j,
        ("p2", "p2"): 0.9267226174737748 - 0.35522285771355794j,
    },
    ("flat", 2, METAL_DEEP): {
        ("p1", "p1"): 0.9290473304245812 - 0.348894765893061j,
        ("p2", "p1"): 0.024832692521190188 + 0.05955488226714988j,
        ("p2", "p2"): 0.929118013557952 - 0.34868469719989315j,
    },
    ("wedge", 1, METAL_DEEP): {
        ("p1", "p1"): 0.700820192383554 + 0.03136016652949589j,
        ("p2", "p1"): 0.026684880675584374 - 0.7035456191620519j,
        ("p2", "p2"): 0.7008198269579506 + 0.03136011400682114j,
    },
    ("wedge", 2, METAL_DEEP): {
        ("p1", "p1"): 0.7011469981727683 + 0.03168759910542668j,
        ("p2", "p1"): 0.02702772487681145 - 0.7032339177628041j,
        ("p2", "p2"): 0.7011468409950194 + 0.03168767326912959j,
    },
    ("wedge", 2, METAL_PLASMONIC): {
        ("p1", "p1"): 0.8065231757869953 - 0.12172362966352918j,
        ("p2", "p1"): -0.16353330692936133 + 0.01607703368872895j,
        ("p2", "p2"): 0.821387722029856 - 0.11759759997426214j,
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
        if abs(p[1] - width / 2) < _TOL:
            return "a__b"
        return "side"

    return sm.structured(edges, volume, surface)


#: S of the split line, (geometry, contrast, n, c, frequency) -> {(to, from):
#: S}: order 4 on the (2n, 2c) mesh (direct). "eps" is eps_r 3 under air,
#: "mu" mu_r 1 beside mu_r 2.5.
SPLIT_LINE_S = {
    ("across", "eps", 4, 2, 3e9): {
        ("p1", "p1"): 0.652828978989151 + 0.45189462783745926j,
        ("p2", "p1"): 0.2720854710527987 - 0.5436035487143068j,
        ("p2", "p2"): 0.7530176122902357 + 0.2517823043821935j,
    },
    ("beside", "mu", 16, 1, 15e9): {
        ("p1", "p1"): 0.7268955786707814 + 0.42223818254558315j,
        ("p2", "p1"): -0.2163215231862367 + 0.433800316564328j,
        ("p2", "p2"): 0.8185961506078079 + 0.24391921819533885j,
    },
}
#: The split line's contrasts, name -> its halves "a" and "b".
SPLIT_LINE_HALVES = {
    "eps": (gfem.Material(eps_r=3.0), gfem.Material()),
    "mu": (gfem.Material(), gfem.Material(mu_r=2.5)),
}


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

#: S of the contrast line, (contrast, ports) -> {(to, from): S at each of
#: `CONTRAST_LINE_FREQUENCIES`}: order 4 on the k = 2 mesh (direct).
CONTRAST_LINE_S = {
    ("mu", "wave"): {
        ("p1", "p1"): (
            -0.01665794169551882 - 0.01385618609280437j,
            -0.02175857215718524 + 0.04064090887948474j,
        ),
        ("p2", "p1"): (
            0.639482222979153 - 0.7684951754246864j,
            -0.8801958499699809 - 0.47227699604203033j,
        ),
        ("p2", "p2"): (
            -0.016657989513285676 - 0.013856128617426867j,
            -0.02175811850684528 + 0.04064115210264601j,
        ),
    },
    ("sigma", "lumped"): {
        ("p1", "p1"): (
            0.9586828985587541 + 0.1054721611772817j,
            0.8019349297480147 + 0.37620687412510545j,
        ),
        ("p2", "p1"): (
            0.011493215414500295 - 0.2037593473320894j,
            -0.14717350929785367 + 0.30646872532831676j,
        ),
        ("p2", "p2"): (
            0.9680087573071381 + 0.06354641923929398j,
            0.8788955045999034 + 0.22573650319897975j,
        ),
    },
}


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
        if abs(p[1]) < _TOL or abs(p[1] - width) < _TOL:
            return "side"
        return ""

    return _model(
        directory,
        f"metal-{geometry}{n}",
        sm.structured(edges, volume, surface),
        materials={"metal": gfem.Material(eps_r=floor), "air": gfem.Material()},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=_lumped("+z", 200.0, 200.0),
    )
