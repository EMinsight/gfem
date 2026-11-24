"""GDS export: the record stream, snapping, keyholes, splitting, errors, read-back."""

import struct
from datetime import datetime

import numpy as np
import pytest

from gfem import design

UM = 1e-6

# GDSII record types (Calma GDSII Stream Format), as (record << 8) | data type
HEADER, BGNLIB, LIBNAME, UNITS, ENDLIB = 0x0002, 0x0102, 0x0206, 0x0305, 0x0400
BGNSTR, STRNAME, ENDSTR = 0x0502, 0x0606, 0x0700
BOUNDARY, LAYER, DATATYPE, XY, ENDEL = 0x0800, 0x0D02, 0x0E02, 0x1003, 0x1100


def _records(data):
    """``(type, payload)`` per record; every record even and at most 32767 B."""
    out, i = [], 0
    while i < len(data):
        size, kind = struct.unpack(">HH", data[i : i + 4])
        assert 4 <= size <= 32767
        assert size % 2 == 0
        out.append((kind, data[i + 4 : i + size]))
        i += size
    assert i == len(data)
    return out


def _boundaries(data):
    """``((layer, datatype), xy)`` per BOUNDARY, ``xy`` closed int database units."""
    records = _records(data)
    out = []
    for k, (kind, _) in enumerate(records):
        if kind == BOUNDARY:
            (k1, layer), (k2, datatype), (k3, xy), (k4, _) = records[k + 1 : k + 5]
            assert (k1, k2, k3, k4) == (LAYER, DATATYPE, XY, ENDEL)
            key = (struct.unpack(">h", layer)[0], struct.unpack(">h", datatype)[0])
            ring = np.frombuffer(xy, ">i4").astype(np.int64).reshape(-1, 2)
            out.append((key, ring))
    return out


def _area2(ring):
    """Twice the signed area of a closed integer ring, exactly."""
    x, y = ring[:-1, 0], ring[:-1, 1]
    return int(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _square(x0, y0, side):
    return np.array(
        [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]]
    )


def _pattern(*polygons, bounds=None):
    return design.LayerPattern.from_polygons(polygons, bounds=bounds)


# --- the record stream ----------------------------------------------------------------


def test_a_square_writes_the_minimal_record_sequence(tmp_path):
    pattern = _pattern(_square(0, 0, 2 * UM))
    path = design.write_gds(tmp_path / "plate.gds", {(1, 0): pattern})
    assert path == tmp_path / "plate.gds"
    records = _records(path.read_bytes())
    assert [kind for kind, _ in records] == [
        HEADER, BGNLIB, LIBNAME, UNITS, BGNSTR, STRNAME,
        BOUNDARY, LAYER, DATATYPE, XY, ENDEL,
        ENDSTR, ENDLIB,
    ]  # fmt: skip
    payload = dict(records)
    assert payload[HEADER] == struct.pack(">h", 600)
    assert payload[LIBNAME] == b"gfem"
    assert payload[STRNAME] == b"plate\0"
    ((key, ring),) = _boundaries(path.read_bytes())
    assert key == (1, 0)
    np.testing.assert_array_equal(
        ring, [[0, 0], [2000, 0], [2000, 2000], [0, 2000], [0, 0]]
    )


# UNITS: the database unit in user units (1 um), then in metres. Reference bytes as
# gdstk 1.0.2 writes them for unit=1e-6 and precision 1e-9 / 5e-9.
@pytest.mark.parametrize(
    ("database_unit", "units"),
    [
        (1e-9, "3e4189374bc6a7f03944b82fa09b5a54"),
        (5e-9, "3f147ae147ae147b3a15798ee2308c3a"),
    ],
)
def test_units_hold_the_database_unit_in_micrometres_and_metres(
    tmp_path, database_unit, units
):
    pattern = _pattern(_square(0, 0, 2 * UM))
    path = design.write_gds(
        tmp_path / "a.gds", {(1, 0): pattern}, database_unit=database_unit
    )
    payload = dict(_records(path.read_bytes()))
    assert payload[UNITS].hex() == units
    ((_, ring),) = _boundaries(path.read_bytes())
    assert ring.max() == round(2 * UM / database_unit)


@pytest.mark.parametrize(
    ("x", "data"),
    # the Calma GDSII Stream Format's own examples
    [(1.0, "4110000000000000"), (0.5, "4080000000000000"), (-2.0, "c120000000000000")],
)
def test_reals_use_the_excess_64_base_16_encoding(x, data):
    from gfem import _gds

    assert _gds._real8(x).hex() == data


@pytest.mark.parametrize(
    ("name", "cell"),
    [
        ("bend-v2.gds", b"bend_v2\0"),
        ("x" * 40 + ".gds", b"x" * 32),
        ("my design.v1.gds", b"my_design_v1"),
    ],
)
def test_the_cell_is_named_after_the_file(tmp_path, name, cell):
    path = design.write_gds(tmp_path / name, {(1, 0): _pattern(_square(0, 0, UM))})
    assert dict(_records(path.read_bytes()))[STRNAME] == cell


def test_an_explicit_cell_name_is_written(tmp_path):
    path = design.write_gds(
        tmp_path / "a.gds", {(1, 0): _pattern(_square(0, 0, UM))}, cell="ring$1?"
    )
    assert dict(_records(path.read_bytes()))[STRNAME] == b"ring$1?\0"


def test_the_file_is_stamped_1970_unless_told_otherwise(tmp_path):
    pattern = _pattern(_square(0, 0, UM))
    path = design.write_gds(tmp_path / "a.gds", {(1, 0): pattern})
    payload = dict(_records(path.read_bytes()))
    for kind in (BGNLIB, BGNSTR):
        assert struct.unpack(">12h", payload[kind]) == (1970, 1, 1, 0, 0, 0) * 2
    when = datetime(2026, 11, 24, 15, 30, 7)  # noqa: DTZ001 (GDS has no zone)
    path = design.write_gds(tmp_path / "b.gds", {(1, 0): pattern}, timestamp=when)
    payload = dict(_records(path.read_bytes()))
    assert struct.unpack(">12h", payload[BGNLIB]) == (2026, 11, 24, 15, 30, 7) * 2


def test_the_same_arguments_give_the_same_bytes(tmp_path):
    hole = _square(6 * UM, 1 * UM, 1 * UM)[::-1]
    holed = design.Polygon(_square(5 * UM, 0, 3 * UM), (hole,))
    pattern = _pattern(_square(0, 0, 3 * UM), holed)
    layers = {(1, 0): pattern, (2, 3): pattern}
    (tmp_path / "again").mkdir()
    first = design.write_gds(tmp_path / "x.gds", layers)
    second = design.write_gds(tmp_path / "again" / "x.gds", layers)
    assert first.read_bytes() == second.read_bytes()


def test_several_layers_and_several_patterns_on_one_layer_share_one_cell(tmp_path):
    left = _pattern(_square(0, 0, UM), _square(2 * UM, 0, UM))
    right = _pattern(_square(4 * UM, 0, UM))
    empty = design.LayerPattern.from_polygons([])
    path = design.write_gds(
        tmp_path / "a.gds", {(1, 0): (left, right), (1, 1): left, (7, 0): empty}
    )
    kinds = [kind for kind, _ in _records(path.read_bytes())]
    assert kinds.count(BGNSTR) == 1
    keys = [key for key, _ in _boundaries(path.read_bytes())]
    assert keys == [(1, 0)] * 3 + [(1, 1)] * 2


def test_a_path_given_as_a_string_is_returned_as_a_path(tmp_path):
    path = design.write_gds(
        str(tmp_path / "a.gds"), {(1, 0): _pattern(_square(0, 0, UM))}
    )
    assert path == tmp_path / "a.gds"
    assert path.is_file()


# --- snapping and clean-up ------------------------------------------------------------


def test_vertices_snap_to_the_grid_and_collinear_and_repeated_ones_go(tmp_path):
    nm = 1e-9
    ring = np.array(
        [
            [0.3 * nm, -0.2 * nm],
            [1000.4 * nm, 0.0],  # collinear: on the bottom edge
            [2000 * nm, 0.1 * nm],
            [2000.2 * nm, 0.2 * nm],  # the same database point
            [2000 * nm, 2000 * nm],
            [0.0, 2000.4 * nm],
        ]
    )
    path = design.write_gds(tmp_path / "a.gds", {(1, 0): _pattern(ring)})
    ((_, written),) = _boundaries(path.read_bytes())
    np.testing.assert_array_equal(
        written, [[0, 0], [2000, 0], [2000, 2000], [0, 2000], [0, 0]]
    )


def test_rings_narrower_than_a_database_unit_are_dropped(tmp_path):
    nm = 1e-9
    island = _square(5 * UM, 0, 0.4 * nm)
    speck = _square(1 * UM, 1 * UM, 0.3 * nm)[::-1]
    holed = design.Polygon(_square(0, 0, 2 * UM), (speck,))
    path = design.write_gds(tmp_path / "a.gds", {(1, 0): _pattern(holed, island)})
    ((_, written),) = _boundaries(path.read_bytes())
    np.testing.assert_array_equal(
        written, [[0, 0], [2000, 0], [2000, 2000], [0, 2000], [0, 0]]
    )


# --- holes ----------------------------------------------------------------------------


def _assert_weakly_simple(ring):
    """No two edges of a closed ring cross properly (touching is allowed)."""
    a, b = ring[:-1], ring[1:]

    def orient(p, q, r):
        return np.sign(
            (q[:, None, 0] - p[:, None, 0]) * (r[None, :, 1] - p[:, None, 1])
            - (q[:, None, 1] - p[:, None, 1]) * (r[None, :, 0] - p[:, None, 0])
        )

    o1, o2 = orient(a, b, a), orient(a, b, b)
    proper = (o1 * o2 < 0) & (o1.T * o2.T < 0)
    assert not proper.any()


@pytest.mark.parametrize(
    "corners",
    [
        [(4, 4)],
        [(1, 1), (6, 2), (2, 6), (6, 6.5)],  # one right of the other: cut to a hole
    ],
)
def test_holes_join_their_ring_by_keyhole_cuts(tmp_path, corners):
    holes = tuple(_square(x * UM, y * UM, 2 * UM)[::-1] for x, y in corners)
    path = design.write_gds(
        tmp_path / "a.gds",
        {(1, 0): _pattern(design.Polygon(_square(0, 0, 10 * UM), holes))},
    )
    ((_, ring),) = _boundaries(path.read_bytes())
    assert _area2(ring) == 2 * (10_000**2 - len(corners) * 2000**2)
    for x, y in corners:  # every hole corner is a vertex of the one ring
        for corner in _square(x * 1000, y * 1000, 2000):
            assert (ring == corner).all(axis=1).any()
    _assert_weakly_simple(ring)


def test_a_hole_outside_its_ring_is_refused_and_no_file_is_written(tmp_path):
    outside = _square(20 * UM, 0, UM)[::-1]
    # from_polygons refuses this; the constructor does not validate
    pattern = design.LayerPattern(
        (design.Polygon(_square(0, 0, 10 * UM), (outside,)),), None
    )
    with pytest.raises(ValueError, match=r"hole 0 of polygon 0 on layer \(1, 0\)"):
        design.write_gds(tmp_path / "a.gds", {(1, 0): pattern})
    assert not (tmp_path / "a.gds").exists()


# --- the vertex cap -------------------------------------------------------------------


def test_a_ring_above_4094_vertices_is_split_into_abutting_pieces(tmp_path):
    # 30 x 30 square holes: the keyholed ring has over 5000 vertices; every edge
    # is axis-aligned, so the cut points are exact and the areas add up exactly
    pitch, side, n = 500, 200, 30
    holes = tuple(
        _square((250 + i * pitch) * 1e-9, (250 + j * pitch) * 1e-9, side * 1e-9)[::-1]
        for i in range(n)
        for j in range(n)
    )
    outer = _square(0, 0, n * pitch * 1e-9)
    small = _square(0, -2 * UM, UM)
    pattern = _pattern(design.Polygon(outer, holes), small)
    path = design.write_gds(tmp_path / "a.gds", {(1, 0): pattern})
    *pieces, (_, last) = _boundaries(path.read_bytes())
    assert len(pieces) > 1
    assert all(len(ring) - 1 <= 4094 for _, ring in pieces)
    assert all(_area2(ring) > 0 for _, ring in pieces)
    assert sum(_area2(ring) for _, ring in pieces) == 2 * (
        (n * pitch) ** 2 - n * n * side**2
    )
    np.testing.assert_array_equal(
        last, [[0, -2000], [1000, -2000], [1000, -1000], [0, -1000], [0, -2000]]
    )


# --- errors ---------------------------------------------------------------------------

SQUARE = design.LayerPattern.from_polygons([_square(0, 0, UM)])


@pytest.mark.parametrize(
    ("layers", "options", "error", "match"),
    [
        (SQUARE, {}, TypeError, r"\{\(1, 0\): pattern\}"),
        ({}, {}, ValueError, "no layer pattern"),
        ({(1, 0): np.ones((4, 4))}, {}, TypeError, r"grid\.pattern\(filtered"),
        ({(1, 0): (SQUARE, np.ones(4))}, {}, TypeError, "not a LayerPattern"),
        ({(1, 0): SQUARE.polygons[0]}, {}, TypeError, "not a LayerPattern"),
        ({(1, 0, 0): SQUARE}, {}, ValueError, r"\(layer, datatype\)"),
        ({1: SQUARE}, {}, ValueError, r"\(layer, datatype\)"),
        ({(40000, 0): SQUARE}, {}, ValueError, r"\[0, 32767\]"),
        ({(1, -1): SQUARE}, {}, ValueError, r"\[0, 32767\]"),
        ({(True, 0): SQUARE}, {}, ValueError, r"\(layer, datatype\)"),
        ({(1.0, 0): SQUARE}, {}, ValueError, r"\(layer, datatype\)"),
        ({(1, 0): SQUARE}, {"cell": "my cell"}, ValueError, r"A-Z a-z 0-9 _ \? \$"),
        ({(1, 0): SQUARE}, {"cell": "x" * 33}, ValueError, "1 to 32 characters"),
        ({(1, 0): SQUARE}, {"cell": ""}, ValueError, "1 to 32 characters"),
        ({(1, 0): SQUARE}, {"database_unit": 0.0}, ValueError, r"\(0, 1e-6\]"),
        ({(1, 0): SQUARE}, {"database_unit": -1e-9}, ValueError, r"\(0, 1e-6\]"),
        ({(1, 0): SQUARE}, {"database_unit": 2e-6}, ValueError, r"\(0, 1e-6\]"),
        ({(1, 0): SQUARE}, {"database_unit": np.nan}, ValueError, r"\(0, 1e-6\]"),
        ({(1, 0): SQUARE}, {"database_unit": np.inf}, ValueError, r"\(0, 1e-6\]"),
        ({(1, 0): SQUARE}, {"database_unit": "1e-9"}, TypeError, "metres"),
        ({(1, 0): SQUARE}, {"timestamp": "1970-01-01"}, TypeError, "datetime"),
        (  # micrometre numbers passed as metres: 3 m from the origin
            {(1, 0): design.LayerPattern.from_polygons([_square(0, 0, 3.0)])},
            {},
            ValueError,
            "coarser database_unit",
        ),
        (
            {
                (1, 0): design.LayerPattern(
                    (design.Polygon(np.full((4, 2), np.nan)),), None
                )
            },
            {},
            ValueError,
            "non-finite",
        ),
    ],
)
def test_bad_arguments_raise_and_leave_no_file(tmp_path, layers, options, error, match):
    with pytest.raises(error, match=match):
        design.write_gds(tmp_path / "a.gds", layers, **options)
    assert not (tmp_path / "a.gds").exists()


def test_a_missing_directory_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        design.write_gds(tmp_path / "missing" / "a.gds", {(1, 0): SQUARE})
    assert not (tmp_path / "missing").exists()


def test_write_gds_is_public_in_gfem_design_only():
    import gfem

    assert "write_gds" in design.__all__
    assert "write_gds" not in gfem.__all__


# --- read back by another library -----------------------------------------------------


@pytest.fixture(scope="module")
def contoured(tmp_path_factory):
    """A layer grid's pattern of a smooth random density: holes, curved edges."""
    import structured_meshes as sm

    import gfem

    path = sm.write_msh22(
        tmp_path_factory.mktemp("square") / "square.msh",
        sm.plate(20.0, 20.0, (("design", 0.22),), (40, 40, 1)),
    )
    grid = design.LayerGrid(
        gfem.Mesh.from_gmsh(path, length_unit="um"), "design", pitch=0.25 * UM
    )
    rng = np.random.default_rng(3)
    k = np.hypot(
        *np.meshgrid(np.fft.fftfreq(grid.shape[1]), np.fft.fftfreq(grid.shape[0]))
    )
    noise = np.fft.ifft2(
        np.fft.fft2(rng.standard_normal(grid.shape)) * np.exp(-((8 * k) ** 2))
    )
    rho = 0.5 + 0.5 * np.real(noise) / np.abs(np.real(noise)).max()
    pattern = grid.pattern(rho)
    assert sum(len(p.holes) for p in pattern) > 0
    return pattern


def _star(n, centre, r_in, r_out):
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = np.where(np.arange(n) % 2, r_in, r_out)
    return centre + radius[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)


def _perimeter(pattern):
    rings = [r for p in pattern for r in (p.exterior, *p.holes)]
    return sum(np.sum(np.hypot(*np.diff(r, axis=0).T)) for r in rings)


@pytest.mark.parametrize("database_unit", [1e-9, 5e-9])
def test_gdstk_reads_back_cell_layers_areas_and_extents(
    tmp_path, contoured, database_unit
):
    gdstk = pytest.importorskip("gdstk")
    ring = _star(6000, (40 * UM, 0), 8 * UM, 10 * UM)  # above the cap, CCW
    star = design.LayerPattern((design.Polygon(np.vstack([ring, ring[:1]])),), None)
    layers = {(1, 0): contoured, (2, 0): star, (1, 1): (star, contoured)}
    path = design.write_gds(
        tmp_path / "bend-v2.gds", layers, database_unit=database_unit
    )
    library = gdstk.read_gds(str(path))
    assert library.unit == pytest.approx(1e-6, rel=1e-15)
    assert library.precision == pytest.approx(database_unit, rel=1e-15)
    (cell,) = library.cells
    assert cell.name == "bend_v2"
    assert all(len(p.points) <= 4094 for p in cell.polygons)
    by_key = {}
    for polygon in cell.polygons:
        by_key.setdefault((polygon.layer, polygon.datatype), []).append(polygon)
    assert set(by_key) == set(layers)
    assert len(by_key[(1, 0)]) == len(contoured)
    assert len(by_key[(2, 0)]) > 1
    for key, value in layers.items():
        patterns = value if isinstance(value, tuple) else (value,)
        area = sum(p.area() for p in by_key[key]) * UM**2
        expected = sum(p.area for p in patterns)
        assert (
            abs(area - expected) <= sum(_perimeter(p) for p in patterns) * database_unit
        )
        low = np.min([p.bounding_box()[0] for p in by_key[key]], axis=0) * UM
        high = np.max([p.bounding_box()[1] for p in by_key[key]], axis=0) * UM
        vertices = np.vstack([p.exterior for pattern in patterns for p in pattern])
        np.testing.assert_allclose(
            low, vertices.min(axis=0), rtol=0, atol=0.5001 * database_unit
        )
        np.testing.assert_allclose(
            high, vertices.max(axis=0), rtol=0, atol=0.5001 * database_unit
        )
