"""The raise table: every result leaf x every model / problem leaf.

Each (problem, result leaf, input leaf) cell is one of three outcomes:
a finite, nonzero gradient; an exact zero, because the problem's physics
never reads that input (or the leaf is a constant of it, like a port's
reference impedance); or `gfem.NonDifferentiableError` at trace time.
Never a silent zero for an input the physics reads. The cells are
generated from the pytrees, so a new leaf without an expected outcome
fails here.

Then the pairs outside v0.1's differentiable set that are not cells of a
result leaf by an input leaf (`OUT_OF_SCOPE`): each raises a named error,
at trace time where the cause is static, on the host in the backward pass
where it depends on the data. Together with the gradient verification
matrix (tests/verification/test_gradient_matrix.py) this is the gate for
v0.1's gradients (``pytest -m verification -k gradient_matrix``).
"""

import dataclasses
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import pytest
import structured_meshes as sm
from gradient_cases import (
    HYBRID_PORTS,
    MIRRORED_TARGET,
    hybrid_line,
    hybrid_mesh,
    mirrored_cavity,
    split_cavity,
)

import gfem
from gfem._adjoint import solve_vjp
from gfem._adjoint.solve_vjp import input_kind
from gfem._engine.electrostatic import ElectrostaticRule
from gfem.shape import boundary_morph

DATA = Path(__file__).parents[1] / "unit" / "data"
GRADIENT, ZERO, RAISES = "gradient", "zero", "raises"

ALL_PROPERTIES = {
    "eps_r": 3.9 * (1 - 0.01j),
    "mu_r": 1.0,
    "tan_delta": 0.01,
    "conductivity": 2.0,
    "thermal_conductivity": 1.4,
    "density": 2200.0,
    "specific_heat": 700.0,
}

# (problem name, result leaf, input kind) -> outcome. Input kinds name the
# model leaf: ("nodes",) or ("material", <region>, <property>).
EXPECTED = {
    ("electrostatic", "capacitance", ("nodes",)): GRADIENT,
    ("electrostatic", "capacitance", ("material", "oxide", "eps_r")): GRADIENT,
    ("electrostatic", "capacitance", ("material", "oxide", "mu_r")): ZERO,
    ("electrostatic", "capacitance", ("material", "oxide", "tan_delta")): ZERO,
    ("electrostatic", "capacitance", ("material", "oxide", "conductivity")): ZERO,
    (
        "electrostatic",
        "capacitance",
        ("material", "oxide", "thermal_conductivity"),
    ): ZERO,
    ("electrostatic", "capacitance", ("material", "oxide", "density")): ZERO,
    ("electrostatic", "capacitance", ("material", "oxide", "specific_heat")): ZERO,
}
# Terminal potentials: their coefficients differentiate like the capacitance;
# their mesh is the input mesh passed through (identity in the nodes).
EXPECTED |= {
    ("electrostatic", f"potential.{t}.{leaf}", kind): (
        EXPECTED[("electrostatic", "capacitance", kind)]
        if leaf == "coefficients" or kind == ("nodes",)
        else ZERO
    )
    for t in ("top",)
    for leaf in ("coefficients", "mesh.nodes")
    for kind in [("nodes",)] + [("material", "oxide", p) for p in ALL_PROPERTIES]
}
# Region energies: each differentiates with respect to what the solve reads
# (the nitride's energy moves with the oxide's permittivity through the
# potential), and is an exact zero for the inputs it never reads.
EXPECTED |= {
    ("electrostatic", f"energy.{region}", kind): (
        ZERO if kind[0] == "material" and kind[2] != "eps_r" else GRADIENT
    )
    for region in ("oxide", "nitride")
    for kind in [("nodes",)] + [("material", "oxide", p) for p in ALL_PROPERTIES]
}
EM_PROPERTIES = ("eps_r", "mu_r", "tan_delta", "conductivity")
# The port mode problem is not differentiable: every input it reads raises
# (whichever result leaf is used), and the thermal properties it never reads
# get exact zeros.
PORT_MODE_OUTPUTS = ("frequency", "gamma", "Z_PV", "residual", "solved")
EXPECTED |= {
    ("port_mode", out, kind): (
        ZERO if kind[0] == "material" and kind[2] not in EM_PROPERTIES else RAISES
    )
    for out in PORT_MODE_OUTPUTS
    for kind in [("nodes",)] + [("material", "fill", p) for p in ALL_PROPERTIES]
}

# Driven: S and the kept field coefficients differentiate with respect to
# every electromagnetic property and the node coordinates, and are exact
# zeros for the thermal properties. z_ref (the port resistances) and the
# kept fields' angular frequency are constants; the kept fields' mesh is the
# input mesh passed through (identity in the nodes, constant otherwise).
_DRIVEN_DEPENDENT = (
    "s.values",
    *(f"e_fields.{k}.coefficients" for k in (0, 1)),
)
_NODE_DEPENDENT = (
    *_DRIVEN_DEPENDENT,
    *(f"e_fields.{k}.mesh.nodes" for k in (0, 1)),
    "e_fields.coordinates",
    "e_fields.mesh.nodes",
)
EXPECTED |= {
    ("driven", out, kind): (
        (GRADIENT if out in _NODE_DEPENDENT else ZERO)
        if kind == ("nodes",)
        else GRADIENT
        if out in _DRIVEN_DEPENDENT and kind[2] in EM_PROPERTIES
        else ZERO
    )
    for out in (
        "s.values",
        "s.z_ref",
        # kept fields (one per excited port)
        *(
            f"e_fields.{k}.{leaf}"
            for k in (0, 1)
            for leaf in ("coefficients", "mesh.nodes", "omega")
        ),
    )
    for kind in [("nodes",)] + [("material", "dielectric", p) for p in ALL_PROPERTIES]
}


# Driven on a periodic model (the C1 line, plain and Bloch): as the
# non-periodic line, region "cell".
def _in_cell(kind: tuple) -> tuple:
    return kind if kind == ("nodes",) else ("material", "cell", kind[2])


EXPECTED |= {
    (problem, out, _in_cell(kind)): outcome
    for problem in ("driven_periodic_plain", "driven_periodic_bloch")
    for (name, out, kind), outcome in list(EXPECTED.items())
    if name == "driven"
}
# A permittivity law on a second region "core" (ADR 0112): one term of each
# kind (Re eps > 0 at the table's 5 GHz), 11 leaves. S and the kept fields'
# coefficients differentiate by every law parameter; z_ref and the fields'
# omega and mesh are constants. The host field and the nodes are as on the
# plain line. ``driven_law_absorbing``: the same behind first-order absorbing
# y walls on both regions (the law's face pieces).
TABLE_LAW = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[
        gfem.Drude(plasma_frequency=3e9, damping=0.5e9),
        gfem.Lorentz(delta_eps=1.0, resonance_frequency=20e9, damping=1e9),
        gfem.Debye(delta_eps=2.0, relaxation_frequency=5e9),
        gfem.DjordjevicSarkar(delta_eps=1.1, low_frequency=1e3, high_frequency=1e12),
    ],
)
LAW_KINDS = [
    ("law", "core", "eps_inf"),
    *(
        ("law", "core", i, f.name)
        for i, term in enumerate(TABLE_LAW.terms)
        for f in dataclasses.fields(term)
    ),
]
EXPECTED |= {
    (problem, out, kind): (
        EXPECTED[("driven", out, kind)]
        if kind[0] != "law"
        else GRADIENT
        if out in _DRIVEN_DEPENDENT
        else ZERO
    )
    for problem in ("driven_law", "driven_law_absorbing")
    for out in sorted({out for name, out, _ in EXPECTED if name == "driven"})
    for kind in [
        ("nodes",),
        *(("material", "dielectric", p) for p in ALL_PROPERTIES),
        *LAW_KINDS,
    ]
}
# A density blend of a constant end and TABLE_LAW on "core" (ADR 0113,
# index scheme): S and the kept fields' coefficients differentiate by its
# density and by the mu_r and conductivity the blend interpolates; the host
# field and the nodes are as on the plain line.
EXPECTED |= {
    ("driven_blend", out, kind): (
        EXPECTED[("driven", out, kind)]
        if kind[1:2] != ("core",)
        else GRADIENT
        if out in _DRIVEN_DEPENDENT
        else ZERO
    )
    for out in sorted({out for name, out, _ in EXPECTED if name == "driven"})
    for kind in [
        ("nodes",),
        *(("material", "dielectric", p) for p in ALL_PROPERTIES),
        *(("material", "core", p) for p in ("mu_r", "conductivity", "blend")),
    ]
}
# The C1 periodic line with the law on its cell (plain periodicity): as
# the plain periodic line in the nodes, the law's leaves as above.
EXPECTED |= {
    ("driven_periodic_law", out, kind): (
        EXPECTED[("driven_periodic_plain", out, kind)]
        if kind[0] != "law"
        else GRADIENT
        if out in _DRIVEN_DEPENDENT
        else ZERO
    )
    for out in sorted({out for name, out, _ in EXPECTED if name == "driven"})
    for kind in [("nodes",), *(("law", "cell", *k[2:]) for k in LAW_KINDS)]
}
# Adaptive sweep (frozen samples, so it traces): S and the kept fields'
# reduced coordinates differentiate like the uniform sweep's S and fields.
# The reduced basis is frozen by definition of the gradient, so a cotangent
# on it raises for every input the solve reads.
ADAPTIVE = gfem.Driven(
    frequency=(4e9, 5e9, 6e9), sweep="adaptive", samples=(4e9, 6e9), keep_fields=5e9
)
EXPECTED |= {
    ("driven_adaptive", out, kind): (
        (
            RAISES
            if out == "e_fields.basis"
            else GRADIENT
            if out in _NODE_DEPENDENT
            else ZERO
        )
        if kind == ("nodes",)
        else ZERO
        if kind[2] not in EM_PROPERTIES
        else GRADIENT
        if out in ("s.values", "e_fields.coordinates")
        else RAISES
        if out == "e_fields.basis"
        else ZERO
    )
    for out in (
        "s.values",
        "s.z_ref",
        "e_fields.basis",
        "e_fields.coordinates",
        "e_fields.omega",
        "e_fields.mesh.nodes",
    )
    for kind in [("nodes",)] + [("material", "dielectric", p) for p in ALL_PROPERTIES]
}
# The law line on the adaptive sweep (ADR 0116): the law's leaves
# differentiate S and the kept fields' reduced coordinates, like the host's
# material; the frozen basis raises; the other inputs as `driven_adaptive`.
EXPECTED |= {
    ("driven_law_adaptive", out, kind): (
        EXPECTED[("driven_adaptive", out, kind)]
        if kind[0] != "law"
        else GRADIENT
        if out in ("s.values", "e_fields.coordinates")
        else RAISES
        if out == "e_fields.basis"
        else ZERO
    )
    for out in (
        "s.values",
        "s.z_ref",
        "e_fields.basis",
        "e_fields.coordinates",
        "e_fields.omega",
        "e_fields.mesh.nodes",
    )
    for kind in [
        ("nodes",),
        *(("material", "dielectric", p) for p in ALL_PROPERTIES),
        *LAW_KINDS,
    ]
}

# Wave ports (design region "core" away from the port faces): S
# differentiates like the lumped-port line's, in the materials and the node
# coordinates; the port modes, and so z_ref (Z_PV), are constants of the
# solve: the design never touches the port faces (nor moves their nodes),
# which is the condition for differentiating at all.
WAVE_PORT_MODE_OUTPUTS = ("frequency", "gamma", "Z_PV", "residual", "solved")
EXPECTED |= {
    ("driven_wave", out, kind): (
        (GRADIENT if out == "s.values" else ZERO)
        if kind == ("nodes",)
        else GRADIENT
        if out == "s.values" and kind[2] in EM_PROPERTIES
        else ZERO
    )
    for out in (
        "s.values",
        "s.z_ref",
        *(
            f"port_modes.{p}.{leaf}"
            for p in ("p1", "p2")
            for leaf in WAVE_PORT_MODE_OUTPUTS
        ),
    )
    for kind in [("nodes",)] + [("material", "core", p) for p in ALL_PROPERTIES]
}

# A shared port surface (two requested modes at each end of a hybrid line):
# S, cross-mode entries included, differentiates like a lone wave port's;
# every port's mode is a constant of the solve. The core carries only a
# permittivity: each cell costs a 4-port solve, and the other properties'
# outcomes are the lone wave ports' above.
EXPECTED |= {
    ("driven_wave_shared", out, kind): GRADIENT if out == "s.values" else ZERO
    for out in (
        "s.values",
        "s.z_ref",
        *(
            f"port_modes.{p}.{leaf}"
            for p in HYBRID_PORTS
            for leaf in WAVE_PORT_MODE_OUTPUTS
        ),
    )
    for kind in [("nodes",), ("material", "core", "eps_r")]
}

# Eigenmode: the eigenvalues (omega, also as each mode field's omega)
# differentiate with respect to every electromagnetic property, the node
# coordinates and the lumped values. The ratios of the mode fields (port Q,
# participations, region, principal and sheet shares) differentiate with respect to
# the electromagnetic properties, sheet fills, node coordinates and lumped
# values (ADR 0102 and its amendment), and the normalized mode fields'
# coefficients with respect to the electromagnetic properties, sheet fills
# and lumped values (never the node coordinates); the principal shares raise
# with respect to the lumped values in this version, as does the backward
# error with respect to anything. The
# cluster labels are integers: exact zeros. The mode fields' mesh is the
# input mesh passed through (identity in the nodes, constant otherwise). A
# periodic model's mode-field outputs raise whatever the input.
_EIGEN_DEPENDENT = ("omega", "e_fields.0.omega")
_EIGEN_CONSTANT = ("cluster", "e_fields.0.mesh.nodes")
_EIGEN_RATIOS = (
    "q_port.",
    "participation.",
    "bulk_participation.",
    "principal_participation.",
    "sheet_participation.",
    "e_fields.0.coefficients",
)


def _eigen_outcome(out, kind, *, periodic=False):
    if kind == ("nodes",):
        if out == "cluster":
            return ZERO
        dependent = (*_EIGEN_DEPENDENT, "e_fields.0.mesh.nodes")
        # the normalized field has no node gradient (ADR 0102)
        ratio = out.startswith(_EIGEN_RATIOS) and not periodic
        field = out == "e_fields.0.coefficients"
        return GRADIENT if out in dependent or (ratio and not field) else RAISES
    if (kind[0] == "material" and kind[2] not in EM_PROPERTIES) or (
        out in _EIGEN_CONSTANT
    ):
        return ZERO
    if out in _EIGEN_DEPENDENT:
        return GRADIENT
    ratio = out.startswith(_EIGEN_RATIOS)
    lumped = kind[0] in ("port", "boundary") and kind[2:] != ("fill", "fill")
    principal = out.startswith("principal_participation.")
    return GRADIENT if ratio and not periodic and not (lumped and principal) else RAISES


EIGEN_KINDS = [
    ("nodes",),
    *(("material", "dielectric", p) for p in ALL_PROPERTIES),
    ("port", 0, "inductance"),
    ("port", 0, "capacitance"),
]
EXPECTED |= {
    ("eigenmode", out, kind): _eigen_outcome(out, kind)
    for out in (
        "omega",
        "error",
        "q_port.p1",
        "q_port.p2",
        "participation.p1",
        "bulk_participation.dielectric",
        "cluster",
        *(f"e_fields.0.{leaf}" for leaf in ("coefficients", "mesh.nodes", "omega")),
    )
    for kind in EIGEN_KINDS
}
# The result carries the model's materials as solved (for measurements): each
# leaf is its input passed through (identity), a constant in the nodes.
EXPECTED |= {
    ("eigenmode", f"materials.dielectric.{out}", kind): (
        GRADIENT if kind[0] == "material" and kind[2] == out else ZERO
    )
    for out in ALL_PROPERTIES
    for kind in EIGEN_KINDS
}

# Eigenmode on a periodic model (the C3 Bragg line, plain at Γ and Bloch at
# the zone edge; no ports): as the non-periodic resonator, region "low".
# Its material field carries only the permittivity and the conductivity
# (the properties above are covered by "eigenmode"): each cell is an
# eigenmode solve and its gradient.
PERIODIC_EIGEN_PROPERTIES = ("eps_r", "conductivity")
EXPECTED |= {
    (problem, out, kind): (
        (
            ZERO
            if kind == ("nodes",)
            else GRADIENT
            if kind[2] == out.removeprefix("materials.low.")
            else ZERO
        )
        if out.startswith("materials.low.")
        else _eigen_outcome(out, kind, periodic=True)
    )
    for problem in ("eigenmode_periodic_plain", "eigenmode_periodic_bloch")
    for out in (
        "omega",
        "error",
        "bulk_participation.low",
        "bulk_participation.high",
        "cluster",
        *(f"e_fields.0.{leaf}" for leaf in ("coefficients", "mesh.nodes", "omega")),
        *(f"materials.low.{p}" for p in PERIODIC_EIGEN_PROPERTIES),
    )
    for kind in [("nodes",)]
    + [("material", "low", p) for p in PERIODIC_EIGEN_PROPERTIES]
}

# Sheet fields: the fill of a sheet on the glued line (model.boundaries[2],
# static materials). Driven S and eigenfrequencies differentiate with respect
# to it like a material; the port mode is not differentiable at all.
FILL = ("boundary", 2, "fill", "fill")
EXPECTED |= {
    ("driven_sheet", out, kind): GRADIENT if out == "s.values" else ZERO
    for out in ("s.values", "s.z_ref")
    for kind in [("nodes",), FILL]
}
EXPECTED |= {
    ("eigenmode_sheet", out, kind): _eigen_outcome(out, kind)
    for out in (
        "omega",
        "error",
        "q_port.p1",
        "q_port.p2",
        "bulk_participation.lower",
        "bulk_participation.upper",
        "cluster",
        *(f"e_fields.0.{leaf}" for leaf in ("coefficients", "mesh.nodes", "omega")),
    )
    for kind in [("nodes",), FILL]
}
EXPECTED |= {
    ("port_mode_sheet", out, kind): RAISES
    for out in PORT_MODE_OUTPUTS
    for kind in [("nodes",), FILL]
}

# Lumped values (ADR 0100): port p1's inductance and capacitance and the
# sheet's resistance, inductance and capacitance on the glued line (static
# materials).
# Driven S, the kept fields and the eigenfrequencies differentiate with
# respect to them; z_ref (p1's resistance) and the fields' omega and mesh do
# not move with them; electrostatics never reads them; the port mode problem
# is not differentiable. Only a model with no Impedance and no lumped-port L
# or C has none of these leaves: every Impedance adds them, R-only ones too.
LUMPED_KINDS = [
    ("port", 0, "inductance"),
    ("port", 0, "capacitance"),
    ("boundary", 2, "resistance"),
    ("boundary", 2, "inductance"),
    ("boundary", 2, "capacitance"),
]
EXPECTED |= {
    ("driven_lumped", out, kind): (
        EXPECTED[("driven", out, ("nodes",))]
        if kind == ("nodes",)
        else GRADIENT
        if out in _DRIVEN_DEPENDENT
        else ZERO
    )
    for out in (
        "s.values",
        "s.z_ref",
        *(
            f"e_fields.{k}.{leaf}"
            for k in (0, 1)
            for leaf in ("coefficients", "mesh.nodes", "omega")
        ),
    )
    for kind in [("nodes",), *LUMPED_KINDS]
}
EXPECTED |= {
    ("driven_lumped_adaptive", out, kind): (
        EXPECTED[("driven_adaptive", out, ("nodes",))]
        if kind == ("nodes",)
        else GRADIENT
        if out in ("s.values", "e_fields.coordinates")
        else RAISES
        if out == "e_fields.basis"
        else ZERO
    )
    for out in (
        "s.values",
        "s.z_ref",
        "e_fields.basis",
        "e_fields.coordinates",
        "e_fields.omega",
        "e_fields.mesh.nodes",
    )
    for kind in [("nodes",), *LUMPED_KINDS]
}
EXPECTED |= {
    ("eigenmode_lumped", out, kind): _eigen_outcome(out, kind)
    for out in (
        "omega",
        "error",
        "q_port.p1",
        "q_port.p2",
        "participation.p1",
        "bulk_participation.lower",
        "bulk_participation.upper",
        "sheet_participation.sheet",
        "cluster",
        *(f"e_fields.0.{leaf}" for leaf in ("coefficients", "mesh.nodes", "omega")),
    )
    for kind in [("nodes",), *LUMPED_KINDS]
}
EXPECTED |= {
    ("electrostatic_lumped", out, kind): (
        EXPECTED[("electrostatic", "capacitance", ("nodes",))]
        if kind == ("nodes",)
        else ZERO
    )
    for out in (
        "capacitance",
        "energy.lower",
        "energy.upper",
        "potential.top.coefficients",
        "potential.top.mesh.nodes",
    )
    for kind in [("nodes",), *LUMPED_KINDS]
}
# the port mode's model: wave port p1, the lumped port p2 holds L and C
PORT_MODE_LUMPED_KINDS = [
    ("port", 1, "inductance"),
    ("port", 1, "capacitance"),
    ("boundary", 2, "resistance"),
    ("boundary", 2, "inductance"),
    ("boundary", 2, "capacitance"),
]
EXPECTED |= {
    ("port_mode_lumped", out, kind): RAISES
    for out in PORT_MODE_OUTPUTS
    for kind in [("nodes",), *PORT_MODE_LUMPED_KINDS]
}


# A permittivity law in a wave port's cross-section (ADR 0114): the port mode
# problem is not differentiable, so every law parameter raises, with the node
# coordinates. One term of each kind, each with Re eps > 0 at 10 GHz.
PORT_MODE_LAW = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[
        gfem.Drude(plasma_frequency=5e9, damping=1e9),
        gfem.Lorentz(delta_eps=1.0, resonance_frequency=50e9, damping=1e9),
        gfem.Debye(delta_eps=0.5, relaxation_frequency=20e9),
        gfem.DjordjevicSarkar(delta_eps=0.3, low_frequency=1e3, high_frequency=1e12),
    ],
)
PORT_MODE_LAW_KINDS = [
    ("law", "fill", "eps_inf"),
    ("law", "fill", 0, "plasma_frequency"),
    ("law", "fill", 0, "damping"),
    ("law", "fill", 1, "delta_eps"),
    ("law", "fill", 1, "resonance_frequency"),
    ("law", "fill", 1, "damping"),
    ("law", "fill", 2, "delta_eps"),
    ("law", "fill", 2, "relaxation_frequency"),
    ("law", "fill", 3, "delta_eps"),
    ("law", "fill", 3, "low_frequency"),
    ("law", "fill", 3, "high_frequency"),
]
EXPECTED |= {
    ("port_mode_law", out, kind): RAISES
    for out in PORT_MODE_OUTPUTS
    for kind in [("nodes",), *PORT_MODE_LAW_KINDS]
}

# The stack with a permittivity law on its nitride (ADR 0111): the table's
# law without its Drude term (a Drude metal has no static permittivity). The
# capacitance, the energies and the potentials differentiate by eps_inf and
# every strength, which the static permittivity reads; every frequency
# parameter is an exact zero (ADR 0011's amendment: a leaf the physics never
# reads). The oxide field and the nodes are as on the plain stack.
ELECTROSTATIC_LAW = dataclasses.replace(TABLE_LAW, terms=TABLE_LAW.terms[1:])
ELECTROSTATIC_LAW_KINDS = [
    ("law", "nitride", "eps_inf"),
    *(
        ("law", "nitride", i, f.name)
        for i, term in enumerate(ELECTROSTATIC_LAW.terms)
        for f in dataclasses.fields(term)
    ),
]
EXPECTED |= {
    ("electrostatic_law", out, kind): (
        EXPECTED[("electrostatic", out, kind)]
        if kind[0] != "law"
        else GRADIENT
        if kind[-1] in ("eps_inf", "delta_eps") and not out.endswith("mesh.nodes")
        else ZERO
    )
    for out in sorted({out for name, out, _ in EXPECTED if name == "electrostatic"})
    for kind in [
        ("nodes",),
        *(("material", "oxide", p) for p in ALL_PROPERTIES),
        *ELECTROSTATIC_LAW_KINDS,
    ]
}


def _field(mesh, region, **overrides):
    # values vary per element, so the terminal potentials depend on them
    n = len(mesh.elements(region))
    ramp = np.linspace(1.0, 1.5, n)
    values = ALL_PROPERTIES | overrides
    return gfem.MaterialField(**{k: v * ramp for k, v in values.items()})


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "stack.msh"
    layers = (("oxide", 0.5), ("nitride", 0.25))
    sm.write_msh22(path, sm.plate(4.0, 3.0, layers, (1, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path)
    return gfem.Model(
        mesh,
        materials={"oxide": _field(mesh, "oxide"), "nitride": gfem.Material(eps_r=7.5)},
        boundaries=[gfem.PEC("bottom")],
    )


@pytest.fixture(scope="module")
def driven_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(4.0, 2.0, 1.0, (2, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": _field(mesh, "dielectric")},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


@pytest.fixture(scope="module")
def law_cases(tmp_path_factory):
    path = tmp_path_factory.mktemp("law_line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(4.0, 2.0, 1.0, (4, 1, 1), core=(1.0, 3.0)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={
            "dielectric": _field(mesh, "dielectric"),
            "core": gfem.Material(eps_r=TABLE_LAW),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )
    absorbing = model.replace(
        boundaries=[*model.boundaries, gfem.Absorbing("side", order=1)]
    )
    problem = gfem.Driven(frequency=5e9, keep_fields=True)
    n = len(mesh.elements("core"))
    blend = gfem.MaterialField.interpolate(
        jnp.linspace(0.3, 0.7, n),
        gfem.Material(eps_r=2.0),
        gfem.Material(eps_r=TABLE_LAW),
        scheme="index",
    )
    return {
        "driven_law": (model, problem),
        "driven_law_absorbing": (absorbing, problem),
        "driven_law_adaptive": (model, ADAPTIVE),
        "driven_blend": (model.with_material("core", blend), problem),
    }


@pytest.fixture(scope="module")
def eigen_model(tmp_path_factory):
    # weakly damped (resistive ports, little conduction): the modes are sharp
    path = tmp_path_factory.mktemp("resonator") / "line.msh"
    sm.write_msh22(path, sm.strip_line(4.0, 2.0, 1.0, (4, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    # p1 also holds an inductance and a capacitance: a participation, two
    # lumped values, and a share of the mode energy outside the dielectric
    # (with the dielectric's alone, its bulk participation is exactly 1)
    return gfem.Model(
        mesh,
        materials={"dielectric": _field(mesh, "dielectric", conductivity=1e-3)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort(
                "p1",
                "port1",
                "+z",
                resistance=1000.0,
                inductance=1e-9,
                capacitance=1e-14,
            ),
            gfem.LumpedPort("p2", "port2", "+z", impedance=1000.0),
        ],
    )


@pytest.fixture(scope="module")
def wave_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("wave") / "line.msh"
    sm.write_msh22(path, sm.strip_line(3.0, 2.0, 1.0, (3, 1, 1), core=(1.0, 2.0)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={
            "dielectric": gfem.Material(eps_r=2.0),
            "core": _field(mesh, "core"),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


@pytest.fixture(scope="module")
def shared_surface_model(tmp_path_factory):
    # the gradient matrix's hybrid line on the coarsest grid that keeps its
    # quadrant: modes 1 and 2 of each lead on one shared port surface
    mesh = hybrid_mesh(tmp_path_factory.mktemp("hybrid"), cells=(4, 2, 2))
    eps = 3.0 * np.linspace(1.0, 1.5, len(mesh.elements("core")))
    return hybrid_line(mesh).with_material("core", gfem.MaterialField(eps_r=eps))


@pytest.fixture(scope="module")
def sheet_mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("sheet") / "sheet.msh"
    sm.write_msh22(path, sm.sheet_line(4.0, 2.0, 1.0, (4, 1, 1)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _sheet_model(mesh, ports):
    n = len(mesh.elements("sheet"))
    fill = gfem.SheetField(fill=np.linspace(0.2, 0.9, n))
    return gfem.Model(
        mesh,
        materials={"lower": gfem.Material(eps_r=2.0), "upper": gfem.Material()},
        boundaries=[
            gfem.PEC("plate1"),
            gfem.PEC("plate2"),
            gfem.Conductor("sheet", 100.0, 1e-6, fill=fill),
        ],
        ports=ports,
    )


@pytest.fixture(scope="module")
def sheet_cases(sheet_mesh):
    lumped = [
        gfem.LumpedPort("p1", "port1", "+y", impedance=1000.0),
        gfem.LumpedPort("p2", "port2", "+y", impedance=1000.0),
    ]
    waves = [gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")]
    return {
        "driven_sheet": (
            _sheet_model(sheet_mesh, lumped),
            gfem.Driven(frequency=20e9),
        ),
        "eigenmode_sheet": (
            _sheet_model(sheet_mesh, lumped),
            gfem.Eigenmode(target=30e9),
        ),
        "port_mode_sheet": (
            _sheet_model(sheet_mesh, waves),
            gfem.PortMode("p1", 20e9),
        ),
    }


#: The lumped values of the lumped cases: port inductance (H) and capacitance
#: (F), sheet resistance (ohm), inductance (H) and capacitance (F) per square.
LUMPED = (2e-9, 5e-14, 2e4, 1e-6, 2e-16)
#: Near the lumped line's sharpest mode, 15.07 + 1.46j GHz (the next ones,
#: 11.7 + 4.2j and 19.8 + 4.8j GHz, are 4.4 and 5.8 GHz off).
LUMPED_TARGET = 15e9
LUMPED_ADAPTIVE = gfem.Driven(
    frequency=(18e9, 20e9, 22e9),
    sweep="adaptive",
    samples=(18e9, 22e9),
    keep_fields=20e9,
)


def _lumped_model(mesh, ports, lower=None):
    sheet = gfem.Impedance(
        "sheet", resistance=LUMPED[2], inductance=LUMPED[3], capacitance=LUMPED[4]
    )
    return gfem.Model(
        mesh,
        materials={
            "lower": gfem.Material(eps_r=2.0) if lower is None else lower,
            "upper": gfem.Material(),
        },
        boundaries=[gfem.PEC("plate1"), gfem.PEC("plate2"), sheet],
        ports=ports,
    )


def _rlc_port(name, surface):
    return gfem.LumpedPort(
        name,
        surface,
        "+y",
        resistance=1000.0,
        inductance=LUMPED[0],
        capacitance=LUMPED[1],
    )


@pytest.fixture(scope="module")
def lumped_cases(sheet_mesh):
    lumped = _lumped_model(
        sheet_mesh,
        [_rlc_port("p1", "port1"), gfem.LumpedPort("p2", "port2", "+y", impedance=1e3)],
    )
    mixed = _lumped_model(
        sheet_mesh, [gfem.WavePort("p1", "port1"), _rlc_port("p2", "port2")]
    )
    return {
        "driven_lumped": (lumped, gfem.Driven(frequency=20e9, keep_fields=True)),
        "driven_lumped_adaptive": (lumped, LUMPED_ADAPTIVE),
        "eigenmode_lumped": (lumped, gfem.Eigenmode(target=LUMPED_TARGET)),
        "electrostatic_lumped": (
            lumped,
            gfem.Electrostatic(terminals={"top": "plate2"}, ground="plate1"),
        ),
        "port_mode_lumped": (mixed, gfem.PortMode("p1", 20e9)),
    }


@pytest.fixture(scope="module")
def wave_cases(wave_model, shared_surface_model):
    return {
        "driven_wave": (wave_model, gfem.Driven(frequency=20e9)),
        "driven_wave_shared": (shared_surface_model, SHARED_SURFACE_PROBLEM),
    }


@pytest.fixture(scope="module")
def driven_periodic_cases(tmp_path_factory):
    # the C1 line (periodic along x) on 3 cells, its port sheets the inner
    # grid planes, a material field over its cell
    path = tmp_path_factory.mktemp("c1") / "c1.msh"
    edges = ((0.0, pc.C1_X1, pc.C1_X2, pc.C1_A), (0.0, pc.C1_W), (0.0, pc.C1_H))
    sheets = (("sheet1", pc.C1_X1), ("sheet2", pc.C1_X2))
    sm.write_msh22(path, sm.periodic_cell(tuple(map(np.asarray, edges)), sheets=sheets))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    model = pc.c1(mesh).with_material("cell", _field(mesh, "cell"))
    return {
        "driven_periodic_plain": (
            model,
            gfem.Driven(frequency=5e9, keep_fields=True),
        ),
        "driven_periodic_law": (
            model.with_material("cell", gfem.Material(eps_r=TABLE_LAW)),
            gfem.Driven(frequency=5e9, keep_fields=True),
        ),
        "driven_periodic_bloch": (
            model,
            gfem.Driven(frequency=5e9, keep_fields=True, wave_vector=pc.C1_K),
        ),
    }


@pytest.fixture(scope="module")
def eigen_periodic_cases(tmp_path_factory):
    # the C3 Bragg line (periodic along x), weakly lossy: a material field
    # over its low layer, the high layer a constant material
    mesh = pc.c3_mesh(tmp_path_factory.mktemp("c3"), (5, 1, 1))
    ramp = np.linspace(1.0, 1.5, len(mesh.elements("low")))
    low = gfem.MaterialField(
        eps_r=ALL_PROPERTIES["eps_r"] * ramp, conductivity=1e-3 * ramp
    )
    model = pc.c3(mesh).with_material("low", low)
    zone_edge = (math.pi / (pc.C3_A * pc.MM), 0.0, 0.0)
    return {
        "eigenmode_periodic_plain": (
            model,
            gfem.Eigenmode(target=pc.C3_PLAIN_TARGET),
        ),
        "eigenmode_periodic_bloch": (
            model,
            gfem.Eigenmode(target=5e9, wave_vector=zone_edge),
        ),
    }


@pytest.fixture(scope="module")
def cases(model, driven_model, eigen_model, wave_cases, request):
    mesh = gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")
    port_model = gfem.Model(
        mesh,
        materials={"fill": _field(mesh, "fill")},
        boundaries=[gfem.PEC("plates")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    return {
        "electrostatic": (model, PROBLEMS["electrostatic"]),
        "electrostatic_law": (
            model.with_material("nitride", gfem.Material(eps_r=ELECTROSTATIC_LAW)),
            PROBLEMS["electrostatic"],
        ),
        "port_mode": (port_model, gfem.PortMode("p1", 10e9)),
        "port_mode_law": (
            port_model.with_material("fill", gfem.Material(eps_r=PORT_MODE_LAW)),
            gfem.PortMode("p1", 10e9),
        ),
        "driven": (driven_model, gfem.Driven(frequency=5e9, keep_fields=True)),
        "driven_adaptive": (driven_model, ADAPTIVE),
        **wave_cases,
        "eigenmode": (eigen_model, gfem.Eigenmode(target=17e9)),
        **request.getfixturevalue("sheet_cases"),
        **request.getfixturevalue("lumped_cases"),
        **request.getfixturevalue("driven_periodic_cases"),
        **request.getfixturevalue("eigen_periodic_cases"),
        **request.getfixturevalue("law_cases"),
    }


#: Both modes of the hybrid line propagate (mode 2 from about 48 GHz). One
#: excitation: S's other rows (cross-mode included) take adjoint loads.
SHARED_SURFACE_PROBLEM = gfem.Driven(frequency=60e9, excite="p1_m1")
PROBLEMS = {
    "electrostatic": gfem.Electrostatic(terminals={"top": "top"}, ground="bottom"),
}


def _cells(cases):
    for problem_name, (model, problem) in cases.items():
        result = gfem.solve(model, problem)
        model_leaves = jax.tree_util.tree_flatten_with_path(model)[0]
        kinds = [input_kind(path, "model") for path, _ in model_leaves] + [
            input_kind(path, "problem")
            for path, _ in jax.tree_util.tree_flatten_with_path(problem)[0]
        ]
        for out_path, _ in jax.tree_util.tree_flatten_with_path(result)[0]:
            out_name = jax.tree_util.keystr(out_path, simple=True, separator=".")
            for index, kind in enumerate(kinds):
                yield problem_name, model, problem, out_name, out_path, index, kind


def _objective(model, problem, out_path, index):
    """The weighted sum of one result leaf as a function of one input leaf."""
    leaves, treedef = jax.tree.flatten((model, problem))

    def objective(x):
        moved, moved_problem = jax.tree.unflatten(
            treedef, [*leaves[:index], x, *leaves[index + 1 :]]
        )
        result = gfem.solve(moved, moved_problem)
        (value,) = [
            v
            for p, v in jax.tree_util.tree_flatten_with_path(result)[0]
            if p == out_path
        ]
        weights = np.random.default_rng(1).uniform(0.5, 1.5, value.shape)
        return jnp.sum(jnp.real(value) * weights)

    return objective, leaves[index]


def test_every_cell_has_an_expected_outcome(cases):
    cells = {(p, o, k) for p, _, _, o, _, _, k in _cells(cases)}
    assert cells == set(EXPECTED)


def test_raise_table(cases):
    for name, model, problem, out_name, out_path, index, kind in _cells(cases):
        expected = EXPECTED[(name, out_name, kind)]
        objective, x = _objective(model, problem, out_path, index)
        cell = f"{name} {out_name} x {kind}"
        if expected == RAISES:
            # raised while tracing, before any host work
            with pytest.raises(gfem.NonDifferentiableError):  # noqa: PT012
                jax.make_jaxpr(jax.grad(objective))(x)
                pytest.fail(f"{cell}: no NonDifferentiableError")
            continue
        grad = np.asarray(jax.grad(objective)(x))
        assert np.isfinite(grad).all(), cell
        if expected == ZERO:
            assert not grad.any(), cell
        else:
            assert np.abs(grad).max() > 0, cell


class _RuleWithoutShapeTerms(ElectrostaticRule):
    """The electrostatic rule as an operator family without shape terms."""

    form = "electrostatic without shape terms"


def test_unsupported_input_error_names_the_input_and_the_opt_out(model):
    # every v0.1 rule differentiates every input its physics reads: a rule
    # of a form the kernel has no node terms for stands in for a future one
    problem = PROBLEMS["electrostatic"]
    rule = _RuleWithoutShapeTerms(model, problem, gfem.SolverOptions())

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.mesh), [nodes]
        )
        return solve_vjp(rule, model.replace(mesh=mesh), problem)["capacitance"][0, 0]

    with pytest.raises(
        gfem.NonDifferentiableError, match=r"mesh\.nodes.*stop_gradient"
    ):
        jax.make_jaxpr(jax.grad(objective))(jnp.asarray(model.mesh.nodes))


# --- out of scope: pairs v0.1 does not differentiate, not leaf x leaf cells ------


def _out_electrostatic_sheet(request):
    # a partly filled sheet is not an electrostatic object (raised by any
    # solve, so under every transformation too)
    mesh = request.getfixturevalue("sheet_mesh")
    model = _sheet_model(mesh, [])
    problem = gfem.Electrostatic(terminals={"top": "plate2"}, ground="plate1")
    jax.grad(lambda m: gfem.solve(m, problem).capacitance[0, 0])(model)


def _sheet_fill_objective(model, problem):
    sheet = model.boundaries[2]

    def objective(fill):
        filled = dataclasses.replace(sheet, fill=gfem.SheetField(fill=fill))
        boundaries = [*model.boundaries[:2], filled]
        s = gfem.solve(model.replace(boundaries=boundaries), problem).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    return objective, sheet.fill.fill


def _out_sheet_beside_a_wave_port(request):
    # the port mode reads the sheet on the port face's edge: a constant of
    # the solve, so the fill may not move there (raised while tracing)
    mesh = request.getfixturevalue("sheet_mesh")
    waves = [gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")]
    objective, fill = _sheet_fill_objective(
        _sheet_model(mesh, waves), gfem.Driven(frequency=20e9)
    )
    jax.make_jaxpr(jax.grad(objective))(fill)


def _out_impedance_beside_a_wave_port(request):
    # the port mode reads the sheet on the port face's edge: its lumped
    # values are constants of the solve there (raised while tracing)
    mesh = request.getfixturevalue("sheet_mesh")
    waves = [gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")]
    problem = gfem.Driven(frequency=20e9)

    def objective(r):
        model = _lumped_model(mesh, waves)
        sheet = gfem.Impedance("sheet", resistance=r, inductance=LUMPED[3])
        model = model.replace(boundaries=[*model.boundaries[:2], sheet])
        return jnp.sum(jnp.abs(gfem.solve(model, problem).s.values) ** 2)

    jax.make_jaxpr(jax.grad(objective))(LUMPED[2])


def _out_lumped_port_resistance(request):
    # a port's resistance is its reference impedance: a fixed value (raised
    # by the constructor)
    model = request.getfixturevalue("driven_model")
    problem = gfem.Driven(frequency=5e9)

    def objective(r):
        ports = [gfem.LumpedPort("p1", "port1", "+z", resistance=r), model.ports[1]]
        s = gfem.solve(model.replace(ports=ports), problem).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    jax.grad(objective)(50.0)


def _out_material_on_a_wave_port_face(request):
    # likewise for a material of a region with elements on a port face
    model = request.getfixturevalue("wave_model")
    problem = gfem.Driven(frequency=20e9)
    n = len(model.mesh.elements("dielectric"))

    def objective(eps):
        m = model.with_material("dielectric", gfem.MaterialField(eps_r=eps))
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return jnp.sum(jnp.abs(gfem.solve(m, problem).s.values) ** 2)

    jax.make_jaxpr(jax.grad(objective))(jnp.full(n, 2.0))


def _out_law_on_a_wave_port_face(request):
    # a law region with elements on a wave port face: the port mode reads it
    model = request.getfixturevalue("wave_model")
    problem = gfem.Driven(frequency=20e9)

    def objective(eps_inf):
        law = dataclasses.replace(TABLE_LAW, eps_inf=eps_inf)
        m = model.with_material("dielectric", gfem.Material(eps_r=law))
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return jnp.sum(jnp.abs(gfem.solve(m, problem).s.values) ** 2)

    jax.make_jaxpr(jax.grad(objective))(2.0)


def _out_law_under_a_bloch_wave_vector(request):
    model, problem = request.getfixturevalue("driven_periodic_cases")[
        "driven_periodic_bloch"
    ]

    def objective(eps_inf):
        law = dataclasses.replace(TABLE_LAW, eps_inf=eps_inf)
        m = model.with_material("cell", gfem.Material(eps_r=law))
        return jnp.sum(jnp.abs(gfem.solve(m, problem).s.values) ** 2)

    jax.make_jaxpr(jax.grad(objective))(2.0)


def _out_law_of_a_blend_end(_request):
    # the ends of a blend are fixed (ADR 0113): a traced law raises while
    # the field is built, with the remedy
    def objective(eps_inf):
        law = dataclasses.replace(TABLE_LAW, eps_inf=eps_inf)
        field = gfem.MaterialField.interpolate(
            jnp.array([0.5]),
            gfem.Material(eps_r=2.0),
            gfem.Material(eps_r=law),
            scheme="index",
        )
        return jnp.sum(field.blend)

    jax.make_jaxpr(jax.grad(objective))(2.0)


def _out_field_with_axes_on_a_wave_port_face(request):
    # a field with axes at the port: the port reads its tensors (the forward
    # takes them), its principal values may not move (raised while tracing)
    model = request.getfixturevalue("wave_model")
    problem = gfem.Driven(frequency=20e9)
    n = len(model.mesh.elements("dielectric"))

    def objective(eps):
        field = gfem.MaterialField(eps_r=eps, axes=np.eye(3))
        m = model.with_material("dielectric", field)
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return jnp.sum(jnp.abs(gfem.solve(m, problem).s.values) ** 2)

    eps = jnp.tile(jnp.array([2.0, 2.5, 3.0]), (n, 1))
    jax.make_jaxpr(jax.grad(objective))(eps)


def _out_material_axes(request):
    # the principal directions are static data, never a design input
    model = request.getfixturevalue("driven_model")
    problem = gfem.Driven(frequency=5e9)
    n = len(model.mesh.elements("dielectric"))

    def objective(axes):
        field = gfem.MaterialField(eps_r=jnp.full((n, 3), 2.0), axes=axes)
        m = model.with_material("dielectric", field)
        return jnp.sum(jnp.abs(gfem.solve(m, problem).s.values) ** 2)

    jax.grad(objective)(jnp.eye(3))


def _out_anisotropic_material_value(request):
    # a Material is a constant, anisotropic or not: the error names the
    # field that holds its principal values as leaves
    model = request.getfixturevalue("driven_model")
    problem = gfem.Driven(frequency=5e9)

    def objective(eps):
        m = model.with_material("dielectric", gfem.Material(eps_r=(eps, 2.5, 3.0)))
        return jnp.sum(jnp.abs(gfem.solve(m, problem).s.values) ** 2)

    jax.grad(objective)(2.0)


def _out_nodes_beside_a_tensor_absorbing_face(request):
    # the node pass transports scalar absorbing coefficients only
    driven = request.getfixturevalue("driven_model")
    model = driven.replace(
        materials={"dielectric": gfem.Material(eps_r=(2.0, 2.5, 3.0))},
        boundaries=[*driven.boundaries, gfem.Absorbing("side")],
    )
    problem = gfem.Driven(frequency=5e9)

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.mesh), [nodes]
        )
        s = gfem.solve(model.replace(mesh=mesh), problem).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    jax.make_jaxpr(jax.grad(objective))(jnp.asarray(model.mesh.nodes))


def _out_material_on_a_shared_wave_port_face(request):
    # the error names the shared port surface and every port on it
    model = request.getfixturevalue("shared_surface_model")
    n = len(model.mesh.elements("fill"))

    def objective(eps):
        m = model.with_material("fill", gfem.MaterialField(eps_r=eps))
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return jnp.sum(jnp.abs(gfem.solve(m, SHARED_SURFACE_PROBLEM).s.values) ** 2)

    jax.make_jaxpr(jax.grad(objective))(jnp.full(n, 2.0))


def _moved_port_objective(model, value):
    """``value(result)`` with one node of port1 moved by ``dx`` along x."""
    mesh = model.mesh
    node = int(mesh.triangles[mesh.triangle_tags == mesh.region("port1").tag][0, 0])

    def objective(dx):
        x = jnp.asarray(mesh.nodes).at[node, 0].add(dx)
        moved = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(mesh), [x])
        return value(model.replace(mesh=moved))

    return objective


def _out_driven_port_nodes_moved(request):
    # shape gradients hold every port's geometry: raised on the host
    model = request.getfixturevalue("driven_model")
    problem = gfem.Driven(frequency=5e9)
    objective = _moved_port_objective(
        model, lambda m: jnp.sum(jnp.abs(gfem.solve(m, problem).s.values) ** 2)
    )
    jax.grad(objective)(1e-6)


def _out_eigenmode_port_nodes_moved(request):
    model = request.getfixturevalue("eigen_model")
    problem = gfem.Eigenmode(target=17e9)
    objective = _moved_port_objective(
        model, lambda m: jnp.real(gfem.solve(m, problem).omega[0]) * 1e-9
    )
    jax.grad(objective)(1e-6)


def _out_eigenmode_field_port_nodes_moved(request):
    # the ratios' node gradient holds the port surfaces too: a moved port node
    # raises on the host, as for omega
    model = request.getfixturevalue("eigen_model")
    problem = gfem.Eigenmode(target=17e9)
    objective = _moved_port_objective(
        model, lambda m: gfem.solve(m, problem).q_port["p1"][0]
    )
    jax.grad(objective)(1e-6)


def _out_greedy_adaptive_sweep(request):
    # the sample set must be frozen (Driven(samples=...)) under a transform
    model = request.getfixturevalue("driven_model")
    problem = gfem.Driven(frequency=(4e9, 5e9, 6e9), sweep="adaptive")
    eps = model.materials["dielectric"].eps_r

    def objective(e):
        m = model.with_material("dielectric", gfem.MaterialField(eps_r=e))
        return jnp.sum(jnp.abs(gfem.solve(m, problem).s.values) ** 2)

    jax.make_jaxpr(jax.grad(objective))(eps)


def _out_forward_mode(request):
    # reverse mode only: jax.jvp of a solve is JAX's own TypeError
    model = request.getfixturevalue("model")
    problem = PROBLEMS["electrostatic"]
    eps = model.materials["oxide"].eps_r

    def objective(e):
        m = model.with_material("oxide", gfem.MaterialField(eps_r=e))
        return gfem.solve(m, problem).capacitance[0, 0]

    jax.jvp(objective, (eps,), (jnp.ones_like(eps),))


def _out_probe_in_the_node_coordinates(request):
    # measurements read fields on the mesh: no node gradients through them
    model = request.getfixturevalue("driven_model")
    problem = gfem.Driven(frequency=5e9, keep_fields=True)
    points = np.array([[2.0, 1.0, 0.5]]) * 1e-3

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.mesh), [nodes]
        )
        frozen = model.replace(mesh=jax.lax.stop_gradient(mesh))
        field = gfem.solve(frozen, problem).e_field(5e9, "p1")
        field = dataclasses.replace(field, mesh=mesh)
        return jnp.sum(jnp.abs(field.at(points)) ** 2)

    jax.make_jaxpr(jax.grad(objective))(jnp.asarray(model.mesh.nodes))


def _out_electric_energy_of_a_traced_field(request):
    # forward only: the measurement takes fixed fields and models
    model = request.getfixturevalue("driven_model")
    problem = gfem.Driven(frequency=5e9, keep_fields=True)
    eps = model.materials["dielectric"].eps_r

    def objective(e):
        m = model.with_material("dielectric", gfem.MaterialField(eps_r=e))
        field = gfem.solve(m, problem).e_field(5e9, "p1")
        return gfem.electric_energy(field, model=m)["dielectric"]

    jax.make_jaxpr(jax.grad(objective))(eps)


def _out_power_flow_of_a_material_in_its_layer(request):
    # the side is the leads: the core field is in the layer power_flow reads
    model = request.getfixturevalue("wave_model")
    problem = gfem.Driven(frequency=20e9, keep_fields=True)
    eps = model.materials["core"].eps_r

    def objective(e):
        m = model.with_material("core", gfem.MaterialField(eps_r=e))
        field = gfem.solve(m, problem).e_field(20e9, "p1")
        return gfem.power_flow(field, "dielectric", model=m)

    jax.make_jaxpr(jax.grad(objective))(eps)


def _out_power_flow_in_the_node_coordinates(request):
    model = request.getfixturevalue("wave_model")
    problem = gfem.Driven(frequency=20e9, keep_fields=True)

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.mesh), [nodes]
        )
        frozen = model.replace(mesh=jax.lax.stop_gradient(mesh))
        field = gfem.solve(frozen, problem).e_field(20e9, "p1")
        field = dataclasses.replace(field, mesh=mesh)
        return gfem.power_flow(field, "core", model=frozen)

    jax.make_jaxpr(jax.grad(objective))(jnp.asarray(model.mesh.nodes))


def _out_power_flow_of_a_reduced_field(request):
    # an adaptive sweep's basis holds no adjoint of the power
    model = request.getfixturevalue("wave_model")
    problem = gfem.Driven(
        frequency=(18e9, 20e9, 22e9),
        sweep="adaptive",
        samples=(18e9, 22e9),
        keep_fields=20e9,
    )
    eps = model.materials["core"].eps_r

    def objective(e):
        m = model.with_material("core", gfem.MaterialField(eps_r=e))
        field = gfem.solve(m, problem).e_field(20e9, "p1")
        return gfem.power_flow(field, "core", model=m)

    jax.make_jaxpr(jax.grad(objective))(eps)


def _out_driven_traced_wave_vector(request):
    # the wave vector is static: a Bloch phase is not a design variable
    model, _ = request.getfixturevalue("driven_periodic_cases")["driven_periodic_bloch"]

    def objective(kx):
        problem = gfem.Driven(frequency=5e9, wave_vector=(kx, 0.0, 0.0))
        return jnp.sum(jnp.abs(gfem.solve(model, problem).s.values) ** 2)

    jax.grad(objective)(pc.C1_K[0])


def _out_periodic_pair_broken(request):
    # a node move that breaks a pair has no forward: raised on the host
    model, problem = request.getfixturevalue("driven_periodic_cases")[
        "driven_periodic_plain"
    ]
    mesh = model.mesh
    node = int(model._periodic[0].second_nodes[0])

    def objective(dy):
        x = jnp.asarray(mesh.nodes).at[node, 1].add(dy)
        moved = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(mesh), [x])
        s = gfem.solve(model.replace(mesh=moved), problem).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    jax.grad(objective)(1e-5)


def _out_morph_of_a_paired_surface(request):
    model, _ = request.getfixturevalue("driven_periodic_cases")["driven_periodic_plain"]
    boundary_morph(model, "x_max")


def _out_bloch_adaptive_sweep(request):
    model, _ = request.getfixturevalue("driven_periodic_cases")["driven_periodic_bloch"]
    problem = gfem.Driven(
        frequency=(4e9, 5e9, 6e9),
        sweep="adaptive",
        samples=(4e9, 6e9),
        wave_vector=pc.C1_K,
    )
    eps = model.materials["cell"].eps_r

    def objective(e):
        m = model.with_material("cell", gfem.MaterialField(eps_r=e))
        return jnp.sum(jnp.abs(gfem.solve(m, problem).s.values) ** 2)

    jax.grad(objective)(eps)


# --- periodic eigenmode rows ----------------------------------------------------------


@pytest.fixture(scope="module")
def bragg_cell(tmp_path_factory):
    """The C3 Bragg line (tests/periodic_cases.py): one periodic pair."""
    return pc.c3(pc.c3_mesh(tmp_path_factory.mktemp("bragg")))


def _bloch_eigenmode(kx):
    return gfem.Eigenmode(target=5e9, wave_vector=(kx, 0.0, 0.0))


def _out_eigenmode_traced_wave_vector(request):
    # the wave vector is a static problem field, never a leaf
    model = request.getfixturevalue("bragg_cell")

    def objective(kx):
        return jnp.real(gfem.solve(model, _bloch_eigenmode(kx)).omega[0])

    jax.make_jaxpr(jax.grad(objective))(math.pi / (pc.C3_A * pc.MM))


def _out_bloch_eigenmode_on_bddc(request):
    # two-sided refinement solves the transposed system on the shift's
    # factors: a Bloch eigenmode solve refuses BDDC + GMRES before assembly
    model = request.getfixturevalue("bragg_cell")
    options = gfem.SolverOptions(linear_solver="bddc_gmres")
    k = math.pi / (pc.C3_A * pc.MM)
    gfem.solve(model, _bloch_eigenmode(k), options=options)


# --- eigenmode mode-field rows (ADR 0102) ---------------------------------------------


def _eigen_ratio_objective(model, problem, options, read):
    """``read(result)`` as a function of the dielectric's permittivity."""
    eps = model.materials["dielectric"].eps_r

    def objective(e):
        m = model.with_material(
            "dielectric", dataclasses.replace(model.materials["dielectric"], eps_r=e)
        )
        return read(gfem.solve(m, problem, options=options))

    return objective, eps


def _trace_eigen_q_port(request, **options):
    # the static refusals: raised while tracing, before any host work
    model = request.getfixturevalue("eigen_model")
    objective, eps = _eigen_ratio_objective(
        model,
        gfem.Eigenmode(target=17e9),
        gfem.SolverOptions(**options),
        lambda r: jnp.sum(r.q_port["p1"]),
    )
    jax.make_jaxpr(jax.grad(objective))(eps)


def _out_eigenmode_error(request):
    model = request.getfixturevalue("eigen_model")
    objective, eps = _eigen_ratio_objective(
        model,
        gfem.Eigenmode(target=17e9),
        gfem.SolverOptions(),
        lambda r: jnp.sum(r.error),
    )
    jax.make_jaxpr(jax.grad(objective))(eps)


def _out_eigenmode_field_bddc(request):
    _trace_eigen_q_port(request, linear_solver="bddc_gmres")


def _out_eigenmode_field_petsc(request):
    # the rule resolves the PETSc linear solver before tracing: without a
    # usable PETSc stack that is the error, not the refusal
    from gfem import _petsc

    try:
        _petsc.require_petsc("linear_solver")
    except gfem.SolverUnavailableError as err:
        pytest.skip(str(err))
    _trace_eigen_q_port(request, linear_solver="petsc")


def _out_eigenmode_field_cuda(request):
    # the GPU is first used by the forward, which the trace does not run
    _trace_eigen_q_port(request, device="cuda")


def _out_eigenmode_field_tolerance(request):
    _trace_eigen_q_port(request, eigen_tolerance=1e-8)


def _out_eigenmode_mode_field_nodes(request):
    # the normalized field (so Field.at, interface_participation) by the node
    # coordinates: never in reach (finite differences over morphed meshes)
    model = request.getfixturevalue("eigen_model")
    problem = gfem.Eigenmode(target=17e9)

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.mesh), [nodes]
        )
        field = gfem.solve(model.replace(mesh=mesh), problem).e_fields[0]
        return jnp.sum(jnp.abs(field.coefficients) ** 2)

    jax.make_jaxpr(jax.grad(objective))(jnp.asarray(model.mesh.nodes))


def _bragg_bulk(request, problem):
    model = request.getfixturevalue("bragg_cell")
    n = len(model.mesh.elements("low"))

    def objective(e):
        m = model.with_material("low", gfem.MaterialField(eps_r=e))
        return gfem.solve(m, problem).bulk_participation["low"][0]

    jax.make_jaxpr(jax.grad(objective))(jnp.full(n, pc.C3_EPS_LOW))


def _out_eigenmode_field_periodic(request):
    _bragg_bulk(request, gfem.Eigenmode(target=pc.C3_PLAIN_TARGET))


def _out_eigenmode_field_bloch(request):
    _bragg_bulk(request, _bloch_eigenmode(math.pi / (pc.C3_A * pc.MM)))


@pytest.fixture(scope="module")
def corner_cavity(tmp_path_factory):
    """The mirrored cavity with a corner region (an energy share)."""
    return mirrored_cavity(tmp_path_factory.mktemp("corner"), corner=True)


def _corner_share(request, *, delta, n_modes, read):
    mesh = request.getfixturevalue("corner_cavity")
    problem = gfem.Eigenmode(target=MIRRORED_TARGET, n_modes=n_modes)

    def objective(eps):
        result = gfem.solve(split_cavity(mesh, eps, delta), problem)
        return read(result.bulk_participation["corner"])

    jax.grad(objective)(jnp.ones(len(mesh.tetrahedra)))


def _out_eigenmode_field_cluster_member(request):
    # TE101 = TE011 on the mirrored mesh: one mode's share, and their sum
    # (symmetric in the pair) too, since each member's field is arbitrary
    with pytest.raises(HOST, match=r"DegenerateModeError.*degenerate cluster"):
        _corner_share(request, delta=0.0, n_modes=2, read=jnp.sum)
    _corner_share(request, delta=0.0, n_modes=2, read=lambda share: share[0])


def _out_eigenmode_field_cut_cluster(request):
    _corner_share(request, delta=0.0, n_modes=1, read=lambda share: share[0])


def _out_eigenmode_field_near_pair(request):
    # split to a relative gap of 2.5e-5: outside cluster_rtol, below 1e-4
    _corner_share(request, delta=1e-4, n_modes=2, read=lambda share: share[0])


def _out_eigenmode_q_port_without_voltage(request):
    # no physical port reads exactly zero voltage on a mode (round-off), so
    # p2's voltage functional is replaced by zero: its q_port is infinite
    from gfem._engine import terms

    monkeypatch = request.getfixturevalue("monkeypatch")
    model = request.getfixturevalue("eigen_model")
    original = terms.port_vectors

    def deaf(space, built, order, data):
        b, v = original(space, built, order, data)
        return (b, np.zeros_like(v)) if data.port.name == "p2" else (b, v)

    monkeypatch.setattr(terms, "port_vectors", deaf)
    objective, eps = _eigen_ratio_objective(
        model,
        gfem.Eigenmode(target=17e9),
        gfem.SolverOptions(),
        lambda r: r.q_port["p2"][0],
    )
    jax.grad(objective)(eps)


def _out_eigenmode_principal_by_a_lumped_value(request):
    # the glued line's lumped model with principal values in "lower": its
    # principal shares by port p1's inductance (no cell checks that pair yet)
    mesh = request.getfixturevalue("sheet_mesh")
    lower = gfem.Material(eps_r=(2.0, 2.4, 1.8))

    def objective(inductance):
        p1 = dataclasses.replace(_rlc_port("p1", "port1"), inductance=inductance)
        p2 = gfem.LumpedPort("p2", "port2", "+y", impedance=1e3)
        model = _lumped_model(mesh, [p1, p2], lower=lower)
        result = gfem.solve(model, gfem.Eigenmode(target=LUMPED_TARGET))
        return jnp.sum(result.principal_participation["lower"][0])

    jax.make_jaxpr(jax.grad(objective))(LUMPED[0])


# (row, raising call, error, message): DegenerateModeError on one mode of a
# degenerate eigenmode cluster is in tests/verification/test_gradient_matrix.py.
HOST = jax.errors.JaxRuntimeError  # a gfem error raised in the backward pass
OUT_OF_SCOPE = [
    (
        "electrostatic x sheet fill",
        _out_electrostatic_sheet,
        NotImplementedError,
        "sheet field",
    ),
    (
        "driven x fill of a sheet beside a wave port",
        _out_sheet_beside_a_wave_port,
        gfem.PortGeometryError,
        "shares an edge with the port face",
    ),
    (
        "driven x Impedance value beside a wave port",
        _out_impedance_beside_a_wave_port,
        gfem.PortGeometryError,
        r"Impedance model\.boundaries\[2\] shares an edge with the port face",
    ),
    (
        "LumpedPort resistance",
        _out_lumped_port_resistance,
        gfem.NonDifferentiableError,
        r"LumpedPort\.resistance is a fixed value.*inductance and capacitance",
    ),
    (
        "driven x material on a wave port face",
        _out_material_on_a_wave_port_face,
        gfem.PortGeometryError,
        "has elements on the port face",
    ),
    (
        "driven x law parameter of a region on a wave port face",
        _out_law_on_a_wave_port_face,
        gfem.PortGeometryError,
        r"has elements on the port face.*permittivity law",
    ),
    (
        "driven x law parameter under a Bloch wave vector",
        _out_law_under_a_bloch_wave_vector,
        gfem.NonDifferentiableError,
        r"eps_r\.eps_inf.*Bloch wave vector",
    ),
    (
        "driven x material field with axes on a wave port face",
        _out_field_with_axes_on_a_wave_port_face,
        gfem.PortGeometryError,
        "has elements on the port face",
    ),
    (
        "material axes (orientation)",
        _out_material_axes,
        gfem.NonDifferentiableError,
        r"axes are fixed.*principal values",
    ),
    (
        "values of an anisotropic Material",
        _out_anisotropic_material_value,
        gfem.NonDifferentiableError,
        r"MaterialField\.uniform.*principal values along its axes",
    ),
    (
        "nodes beside an anisotropic absorbing face",
        _out_nodes_beside_a_tensor_absorbing_face,
        gfem.NonDifferentiableError,
        r"Absorbing surface 'side' has an anisotropic material.*stop_gradient",
    ),
    (
        "driven x material on a shared wave port face",
        _out_material_on_a_shared_wave_port_face,
        gfem.PortGeometryError,
        (
            r"^WavePorts 'p1_m1', 'p1_m2' on surface 'port1': region 'fill' has "
            "elements on the port face"
        ),
    ),
    (
        "driven x nodes of a port surface",
        _out_driven_port_nodes_moved,
        HOST,
        r"PortGeometryError.*port1",
    ),
    (
        "eigenmode x nodes of a port surface",
        _out_eigenmode_port_nodes_moved,
        HOST,
        r"PortGeometryError.*port1",
    ),
    (
        "greedy adaptive sweep under a transformation",
        _out_greedy_adaptive_sweep,
        NotImplementedError,
        "samples=",
    ),
    ("forward mode", _out_forward_mode, TypeError, "jvp"),
    (
        "probe x node coordinates",
        _out_probe_in_the_node_coordinates,
        gfem.NonDifferentiableError,
        r"mesh node coordinates.*stop_gradient",
    ),
    (
        "electric energy of a field",
        _out_electric_energy_of_a_traced_field,
        gfem.NonDifferentiableError,
        r"gfem\.electric_energy is not differentiable",
    ),
    (
        "power_flow x material in its layer",
        _out_power_flow_of_a_material_in_its_layer,
        gfem.NonDifferentiableError,
        r"materials\['core'\]\.eps_r.*element layer.*move the surface",
    ),
    (
        "power_flow x node coordinates",
        _out_power_flow_in_the_node_coordinates,
        gfem.NonDifferentiableError,
        r"mesh node coordinates.*stop_gradient",
    ),
    (
        "power_flow x reduced field",
        _out_power_flow_of_a_reduced_field,
        gfem.NonDifferentiableError,
        r"adaptive sweep.*frozen reduced basis.*sweep=\"uniform\"",
    ),
    # driven on a periodic model
    (
        "driven x wave_vector",
        _out_driven_traced_wave_vector,
        gfem.NonDifferentiableError,
        "wave_vector must be fixed",
    ),
    (
        "periodic model x node move that breaks a pair",
        _out_periodic_pair_broken,
        HOST,
        r"ValueError.*breaks Periodic\(",
    ),
    (
        "boundary_morph x paired moving surface",
        _out_morph_of_a_paired_surface,
        ValueError,
        "cannot be the moving surface",
    ),
    (
        "Bloch driven x adaptive sweep",
        _out_bloch_adaptive_sweep,
        NotImplementedError,
        "sweep='uniform'",
    ),
    # periodic eigenmode rows
    (
        "eigenmode x wave_vector",
        _out_eigenmode_traced_wave_vector,
        gfem.NonDifferentiableError,
        "wave_vector must be fixed",
    ),
    (
        "Bloch eigenmode x bddc_gmres",
        _out_bloch_eigenmode_on_bddc,
        ValueError,
        "linear_solver='direct'",
    ),
    # eigenmode mode-field outputs (ADR 0102)
    (
        "eigenmode backward error",
        _out_eigenmode_error,
        gfem.NonDifferentiableError,
        "'error'",
    ),
    (
        "eigenmode mode-field output x bddc_gmres",
        _out_eigenmode_field_bddc,
        gfem.NonDifferentiableError,
        'linear_solver="bddc_gmres"',
    ),
    (
        "eigenmode mode-field output x petsc",
        _out_eigenmode_field_petsc,
        gfem.NonDifferentiableError,
        'linear_solver="petsc"',
    ),
    (
        "eigenmode mode-field output x cuda",
        _out_eigenmode_field_cuda,
        gfem.NonDifferentiableError,
        'device="cuda"',
    ),
    (
        "eigenmode mode-field output x eigen_tolerance above 1e-10",
        _out_eigenmode_field_tolerance,
        gfem.NonDifferentiableError,
        "eigen_tolerance = 1e-08",
    ),
    (
        "eigenmode mode-field output x plain periodic model",
        _out_eigenmode_field_periodic,
        gfem.NonDifferentiableError,
        "periodic model",
    ),
    (
        "eigenmode mode-field output x Bloch wave vector",
        _out_eigenmode_field_bloch,
        gfem.NonDifferentiableError,
        "Bloch wave vector",
    ),
    (
        "eigenmode mode-field output x nodes of a port surface",
        _out_eigenmode_field_port_nodes_moved,
        HOST,
        r"PortGeometryError.*port1",
    ),
    (
        "eigenmode mode field x node coordinates",
        _out_eigenmode_mode_field_nodes,
        gfem.NonDifferentiableError,
        r"'e_fields.*' with respect to the mesh node coordinates.*finite differences",
    ),
    (
        "eigenmode mode-field output x member of a cluster",
        _out_eigenmode_field_cluster_member,
        HOST,
        "DegenerateModeError.*degenerate cluster",
    ),
    (
        "eigenmode mode-field output x cluster cut by n_modes",
        _out_eigenmode_field_cut_cluster,
        HOST,
        "DegenerateModeError.*n_modes",
    ),
    (
        "eigenmode mode-field output x nearly degenerate mode",
        _out_eigenmode_field_near_pair,
        HOST,
        "DegenerateModeError.*nearly degenerate",
    ),
    (
        "eigenmode q_port of a mode with no voltage on the port",
        _out_eigenmode_q_port_without_voltage,
        HOST,
        r"NonDifferentiableError.*1 / q_port",
    ),
    (
        "law parameter of a blend's end",
        _out_law_of_a_blend_end,
        gfem.NonDifferentiableError,
        "differentiate the law on a `Material` region",
    ),
    (
        "eigenmode principal_participation x lumped value",
        _out_eigenmode_principal_by_a_lumped_value,
        gfem.NonDifferentiableError,
        r"'principal_participation'.*model\.ports\[0\]\.inductance.*stop_gradient",
    ),
]


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [row[1:] for row in OUT_OF_SCOPE],
    ids=[row[0] for row in OUT_OF_SCOPE],
)
def test_out_of_scope_pairs_raise_named_errors(request, call, error, match):
    with pytest.raises(error, match=match):
        call(request)


# --- the frequency path beside the mode-field rows --------------------------------


def _omega_options(request, **options):
    # rows 3 to 6: the refusals are static, so tracing is the whole check
    model = request.getfixturevalue("eigen_model")
    objective, eps = _eigen_ratio_objective(
        model,
        gfem.Eigenmode(target=17e9),
        gfem.SolverOptions(**options),
        lambda r: jnp.real(r.omega[0]) * 1e-9,
    )
    jax.make_jaxpr(jax.grad(objective))(eps)


def _omega_petsc(request):
    from gfem import _petsc

    try:
        _petsc.require_petsc("linear_solver")
    except gfem.SolverUnavailableError as err:
        pytest.skip(str(err))
    _omega_options(request, linear_solver="petsc")


def _omega_bragg(request, problem):
    model = request.getfixturevalue("bragg_cell")
    n = len(model.mesh.elements("low"))

    def objective(e):
        m = model.with_material("low", gfem.MaterialField(eps_r=e))
        return jnp.real(gfem.solve(m, problem).omega[0]) * 1e-9

    jax.make_jaxpr(jax.grad(objective))(jnp.full(n, pc.C3_EPS_LOW))


def _omega_corner(request, *, delta, read):
    # rows 10 and 12 refuse in the backward: these run it
    mesh = request.getfixturevalue("corner_cavity")
    problem = gfem.Eigenmode(target=MIRRORED_TARGET, n_modes=2)

    def objective(eps):
        result = gfem.solve(split_cavity(mesh, eps, delta), problem)
        return read(jnp.real(result.omega)) * 1e-9

    grad = jax.grad(objective)(jnp.ones(len(mesh.tetrahedra)))
    assert np.isfinite(np.asarray(grad)).all()
    assert np.abs(np.asarray(grad)).max() > 0


#: The models of the mode-field rows 3 to 12 (ticket 23's numbering) with
#: an omega-only objective: the new refusals stay off the frequency path.
#: Rows 9 (a moved port node) and 11 (a cluster cut by n_modes) are left
#: out: omega raises there too, by its own rows ("eigenmode x nodes of a
#: port surface", and tests/verification/test_gradient_matrix.py). On the
#: cluster of row 10 omega differentiates as a sum over the cluster.
OMEGA_BESIDE_ROWS = [
    ("bddc_gmres", lambda r: _omega_options(r, linear_solver="bddc_gmres")),
    ("petsc", _omega_petsc),
    ("cuda", lambda r: _omega_options(r, device="cuda")),
    ("eigen_tolerance above 1e-10", lambda r: _omega_options(r, eigen_tolerance=1e-8)),
    (
        "plain periodic model",
        lambda r: _omega_bragg(r, gfem.Eigenmode(target=pc.C3_PLAIN_TARGET)),
    ),
    (
        "Bloch wave vector",
        lambda r: _omega_bragg(r, _bloch_eigenmode(math.pi / (pc.C3_A * pc.MM))),
    ),
    ("cluster sum", lambda r: _omega_corner(r, delta=0.0, read=jnp.sum)),
    (
        "nearly degenerate mode",
        lambda r: _omega_corner(r, delta=1e-4, read=lambda omega: omega[0]),
    ),
]


@pytest.mark.parametrize(
    "call",
    [row[1] for row in OMEGA_BESIDE_ROWS],
    ids=[row[0] for row in OMEGA_BESIDE_ROWS],
)
def test_omega_on_the_mode_field_rows_models_still_differentiates(request, call):
    call(request)


def test_stop_gradient_opts_a_read_input_out(model):
    problem = PROBLEMS["electrostatic"]

    def objective(m):
        frozen = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return gfem.solve(frozen, problem).capacitance[0, 0]  # in series with oxide

    grads = jax.grad(objective)(model)
    assert not np.asarray(grads.mesh.nodes).any()
    assert np.abs(np.asarray(grads.materials["oxide"].eps_r)).max() > 0


class _RuleWithExtraOutput(ElectrostaticRule):
    """The electrostatic rule plus an output its backward cannot handle."""

    def output_specs(self, model):
        return {**super().output_specs(model), "extra": jax.ShapeDtypeStruct((), float)}

    def forward(self, model, *, keep):
        outputs, entry = super().forward(model, keep=keep)
        return {**outputs, "extra": np.trace(outputs["capacitance"])}, entry


def test_nondifferentiable_output_raises_unless_stop_gradient(model):
    problem = PROBLEMS["electrostatic"]
    rule = _RuleWithExtraOutput(model, problem, gfem.SolverOptions())
    n = len(model.mesh.elements("oxide"))

    def objective(eps, *, stop):
        m = model.with_material("oxide", gfem.MaterialField(eps_r=eps))
        out = solve_vjp(rule, m, problem)
        extra = jax.lax.stop_gradient(out["extra"]) if stop else out["extra"]
        return out["capacitance"][0, 0] + extra

    eps = jnp.full(n, 3.9)
    with pytest.raises(gfem.NonDifferentiableError, match="'extra'"):
        jax.make_jaxpr(jax.grad(lambda e: objective(e, stop=False)))(eps)
    grad = jax.grad(lambda e: objective(e, stop=True))(eps)

    def c_top(e):
        m = model.with_material("oxide", gfem.MaterialField(eps_r=e))
        return gfem.solve(m, problem).capacitance[0, 0]

    assert np.abs(np.asarray(grad)).max() > 0
    np.testing.assert_allclose(grad, jax.grad(c_top)(eps), rtol=1e-14)
