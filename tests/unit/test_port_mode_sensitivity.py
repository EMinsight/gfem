"""The section kernel against central differences of the assembled forms.

``F = Re q^T (A - lam B) x`` for fixed random complex vectors ``q`` and ``x``
and a fixed ``lam``, on a section whose coefficient moves: the kernel's
cotangent contracted with the move must equal ``dF``. The forms are
re-assembled by `PortModeSpace` at each point, so the check covers the form
list, the quadrature and the stretch of a perfectly matched layer.
"""

import dataclasses
from pathlib import Path

import anisotropy
import numpy as np
import pml_cases
import pytest

import gfem
from gfem import _pml
from gfem._boundaries import Conductor
from gfem._engine.port_mode import PortModeSpace
from gfem._engine.port_mode_sensitivity import (
    FormPair,
    SectionKernel,
    impedance_cotangents,
    integrated,
    law_cotangents,
    owner_tetrahedra,
)
from gfem._port_section import port_section

DATA = Path(__file__).parent / "data"
ORDER = 2
LAW = gfem.PermittivityLaw(
    eps_inf=1.5,
    terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=12e9, damping=1e8)],
)
K0 = 0.35  # per local unit
OMEGA_OF = lambda section: K0 * 299792458.0 / section.scale  # noqa: E731


def guide(mesh_name, unit, materials, boundaries, ports=None):
    mesh = gfem.Mesh.from_gmsh(DATA / mesh_name, length_unit=unit)
    model = gfem.Model(
        mesh,
        materials=materials,
        boundaries=boundaries,
        ports=ports or [gfem.WavePort("p1", "port1")],
    )
    return model, port_section(model, model.ports[0])


def lossy_slab():
    return guide(
        "slab_waveguide.msh",
        "mm",
        {"slab": gfem.Material(eps_r=4.0, tan_delta=0.03), "air": gfem.Material()},
        [gfem.PEC("walls")],
    )


def vectors(section, seed=0):
    space = PortModeSpace(section, ORDER)
    n = len(space.free)
    rng = np.random.default_rng(seed)
    cplx = lambda: rng.standard_normal(n) + 1j * rng.standard_normal(n)  # noqa: E731
    return space, cplx(), cplx(), complex(rng.standard_normal() + 0.3j)


def form_value(section, q, x, lam):
    a, b = PortModeSpace(section, ORDER).pencil.at(K0)
    return complex(q @ (a @ x) - lam * (q @ (b @ x)))


def central(section, build, q, x, lam, h=1e-2):  # noqa: PLR0917
    up, dn = build(section, h), build(section, -h)
    return (form_value(up, q, x, lam).real - form_value(dn, q, x, lam).real) / (2 * h)


def random_like(shape, seed, real=False):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(shape)
    return d if real else d + 1j * rng.standard_normal(shape)


def cotangents(section, seed=0):
    space, q, x, lam = vectors(section, seed)
    return SectionKernel(space, K0).cotangents([(q, x, lam)]), q, x, lam


@pytest.mark.parametrize("name", ["eps", "sigma", "nu"])
def test_volume_coefficients_match_the_assembled_forms(name):
    _, section = lossy_slab()
    cot, q, x, lam = cotangents(section)
    n = len(section.triangles)
    d_tt, d_nn = random_like((n, 2, 2), 1), random_like(n, 2)
    if name == "sigma":
        d_tt, d_nn = d_tt.real, d_nn.real
    scale = (
        scipy_z0(section) if name == "sigma" else 1.0
    )  # the item's own weight, outside the kernel's cotangent
    field = {
        "eps": ("eps_tt", "eps_nn"),
        "sigma": ("sigma_tt", "sigma_nn"),
        "nu": ("nu_tt", "nu_nn"),
    }[name]

    def build(sec, h):
        return dataclasses.replace(
            sec,
            **{
                field[0]: getattr(sec, field[0]) + h * d_tt,
                field[1]: getattr(sec, field[1]) + h * d_nn,
            },
        )

    fd = central(section, build, q, x, lam)
    c_tt, c_nn = getattr(cot, field[0]), getattr(cot, field[1])
    got = float(np.real(np.sum(c_tt * d_tt) + np.sum(c_nn * d_nn))) * scale
    assert got == pytest.approx(fd, rel=1e-9)


def scipy_z0(section):
    import scipy.constants

    return scipy.constants.mu_0 * scipy.constants.c * section.scale


def test_a_pml_cut_strip_includes_the_stretch(tmp_path):
    path = pml_cases.strip_guide(
        tmp_path / "strip.msh", 0.4, quarter=False, length=0.15
    )
    strip = pml_cases.strip_model(path)
    stretch = _pml.stretch(strip, [pml_cases.STRIP_F])
    section = port_section(strip, strip.ports[0], stretch)
    assert section.layer is not None
    cot, q, x, lam = cotangents(section)
    n = len(section.triangles)
    for field in ("eps", "nu", "sigma"):
        d_tt, d_nn = random_like((n, 2, 2), 1), random_like(n, 2)
        if field == "sigma":
            d_tt, d_nn = 1e3 * d_tt.real, 1e3 * d_nn.real
        names = (f"{field}_tt", f"{field}_nn")

        def build(sec, h, names=names, d_tt=d_tt, d_nn=d_nn):
            return dataclasses.replace(
                sec,
                **{
                    names[0]: getattr(sec, names[0]) + h * d_tt,
                    names[1]: getattr(sec, names[1]) + h * d_nn,
                },
            )

        fd = central(section, build, q, x, lam)
        scale = scipy_z0(section) if field == "sigma" else 1.0
        got = float(
            np.real(
                np.sum(getattr(cot, names[0]) * d_tt)
                + np.sum(getattr(cot, names[1]) * d_nn)
            )
        )
        assert got * scale == pytest.approx(fd, rel=1e-9), field


def test_an_absorbing_edge_admittance_matches_the_assembled_forms():
    _, section = guide(
        "waveguide.msh",
        "mm",
        {"air": gfem.Material(eps_r=2.0)},
        [gfem.Absorbing("walls")],
    )
    assert len(section.absorbing_edges)
    cot, q, x, lam = cotangents(section)
    d = random_like((len(section.absorbing_edges), 1), 4, real=True)

    def build(sec, h):
        # a common move of both components of every edge's admittance
        return dataclasses.replace(
            sec, absorbing_admittance=sec.absorbing_admittance + h * d
        )

    fd = central(section, build, q, x, lam)
    got = float(np.real(np.sum((cot.y_tau + cot.y_n) * d[:, 0])))
    assert got == pytest.approx(fd, rel=1e-9)
    # and each component alone
    d2 = np.zeros((len(d), 2))
    d2[:, 0] = d[:, 0]

    def build2(sec, h):
        return dataclasses.replace(
            sec, absorbing_admittance=sec.absorbing_admittance + h * d2
        )

    fd2 = central(section, build2, q, x, lam)
    assert float(np.real(np.sum(cot.y_tau * d[:, 0]))) == pytest.approx(fd2, rel=1e-9)


def plates(item):
    return guide(
        "parallel_plate.msh",
        "um",
        {"fill": gfem.Material(eps_r=4.0, tan_delta=0.01)},
        [item],
        [gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


def test_sheet_fills_and_impedance_values_match_the_assembled_forms():
    item = gfem.Impedance("plates", resistance=2.0, inductance=3e-12, capacitance=4e-15)
    _, section = plates(item)
    section = dataclasses.replace(
        section, impedance_fill=np.linspace(0.3, 1.0, len(section.impedance_edges))
    )
    cot, q, x, lam = cotangents(section)
    d = random_like(len(section.impedance_edges), 5, real=True)

    def build(sec, h):
        return dataclasses.replace(sec, impedance_fill=sec.impedance_fill + h * d)

    fd = central(section, build, q, x, lam)
    assert float(np.real(np.sum(cot.fill * d))) == pytest.approx(fd, rel=1e-9)
    values = impedance_cotangents(section, cot)
    for name in ("resistance", "inductance", "capacitance"):

        def moved(sec, t, name=name):
            ((it, factor),) = sec.impedance_terms
            value = getattr(it, name)
            it2 = dataclasses.replace(it, **{name: value * (1 + t)})
            return dataclasses.replace(sec, impedance_terms=((it2, factor),))

        def at(t):
            return form_value(moved(section, t), q, x, lam).real

        h = 5e-3
        fd = (-at(2 * h) + 8 * at(h) - 8 * at(-h) + at(-2 * h)) / (12 * h)
        value = getattr(section.impedance_terms[0][0], name)
        assert values[0, name] * value == pytest.approx(fd, rel=1e-8), name


def test_a_conductor_edge_cpw_section_matches_the_assembled_forms():
    _, section = guide(
        "cpw_section.msh",
        "um",
        {"si": gfem.Material(eps_r=11.9), "air": gfem.Material()},
        [gfem.PEC("walls"), Conductor("metal", conductivity=5.8e7)],
    )
    assert len(section.impedance_edges)
    cot, q, x, lam = cotangents(section)
    d = random_like(len(section.impedance_edges), 6, real=True)
    fill = np.ones(len(d))

    def build(sec, h):
        return dataclasses.replace(sec, impedance_fill=fill + h * d)

    fd = central(section, build, q, x, lam)
    assert float(np.real(np.sum(cot.fill * d))) == pytest.approx(fd, rel=1e-9)
    n = len(section.triangles)
    d_tt, d_nn = random_like((n, 2, 2), 1), random_like(n, 2)

    def eps(sec, h):
        return dataclasses.replace(
            sec, eps_tt=sec.eps_tt + h * d_tt, eps_nn=sec.eps_nn + h * d_nn
        )

    fd = central(section, eps, q, x, lam)
    got = float(np.real(np.sum(cot.eps_tt * d_tt) + np.sum(cot.eps_nn * d_nn)))
    assert got == pytest.approx(fd, rel=1e-9)


def test_law_parameters_match_the_assembled_forms():
    def build(eps_inf=LAW.eps_inf, delta=1.0, res=12e9, damping=1e8):
        law = gfem.PermittivityLaw(
            eps_inf=eps_inf,
            terms=[
                gfem.Lorentz(delta_eps=delta, resonance_frequency=res, damping=damping)
            ],
        )
        return guide(
            "waveguide.msh",
            "mm",
            {"air": gfem.Material(eps_r=law)},
            [gfem.Absorbing("walls")],
        )[1]

    section = build()
    assert section.laws
    cot, q, x, lam = cotangents(section)
    kinds = law_cotangents(section, cot, OMEGA_OF(section))
    region = section.laws[0][0]
    for key, kw, step in (
        ((region, "eps_inf"), "eps_inf", 1e-4),
        ((region, 0, "delta_eps"), "delta", 1e-4),
        ((region, 0, "damping"), "damping", 1e4),
    ):
        base = {"eps_inf": 1.5, "delta": 1.0, "damping": 1e8}[kw]
        up = form_value(build(**{kw: base + step}), q, x, lam).real
        dn = form_value(build(**{kw: base - step}), q, x, lam).real
        assert kinds[("law", *key)] == pytest.approx(
            (up - dn) / (2 * step), rel=1e-7
        ), key


@pytest.mark.parametrize(
    "form", ["power_g", "power_t", "gauge_n", "gauge_t", "coupling_g", "coupling_t"]
)
def test_the_explicit_forms_move_with_the_inverse_permeability(form):
    _, section = lossy_slab()
    space = PortModeSpace(section, ORDER)
    n = len(space.free)
    rng = np.random.default_rng(8)
    cplx = lambda m: rng.standard_normal(m) + 1j * rng.standard_normal(m)  # noqa: E731
    right = cplx(n)
    left = (
        None
        if form.startswith("gauge")
        else cplx(space.n_t if form.startswith("coupling") else n)
    )
    weight = 0.7 - 0.4j
    pair = FormPair(form, left, right, weight)
    cot = SectionKernel(space, K0).explicit([pair])
    d = random_like((len(section.triangles), 2, 2), 9, real=True)

    def value(sec):
        sp = PortModeSpace(sec, ORDER)
        if form.startswith("power"):
            g, t = sp.power_forms
            f = g if form.endswith("_g") else t
            return weight * (left @ (f @ right))
        if form.startswith("gauge"):
            ln, lt = sp.gauge_forms
            return weight * ((ln if form.endswith("_n") else lt) @ right)
        g, t = sp.coupling_forms
        f = g if form.endswith("_g") else t
        return weight * (left @ (f @ right))

    h = 1e-2
    up = dataclasses.replace(section, nu_tt=section.nu_tt + h * d)
    dn = dataclasses.replace(section, nu_tt=section.nu_tt - h * d)
    fd = (value(up).real - value(dn).real) / (2 * h)
    assert float(np.real(np.sum(cot.nu_tt * d))) == pytest.approx(fd, rel=1e-8)


def test_explicit_forms_carry_the_stretch(tmp_path):
    path = pml_cases.strip_guide(
        tmp_path / "strip.msh", 0.4, quarter=False, length=0.15
    )
    strip = pml_cases.strip_model(path)
    section = port_section(
        strip, strip.ports[0], _pml.stretch(strip, [pml_cases.STRIP_F])
    )
    space = PortModeSpace(section, ORDER)
    n = len(space.free)
    rng = np.random.default_rng(8)
    right = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    left = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    d = random_like((len(section.triangles), 2, 2), 9, real=True)
    for form in ("power_g", "power_t", "gauge_n", "coupling_t"):
        pair = FormPair(
            form,
            None
            if form.startswith("gauge")
            else (left[: space.n_t] if form.startswith("coupling") else left),
            right,
            1.0 + 0.5j,
        )
        cot = SectionKernel(space, K0).explicit([pair])

        def value(sec, form=form, pair=pair):
            sp = PortModeSpace(sec, ORDER)
            if form.startswith("power"):
                g, t = sp.power_forms
                return pair.weight * (
                    pair.left @ ((g if form.endswith("_g") else t) @ right)
                )
            if form.startswith("gauge"):
                return pair.weight * (sp.gauge_forms[0] @ right)
            return pair.weight * (pair.left @ (sp.coupling_forms[1] @ right))

        h = 1e-2
        up = dataclasses.replace(section, nu_tt=section.nu_tt + h * d)
        dn = dataclasses.replace(section, nu_tt=section.nu_tt - h * d)
        fd = (value(up).real - value(dn).real) / (2 * h)
        assert float(np.real(np.sum(cot.nu_tt * d))) == pytest.approx(fd, rel=1e-8), (
            form
        )


# --- the 3D owner tetrahedra ----------------------------------------------------------


@pytest.fixture(scope="module")
def turned(tmp_path_factory):
    mesh = anisotropy.wr28(
        tmp_path_factory.mktemp("wr28"), length=10.0, cells=(6, 3, 4)
    )
    return mesh


def turned_section(mesh, eps, mu, angle):
    model = gfem.Model(
        mesh,
        materials={
            "fill": gfem.Material(eps_r=eps, mu_r=mu, axes=anisotropy.turned(2, angle))
        },
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    return model, port_section(model, model.ports[0])


def test_turned_axes_moments_give_the_principal_value_derivatives(turned):
    eps, mu, angle = (2.0, 4.0, 3.0), (1.0, 2.0, 1.5), 30.0
    model, section = turned_section(turned, eps, mu, angle)
    space = PortModeSpace(section, ORDER)
    n = len(space.free)
    rng = np.random.default_rng(1)
    q = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    lam = 0.3 - 0.1j
    cot = SectionKernel(space, K0).cotangents([(q, x, lam)])
    owners = owner_tetrahedra(model, section)
    n_tets = len(model.mesh.volume_elements)
    out = integrated(section, cot, owners, n_tets)
    axes = np.asarray(anisotropy.turned(2, angle), float)
    for k in range(3):
        a = axes[k]
        for prop, name, slope in (
            ("eps", "mass", lambda _: 1.0),
            ("mu", "curl_curl", lambda v: -1.0 / v**2),
        ):
            vals = {"eps": list(eps), "mu": list(mu)}
            h = 1e-5
            base = vals[prop][k]

            def at(v, prop=prop, k=k):
                new = {"eps": list(eps), "mu": list(mu)}
                new[prop][k] = v
                return form_value(
                    turned_section(turned, tuple(new["eps"]), tuple(new["mu"]), angle)[
                        1
                    ],
                    q,
                    x,
                    lam,
                ).real

            fd = (at(base + h) - at(base - h)) / (2 * h)
            moment = out.moments[name].sum(axis=0)
            got = float(np.real(a @ moment @ a)) * slope(base)
            assert got == pytest.approx(fd, rel=1e-8), (prop, k)
            # the scalar value of an isotropic move is the trace
            assert np.trace(moment) == pytest.approx(out.values[name].sum())


def test_a_uniform_eps_scale_of_the_face_tetrahedra_matches_the_mode():
    """``d lam / d eps`` of a mode: the owner-tetrahedron values, re-solved."""
    from gfem._sweep.port_mode import _Greedy

    model, section = lossy_slab()
    f = 12e9
    space = PortModeSpace(section, ORDER)
    g = _Greedy(space, np.array([f]), [1], 1e-6, [section.port])
    cand = g._solve(0)
    k0, lam, x = float(g.k0[0]), complex(cand.lam[0]), cand.vecs[:, 0].copy()
    a, b = space.pencil.at(k0)
    q = x / (x @ (b @ x))
    cot = SectionKernel(space, k0).cotangents([(q, x, lam)])
    owners = owner_tetrahedra(model, section)
    out = integrated(section, cot, owners, len(model.mesh.volume_elements))
    eps_slab = 4.0
    tan = 0.03
    tets = np.unique(owners)
    slab_tets = np.isin(
        model.mesh.tetrahedron_tags[tets], [model.mesh.region("slab").tag]
    )
    # mass coefficient eps (1 - i tan): d/d eps of the slab tetrahedra
    got = (out.values["mass"][tets[slab_tets]] * (1 - 1j * tan)).sum()

    def lam_at(eps):
        _, sec = guide(
            "slab_waveguide.msh",
            "mm",
            {"slab": gfem.Material(eps_r=eps, tan_delta=tan), "air": gfem.Material()},
            [gfem.PEC("walls")],
        )
        sp = PortModeSpace(sec, ORDER)
        gg = _Greedy(sp, np.array([f]), [1], 1e-6, [sec.port])
        return complex(gg._solve(0).lam[0])

    h = 1e-4
    fd = (lam_at(eps_slab + h) - lam_at(eps_slab - h)) / (2 * h)
    assert got == pytest.approx(fd, rel=1e-7)
    del a


def test_the_absorbing_edge_values_reach_the_owner_tetrahedra():
    """``d lam / d eps`` of an absorbing-walled guide: mass plus admittance values."""
    from gfem._sweep.port_mode import _Greedy

    f, eps = 12e9, 2.0

    def build(e):
        return guide(
            "waveguide.msh",
            "mm",
            {"air": gfem.Material(eps_r=e)},
            [gfem.Absorbing("walls")],
        )

    model, section = build(eps)
    space = PortModeSpace(section, ORDER)
    g = _Greedy(space, np.array([f]), [1], 1e-6, [section.port])
    cand = g._solve(0)
    k0, lam, x = float(g.k0[0]), complex(cand.lam[0]), cand.vecs[:, 0].copy()
    _, b = space.pencil.at(k0)
    q = x / (x @ (b @ x))
    cot = SectionKernel(space, k0).cotangents([(q, x, lam)])
    owners = owner_tetrahedra(model, section)
    out = integrated(section, cot, owners, len(model.mesh.volume_elements))
    assert np.abs(out.values["absorbing"]).sum() > 0
    # Y = sqrt(eps / mu): dY / d eps = 1 / (2 sqrt(eps))
    got = out.values["mass"].sum() + out.values["absorbing"].sum() / (2 * np.sqrt(eps))

    def lam_at(e):
        _, sec = build(e)
        sp = PortModeSpace(sec, ORDER)
        gg = _Greedy(sp, np.array([f]), [1], 1e-6, [sec.port])
        return complex(gg._solve(0).lam[0])

    h = 1e-3
    fd = (lam_at(eps + h) - lam_at(eps - h)) / (2 * h)
    assert got == pytest.approx(fd, rel=1e-7)
