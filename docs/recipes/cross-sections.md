# Measure cross sections

{py:obj}`gfem.cross_sections` gives the extinction, absorption and
scattering {term}`cross sections` of the {term}`scatterer` under an
{term}`incident field`: the power removed from the incident field, the power
the scatterer absorbs and the power the {term}`scattered field` carries out,
each divided by the {term}`incident intensity`, in m².

They come from one exact power balance of the solved fields, read on the
{term}`truncation layer` (the physical elements touching the
{term}`perfectly matched layer` or an absorbing face). With $u$ the total
field, $b$ the incident load, $e_s$ the scattered field and $\operatorname{flux}$
the residual form of {py:obj}`gfem.power_flow` out of the elements $S$ inside
that layer:

$$
P_\text{sca} = \operatorname{flux}(e_s), \qquad
P_\text{abs} = \frac{\operatorname{Im}(u_S^H b)}{2\omega\mu_0} - \operatorname{flux}(u),
\qquad P_\text{ext} = P_\text{abs} + P_\text{sca}.
$$

Any set of elements holding the scatterer gives the same three numbers, so
there is no surface to choose, and extinction is absorption plus scattering
exactly. A Poynting integral of point-sampled fields over a sphere does not
have these properties: its two sides differ by percents and it finds
absorption where there is none.

## Cross sections of a sphere

The sphere of {doc}`incident-fields` ($\varepsilon = 4 - i$, radius 0.25 µm,
$ka = \pi/2$) under the same two plane waves, solved at order 3:

```python
import urllib.request

import numpy as np

import gfem

url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/sphere.msh"
)
urllib.request.urlretrieve(url, "sphere.msh")
mesh = gfem.Mesh.from_gmsh("sphere.msh", length_unit="um")

air = gfem.Material()
model = gfem.Model(
    mesh,
    materials={
        "sphere": gfem.Material(eps_r=4.0, tan_delta=0.25),
        "air": air,
        "pml": air,
    },
    pml=gfem.PML("pml"),
    background=air,
)
theta = np.radians(30)
waves = (
    gfem.PlaneWave("te", theta=theta, phi=0.0, polarization="TE"),
    gfem.PlaneWave("tm", theta=theta, phi=0.0, polarization="TM"),
)
f = 299792458.0 / 1e-6
problem = gfem.Driven(f, sources=waves, keep_fields=True)
result = gfem.solve(model, problem, options=gfem.SolverOptions(order=3))

for name in ("te", "tm"):
    xs = gfem.cross_sections(result, name, model=model)
    ext, ab, sca = (
        float(x[0]) * 1e12 for x in (xs.extinction, xs.absorption, xs.scattering)
    )
    print(
        f"{name}: extinction {ext:.4f}, absorption {ab:.4f}, scattering {sca:.4f} um^2"
    )
```

```text
te: extinction 0.6012, absorption 0.2214, scattering 0.3798 um^2
tm: extinction 0.6019, absorption 0.2224, scattering 0.3795 um^2
```

A sphere scatters both polarizations alike: the two columns differ by the
mesh only.

## Compare with Mie

Straight tetrahedra cut the sphere's surface, so compare with the Mie
series at the radius of the sphere's meshed volume, not the nominal one
(against it the faceted sphere is 2-3 % off). The Bohren-Huffman series,
with their $e^{-i\omega t}$ index $m = \sqrt{\varepsilon' + i\varepsilon''}$
for gfem's $\varepsilon = \varepsilon' - i\varepsilon''$:

```python
from scipy.special import spherical_jn, spherical_yn


def mie(eps, x):
    """Extinction, absorption, scattering efficiencies at size parameter x."""
    n = np.arange(1, int(x + 4 * x ** (1 / 3) + 8))
    m = np.sqrt(np.conj(eps))

    def psi(z):
        return z * spherical_jn(n, z)

    def dpsi(z):
        return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)

    h = spherical_jn(n, x) + 1j * spherical_yn(n, x)
    xi = x * h
    dxi = h + x * (spherical_jn(n, x, True) + 1j * spherical_yn(n, x, True))
    a = (m * psi(m * x) * dpsi(x) - psi(x) * dpsi(m * x)) / (
        m * psi(m * x) * dxi - xi * dpsi(m * x)
    )
    b = (psi(m * x) * dpsi(x) - m * psi(x) * dpsi(m * x)) / (
        psi(m * x) * dxi - m * xi * dpsi(m * x)
    )
    q_ext = 2 / x**2 * np.sum((2 * n + 1) * np.real(a + b))
    q_sca = 2 / x**2 * np.sum((2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2))
    return q_ext, q_ext - q_sca, q_sca


p = np.asarray(mesh.nodes)[mesh.tetrahedra[mesh.elements("sphere")]]
volume = (
    abs(
        np.einsum(
            "ij,ij->i",
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
            p[:, 3] - p[:, 0],
        )
    ).sum()
    / 6
)
radius = (3 * volume / (4 * np.pi)) ** (1 / 3)
k = 2 * np.pi / 1e-6
reference = np.array(mie(4 - 1j, k * radius)) * np.pi * radius**2 * 1e12
xs = gfem.cross_sections(result, "te", model=model)
measured = np.array(
    [float(x[0]) * 1e12 for x in (xs.extinction, xs.absorption, xs.scattering)]
)
print(f"Mie at a_vol = {radius * 1e6:.4f} um: {np.round(reference, 4)} um^2")
print(f"error: {np.round(100 * (measured / reference - 1), 2)} %")
```

```text
Mie at a_vol = 0.2410 um: [0.6053 0.2263 0.3789] um^2
error: [-0.67 -2.19  0.23] %
```

The absorption converges slowest: it lives inside the sphere, where the
wavelength is half the vacuum one. Refining the sphere's elements to about
a sixth of that inner wavelength brings each of the three within 0.5 % of
Mie; the rest is set by the exterior mesh and the layer.

## The intensity of each source

The powers are divided by:

- {py:obj}`gfem.PlaneWave`: $|E_0|^2 / (2\eta)$, $\eta$ the wave impedance
  of the background (`xs.intensity` holds it);
- {py:obj}`gfem.IncidentField`: the `intensity=` you pass, W/m², a number or
  one per kept frequency: gfem cannot tell the intensity of a field it only
  evaluates. Passing `intensity=` for a plane wave is refused, so each source
  has one normalization.

## Absorption per region

Cross sections are totals over the scatterer. For one region's absorption,
use {py:obj}`gfem.power_flow` on the source's total field
({doc}`incident-fields`): `-power_flow(result.e_field(f, "te"), "sphere",
model=model)` is the absorbed power in W, `absorption * intensity` when the
region is the whole scatterer.

## Symmetry-reduced models

A model cut by a wall the incident field satisfies (a {py:obj}`gfem.PEC`
plane where its $E$ is normal, a magnetic one where its $H$ is) is a
mirror-symmetric model's half, and {py:obj}`gfem.cross_sections` reports the
meshed half: multiply by $2^n$ for $n$ mirror planes. gfem does not scale
it, since it cannot tell a symmetry plane from a physical wall.

## Gradients

The cross sections are differentiable through the fields: under
{py:obj}`jax.grad`, a {py:obj}`gfem.MaterialField` in the scatterer gets
its gradient, so maximizing absorption is an objective. The measurement
reads only background elements, the load and the incident interpolant,
none of which depends on the design. A design region with an element in the
truncation layer raises {py:obj}`gfem.NonDifferentiableError`; the source,
its intensity and the frequency are static.

## Limits

- **Incident-field columns only**: a port column has no cross sections.
- **Kept fields**: solve with `keep_fields=True` (or the frequencies to
  keep).
- **One intensity per source**: an {py:obj}`gfem.IncidentField` needs
  `intensity=`, a positive number or one per kept frequency; a plane wave
  takes none.
- **A background truncation layer**: a scatterer that reaches the truncation
  layer has no finite cross section and is refused; leave one background
  element between them, or measure per region with
  {py:obj}`gfem.power_flow`.
- **Lossless backgrounds**: in a lossy background the absorption would hold
  the background's own loss of the incident field, so it is refused.
- **Wave ports outside the side**: the incident load next to a wave port
  holds the port's terms, so a side (or the region inside the truncation
  layer) whose elements reach a wave port's face is refused; keep the port at
  least one element outside it, or read the scattered field.
- **Power flow of an incident column**: {py:obj}`gfem.power_flow` refuses the
  incident part, which balances no load, and a side holding layer elements,
  where the kept field is the scattered one.
