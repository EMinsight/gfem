# Conventions

Which sign, unit and normalization is behind a number gfem returns? The full
statement is the {py:obj}`gfem.conventions` module, the one place where each
convention is written out. This page shows each of them on a solve, so you
can tell which one you are looking at, and names the constant that records
it. The device is the parallel-plate line of {doc}`get-started`.

| Constant | Value | What it fixes |
| --- | --- | --- |
| `gfem.conventions.TIME_DEPENDENCE` | `exp(+iωt)` | the time dependence of every complex phasor |
| `gfem.conventions.PHASOR` | `peak` | a phasor's modulus is the amplitude of the oscillation, not its RMS value |
| `gfem.conventions.WAVES` | `power` | S relates power waves |
| `gfem.conventions.INCIDENT_POWER` | 1 W | the incident power behind every driven field |
| `gfem.conventions.MODE_ENERGY` | 1 J | the electric energy every mode field stores |
| `gfem.conventions.BLOCH_PHASE` | `E(r + a) = exp(-i k·a) E(r)` | the phase across a periodic pair |
| `gfem.conventions.INTERIOR_SHEET_ADMITTANCE` | 0.5 | a thin `Conductor` film on an interior surface, in units of σt |
| `gfem.conventions.EXTERIOR_THICKNESS_FACTOR` | 2 | how often a `Conductor`'s thickness counts on an exterior surface |

## Units

Everything you pass to gfem and everything it returns is in SI units: metres,
hertz, ohms, farads, watts, V/m. The one exception is the mesh file, whose
length unit you state once when you read it; gfem converts the coordinates.

```python
import urllib.request

import numpy as np

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/line.msh"
urllib.request.urlretrieve(url, "line.msh")
mesh = gfem.Mesh.from_gmsh("line.msh", length_unit="mm")
print(mesh.length_unit)  # the unit of the file
print(mesh.imported_nodes.max(axis=0))  # the far corner of the line, m
```

```text
mm
[0.04  0.006 0.001]
```

## Time dependence

The physical field is Re(E exp(+iωt)), the electrical engineering convention
(`TIME_DEPENDENCE`): write j for i and the formulas of a circuit textbook
apply. A wave travelling along +x is then exp(−iβx), so the phase of a
transmission falls with distance, and with frequency. On the line with its
ceramic section, from 2 to 8 GHz:

```python
z_line = 376.73 / 6  # ohm
plates = [gfem.PEC("bottom"), gfem.PEC("top")]
ports = [
    gfem.LumpedPort("p1", "port1", "+z", impedance=z_line),
    gfem.LumpedPort("p2", "port2", "+z", impedance=z_line),
]
ceramic = {"feed": gfem.Material(), "design": gfem.Material(eps_r=10.0)}
model = gfem.Model(mesh, materials=ceramic, boundaries=plates, ports=ports)

band = gfem.Driven(frequency=np.linspace(2e9, 8e9, 7), excite="p1", keep_fields=5e9)
result = gfem.solve(model, band)
print(np.unwrap(np.angle(result.s["p2", "p1"])))  # the phase of S21, rad
```

```text
[ 3.05180459  0.79326538 -0.47316251 -2.65371362 -4.14297302 -5.8252978
 -7.83283125]
```

A code that uses exp(−iωt) gives the complex conjugate of every S entry and
every field.

## S-parameters and power

S relates {term}`power waves` (`WAVES`), normalized to each port's
{term}`reference impedance`: abs(a)² is the power a wave carries, in W.
Phasors are peak values (`PHASOR`), so a time-averaged power carries a
factor ½, as in abs(V)² / (2R). A kept driven field is the field for
`INCIDENT_POWER`, 1 W, sent into the excited port, with every other port
matched. The power that field carries through the feed line should then be
what port 1 does not reflect, 1 − abs(S11)² W:

```python
field = result.e_field(5e9, "p1")
through = gfem.power_flow(field, lambda c: c[:, 0] < 5e-3, model=model)
print(through)  # W crossing x = 5 mm towards the ceramic
print(1 - abs(result.s["p1", "p1"][3]) ** 2)  # what port 1 does not reflect at 5 GHz
```

```text
0.8123995326128145
0.8124000081969966
```

## Sign of loss

Under exp(+iωt) a lossy permittivity has a negative imaginary part,
ε = ε′ (1 − i tan δ), and a mode that decays has Im ω > 0. Its quality factor
is abs(ω) / (2 abs(Im ω)). With the ports removed and a
{term}`loss tangent` of 1e-3 in the ceramic, the line's first resonance:

```python
lossy = {"feed": gfem.Material(), "design": gfem.Material(eps_r=10.0, tan_delta=1e-3)}
resonator = gfem.Model(mesh, materials=lossy, boundaries=plates)
modes = gfem.solve(resonator, gfem.Eigenmode(target=2e9))
omega = modes.omega[0]  # rad/s, complex
print(omega.real / 2e9 / np.pi, omega.imag > 0)  # GHz, and whether it decays
print(modes.q[0], abs(omega) / (2 * omega.imag))
```

```text
2.1419711636402075 True
1251.6804868146692 1251.6804868146692
```

## Dispersive materials

A {term}`permittivity law` follows the same sign of loss. Its parameters are
ordinary frequencies in hertz (ω = 2π f), and under exp(+iωt) its terms are:

| Term | Adds to ε∞ |
| --- | --- |
| {py:obj}`gfem.Drude` | −ωp² / (ω (ω − iγ)) |
| {py:obj}`gfem.Lorentz` | Δε ω0² / (ω0² − ω² + iγω) |
| {py:obj}`gfem.Debye` | Δε / (1 + i f / f_r) |
| {py:obj}`gfem.DjordjevicSarkar` | Δε ln((f2 + i f) / (f1 + i f)) / ln(f2 / f1) |

Here ωp, ω0 and γ are 2π times the term's plasma, resonance and damping
frequencies, and f_r, f1 and f2 are its relaxation and corner frequencies.
With every strength positive and every damping at least 0, each term is
passive alone, so Im ε ≤ 0 at every real frequency. A Lorentz medium just
below its resonance:

```python
law = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=10e9, damping=1e9)],
)
eps = complex(law(9e9))
print(eps.real, eps.imag)  # at 9 GHz: Im eps < 0, a loss
print(law(0.0).real)  # the static permittivity, eps_inf + delta_eps
```

```text
6.298642533936651 -2.0361990950226243
3.0
```

Most optics papers write their fits under exp(−iωt). There the same law is
the complex conjugate: keep the parameter values, and read their −iγ as +iγ.
`law(0)` is the {term}`static permittivity`, which the electrostatic problem
reads. A mode's energy weighs a law region with its
{term}`energy permittivity` instead of Re ε (see below). How to write a law:
{doc}`recipes/dispersive-materials`.

## Eigenmode fields

A mode has no source to set its amplitude, so gfem scales every mode field to
store `MODE_ENERGY`, 1 J, of electric energy: ½ Re ∫ D\* · E dV, plus
C|V|²/2 in each lumped port with a capacitance C, plus ½ ∫ C_s |E_t|² dS on
each {py:obj}`gfem.Impedance` sheet with a capacitance C_s per square. For a
lossless mode this is its total energy. In a region with a permittivity
law, the volume term weighs |E|² with the energy permittivity at the mode's
real frequency, so it stays the total energy of a lossless mode. Elsewhere,
the time-averaged electric energy `gfem.electric_energy` measures is half of
the volume term. Participation
ratios are shares of the same energy. Over all the volumes, the capacitive
ports and the capacitive sheets they add up to 1, and over the volumes alone
when the model has neither, as here:

```python
energy = gfem.electric_energy(modes.e_field(0), model=resonator)  # J, per volume
print(2 * sum(energy.values()))  # J
print(sum(modes.bulk_participation.values())[0])
```

```text
1.0
1.0000000000000002
```

The inductive participations of a lumped port with an inductance L and of an
inductive sheet are shares of the same energy too. For a lossless mode, a
sheet's `sheet_participation` is −2 d ln ω / d ln L_s on any mesh. A port's
`participation` is −2 d ln ω / d ln L when the field is uniform across the
port; otherwise it is read from the port's mean voltage and differs slightly.

## Periodic cells

Across a periodic pair of translation a, a Bloch wave vector k sets
`BLOCH_PHASE`: `E(r + a) = exp(-i k·a) E(r)`, so the phase of a Bloch wave
travels along +k. {doc}`recipes/periodic-cell` shows it on a cell, and
"Floquet ports" below gives the conventions of a cell lit by a plane wave.

## Floquet ports

A {py:obj}`gfem.FloquetPort` is one diffraction order in one polarization of
a plane wave on the open face of a periodic cell
({doc}`recipes/floquet-ports`). The cell here is one period of the
{doc}`metagrating tutorial <tutorials/metagrating>`'s grating with its
design layers left as air: a flat interface between glass (n = 1.444) below
and air above, 2.023 µm along x and 0.08 µm along y.

**The lattice frame.** The two pairs that rim a face, in `boundaries` order,
have translations a1 and a2. The face's frame is ê1 = a1 / |a1|, ê3 along
a1 × a2 and ê2 = ê3 × ê1; here ê1 = +x, ê2 = +y, ê3 = +z on both faces.
Order (m, n) has the in-plane wave vector κ = k_t + m b1 + n b2, with
b_i · a_j = 2π δ_ij and k_t the problem's `wave_vector` (zero at normal
incidence). An order propagates in the medium behind the face when
|κ| < k0 n; at 1.55 µm the orders −1, 0 and +1 do, at 3.1 µm only the
specular one:

```python
c = 299_792_458.0  # m/s
url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/metagrating.msh"
urllib.request.urlretrieve(url, "metagrating.msh")
cell_mesh = gfem.Mesh.from_gmsh("metagrating.msh", length_unit="um")
n_glass = 1.444
air = gfem.Material()
glass = gfem.Material(eps_r=n_glass**2)
flat = {"glass": glass, "lower": air, "upper": air, "air": air}
pairs = [gfem.Periodic(("x_min", "x_max")), gfem.Periodic(("y_min", "y_max"))]
cell = gfem.Model(cell_mesh, materials=flat, boundaries=pairs)
print(np.asarray(cell.periodic_translations))  # a1, a2, m
print(cell.propagating_orders("top", c / 1.55e-6))
print(cell.propagating_orders("top", c / 3.1e-6))
```

```text
[[2.0233813e-06 0.0000000e+00 0.0000000e+00]
 [0.0000000e+00 8.0000000e-08 0.0000000e+00]]
((0, 0), (-1, 0), (1, 0))
((0, 0),)
```

The TE reference of an order is ê3 × κ̂ (E normal to its plane of
incidence, s) and the TM reference is κ̂ (E in it, p), both tangential to
the face. At κ = 0, κ̂ is ê1: at normal incidence TE is E along ê2 (+y here)
and TM is E along ê1 (+x).

**What S is.** A Floquet entry compares the tangential E of two orders at
their face planes, scaled to power: S_{p←q} = (c_p / c_q) √(Z_q / Z_p), with
c the order's tangential-E coefficient and Z its wave impedance in the medium
behind its own face, η / cos θ for TE and η cos θ for TM. |S|² is therefore
the {term}`diffraction efficiency`, and for a lossless cell the efficiencies
of every propagating order on both faces, in both polarizations, add up to 1.
The reflection of a flat interface is the tangential-E coefficient
(n1 − n2) / (n1 + n2) in both polarizations at normal incidence. The
Fresnel r_p of many optics texts has the opposite sign. With the ports'
`offset` moving their reference planes to the interface, at 3.1 µm:

```python
f = c / 3.1e-6  # Hz: only the specular order propagates
ports = [
    gfem.FloquetPort(f"glass {pol}", "bottom", polarization=pol, offset=0.5e-6)
    for pol in ("TE", "TM")
] + [
    gfem.FloquetPort(f"air {pol}", "top", polarization=pol, offset=1.2e-6)
    for pol in ("TE", "TM")
]
ports.append(gfem.FloquetPort("air -1", "top", order=(-1, 0), polarization="TE"))
interface = cell.replace(ports=ports)
direct = gfem.SolverOptions(linear_solver="direct")
lit = gfem.Driven(f, excite=["glass TE", "glass TM"])
floquet = gfem.solve(interface, lit, options=direct)
s = floquet.s
print(complex(s["glass TE", "glass TE"][0]), complex(s["glass TM", "glass TM"][0]))
print((n_glass - 1) / (n_glass + 1))  # (n1 - n2) / (n1 + n2), from the glass
print(abs(s["glass TE", "glass TE"][0]) ** 2 + abs(s["air TE", "glass TE"][0]) ** 2)
```

```text
(0.18167034107937008+2.0236696358788642e-07j) (0.1816703863297391+2.0964626335623908e-07j)
0.18166939443535188
0.9999999997457327
```

**Rows that do not propagate.** The (−1, 0) order is evanescent at 3.1 µm. Its
row of S reads exactly 0 at every frequency where it does not propagate, and
its reference impedance is NaN there. `diffraction_orders` gives its κ, here
−b1 along −x, and whether it propagates. The reference impedances of the
propagating rows are η / n of their medium (TE and TM are equal at normal
incidence):

```python
minus_one = floquet.diffraction_orders["air -1"]
print(np.asarray(minus_one.wave_vector[0]), bool(minus_one.propagating[0]))
print(complex(s["air -1", "glass TE"][0]))
at = {name: i for i, name in enumerate(s.ports)}
print(np.asarray(s.z_ref)[0, [at["glass TE"], at["air TE"], at["air -1"]]])  # ohm
```

```text
[-3105289.79977531        0.                0.        ] False
0j
[260.89356884 376.73031341          nan]
```

**Phase.** Under exp(+iωt) a plane wave crossing an empty cell of thickness L
comes out as exp(−iβL), a delay. With the glass made air and no offsets, the
1.7 µm cell:

```python
empty = cell.with_material("glass", air).replace(
    ports=[
        gfem.FloquetPort("down", "top", polarization="TE"),
        gfem.FloquetPort("out", "bottom", polarization="TE"),
    ]
)
s_empty = gfem.solve(empty, gfem.Driven(f, excite="down"), options=direct).s
print(complex(s_empty["out", "down"][0]))
print(np.exp(-1j * 2 * np.pi * f / c * 1.7e-6))  # exp(-i beta L)
```

```text
(-0.9541393055579418+0.2993629662200534j)
(-0.954139256400049+0.2993631229733574j)
```

The in-plane phase of every order is referred to the model origin.
Translating the whole model in-plane by r0 multiplies S_{p←q} by
exp(−i (κ_p − κ_q) · r0): entries between two specular orders do not change,
entries between different orders do.

**Bloch reciprocity.** A reversed wave has the opposite in-plane wave vector,
so order (m, n) maps to (−m, −n), and the TE and TM references, odd in κ̂,
flip sign with it. With σ = −1 for an order with κ ≠ 0 and +1 for κ = 0,
{term}`Bloch reciprocity` reads S_{p←q}(k) = σ_p σ_q S_{−q←−p}(−k), where
−p is port p's face and polarization with the opposite order. The plain
transpose never holds as it stands; the
{doc}`metagrating tutorial <tutorials/metagrating>` checks the signed form
on its design.

**Circular polarization.** No port is circular: combine columns. With the TE
and TM ports of the same order on the top face (incidence along −ê3),
(S[:, TM] + i S[:, TE]) / √2 is the response to right-handed circular
incidence in the IEEE sense (the field turns counterclockwise seen with the
wave coming towards you). From the bottom face, incidence along +ê3, the same
combination is left-handed, and (S[:, TM] − i S[:, TE]) / √2 is right-handed.

## Thin metals

A {py:obj}`gfem.Conductor` with a thickness describes one face of a film. On
a surface inside the mesh a film much thinner than its skin depth is
therefore a sheet of admittance `INTERIOR_SHEET_ADMITTANCE` times σt, half
the film's conductance: give `thickness=2 * t` for the whole film. On the
outer surface of the mesh the thickness already counts
`EXTERIOR_THICKNESS_FACTOR` times, and the film is σt. The module's "Surface
conductors" section has the formula.

## Next

- {py:obj}`gfem.conventions`: every convention in full, with the formulas
  for power waves, wave ports and energy densities.
- {doc}`recipes/solve-a-device`: the solves this page reads its numbers from.
- {doc}`recipes/measurements`: power through a surface and energy in a region.
