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

## Eigenmode fields

A mode has no source to set its amplitude, so gfem scales every mode field to
store `MODE_ENERGY`, 1 J, of electric energy: ½ Re ∫ D\* · E dV, which for a
lossless mode is its total energy. The time-averaged electric energy
`gfem.electric_energy` measures is half of that. Participation ratios are
shares of the same energy, so over all the volumes they add up to 1:

```python
energy = gfem.electric_energy(modes.e_field(0), model=resonator)  # J, per volume
print(2 * sum(energy.values()))  # J
print(sum(modes.bulk_participation.values())[0])
```

```text
1.0
1.0000000000000002
```

## Periodic cells

Across a periodic pair of translation a, a Bloch wave vector k sets
`BLOCH_PHASE`: `E(r + a) = exp(-i k·a) E(r)`, so the phase of a Bloch wave
travels along +k. {doc}`recipes/periodic-cell` shows it on a cell.

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
