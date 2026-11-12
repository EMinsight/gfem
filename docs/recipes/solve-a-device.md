# Solve a device: S-parameters, modes and capacitance

Get the three answers gfem gives for one device: what it transmits between its
ports, where it resonates and how fast each resonance decays, and the
capacitance between its conductors. Each is a problem type solved on the same
{term}`model`: {py:obj}`gfem.Driven` for {term}`S-parameters`,
{py:obj}`gfem.Eigenmode` for modes, {py:obj}`gfem.Electrostatic` for
capacitance.

The device is the line of {doc}`../get-started`: two parallel plates 40 mm
long, 6 mm wide and 1 mm apart, with a ceramic of relative permittivity 10
between them over the middle 20 mm. The solves are full-wave and linear, in
the frequency domain. Every value is in SI units and every phase follows
exp(+iωt); {doc}`../conventions` shows each convention on this same line.

## S-parameters over a band

How much of the power sent into port 1 reaches port 2 between 2 and 8 GHz?
Each end of the line carries a {term}`lumped port` matched to the empty line.

```python
import urllib.request

import numpy as np

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/line.msh"
urllib.request.urlretrieve(url, "line.msh")
mesh = gfem.Mesh.from_gmsh("line.msh", length_unit="mm")

z_line = 376.73 / 6  # ohm: the 6 mm wide, 1 mm high parallel-plate line in air
plates = [gfem.PEC("bottom"), gfem.PEC("top")]
ports = [
    gfem.LumpedPort("p1", "port1", "+z", impedance=z_line),
    gfem.LumpedPort("p2", "port2", "+z", impedance=z_line),
]
ceramic = {"feed": gfem.Material(), "design": gfem.Material(eps_r=10.0)}
model = gfem.Model(mesh, materials=ceramic, boundaries=plates, ports=ports)

result = gfem.solve(model, gfem.Driven(frequency=np.linspace(2e9, 8e9, 7)))
s = result.s
print(result.frequency / 1e9)  # GHz
print(abs(s["p2", "p1"]))  # |S21|
print(s.db("p2", "p1"))  # 20 log10 |S21|, dB
print(s.z_ref[0])  # ohm, the reference impedance of each port at 2 GHz
```

```text
[2. 3. 4. 5. 6. 7. 8.]
[0.8306221  0.68797765 0.64567246 0.90133223 0.57685516 0.97922016
 0.60517486]
[-1.61193031 -3.24851336 -3.79975481 -0.90230198 -4.7786644  -0.18239307
 -4.36238245]
[62.78833333 62.78833333]
```

- `frequency` is in Hz: one value, a sequence or an array. Each frequency is
  one full solve; for hundreds of frequencies see {doc}`frequency-sweep`.
- Every port is excited in turn unless you pass `excite="p1"` (one name or
  several), which solves only those columns of S.
- `result.s["p2", "p1"]` is S21, the wave leaving `"p2"` for a wave sent into
  `"p1"`: a complex array with one entry per frequency. `s.db(to, source)` is
  its magnitude in dB, and `s.values` the whole array, indexed frequency,
  output port, excited port.
- S is in {term}`power waves`, normalized to `s.z_ref`: here each port's own
  resistance, whatever it is. Nothing is referred to 50 Ω unless you ask with
  `s.renormalize(50.0)`.
- The ceramic section is a line of impedance 62.8 Ω / √10, 20 mm long,
  between two matched lines. The textbook formula for it,
  |S21| = 2 / |2 cos θ + i (Z₁/Z₂ + Z₂/Z₁) sin θ| with θ = 2π f √10 · 20 mm / c,
  agrees with every printed value within 0.00002.

## Walls and metals

A surface of the mesh that the model does not name is a magnetic wall: the
tangential magnetic field vanishes on it, as on a symmetry plane. That is what
closes the sides of this line, so the empty line carries a plane wave between
its plates. {py:obj}`gfem.PMC` says the same thing in writing. With both
volumes filled with air the line is matched end to end, and S21 is the phase
of 40 mm of free space, exp(−i k L):

```python
air = {"feed": gfem.Material(), "design": gfem.Material()}
at_5ghz = gfem.Driven(frequency=5e9, excite="p1")

walls = [*plates, gfem.PMC("side")]
unnamed = gfem.Model(mesh, materials=air, boundaries=plates, ports=ports)
named = gfem.Model(mesh, materials=air, boundaries=walls, ports=ports)
for empty_line in (unnamed, named):
    s21 = gfem.solve(empty_line, at_5ghz).s["p2", "p1"][0]
    print(abs(s21), np.angle(s21))  # |S21| and its phase, rad

k = 2 * np.pi * 5e9 / 299792458.0  # rad/m
print(np.angle(np.exp(-1j * k * 40e-3)))  # the phase of exp(-i k L), rad
```

```text
0.9999999272390561 2.091495024608073
0.9999999272390561 2.091495024608073
2.0914952632762236
```

The plates above are perfect conductors ({py:obj}`gfem.PEC`: the tangential
electric field vanishes). A real metal is a {py:obj}`gfem.Conductor` with its
conductivity in S/m, and an optional thickness. It absorbs: with a copper top
plate, how much power is left after 40 mm?

```python
copper = [gfem.PEC("bottom"), gfem.Conductor("top", 5.8e7)]
lossy_line = gfem.Model(mesh, materials=air, boundaries=copper, ports=ports)
s21 = gfem.solve(lossy_line, at_5ghz).s["p2", "p1"][0]
print(abs(s21) ** 2)  # fraction of the incident power that reaches port 2

r_s = np.sqrt(np.pi * 5e9 * 4e-7 * np.pi / 5.8e7)  # ohm: copper's surface resistance
alpha = r_s / (2 * 376.73 * 1e-3)  # Np/m: one lossy plate of a 1 mm high line
print(np.exp(-2 * alpha * 40e-3))  # the same fraction from the attenuation formula
```

```text
0.9980422163744983
0.9980431596715874
```

The other boundaries follow the same pattern, a class and a surface name:

- {py:obj}`gfem.Impedance` is a sheet given by its resistance, inductance and
  capacitance per square: a resistive film, a kinetic inductance.
- {py:obj}`gfem.Absorbing` lets waves leave through a surface of the mesh that
  stands for open space.
- {py:obj}`gfem.Periodic` pairs two opposite surfaces of a repeating cell
  ({doc}`periodic-cell`).

A surface given two conditions, or a name the mesh does not have, raises when
the model is built, before anything is solved.

## Ports

A port is where power enters and where S is measured. gfem has two kinds.

- A {term}`lumped port` ({py:obj}`gfem.LumpedPort`) is a resistor across a
  face that spans from one conductor to the other, with a direction: the one
  the port voltage is counted along. Its field is uniform across the face. Use
  it where the face is small against the wavelength and the line under it is
  simple: a parallel-plate or microstrip feed, the gaps of a coplanar line
  (two faces with opposite directions), a junction. Without a value the
  resistance is 50 Ω; `inductance=` and `capacitance=` add a parallel L and C.
- A {term}`wave port` ({py:obj}`gfem.WavePort`) feeds a mode of the port's
  cross-section, which gfem solves first on the port's face
  (`gfem.WavePort("p1", "port1")`). Use it where the field of the guide
  matters: a dielectric waveguide, a hollow guide, a line whose impedance you
  do not know. The face must be an exterior surface of the mesh. `mode=2`
  asks for the next mode, and several wave ports on one face each take a
  different mode ({doc}`shared-port-surface`).

The order of `ports` in the model is the order of the rows and columns of
`s.values`; by name, as above, the order does not matter.

## Resonances and Q

Take the ports away and both ends of the line are open (magnetic walls): it
becomes a resonator. Where are its first three resonances, and what Q does
the ceramic's {term}`loss tangent` of 1e-3 leave them?

```python
lossy = {"feed": gfem.Material(), "design": gfem.Material(eps_r=10.0, tan_delta=1e-3)}
resonator = gfem.Model(mesh, materials=lossy, boundaries=plates)

modes = gfem.solve(resonator, gfem.Eigenmode(target=3e9, n_modes=3))
print(modes.frequency / 1e9)  # GHz
print(modes.q)
p = modes.bulk_participation["design"]
print(p)  # share of each mode's electric energy in the ceramic
print(1 / (1e-3 * p))  # the Q that this share and the loss tangent give
print(modes.error.max() < 1e-9)
```

```text
[2.14197116 4.19062391 5.96307245]
[1251.68048682 1379.50404589 1748.29991075]
[0.79892612 0.72489834 0.57198431]
[1251.68018722 1379.50377406 1748.29969626]
True
```

- `target` is the frequency, in Hz, to search around, and `n_modes` how many
  modes to return: the nearest ones, nearest first. Aim at the band you care
  about, not at zero.
- `modes.frequency` is in Hz and `modes.q` is the quality factor; both come
  from the complex `modes.omega` (rad/s).
- `modes.bulk_participation` gives, per volume, the share of each mode's
  electric energy stored in it. Each lossy volume adds its share times its
  loss tangent to 1/Q, which is the fourth printed line.
- `modes.error` is each mode's residual, relative to the size of the
  equations: the modes are solved to `SolverOptions(eigen_tolerance=...)`.
- By hand: with a = 10 mm, n = √10 and k = 2π f / c, the open-ended line
  resonates where tan(k a) tan(n k a) = n (modes 1 and 3) or
  tan(k a) = −n tan(n k a) (mode 2). The roots are within 5 kHz of the
  printed frequencies.

## Coupling Q of a port

A resistive port loads a resonance. How strongly do two 10 kΩ ports, far above
the line's impedance, couple to the first two modes of the lossless line?

```python
weak_ports = [
    gfem.LumpedPort("p1", "port1", "+z", impedance=1e4),
    gfem.LumpedPort("p2", "port2", "+z", impedance=1e4),
]
coupled = gfem.Model(mesh, materials=ceramic, boundaries=plates, ports=weak_ports)

modes = gfem.solve(coupled, gfem.Eigenmode(target=3e9, n_modes=2))
print(modes.q_port["p1"], modes.q_port["p2"])
print(modes.q)
print(1 / (1 / modes.q_port["p1"] + 1 / modes.q_port["p2"]))
```

```text
[665.27717644 792.96160529] [665.27717644 792.96160529]
[332.63896398 396.48111753]
[332.63858822 396.48080264]
```

`modes.q_port` holds one coupling Q per resistive lumped port, from the power
its resistor draws out of the mode. The device is symmetric, so both ports
couple equally. Nothing else is lossy here, so the total Q is the two in
parallel, half of each, which the last two lines compare.

## Capacitance

What is the capacitance between the plates? An electrostatic problem takes
no ports and no walls: it holds each terminal at a potential and grounds the
rest.

```python
plates_only = gfem.Model(mesh, materials=ceramic)
problem = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")
static = gfem.solve(plates_only, problem)
print(static.capacitance[0, 0] * 1e12)  # pF

eps_0 = 8.8541878128e-12  # F/m
print(eps_0 * (6e-3 / 1e-3) * (20e-3 + 10.0 * 20e-3) * 1e12)  # two plate capacitors, pF
```

```text
11.6875279208162
11.687527912896002
```

- `terminals` maps a name of your choice to the surface, or surfaces, of one
  conductor. `ground` names the conductors held at 0 V.
- `static.capacitance` is the Maxwell capacitance matrix in F, one row and
  column per terminal in the order of `static.terminals`.
- The hand value is ε₀ (w / h) (L_air + 10 L_ceramic): the air and ceramic
  parts are plate capacitors in parallel, and the magnetic walls on the sides
  leave no fringing field.

## Options

The numerical settings are apart from the model and the problem, in
{py:obj}`gfem.SolverOptions`, passed as `gfem.solve(model, problem, options=options)`:

- `order` is the {term}`element order`, 2 by default. Order 3 is more
  accurate on the same mesh and costs more time and memory.
- `linear_solver` is `"auto"` by default: an iterative solver for driven and
  electrostatic problems, a direct one for eigenmodes. `"direct"` suits small
  and medium models on one machine.
- `tolerance` is where the iterative solver stops (1e-6, relative to the
  right-hand side). It is not the error of S.
- `gfem.SolverOptions.preset("fast")` and `preset("accurate")` set these in
  one word.

## Check your results

Three checks need no reference. A lossless reciprocal device conserves power,
|S11|² + |S21|² = 1, and S is symmetric. And a result that the mesh resolves
does not move when the {term}`element order` goes up:

```python
s11, s21 = s["p1", "p1"], s["p2", "p1"]
print(abs(s11) ** 2 + abs(s21) ** 2)  # 1 for a lossless device
print(abs(s["p1", "p2"] - s21).max() < 1e-10)  # reciprocity

order_3 = gfem.SolverOptions(order=3)
finer = gfem.solve(model, at_5ghz, options=order_3)
print(abs(s21[3]), abs(finer.s["p2", "p1"][0]))  # |S21| at 5 GHz, orders 2 and 3
```

```text
[0.99999941 0.99999999 1.00000047 0.99999978 1.00000003 1.00000028
 1.        ]
True
0.9013322305443373 0.9013341565407679
```

- Here the power balance holds to better than 1e-6. A larger gap in a model
  you believe lossless points at a boundary that absorbs, or at a port that
  is not the one you meant.
- The two orders differ in the sixth digit: this mesh resolves the line.
  When they differ by more than you can accept, refine the mesh where the
  error is: {doc}`mesh-refinement`.
- For a resonance, solve again with another `target` nearby and a larger
  `n_modes`: a mode that belongs to the device comes back at the same
  frequency.

## Limits

- Below the low-frequency floor a driven solve loses digits and warns with
  {py:obj}`gfem.LowFrequencyWarning`; for the capacitance of a device far
  smaller than the wavelength, solve {py:obj}`gfem.Electrostatic`. See
  "Limits" in {doc}`../capabilities`.
- The mesh is straight-sided tetrahedra read from gmsh. A curved surface is
  as round as its facets, and raising the element order does not round it.
- Materials are linear and time-invariant; there is no time-domain solve and
  no magnetostatic one. {doc}`../capabilities` lists what gfem solves and what
  it refuses.
- An eigenmode solve returns the `n_modes` modes nearest `target`, not every
  mode of a band: ask for more modes than you expect and check that the
  farthest one lies outside the band. A model with a wave port has no
  eigenmode solve; it raises.

## Next

- {doc}`frequency-sweep`: hundreds of frequencies from a few solves.
- {doc}`measurements`: field values at points, the power through a surface,
  surface-loss Q.
- {doc}`prepare-a-mesh`: build the mesh of your own device in gmsh.
- {doc}`mesh-refinement`: refine the mesh where the error is.
- {doc}`s-parameter-gradient`: the gradient of any of these results with
  respect to the materials, at the cost of one {term}`adjoint solve`.
- {doc}`../conventions`: the signs and normalizations behind every number
  above.
