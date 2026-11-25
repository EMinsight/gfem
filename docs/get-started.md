
# Get started

gfem computes S-parameters, eigenmodes and capacitances of 3D devices meshed with
[gmsh](https://gmsh.info), and their gradients with respect to materials and shape.
This page installs gfem, runs a first solve, differentiates it and points you to
the next pages.

## Install

{{ install }}

gfem needs Python >= {{ python }} on Linux x86_64 (glibc >= 2.28), macOS arm64 or
Windows x86_64. The plain install solves on the CPU.

gfem is {{ status }}, and its API may change before 1.0. To keep a project
working, pin a release (`pip install "gfem==0.2.0"`); the
{doc}`project/release-notes` list the releases.

## A first solve

A parallel-plate line 40 mm long, 6 mm wide and 1 mm high joins two lumped ports
matched to it. Its middle 20 mm is filled with a ceramic of relative permittivity 10.
How much of the incident power reaches port 2 at 2, 5 and 8 GHz? The snippet
downloads the mesh, which the {doc}`topology tutorial <tutorials/topology>` also uses.

```python
import urllib.request

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/line.msh"
urllib.request.urlretrieve(url, "line.msh")
mesh = gfem.Mesh.from_gmsh("line.msh", length_unit="mm")

z_line = 376.73 / 6  # ohm: the 6 mm wide, 1 mm high parallel-plate line in air
model = gfem.Model(
    mesh,
    materials={"feed": gfem.Material(), "design": gfem.Material(eps_r=10.0)},
    boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
    ports=[
        gfem.LumpedPort("p1", "port1", "+z", impedance=z_line),
        gfem.LumpedPort("p2", "port2", "+z", impedance=z_line),
    ],
)
result = gfem.solve(model, gfem.Driven(frequency=(2e9, 5e9, 8e9), excite="p1"))
print(abs(result.s["p2", "p1"]))  # |S21| at 2, 5 and 8 GHz
```

```text
[0.83062238 0.90133217 0.60517471]
```

What each part does:

- {py:obj}`gfem.Mesh.from_gmsh` reads the mesh. Its node coordinates are in mm, so
  `length_unit="mm"`; everything you pass to gfem is in SI units.
- {py:obj}`gfem.Model` refers to the mesh's named regions. The volumes `"feed"`
  (air, both ends) and `"design"` (the middle 20 mm) take a
  {py:obj}`gfem.Material`. The plates `"bottom"` and `"top"` are perfect conductors
  ({py:obj}`gfem.PEC`). The surfaces left unassigned, here the side walls, are
  magnetic walls, so the empty line carries a TEM wave.
- Each {py:obj}`gfem.LumpedPort` is a resistor across a port face (`"port1"` at
  $x = 0$, `"port2"` at $x = 40$ mm), driven along $+z$. Its impedance is the empty
  line's, $\eta_0 h / w \approx 62.8$ Ω.
- {py:obj}`gfem.Driven` asks for a frequency-domain solve at three frequencies in Hz,
  with port `"p1"` excited. `result.s` holds the S-parameters as waves
  normalized to the port impedances; see {py:obj}`gfem.conventions`.

You can check the numbers by hand. The ceramic section is a line of impedance
$62.8\ \Omega / \sqrt{10}$ and length 20 mm, and the textbook formula for a section between two
matched lines gives $|S_{21}|$ = 0.831, 0.901 and 0.605.

## Differentiate it

`gfem.solve` is a JAX function of the model's materials and mesh. Here `jax.grad`
gives the derivative of the transmitted power $|S_{21}|^2$ at 5 GHz with respect to the
ceramic's permittivity, with one extra linear solve:

```python
import jax

at_5ghz = gfem.Driven(frequency=5e9, excite="p1")


def transmission(ceramic):
    """|S21|^2 at 5 GHz with this ceramic in the middle section."""
    s21 = gfem.solve(model.with_material("design", ceramic), at_5ghz).s["p2", "p1"]
    return abs(s21[0]) ** 2


ceramic = gfem.MaterialField.uniform("design", gfem.Material(eps_r=10.0))
power, grad = jax.value_and_grad(transmission)(ceramic)
print(power, grad.eps_r)  # |S21|^2 and its derivative with respect to eps_r
```

```text
0.8123998572975817 -0.30017734440224814
```

A {py:obj}`gfem.Material` holds constants: give it a value JAX is differentiating
and it raises {py:obj}`gfem.NonDifferentiableError`. Wrap the material in
{py:obj}`gfem.MaterialField.uniform`, as above, to differentiate with respect to it.

## Your own device

gfem reads gmsh `.msh` files whose volumes and surfaces are named physical
groups. {doc}`recipes/prepare-a-mesh` builds this line in gmsh, and
{doc}`recipes/gsim` starts from a GDS layout.

## Optional extras

An extra goes in brackets after `gfem` in the same install line:

{{ install_extras }}

| Extra | What it adds | Platforms |
| --- | --- | --- |
| `mkl` | A faster CPU direct solver (Intel MKL PARDISO) for `linear_solver="direct"`, with the same results, except for an ungauged solve (such as `gauge="none"`) far below the low-frequency floor ({py:obj}`gfem.Driven`, Notes). Intel's licence forbids hosted (SaaS) use: read {doc}`project/licence` before you install it. | Linux x86_64, Windows x86_64 |
| `cuda12` | The direct solver on an NVIDIA GPU (cuDSS); see {doc}`recipes/gpu-direct-solver`. Use `cuda13` instead for CUDA 13; install one of the two, not both. | Linux x86_64, Windows x86_64 |
| `xarray` | Results as xarray datasets (`.to_xarray()`). | all |
| `viz` | Fields as PyVista grids (`.to_pyvista()`). | all |
| `mpi` | One driven, electrostatic or eigenmode solve across the ranks of an MPI job; Open MPI comes from your machine. See {doc}`recipes/distributed-solve`. | Linux x86_64 |
| `log` | Coloured log lines and progress rows while a solve runs; see {doc}`recipes/watch-a-solve`. | all |
| `sax` | SAX, to place solves in SAX circuits ({py:obj}`gfem.SaxComponent`); gfem builds the S dictionary without it. | all |
| `skrf` | scikit-rf, to export S as a scikit-rf network (`.to_skrf()`). | all |

On a platform outside its list, an extra installs nothing and gfem solves with
the plain install's CPU solvers.

PETSc and SLEPc solvers are not an extra: they need a source build, which
{doc}`recipes/petsc-slepc` does with one script.

Importing gfem turns on JAX double precision (`jax_enable_x64`) for your whole
program, because every gfem solve needs it.

## Next steps

- The {doc}`topology tutorial <tutorials/topology>` turns this same line, seen
  from above as a 2D model, into a band-stop filter by optimizing the ceramic
  pattern, and the
  {doc}`resistive film <tutorials/sheet>` tutorial lays out a 300 Ω film to make
  a matched 3 dB attenuator. The
  {doc}`shape <tutorials/shape>` tutorial moves a coplanar line's conductor walls
  to match it to 50 Ω, the {doc}`transmon <tutorials/transmon>` tutorial
  takes the gradient of a superconducting qubit's dispersive shift with respect
  to its pad gap, the {doc}`bend <tutorials/bend>` tutorial designs a
  photonic waveguide bend on a 2D model, and the
  {doc}`3D bend <tutorials/bend3d>` tutorial designs it again on the real stack,
  a 220 nm silicon guide in oxide. The {doc}`metal bend <tutorials/metal-bend>`
  tutorial places gold in the same bend, over a band of wavelengths, and the
  {doc}`metagrating <tutorials/metagrating>` tutorial designs a silicon
  grating on glass that sends a plane wave into the air at 50°. The
  {doc}`photonic crystal <tutorials/photonic-crystal>` tutorial computes the
  band diagram of a square lattice of alumina rods: its band gap, how fast
  light crosses it, and where more permittivity would widen the gap. The
  {doc}`metalens <tutorials/metalens>` tutorial tabulates how a silicon nitride
  ridge transmits light against its width and builds from that table a lens
  that focuses green and red light on one line. The
  {doc}`nanosphere <tutorials/nanosphere>` tutorial puts a point dipole
  beside a silicon and a gold sphere: its Purcell factor, the share of its
  power the gold quenches, and where its light goes. The
  {doc}`meta-atom <tutorials/meta-atom>` tutorial places a high-index
  material in a small cube to scatter the strongest magnetic dipole. The
  {doc}`grating coupler <tutorials/grating>` tutorial shapes the trenches of
  a silicon grating so that more of a guide's light reaches a fiber above it.
  The
  {doc}`heater <tutorials/heater>` tutorial heats a silicon waveguide with a
  TiN strip: the power for a $\pi$ phase shift, the crosstalk to a neighbouring
  guide, and how fast the phase follows the current. The
  {doc}`3D heater <tutorials/heater3d>` tutorial heats a whole waveguide spiral
  in 3D and shows how its phase is shared between the passes, and
  {doc}`recipes/electromagnetic-heating` adds heat from a driven solve's
  electromagnetic loss. The
  {doc}`modulator <tutorials/modulator>` tutorial computes the voltage a
  lithium niobate push-pull modulator needs for a $\pi$ phase shift, and
  co-designs its electrode gap and ridge width. The
  {doc}`transducer <tutorials/transducer>` tutorial launches the shear plate
  wave of a thin lithium niobate film with interdigital fingers: the admittance
  they present, the strain that reaches two waveguide arms and in which phase. The
  {doc}`acousto-optic <tutorials/acousto-optic>` tutorial adds the light: the index
  shift of the two arms per volt and the push-pull $V_\pi L$.
- {doc}`recipes/solve-a-device` takes this line further: a band of
  frequencies, real metals, its resonances and their Q, its capacitance.
- {doc}`recipes/prepare-a-mesh` builds the mesh of your own device in gmsh.
- {doc}`recipes/save-and-export` writes S to a Touchstone file, a field to
  ParaView, and the run to a job file that solves again on another machine.
- The {doc}`recipes <recipes/index>` start from a task: a gradient of S-parameters,
  eigenmodes or capacitance, a periodic cell and the plane waves it diffracts,
  a measurement on the fields.
- {doc}`capabilities` lists what gfem solves and differentiates;
  {doc}`project/scaling` shows how large a model runs, and how fast.
- {doc}`conventions` shows the units, signs and normalizations on a solve, and
  {doc}`glossary` explains the terms these pages use.
