# Modes and ports

gfem solves two different eigenproblems, and both return "modes". A
{term}`port mode` is a wave travelling along a guide: the frequency is given and
its propagation constant is found. A {term}`resonant mode` is a field that rings
in a structure: its frequency is found. This page says which is which, when
each is real or complex, and what a {term}`port` does in a solve.

| | Port mode | Resonant mode |
| --- | --- | --- |
| Problem | {py:obj}`gfem.PortMode` | {py:obj}`gfem.Eigenmode` |
| Domain | a port's cross-section (2D, or 1D for a {term}`line port`) | the whole model |
| Given | the frequency $\omega$ | the Bloch wave vector, on a periodic cell |
| Found | the propagation constant $\gamma = \alpha + i\beta$ | the angular frequency $\omega$ |
| Result | `PortModeResult.gamma`, `n_eff` | `EigenmodeResult.omega`, `frequency`, `q` |
| Lossless and bound | $\gamma = i\beta$, $\alpha = 0$ | $\omega$ real, $Q = \infty$ |
| Losing power | $\alpha > 0$: the wave decays along the guide | $\mathrm{Im}\,\omega > 0$: the field decays in time |

```python
modes = gfem.solve(model, gfem.PortMode("p1", np.linspace(180e12, 200e12, 21)))
modes.gamma  # 1/m, alpha + i beta: the frequency was given
modes.n_eff  # gamma / (i k0)

resonances = gfem.solve(model, gfem.Eigenmode(target=193e12, n_modes=3))
resonances.omega  # rad/s, complex: the frequency was found
resonances.q  # |omega| / (2 Im omega)
```

## Port modes: bound and leaky

A port mode varies along the guide as $e^{-\gamma s}$ (see {doc}`conventions`).
Two things make $\alpha$ positive, and they are not the same:

- **Loss.** A lossy material or a conductor in the cross-section absorbs
  power as the wave travels.
- **Leakage.** The mode is not held by the guide and radiates sideways as it
  travels: a strip waveguide over a buried oxide too thin to isolate it from
  the substrate, a bend, a guide above the light line of a cladding. Such a
  mode is a {term}`leaky mode`. Its field decays along the guide but grows
  away from it, into the region it radiates into, so the cross-section must
  end in an absorbing edge or a {term}`perfectly matched layer`; behind a
  perfect conductor its radiation would reflect back and the section would
  have no leaky mode to find.

A {term}`bound mode` is one whose field decays away from the guide: below the
light line of every cladding, $\beta > k_0 n$ for each cladding index $n$. A
lossless bound mode has $\alpha = 0$. The fundamental mode of a silicon strip
waveguide is bound, even though the structure is open: nothing about it
radiates. A section that is open is not enough to make a mode leaky.

## Resonant modes: closed and open

An {term}`eigenmode problem` finds the frequencies at which a source-free field
can exist. In a closed, lossless structure, such as a cavity with perfectly
conducting walls and lossless filling, these frequencies are real.

When the structure loses power, $\omega$ is complex and
$\mathrm{Im}\,\omega > 0$ (time dependence $e^{+i\omega t}$, so the field decays
as $e^{-\mathrm{Im}\,\omega\, t}$). Material loss does this, and so does
anything that lets the field leave: a resistive {term}`lumped port`, a
first-order `Absorbing` boundary, a perfectly matched layer. A resonant mode of
an open structure is a {term}`quasi-normal mode` (QNM). Its field decays in
time and, because the power it loses travels outwards, grows away from the
structure; the perfectly matched layer damps that growth inside the layer.
Its quality factor is

$$
Q = \frac{|\omega|}{2\, \mathrm{Im}\,\omega}.
$$

On a periodic cell with a fixed Bloch wave vector, `Eigenmode(wave_vector=...)`
gives the modes of that wave vector; one problem per wave vector traces the
band diagram $\omega(k)$. In a lossless cell a band of bound modes is real and a
band of leaky modes is complex.

`Eigenmode` does not take a {term}`wave port`: the port's propagation constant
would have to be found at a complex frequency. Terminate the guide with a
perfectly matched layer, or find the resonance from a {term}`driven problem`
sweep through the port.

## One leak, seen from both sides

A leaky mode is one physical wave. Fix the frequency and it decays in space
($\alpha > 0$, a port mode); fix the wave number and it decays in time
($\mathrm{Im}\,\omega > 0$, a resonant mode of a periodic cell). For weak
leakage, $\alpha \ll \beta$, the two rates are tied by the group velocity
$v_g = d\omega / d\beta$ along the guide:

$$
\mathrm{Im}\,\omega \approx v_g\, \alpha .
$$

The same holds for the losses of a bound mode. A guide coupled to a cavity
mixes the two pictures: the guide's own mode stays bound, while the cavity's
resonance becomes a quasi-normal mode whose $\mathrm{Im}\,\omega$ counts the
power it loses into the guide.

## The three roles of a port

A {term}`port` does three things, and only one of them depends on the problem.

1. **It is a boundary, always.** A wave port absorbs its own mode without
   reflection; a lumped port is a resistor, with an optional inductance and
   capacitance. Both are in the operator of every problem on the model, so
   they set the resonances and their Q. In an `Eigenmode` problem a lumped
   port is only this boundary, and a resistive one adds a coupling Q
   (`EigenmodeResult.q_port`).
2. **It is a source, when excited.** A {term}`driven problem` feeds each
   {term}`excited port <Excited port>` in turn with 1 W, one column of the
   {term}`S-parameters` each. `Driven(excite=...)` names those ports; the
   default excites every port.
3. **It is read, always.** S has a row for every port, excited or not.

```python
# Feed p1 only; p2 still terminates the device and S still reads it
result = gfem.solve(model, gfem.Driven(frequencies, excite="p1"))
```

Changing `excite=` changes only the right-hand side, never the operator: a
port that is not excited loads the device exactly as an excited one does.
