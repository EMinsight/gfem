r"""Run the meta-atom design and write its run record.

Usage: python tools/meta_atom_run.py --out DIR [--profile full|smoke]

A density design in a 0.35 um cube of air-to-silicon material
(``docs/tutorials/data/meta_atom.msh``, built by tools/meta_atom_mesh.py)
that maximizes the absolute magnetic-dipole partial scattering cross
section :math:`\sigma^M_1` of a plane wave at 1 um, 40 steps of Adam
through a Helmholtz filter and a projection whose sharpness doubles every
8 steps, each step one order-2 solve and :func:`gfem.multipoles` outside
:func:`jax.jit`. It also checks the first gradient against a finite
difference, re-solves the final design at order 3, and compares the sum of
the partial cross sections with :func:`gfem.cross_sections`. The docs page
``docs/tutorials/meta-atom.ipynb`` reads the record and solves nothing; it
refuses a record whose script hashes differ from these files', so editing
one forces a rerun. ``just meta-atom-data`` writes ``meta_atom.json`` and
``meta_atom.npz`` into ``docs/tutorials/data``; the full run takes about
10 min and 4 GiB on 8 threads.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import bend3d_run as provenance
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("tools/meta_atom_run.py", "tools/meta_atom_mesh.py")
RECORD = ("meta_atom.json", "meta_atom.npz")
MAX_RECORD_BYTES = 1_000_000

WAVELENGTH_UM = 1.0
EPS_SILICON = 12.25
FILTER_UM = 0.05
LEARNING_RATE = 0.05
START_DENSITY = 0.25
BETAS = (1.0, 2.0, 4.0, 8.0, 16.0)
PROFILES = {"full": 8, "smoke": 1}  # steps per beta
FD_BETA = 2.0
FD_STEP = 1e-2
SNAPSHOTS = (0, 8, 24, 39)  # steps whose physical density is kept


def sha256s():
    """The sha256 of the run script and the mesh builder, by path."""
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SCRIPTS}


def run(profile):  # noqa: PLR0915
    """Run the design; return ``(settings, observations, arrays)``."""
    import jax
    import jax.numpy as jnp
    import optax

    import gfem
    from gfem import design

    jax.config.update("jax_enable_x64", True)
    um = 1e-6
    tic = time.perf_counter()
    steps_s = {}
    mesh = gfem.Mesh.from_gmsh(ROOT / "docs/tutorials/data/meta_atom.msh", "um")
    air = gfem.Material(name="air")
    silicon = gfem.Material(eps_r=EPS_SILICON, name="silicon")
    model = gfem.Model(
        mesh,
        materials={"design": air, "air": air, "pml": air},
        pml=gfem.PML("pml"),
        background=air,
    )
    frequency = 299792458.0 / (WAVELENGTH_UM * um)
    wave = gfem.PlaneWave("z", theta=0, polarization="TM")
    problem = gfem.Driven(frequency, sources=(wave,), keep_fields=True)
    blur = design.helmholtz_filter(mesh, "design", radius=FILTER_UM * um)
    n = len(mesh.elements("design"))
    um2 = 1e12

    def solve(density, order):
        placed = model.with_material(
            "design", gfem.MaterialField.interpolate(density, air, silicon)
        )
        options = gfem.SolverOptions(order=order, linear_solver="direct")
        result = gfem.solve(placed, problem, options=options)
        return placed, result, gfem.multipoles(result, "z", model=placed)

    def physical(rho, beta):
        return design.project(blur(rho), beta)

    def objective(rho, beta):
        _, _, mp = solve(physical(rho, beta), 2)
        partial = jnp.concatenate(
            [mp.magnetic_scattering[0, :2], mp.electric_scattering[0, :2]]
        )
        return -um2 * mp.magnetic_scattering[0, 0], um2 * jnp.append(
            partial, mp.scattering[0]
        )

    value_and_grad = jax.value_and_grad(objective, has_aux=True)
    per_beta = PROFILES[profile]
    rho = jnp.full(n, START_DENSITY)
    optimizer = optax.adam(LEARNING_RATE)
    state = optimizer.init(rho)
    history, snapshots = [], {}
    for beta in BETAS:
        for _ in range(per_beta):
            step = len(history)
            t0 = time.perf_counter()
            (_, parts), g = value_and_grad(rho, beta)
            if step in SNAPSHOTS or step == len(BETAS) * per_beta - 1:
                snapshots[step] = np.asarray(physical(rho, beta))
            updates, state = optimizer.update(g, state)
            rho = jnp.clip(optax.apply_updates(rho, updates), 0.0, 1.0)
            history.append((beta, *np.asarray(parts)))
            steps_s[f"step{step}_s"] = round(time.perf_counter() - t0, 1)
            print(f"step {step} beta {beta:g}: {history[-1][1]:.4f} um^2", flush=True)
    history = np.array(history)  # beta, MD, MQ, ED, EQ, sum
    n_steps = len(history)

    # the gradient of the objective against a fourth-order central difference
    t0 = time.perf_counter()
    rng = np.random.default_rng(1)
    start = jnp.full(n, START_DENSITY) + 0.05 * jnp.asarray(rng.standard_normal(n))
    direction = rng.standard_normal(n)
    direction = jnp.asarray(direction / np.linalg.norm(direction))
    (_, _), g0 = value_and_grad(start, FD_BETA)
    adjoint = float(jnp.sum(g0 * direction))

    def f(shift):
        return float(objective(start + shift * direction, FD_BETA)[0])

    h = FD_STEP
    difference = (8 * (f(h) - f(-h)) - (f(2 * h) - f(-2 * h))) / (12 * h)
    fd_error = abs(adjoint - difference) / abs(difference)
    steps_s["fd_s"] = round(time.perf_counter() - t0, 1)

    # the final design at both orders, and the cross sections at order 3
    beta_final = BETAS[-1]
    final = physical(rho, beta_final)
    out = {}
    for order in (2, 3):
        t0 = time.perf_counter()
        placed, result, mp = solve(final, order)
        xs = gfem.cross_sections(result, "z", model=placed)
        out[order] = {
            "md": um2 * float(mp.magnetic_scattering[0, 0]),
            "ed": um2 * float(mp.electric_scattering[0, 0]),
            "sum": um2 * float(mp.scattering[0]),
            "cross_sections": um2 * float(xs.scattering[0]),
            "dof": int(result.e_fields[0].coefficients.shape[0]),
            "lmax": int(mp.lmax),
        }
        steps_s[f"order{order}_s"] = round(time.perf_counter() - t0, 1)
    k = 2 * np.pi / WAVELENGTH_UM
    start_value = float(history[0, 1])
    final_value = float(out[2]["md"])
    centres = np.asarray(mesh.nodes)[mesh.tetrahedra[mesh.elements("design")]].mean(1)
    settings = {
        "wavelength_um": WAVELENGTH_UM,
        "eps_silicon": EPS_SILICON,
        "filter_radius_um": FILTER_UM,
        "learning_rate": LEARNING_RATE,
        "start_density": START_DENSITY,
        "betas": list(BETAS),
        "steps": n_steps,
        "design_elements": n,
        "tetrahedra": len(mesh.tetrahedra),
        "fd": {
            "beta": FD_BETA,
            "step": FD_STEP,
            "adjoint": adjoint,
            "difference": difference,
            "error": fd_error,
        },
        "order2": out[2],
        "order3": out[3],
        "order_gap": abs(out[3]["md"] / out[2]["md"] - 1),
        "sum_gap": abs(out[3]["sum"] / out[3]["cross_sections"] - 1),
        "single_channel_um2": 6 * np.pi / k**2,
        "start_um2": start_value,
        "final_um2": final_value,
        "gain": final_value / start_value,
    }
    observations = {"wall_s": round(time.perf_counter() - tic, 1), "steps_s": steps_s}
    arrays = {
        "history": history,
        "centres": centres,
        "density_final": np.asarray(final),
        "snapshot_steps": np.array(sorted(snapshots)),
        "snapshots": np.array([snapshots[s] for s in sorted(snapshots)]),
    }
    return settings, observations, arrays


def main(argv=None):
    """Run the design and write ``meta_atom.json`` and ``meta_atom.npz``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="the record's folder")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="full")
    args = parser.parse_args(argv)
    memory, start = provenance.PeakMemory(), provenance._utc()  # noqa: SLF001
    commit = provenance.clean_commit(ROOT) if args.profile == "full" else None
    if commit is None:
        commit = provenance.run_commit(ROOT)[0]
    hashes = sha256s()
    settings, observations, arrays = run(args.profile)
    observations["peak_rss_bytes"] = memory.peak_bytes
    record = {
        "profile": args.profile,
        "provenance": {
            "commit": commit,
            "sha256": hashes,
            "start_utc": start,
            "end_utc": provenance._utc(),  # noqa: SLF001
            "command": " ".join(["python", "tools/meta_atom_run.py", *sys.argv[1:]]),
            "environment": provenance.environment(),
        },
        "settings": settings,
        "observations": observations,
    }
    text = (json.dumps(record, indent=1) + "\n").encode()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    if len(text) + len(buffer.getvalue()) > MAX_RECORD_BYTES:
        raise ValueError("the run record is over its size cap")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, payload in zip(RECORD, (text, buffer.getvalue()), strict=True):
        (args.out / name).write_bytes(payload)
    print(f"wall {observations['wall_s']} s, peak {memory.peak_bytes / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
