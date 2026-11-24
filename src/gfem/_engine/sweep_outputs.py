"""A driven forward's sweep outputs: one value at the frequencies it solved.

`SweepOutputs` holds S, ``z_ref``, the Krylov iterations and reuse flags,
the kept fields (a uniform sweep's ``e_field``, an adaptive sweep's reduced
fields) and the wave ports' modes at the grid indices ``rows``, in order: every
per-frequency array has ``len(rows)`` rows, the port mode fields'
coordinates included (`PortModeRows.field`), so trimming to the solved
frequencies is one ``rows``. Frequency workers' blocks join into one value
(`SweepOutputs.join`), with the wave ports' modes of the parent's sweep, and
a uniform sweep's partial result is the value at its solved rows, carried by
the error that reports it.

It also owns the names at the ``custom_vjp`` seam (`gfem._adjoint.solve_vjp`),
which stays a dict: `SweepOutputs.outputs` is the forward's
dict, `SweepOutputs.read` its inverse (also of the JAX path's dict, which
has no host-only key), `SweepOutputs.specs` each output's shape and dtype,
and `DIFFERENTIABLE` the outputs a backward pass differentiates. The rule
(`gfem._engine.driven.DrivenRule`) builds the public result from the
value's fields.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gfem._engine.wave_ports import WavePortCoupling
from gfem.results import PortModeField

__all__ = ["DIFFERENTIABLE", "PortModeRows", "SweepOutputs"]

#: Outputs whose cotangents the backward pass handles. ``z_ref`` and the
#: wave ports' modes (``port_gamma``, ``port_z_pv``) follow a design at a wave
#: port face (the port pass); the modes' ``port_residual`` is a diagnostic, and
#: refused beside such a design. An adaptive sweep's kept
#: fields differentiate through their reduced coordinates; its basis
#: ``e_basis`` is frozen, so not differentiable.
DIFFERENTIABLE = frozenset(
    {
        "s",
        "z_ref",
        "e_field",
        "e_coordinates",
        "port_waves",
        "emitted_power",
        "port_gamma",
        "port_z_pv",
        "port_residual",
        "port_solved",
    }
)


def _at(value: np.ndarray, rows: tuple[int, ...]) -> np.ndarray:
    """``value``'s rows ``rows``: ``value`` itself when they are all of them."""
    if rows == tuple(range(len(value))):
        return value
    return value[list(rows)]


@dataclasses.dataclass(frozen=True, eq=False)
class PortModeRows:
    """The wave ports' modes at a sweep's rows, ``(len(rows), n_wave)`` each.

    Attributes:
        gamma: Propagation constants, 1/m.
        z_pv: Complex power-voltage impedances, ohm.
        residual: Backward errors of the modes.
        solved: True where a mode is an exact solve.
        sweeps: Each wave port's `gfem._sweep.port_mode.PortModeSweep` over
            the whole grid (host only: ``None`` when read from JAX).
        rows: The grid indices of the rows.
    """

    gamma: np.ndarray
    z_pv: np.ndarray
    residual: np.ndarray
    solved: np.ndarray
    sweeps: tuple[Any, ...] | None
    rows: tuple[int, ...]

    @classmethod
    def of(cls, sweeps: Sequence[Any], rows: tuple[int, ...]) -> PortModeRows | None:
        """The modes at ``rows`` of the wave ports' whole-grid ``sweeps``.

        One `gfem._sweep.port_mode.PortModeSweep` per wave port, in model
        port order; ``None`` without any.
        """
        if not sweeps:
            return None
        sweeps = tuple(sweeps)
        return cls(
            gamma=_at(np.stack([s.gamma for s in sweeps], axis=1), rows),
            z_pv=_at(np.stack([s.z_pv for s in sweeps], axis=1), rows),
            residual=_at(np.stack([s.residual for s in sweeps], axis=1), rows),
            solved=_at(np.stack([s.solved for s in sweeps], axis=1), rows),
            sweeps=sweeps,
            rows=rows,
        )

    def field(self, k: int) -> PortModeField | None:
        """Wave port ``k``'s mode field at the rows; ``None`` without its sweep.

        The coordinates are sliced with the rows that sliced the other
        arrays (all rows: the sweep's own coordinates).
        """
        if self.sweeps is None:
            return None
        sweep = self.sweeps[k]
        return PortModeField(sweep.space, sweep.basis, _at(sweep.coords, self.rows))


@dataclasses.dataclass(frozen=True, eq=False)
class SweepOutputs:
    """A driven forward's outputs at the grid frequencies ``rows``.

    Attributes:
        rows: The grid indices of the rows, in order.
        s: ``(len(rows), n_ports, n_excited)`` complex.
        z_ref: ``(len(rows), n_ports)`` ohm.
        iterations: Krylov iterations per solve (per frequency, or per
            adaptive sample) and excited port (host only).
        reused: Per solve, whether it ran on a preconditioner built for
            another matrix (host only).
        e_field: A uniform sweep's kept fields at the rows, ``(ndof,
            n_kept * n_excited)`` complex (frequency-major), or ``None``.
        reduced: An adaptive sweep's kept fields ``(e_basis,
            e_coordinates)``, or ``None``.
        ports: The wave ports' modes at the rows, or ``None`` without any.
        sweep: An adaptive sweep's greedy information ``{"samples",
            "errors", "converged", "basis_size"}`` (host only), or ``None``.
        port_waves: The outgoing wave at each port for each source column,
            ``(len(rows), n_ports, n_sources)``, or ``None`` without sources.
        emitted_power: ``(len(rows), n_dipoles)`` W, or ``None`` without
            dipoles.
    """

    rows: tuple[int, ...]
    s: np.ndarray
    z_ref: np.ndarray
    iterations: list[tuple[int, ...]] | None
    reused: list[bool] | None
    e_field: np.ndarray | None
    reduced: tuple[np.ndarray, np.ndarray] | None
    ports: PortModeRows | None
    sweep: Mapping[str, Any] | None
    port_waves: np.ndarray | None = None
    emitted_power: np.ndarray | None = None

    @classmethod
    def uniform(
        cls,
        rows: tuple[int, ...],
        *,
        s: np.ndarray,
        z_ref: np.ndarray,
        iterations: list[tuple[int, ...]],
        reused: list[bool],
        e_field: np.ndarray | None,
        waves: Sequence[WavePortCoupling],
        port_waves: np.ndarray | None = None,
        emitted_power: np.ndarray | None = None,
    ) -> SweepOutputs:
        """A uniform sweep's outputs at ``rows``.

        ``s``, ``z_ref``, ``port_waves``, ``emitted_power`` and the wave
        ports' modes are over the grid; the kept fields ``e_field`` are at
        ``rows`` already.
        """
        return cls(
            rows=rows,
            s=_at(s, rows),
            z_ref=_at(z_ref, rows),
            iterations=iterations,
            reused=reused,
            e_field=e_field,
            reduced=None,
            ports=PortModeRows.of([c.sweep for c in waves], rows),
            sweep=None,
            port_waves=None if port_waves is None else _at(port_waves, rows),
            emitted_power=None if emitted_power is None else _at(emitted_power, rows),
        )

    @classmethod
    def adaptive(
        cls,
        *,
        s: np.ndarray,
        z_ref: np.ndarray,
        iterations: list[tuple[int, ...]],
        reused: list[bool],
        sweep: Mapping[str, Any],
        reduced: tuple[np.ndarray, np.ndarray] | None,
        waves: Sequence[WavePortCoupling],
        port_waves: np.ndarray | None = None,
        emitted_power: np.ndarray | None = None,
    ) -> SweepOutputs:
        """An adaptive sweep's outputs: every grid frequency has a row."""
        rows = tuple(range(len(s)))
        return cls(
            rows=rows,
            s=s,
            z_ref=z_ref,
            iterations=iterations,
            reused=reused,
            e_field=None,
            reduced=reduced,
            ports=PortModeRows.of([c.sweep for c in waves], rows),
            sweep=sweep,
            port_waves=port_waves,
            emitted_power=emitted_power,
        )

    @classmethod
    def join(
        cls,
        blocks: Sequence[tuple[Sequence[int] | np.ndarray, SweepOutputs]],
        *,
        sweeps: Sequence[Any] = (),
    ) -> SweepOutputs:
        """Frequency workers' blocks joined in order.

        The wave ports' modes are the parent's: it swept them once over the
        whole grid, and the blocks come without.

        Args:
            blocks: Per block, the grid indices of its frequencies and its
                outputs, whose rows index those.
            sweeps: Each wave port's port mode sweep over the whole grid, in
                model port order; empty without wave ports.

        Raises:
            ValueError: A block has its own port modes or an adaptive
                sweep's outputs (a frequency worker sweeps neither).
        """
        for _, block in blocks:
            if block.ports is not None or block.sweep is not None:
                raise ValueError(
                    "Only uniform sweeps join over frequency workers, and a "
                    "block's own port modes are not the whole grid's: the "
                    "parent's port mode sweeps go in sweeps."
                )
        fields = [b.e_field for _, b in blocks if b.e_field is not None]
        waves = [b.port_waves for _, b in blocks if b.port_waves is not None]
        powers = [b.emitted_power for _, b in blocks if b.emitted_power is not None]
        rows = tuple(int(grid[r]) for grid, b in blocks for r in b.rows)
        return cls(
            rows=rows,
            s=np.concatenate([b.s for _, b in blocks]),
            z_ref=np.concatenate([b.z_ref for _, b in blocks]),
            iterations=[its for _, b in blocks for its in b.iterations or ()],
            reused=[r for _, b in blocks for r in b.reused or ()],
            e_field=np.concatenate(fields, axis=1) if fields else None,
            reduced=None,
            ports=PortModeRows.of(sweeps, rows),
            sweep=None,
            port_waves=np.concatenate(waves) if waves else None,
            emitted_power=np.concatenate(powers) if powers else None,
        )

    def without_port_modes(self) -> SweepOutputs:
        """These outputs without the wave ports' modes.

        What a frequency worker returns: the modes are its parent's, and a
        port mode space does not pickle.
        """
        return dataclasses.replace(self, ports=None)

    def outputs(self) -> dict[str, object]:
        """The forward outputs at the ``custom_vjp`` seam, by name.

        ``"s"``, ``"z_ref"``, ``"iterations"``, ``"reused"``, ``"sweep"``, the
        wave ports' ``"port_gamma"``, ``"port_z_pv"``, ``"port_residual"``,
        ``"port_solved"`` and ``"port_sweeps"``, ``"port_waves"``,
        ``"emitted_power"``, then
        ``"e_field"`` or ``"e_basis"`` and ``"e_coordinates"``; only the ones
        present.
        """
        outputs: dict[str, object] = {"s": self.s, "z_ref": self.z_ref}
        if self.iterations is not None:
            outputs["iterations"] = self.iterations
        if self.reused is not None:
            outputs["reused"] = self.reused
        if self.sweep is not None:
            outputs["sweep"] = self.sweep
        if self.ports is not None:
            outputs["port_gamma"] = self.ports.gamma
            outputs["port_z_pv"] = self.ports.z_pv
            outputs["port_residual"] = self.ports.residual
            outputs["port_solved"] = self.ports.solved
            if self.ports.sweeps is not None:
                outputs["port_sweeps"] = list(self.ports.sweeps)
        if self.port_waves is not None:
            outputs["port_waves"] = self.port_waves
        if self.emitted_power is not None:
            outputs["emitted_power"] = self.emitted_power
        if self.e_field is not None:
            outputs["e_field"] = self.e_field
        if self.reduced is not None:
            outputs["e_basis"], outputs["e_coordinates"] = self.reduced
        return outputs

    @classmethod
    def read(cls, outputs: Mapping[str, Any]) -> SweepOutputs:
        """The value of forward outputs by name (`outputs`).

        A host-only output that is absent (on the JAX path: ``"iterations"``,
        ``"reused"``, ``"port_sweeps"``, ``"sweep"``) reads as ``None``. The
        outputs of a complete sweep only cross the seam, so the rows are every
        row.
        """
        rows = tuple(range(len(outputs["s"])))
        ports = None
        if "port_gamma" in outputs:
            sweeps = outputs.get("port_sweeps")
            ports = PortModeRows(
                gamma=outputs["port_gamma"],
                z_pv=outputs["port_z_pv"],
                residual=outputs["port_residual"],
                solved=outputs["port_solved"],
                sweeps=None if sweeps is None else tuple(sweeps),
                rows=rows,
            )
        reduced = None
        if "e_basis" in outputs:
            reduced = (outputs["e_basis"], outputs["e_coordinates"])
        return cls(
            rows=rows,
            s=outputs["s"],
            z_ref=outputs["z_ref"],
            iterations=outputs.get("iterations"),
            reused=outputs.get("reused"),
            e_field=outputs.get("e_field"),
            reduced=reduced,
            ports=ports,
            sweep=outputs.get("sweep"),
            port_waves=outputs.get("port_waves"),
            emitted_power=outputs.get("emitted_power"),
        )

    @staticmethod
    def specs(
        *,
        n_freq: int,
        n_ports: int,
        n_excited: int,
        n_wave: int,
        ndof: int | None = None,
        n_fields: int = 0,
        reduced: int | None = None,
        complex_z_ref: bool = False,
        n_sources: int = 0,
        n_dipoles: int = 0,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        """Shape and dtype of each output a JAX transformation reads.

        Args:
            n_freq: Grid frequencies.
            n_ports: The model's ports (the rows of S).
            n_excited: Excited ports (the columns of S).
            n_wave: Wave ports.
            ndof: Kept fields' coefficients; ``None`` keeps no field.
            n_fields: Kept fields, ``n_kept`` times the excited ports and
                the sources.
            reduced: An adaptive sweep's padded basis size (its kept fields
                are reduced); ``None`` for a uniform sweep.
            complex_z_ref: ``z_ref`` is complex
                (`gfem._engine.port_network.complex_reference`).
            n_sources: Source columns; their ``port_waves`` with ports.
            n_dipoles: Point dipoles among them: their ``emitted_power``.
        """
        specs = {
            "s": jax.ShapeDtypeStruct((n_freq, n_ports, n_excited), jnp.complex128),
            "z_ref": jax.ShapeDtypeStruct(
                (n_freq, n_ports), jnp.complex128 if complex_z_ref else jnp.float64
            ),
        }
        if n_sources:
            specs["port_waves"] = jax.ShapeDtypeStruct(
                (n_freq, n_ports, n_sources), jnp.complex128
            )
        if n_dipoles:
            specs["emitted_power"] = jax.ShapeDtypeStruct(
                (n_freq, n_dipoles), jnp.float64
            )
        if n_wave:
            shape = (n_freq, n_wave)
            specs["port_gamma"] = jax.ShapeDtypeStruct(shape, jnp.complex128)
            specs["port_z_pv"] = jax.ShapeDtypeStruct(shape, jnp.complex128)
            specs["port_residual"] = jax.ShapeDtypeStruct(shape, jnp.float64)
            specs["port_solved"] = jax.ShapeDtypeStruct(shape, jnp.bool_)
        if ndof is not None:
            if reduced is not None:
                specs["e_basis"] = jax.ShapeDtypeStruct((ndof, reduced), jnp.float64)
                specs["e_coordinates"] = jax.ShapeDtypeStruct(
                    (n_fields, reduced), jnp.complex128
                )
            else:
                specs["e_field"] = jax.ShapeDtypeStruct(
                    (ndof, n_fields), jnp.complex128
                )
        return specs
