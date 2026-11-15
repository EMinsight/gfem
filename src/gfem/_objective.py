"""Objective classification: what an objective ``J(result, model)`` reads (ADR 0143).

`classify` traces the user's objective on a solved result and returns its
estimate kind, value, read entries with their cotangents, and the curvature
of the objective over those entries (the 2x2 spectral norms of its Hessian's
(Re, Im) blocks, for the remainder term of ADR 0129). Two levels (ADR 0143
amendment):

1. **Kind, refusals and mixtures** come from structural dependence on whole
   result leaves and on recorded measurement outputs: the objective's jaxpr,
   pruned by dead-code elimination. Each leaf of the result is an entry leaf
   (`ENTRY_KINDS`: its elements are the objective's scalar inputs), a refused
   leaf (`REFUSED_LEAVES`, or a field's coefficients: read through a
   recognized measurement only), or a constant.
2. **Read entries** are the entries of the live leaves whose NaN reaches the
   objective's value, or where its cotangent is nonzero.

Cotangents are the derivative of the objective's value, gfem's contract
``dJ/dx - i dJ/dy`` per complex entry, with ``stop_gradient`` evaluated as the
identity: the user's gradient choice does not change the quantity's error.

A recognized measurement reads a field only under the `Recorder`: a pass of
`jax.eval_shape` discovers which outputs the objective reads, and the jaxpr
takes those outputs as inputs, so no host kernel runs while tracing. Each
measured output is evaluated once, eagerly, for its value. A new kind plugs
in by an entry of `ENTRY_KINDS` (a result leaf) or of `MEASUREMENT_KINDS` (a
measurement, through `Recorder.output`), and an estimate kernel in
`gfem.refinement`.

The classifier relies on JAX internals (``dce_jaxpr``, ``eval_jaxpr``) at the
pinned JAX version; its unit tests assert a known classification table, so a
JAX upgrade that changes them fails loudly.
"""

from __future__ import annotations

import contextvars
import dataclasses
import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as jcore
from jax._src.interpreters import partial_eval as pe

from gfem._checks import is_staging, is_traced
from gfem.errors import MeshRefinementUnsupportedError, NonDifferentiableError

__all__ = [
    "ENTRY_KINDS",
    "MEASUREMENT_KINDS",
    "Objective",
    "Read",
    "Recorder",
    "aligned",
    "classify",
    "recording",
]

#: The result leaves an objective may read, by result type and leaf path
#: (`jax.tree_util.keystr`, matched as a prefix): their estimate kind and
#: whether the tolerance floor scales an entry absolutely (1) or by its value.
ENTRY_KINDS: dict[tuple[str, str], tuple[str, bool]] = {
    ("DrivenResult", ".s.values"): ("objective of s", True),
    ("ElectrostaticResult", ".capacitance"): ("objective of capacitance", False),
    ("ElectrostaticResult", ".energy["): ("objective of electrostatic energy", False),
    ("EigenmodeResult", ".omega"): ("objective of omega", False),
    ("EigenmodeResult", ".bulk_participation["): ("objective of participation", False),
}

#: The leaves an objective may not read, by path prefix: the feature its
#: `MeshRefinementUnsupportedError` names.
REFUSED_LEAVES: dict[str, str] = {
    ".q_port[": "q_port",
    ".participation[": "participation",
    ".principal_participation[": "principal_participation",
    ".sheet_participation[": "sheet_participation",
    ".error": "eigen error",
    ".port_modes[": "port mode output",
}

#: A recognized measurement's estimate kind, by (measurement, field kind).
MEASUREMENT_KINDS: dict[tuple[str, str], str] = {
    ("power_flow", "driven"): "objective of absorbed power",
    ("region_mean", "driven"): "objective of field means",
    ("region_mean", "potential"): "objective of potential means",
}

#: The one admitted mixture of kinds (ADR 0142), and the kind it takes.
_MIXTURE = (
    frozenset({"objective of omega", "objective of participation"}),
    "objective of participation",
)

#: Primitives that branch on their operands: an objective applying one to a
#: read entry is not smooth there (ADR 0143 amendment). ``select_n`` and
#: ``cond`` count when their predicate depends on a read entry.
_BRANCHING = frozenset(
    {
        "max",
        "min",
        "reduce_max",
        "reduce_min",
        "argmax",
        "argmin",
        "sort",
        "clamp",
        "cummax",
        "cummin",
        "top_k",
    }
)

_FIELD_KINDS = {
    "DrivenResult": "driven",
    "ElectrostaticResult": "potential",
    "EigenmodeResult": "mode",
}

_TRACE_ADVICE = (
    "gfem traces the objective with JAX to find what it reads: write it with "
    "jax.numpy, with no Python branch (if, bool(), float()) or NumPy conversion "
    "of a value of the result."
)

_REMEDY = "Estimate the quantities it reads one by one, or use gfem.FieldEstimator()."


@dataclasses.dataclass(frozen=True)
class Read:
    """One scalar an objective reads: a result entry or a measurement output.

    Attributes:
        leaf: The result leaf's path (``".s.values"``), or the measurement
            output's key (`Recorder.output`).
        index: The entry's index in the leaf, ``()`` for a 0-d leaf; a
            measurement output's component.
        value: The entry's value.
        cotangent: The objective's cotangent on it, ``dJ/dx - i dJ/dy`` for a
            complex entry, ``dJ/dx`` for a real one.
        is_complex: Whether the entry is complex.
    """

    leaf: str
    index: tuple[int, ...]
    value: complex
    cotangent: complex
    is_complex: bool


@dataclasses.dataclass(frozen=True, eq=False)
class Objective:
    """An objective classified on one result (`classify`).

    Attributes:
        kind: Its estimate kind (``gfem.refinement.ESTIMATE_KINDS``).
        value: The objective's value on the result.
        reads: Each scalar it reads, in a fixed order.
        curvature: ``(n, n)``, ``||H_ab||_2`` of the (Re, Im) blocks of the
            objective's Hessian over the read scalars ``a``, ``b`` (a real
            scalar has a one-row block).
        diagonal: Per read scalar, its own (Re, Im) block ``H_aa`` of the
            Hessian (2 x 2 for a complex entry, 1 x 1 for a real one): the
            blocks a rewrite of the objective shifts (ADR 0135).
        floor_scales: Per read, its scale in the tolerance floor: 1 for an
            absolute entry (S), its magnitude otherwise.
        measurements: Per measurement key the objective reads, its record
            (`Recorder.output`).
    """

    kind: str
    value: float
    reads: tuple[Read, ...]
    curvature: np.ndarray
    floor_scales: tuple[float, ...]
    measurements: Mapping[str, Mapping[str, Any]]
    diagonal: tuple[np.ndarray, ...] = ()

    @property
    def sensitivity(self) -> float:
        """``sum_a |ct_a| s_a``: the objective's first-order sensitivity scale."""
        return float(
            sum(
                abs(r.cotangent) * s
                for r, s in zip(self.reads, self.floor_scales, strict=True)
            )
        )


# --- the recorder -----------------------------------------------------------------

_RECORDER: contextvars.ContextVar[Recorder | None] = contextvars.ContextVar(
    "gfem_objective_recorder", default=None
)


def recording() -> Recorder | None:
    """The `Recorder` of the objective being classified, else ``None``."""
    return _RECORDER.get()


class Recorder:
    """What the measurements an objective calls see while it is classified.

    Under a recorder, a recognized measurement does not touch the field: it
    names the field it was given (by the identity of its coefficients,
    `label`) and returns `output`, a scalar input of the traced objective.
    Other field reads record what they did, so a refusal can name it.

    Attributes:
        model: The model the objective was called with.
        discover: Whether this is the abstract pass (`jax.eval_shape`),
            which records each output's key, or the jaxpr's, which reads them.
    """

    def __init__(
        self,
        labels: Mapping[int, str],
        field_kinds: Mapping[str, str],
        fields: Mapping[str, Any],
        model: Any,
        *,
        discover: bool,
        specs: dict[str, dict[str, Any]],
        outputs: Mapping[str, jax.Array] | None = None,
        constants: Mapping[int, Any] | None = None,
    ) -> None:
        """Hold the field labels and what the measurements record."""
        self._labels = labels
        self._field_kinds = field_kinds
        self._fields = fields
        self._specs = specs
        self._outputs = outputs or {}
        self._constants = constants or {}
        self.model = model
        self.discover = discover
        self.point_reads: set[str] = set()

    def label(self, field: Any, who: str) -> str:
        """The label of a field of the classified result.

        Raises:
            MeshRefinementUnsupportedError: ``field`` is not one of the
                result's fields (``"field coefficients"``), or is an eigen
                mode field (``"normalized mode field"``).
        """
        label = self._labels.get(id(getattr(field, "coefficients", None)))
        if label is None:
            raise MeshRefinementUnsupportedError(
                "field coefficients",
                f"The objective calls {who} on a field that is not one of the "
                "fields of its own result argument, so its estimate has no "
                "dual for it: read fields from the result the objective is "
                "given.",
            )
        if self._field_kinds[label] == "mode":
            raise MeshRefinementUnsupportedError(
                "normalized mode field",
                f"The objective calls {who} on an eigen mode field ({label}), "
                "whose normalization makes its error one no estimate covers. "
                f"{_REMEDY}",
            )
        return label

    def point(self, field: Any) -> None:
        """Record that ``field`` is read at points (`gfem.Field.at`)."""
        label = self._labels.get(id(getattr(field, "coefficients", None)))
        if label is not None:
            self.point_reads.add(label)

    def field(self, label: str) -> Any:
        """The concrete field of ``label``."""
        return self._fields[label]

    def output(
        self,
        key: str,
        shape: tuple[int, ...],
        dtype: Any,
        evaluate: Callable[[], Any],
        **spec: Any,
    ) -> jax.Array:
        """A recognized measurement's output, as the traced objective's input.

        Args:
            key: Unique per (measurement, field, side or region).
            shape: The output's shape; each element is one scalar input.
            dtype: Its dtype.
            evaluate: The measurement's concrete value, run once eagerly.
            spec: What the estimate kernel needs to know of it; must hold
                ``measurement`` and ``label`` (its field's).
        """
        names = [f"{key}#{i}" for i in range(int(np.prod(shape, dtype=int)))]
        if self.discover:
            self._specs.setdefault(
                key,
                dict(spec, names=names, shape=shape, dtype=dtype, evaluate=evaluate),
            )
            return jnp.zeros(shape, dtype)
        values = [self._outputs[n] for n in names]
        return jnp.stack(values).reshape(shape) if shape else values[0]

    def constant(self, x: Any) -> Any:
        """``x`` as the concrete value, if it is a traced leaf held constant.

        A wave port's ``gamma`` is a constant of the solve when it moves a
        reference plane (`gfem.DrivenResult.with_offsets`): the objective is
        then still one of S.
        """
        return self._constants.get(id(x), x)


def constant(x: Any) -> Any:
    """`Recorder.constant` of the active recorder; ``x`` itself outside one."""
    recorder = recording()
    return x if recorder is None else recorder.constant(x)


# --- splitting a result into the objective's inputs ---------------------------------


@dataclasses.dataclass
class _Slot:
    path: str
    how: str  # "entries", "array" or "const"
    leaf: Any
    kind: str | None = None  # an entry leaf's kind


class _Split:
    """A result's leaves as the objective's traced inputs (see module docstring)."""

    def __init__(self, result: Any) -> None:
        self.result = result
        rtype = type(result).__name__
        flat, self.treedef = jax.tree_util.tree_flatten_with_path(result)
        self.slots: list[_Slot] = []
        self.entries: list[tuple[str, tuple[int, ...]]] = []
        self.entry_values: list[jax.Array] = []
        self.entry_kinds: list[str] = []
        self.entry_absolute: list[bool] = []
        self.arrays: list[str] = []
        self.array_values: list[jax.Array] = []
        self.field_kinds: dict[str, str] = {}
        self.fields: dict[str, Any] = {}
        self.port_mode_leaves: list[int] = []
        for path, leaf in flat:
            p = jax.tree_util.keystr(path)
            entry = next(
                (
                    v
                    for (r, prefix), v in ENTRY_KINDS.items()
                    if r == rtype and p.startswith(prefix)
                ),
                None,
            )
            if p.endswith(".coefficients") and rtype in _FIELD_KINDS:
                label = p.removesuffix(".coefficients")
                self.field_kinds[label] = _FIELD_KINDS[rtype]
                self._array(p, leaf)
            elif entry is not None:
                a = jnp.asarray(leaf)
                self.slots.append(_Slot(p, "entries", a.shape, entry[0]))
                for index in np.ndindex(a.shape):
                    self.entries.append((p, tuple(int(i) for i in index)))
                    self.entry_values.append(a[index])
                    self.entry_kinds.append(entry[0])
                    self.entry_absolute.append(entry[1])
            elif any(p.startswith(prefix) for prefix in REFUSED_LEAVES) and _is_array(
                leaf
            ):
                if p.startswith(".port_modes["):
                    self.port_mode_leaves.append(len(self.arrays))
                self._array(p, leaf)
            else:
                self.slots.append(_Slot(p, "const", leaf))

    def _array(self, path: str, leaf: Any) -> None:
        self.slots.append(_Slot(path, "array", len(self.arrays)))
        self.arrays.append(path)
        self.array_values.append(jnp.asarray(leaf))

    def build(
        self, entries: Sequence[jax.Array], arrays: Sequence[jax.Array]
    ) -> tuple[Any, dict[int, str]]:
        """The result on traced inputs, and its fields' labels by coefficients id."""
        leaves = []
        labels: dict[int, str] = {}
        k = 0
        for slot in self.slots:
            if slot.how == "const":
                leaves.append(slot.leaf)
            elif slot.how == "array":
                a = arrays[slot.leaf]
                leaves.append(a)
                label = slot.path.removesuffix(".coefficients")
                if label in self.field_kinds:
                    labels[id(a)] = label
            else:
                n = int(np.prod(slot.leaf, dtype=int))
                part = list(entries[k : k + n])
                k += n
                leaves.append(
                    jnp.stack(part).reshape(slot.leaf) if slot.leaf else part[0]
                )
        return self.treedef.unflatten(leaves), labels


def _is_array(leaf: Any) -> bool:
    return isinstance(leaf, (np.ndarray, jax.Array)) and np.ndim(leaf) >= 0


def _concrete_fields(split: _Split) -> dict[str, Any]:
    return {label: _field_at(split.result, label) for label in split.field_kinds}


def _field_at(result: Any, label: str) -> Any:
    """The field of ``result`` at a label: ``".e_fields[2]"``, ``".potential['a']"``."""
    attribute, _, key = label[1:].partition("[")
    container = getattr(result, attribute)
    key = key.removesuffix("]")
    return container[int(key)] if key.isdigit() else container[key.strip("'\"")]


# --- classification ---------------------------------------------------------------


def classify(
    objective: Callable[[Any, Any], Any], result: Any, model: Any
) -> Objective:
    """Classify ``objective(result, model)``: kind, value, reads and curvature.

    Raises:
        TypeError: The objective is not traceable (Python control flow or a
            NumPy conversion on a value of the result), or does not return a
            real 0-d value.
        ValueError: The objective is not finite at the result, or reads no
            entry of it.
        NonDifferentiableError: Its value carries a tracer of an enclosing
            JAX transformation.
        MeshRefinementUnsupportedError: It reads something no estimate
            covers (``feature`` names it), or mixes two estimate kinds.
    """
    trace = _Trace(objective, result, model)
    kind = trace.kind()
    _walk(trace.pruned, [True] * len(trace.pruned.invars))
    value = _value(objective, result, model)
    if not np.isfinite(value):
        _refuse_not_finite(result, kind, value)
    flat = trace.inputs()
    stripped = _identity_stop_gradient(trace.closed)
    scalars = trace.scalars()
    cotangents = _cotangents(stripped, flat, scalars)
    read = [i for i in scalars if cotangents[i] != 0]
    zero = [i for i in scalars if cotangents[i] == 0]
    read = sorted({*read, *_nan_reads(trace, flat, zero)})
    reads, scales = [], []
    curvature, diagonal = _curvature(stripped, flat, read)
    for i in read:
        leaf, index, absolute = trace.name(i)
        x = complex(np.asarray(flat[i]))
        is_complex = bool(jnp.iscomplexobj(flat[i]))
        reads.append(Read(leaf, index, x, cotangents[i], is_complex))
        scales.append(1.0 if absolute else abs(x))
    return Objective(
        kind=kind,
        value=value,
        reads=tuple(reads),
        curvature=curvature,
        diagonal=diagonal,
        floor_scales=tuple(scales),
        measurements={
            key: {k: v for k, v in spec.items() if k != "evaluate"}
            for key, spec in trace.specs.items()
            if any(r.leaf == key for r in reads)
        },
    )


class _Trace:
    """The objective's jaxpr on a result's split inputs, and its pruning.

    The inputs are, in order: the entries, the arrays (fields and refused
    leaves), and the recorded measurement outputs (`_Split`, `Recorder`).
    """

    def __init__(
        self, objective: Callable[[Any, Any], Any], result: Any, model: Any
    ) -> None:
        self.split = split = _Split(result)
        self.model = model
        self.objective = objective
        self.fields = _concrete_fields(split)
        self.specs: dict[str, dict[str, Any]] = {}
        self.point_reads: set[str] = set()
        entries, arrays = list(split.entry_values), list(split.array_values)
        try:
            with _tracing_errors():
                self._discovering = True
                jax.eval_shape(lambda e, a: self._run(e, a, {}), entries, arrays)
                self._discovering = False
                outputs = {
                    n: jnp.zeros((), s["dtype"])
                    for s in self.specs.values()
                    for n in s["names"]
                }
                self.closed = jax.make_jaxpr(self._run)(entries, arrays, outputs)
        except _Unsupported as err:
            raise err.error from None
        out = self.closed.out_avals
        if (
            len(out) != 1
            or out[0].shape != ()
            or not jnp.issubdtype(out[0].dtype, jnp.floating)
        ):
            got = f"{out[0].dtype} of shape {out[0].shape}" if len(out) == 1 else out
            raise TypeError(f"{_REAL_ADVICE} Got {got}.")
        self.names = list(outputs)
        self.n_entries, self.n_arrays = len(entries), len(arrays)
        self.pruned, self.used = pe.dce_jaxpr(
            self.closed.jaxpr, [True], instantiate=False
        )
        self.live = [i for i, u in enumerate(self.used) if u]

    def _run(
        self,
        entries: Sequence[jax.Array],
        arrays: Sequence[jax.Array],
        outputs: Mapping[str, jax.Array],
    ) -> Any:
        """The objective on the inputs; ``outputs`` ``{}`` discovers them."""
        split = self.split
        res, labels = split.build(entries, arrays)
        recorder = Recorder(
            labels,
            split.field_kinds,
            self.fields,
            self.model,
            discover=self._discovering,
            specs=self.specs,
            outputs=outputs,
            constants={
                id(arrays[i]): split.array_values[i] for i in split.port_mode_leaves
            },
        )
        token = _RECORDER.set(recorder)
        try:
            return self.objective(res, self.model)
        finally:
            _RECORDER.reset(token)
            self.point_reads |= recorder.point_reads

    def _is_entry(self, i: int) -> bool:
        return i < self.n_entries

    def _is_output(self, i: int) -> bool:
        return i >= self.n_entries + self.n_arrays

    def kind(self) -> str:
        """The estimate kind of the live inputs; raises for a refused one."""
        split = self.split
        _refuse_arrays(
            split,
            [
                i - self.n_entries
                for i in self.live
                if not (self._is_entry(i) or self._is_output(i))
            ],
            self.point_reads,
        )
        kinds = {split.entry_kinds[i] for i in self.live if self._is_entry(i)}
        for i in self.live:
            if self._is_output(i):
                spec = self.specs[self.name(i)[0]]
                field_kind = split.field_kinds[spec["label"]]
                kinds.add(MEASUREMENT_KINDS[spec["measurement"], field_kind])
        if not kinds:
            raise ValueError(
                "The objective reads nothing gfem solved: it is a constant of "
                "the result. Read the result's S, capacitance, energy, omega or "
                "participation, or a field through gfem.power_flow or "
                "gfem.region_mean, from the result argument the objective is "
                "given."
            )
        return _kind(kinds)

    def scalars(self) -> list[int]:
        """The live scalar inputs: entries and measurement outputs."""
        return [i for i in self.live if self._is_entry(i) or self._is_output(i)]

    def name(self, i: int) -> tuple[str, tuple[int, ...], bool]:
        """A scalar input's leaf (or output key), index, and floor absoluteness."""
        if self._is_entry(i):
            leaf, index = self.split.entries[i]
            return leaf, index, self.split.entry_absolute[i]
        key, _, component = self.names[i - self.n_entries - self.n_arrays].partition(
            "#"
        )
        return key, (int(component),), False

    def inputs(self) -> list[Any]:
        """The concrete inputs: the result's, and each measurement run once."""
        measured = {
            n: jnp.asarray(v)
            for spec in self.specs.values()
            for n, v in zip(
                spec["names"], np.ravel(np.asarray(spec["evaluate"]())), strict=True
            )
        }
        return [
            *self.split.entry_values,
            *self.split.array_values,
            *(measured[n] for n in self.names),
        ]


_REAL_ADVICE = (
    "An objective is one real number: return a real 0-d value, for example "
    "its .real, jnp.abs(...) or jnp.abs(...) ** 2."
)


def _value(objective: Callable[[Any, Any], Any], result: Any, model: Any) -> float:
    """The objective's value, eagerly; its type and outer tracers checked."""
    out = objective(result, model)
    if is_traced(out) or is_staging():
        raise NonDifferentiableError(
            "The objective's value carries a tracer of an enclosing jax.grad, "
            "jax.jit or jax.vmap: mesh refinement estimates a fixed design, "
            "which has no derivative. Estimate outside the transformed "
            "function, with concrete values."
        )
    a = np.asarray(out)
    if a.shape != () or np.iscomplexobj(a) or not np.issubdtype(a.dtype, np.number):
        raise TypeError(f"{_REAL_ADVICE} Got a {a.dtype} value of shape {a.shape}.")
    return float(a)


class _Unsupported(Exception):  # noqa: N818 - carries an error out of a JAX trace
    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


class _tracing_errors:  # noqa: N801 - a context manager used as a block
    """Turn a JAX tracing error raised by the objective into `TypeError`."""

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        tb: types.TracebackType | None,
    ) -> bool:
        if error is None:
            return False
        if isinstance(error, MeshRefinementUnsupportedError):
            raise _Unsupported(error)
        if isinstance(
            error,
            (
                jax.errors.ConcretizationTypeError,
                jax.errors.TracerBoolConversionError,
                jax.errors.TracerArrayConversionError,
                jax.errors.TracerIntegerConversionError,
            ),
        ):
            raise TypeError(_TRACE_ADVICE) from error
        return False


def _kind(kinds: set[str]) -> str:
    if len(kinds) == 1:
        return next(iter(kinds))
    if kinds == _MIXTURE[0]:
        return _MIXTURE[1]
    raise MeshRefinementUnsupportedError(
        "mixed objective",
        f"The objective reads quantities of different estimate kinds "
        f"({', '.join(sorted(kinds))}), which one estimate does not combine. "
        "Estimate each as its own objective, together in a tuple of "
        "QuantityEstimators.",
    )


def _refuse_arrays(split: _Split, live: list[int], point_reads: set[str]) -> None:
    """Raise for a live field or refused leaf (ADR 0143, ticket 11 Q10)."""
    for i in live:
        path = split.arrays[i]
        label = path.removesuffix(".coefficients")
        if label in split.field_kinds:
            if split.field_kinds[label] == "mode":
                raise MeshRefinementUnsupportedError(
                    "normalized mode field",
                    f"The objective reads the eigen mode field {label}, whose "
                    f"normalization makes its error one no estimate covers. {_REMEDY}",
                )
            if label in point_reads:
                raise MeshRefinementUnsupportedError(
                    "point value",
                    f"The objective reads {label} at points (Field.at), and a "
                    "point value has no estimate (ADR 0133): read the mean "
                    "over a small region with gfem.region_mean.",
                )
            raise MeshRefinementUnsupportedError(
                "field coefficients",
                f"The objective reads the coefficients of {label} directly. "
                "Read fields through gfem.power_flow or gfem.region_mean, "
                "whose duals the estimate knows.",
            )
        feature = next(f for p, f in REFUSED_LEAVES.items() if path.startswith(p))
        raise MeshRefinementUnsupportedError(
            feature,
            f"The objective reads {path.lstrip('.')}, and {feature} has no "
            f"estimate in this version of gfem. {_REMEDY}",
        )


def _refuse_not_finite(result: Any, kind: str, value: float) -> None:
    if kind == "objective of s":
        s = result.s
        undefined = [
            p
            for i, p in enumerate(s.ports)
            if np.isnan(np.asarray(s.z_ref)[:, i]).any()
        ]
        if undefined:
            raise MeshRefinementUnsupportedError(
                "undefined S row",
                "The objective reads S of a port without a resistance "
                f"({', '.join(map(repr, undefined))}): its row of S is "
                "undefined (NaN), as it is for the gradient. Give the port a "
                "resistance, or leave its row out.",
            )
    raise ValueError(
        f"The objective is {value} at the result, not a finite number, so it "
        "has no error to estimate. (y() and z() of a wave port result are NaN: "
        "a wave port has no reference impedance.)"
    )


# --- jaxpr structure: branches and custom derivatives ------------------------------


def _sub_jaxprs(eqn: Any) -> list[Any]:
    """The closed jaxprs a call-like equation runs on its own operands."""
    name = eqn.primitive.name
    if name in ("pjit", "jit", "closed_call", "core_call", "remat", "checkpoint"):
        sub = eqn.params.get("jaxpr", eqn.params.get("call_jaxpr"))
        return [sub] if sub is not None else []
    return []


def _walk(jaxpr: Any, dependent: list[bool]) -> list[bool]:
    """Forward dependence on read values through ``jaxpr``, raising on the way.

    Raises for a branch on a read value or a custom derivative (ADR 0143
    amendment). On the objective's pruned jaxpr every input is read.
    """
    dep: dict[Any, bool] = {}

    def read(v: Any) -> bool:
        return False if isinstance(v, jcore.Literal) else dep.get(v, False)

    dep.update(zip(jaxpr.invars, dependent, strict=True))
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        ins = [read(v) for v in eqn.invars]
        if name.startswith(("custom_jvp_call", "custom_vjp_call")) and any(ins):
            raise MeshRefinementUnsupportedError(
                "custom derivative in objective",
                "The objective holds a jax.custom_jvp or jax.custom_vjp rule "
                "of its own on what it reads. The estimate needs the true "
                "derivative of the objective's value, and a user rule was not "
                "verified: write the objective without it.",
            )
        branching = (
            (name in _BRANCHING and any(ins))
            or (name == "select_n" and ins[0])
            or (name == "cond" and ins[0])
            or (name == "while" and any(ins))
        )
        if branching:
            raise MeshRefinementUnsupportedError(
                "non-smooth objective",
                f"The objective branches on a value it reads ({name}: max, "
                "min, sort, or where/cond on a predicate of the result). At a "
                "near-tie its derivative follows one branch only, and the "
                "other's error is missed: no estimate is verified for that. "
                "Estimate each branch as its own objective.",
            )
        subs = _sub_jaxprs(eqn)
        if subs:
            outs = _walk(subs[0].jaxpr, ins)
        elif name == "cond":
            outs = [any(ins)] * len(eqn.outvars)
            for branch in eqn.params["branches"]:
                _walk(branch.jaxpr, ins[1:])
        else:
            outs = [any(ins)] * len(eqn.outvars)
        dep.update(zip(eqn.outvars, outs, strict=True))
    return [read(v) for v in jaxpr.outvars]


# --- values: stop_gradient as identity, cotangents, NaN reach, curvature -----------


def _identity_stop_gradient(closed: Any) -> Any:
    """``closed`` with every ``stop_gradient`` (in nested calls too) the identity."""

    def run(jaxpr: Any, consts: Sequence[Any], *args: Any) -> list[Any]:
        env: dict[Any, Any] = {}

        def read(v: Any) -> Any:
            return v.val if isinstance(v, jcore.Literal) else env[v]

        env.update(
            zip([*jaxpr.constvars, *jaxpr.invars], [*consts, *args], strict=True)
        )
        for eqn in jaxpr.eqns:
            ins = [read(v) for v in eqn.invars]
            subs = _sub_jaxprs(eqn)
            if eqn.primitive.name == "stop_gradient":
                outs = ins
            elif subs:
                outs = run(subs[0].jaxpr, subs[0].consts, *ins)
            else:
                params = eqn.primitive.get_bind_params(eqn.params)
                with eqn.ctx.manager:
                    outs = eqn.primitive.bind(*ins, **params)
                if not eqn.primitive.multiple_results:
                    outs = [outs]
            env.update(zip(eqn.outvars, outs, strict=True))
        return [read(v) for v in jaxpr.outvars]

    avals = [
        jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype) for v in closed.jaxpr.invars
    ]
    return jax.make_jaxpr(lambda *a: run(closed.jaxpr, closed.consts, *a)[0])(*avals)


def _evaluate(closed: Any, *args: Any) -> jax.Array:
    return jcore.eval_jaxpr(closed.jaxpr, closed.consts, *args)[0]


def _cotangents(closed: Any, flat: list[Any], scalars: list[int]) -> dict[int, complex]:
    """The cotangent of each scalar input of ``scalars`` (gfem's contract)."""
    value, pullback = jax.vjp(lambda *xs: _evaluate(closed, *xs), *flat)
    cts = pullback(jnp.ones((), value.dtype))
    out = {}
    for i in scalars:
        ct = complex(np.asarray(cts[i]))
        out[i] = ct if np.isfinite(ct) else 0j
    return out


def _nan_reads(trace: _Trace, flat: list[Any], candidates: list[int]) -> list[int]:
    """The candidate inputs whose NaN reaches the pruned objective's value.

    Vectorized over the candidates and jitted: a loop over them is quadratic
    in the entries (ticket 20's scaling).
    """
    if not candidates:
        return []
    keep = trace.live
    position = {i: keep.index(i) for i in candidates}
    args = [flat[i] for i in keep]

    def one(k: jax.Array) -> jax.Array:
        xs = list(args)
        for j, i in enumerate(candidates):
            p = position[i]
            xs[p] = jnp.where(k == j, jnp.full_like(xs[p], jnp.nan), xs[p])
        return jcore.eval_jaxpr(trace.pruned, trace.closed.consts, *xs)[0]

    reached = jax.jit(jax.vmap(one))(jnp.arange(len(candidates)))
    return [
        i for i, r in zip(candidates, np.isnan(np.asarray(reached)), strict=True) if r
    ]


def _curvature(
    closed: Any, flat: list[Any], read: list[int]
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """``||H_ab||_2`` over the (Re, Im) blocks of the read scalars ``read``.

    Also each read's own block ``H_aa`` (`Objective.diagonal`).
    """
    if not read:
        return np.zeros((0, 0)), ()
    is_complex = [bool(jnp.iscomplexobj(flat[i])) for i in read]
    widths = [2 if c else 1 for c in is_complex]
    starts = np.concatenate([[0], np.cumsum(widths)])

    def real_function(v: jax.Array) -> jax.Array:
        xs = list(flat)
        for k, i in enumerate(read):
            s = starts[k]
            if is_complex[k]:
                xs[i] = (v[s] + 1j * v[s + 1]).astype(flat[i].dtype)
            else:
                xs[i] = v[s].astype(flat[i].dtype)
        return jnp.real(_evaluate(closed, *xs))

    v0 = []
    for k, i in enumerate(read):
        x = jnp.asarray(flat[i])
        v0 += [jnp.real(x), jnp.imag(x)] if is_complex[k] else [jnp.real(x)]
    hessian = np.asarray(
        jax.jit(jax.hessian(real_function))(jnp.stack(v0).astype(float))
    )
    n = len(read)
    out = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            block = hessian[starts[a] : starts[a + 1], starts[b] : starts[b + 1]]
            out[a, b] = np.linalg.norm(block, 2)
    diagonal = tuple(
        np.nan_to_num(hessian[starts[a] : starts[a + 1], starts[a] : starts[a + 1]])
        for a in range(n)
    )
    return np.nan_to_num(out), diagonal


# --- the mode-aligned view ------------------------------------------------------------

#: The `EigenmodeResult` attributes with one entry per mode along axis 0.
_MODE_AXIS = ("omega", "error", "cluster")
_MODE_MAPS = (
    "q_port",
    "participation",
    "bulk_participation",
    "principal_participation",
    "sheet_participation",
)


def aligned(result: Any, modes: Mapping[int, int]) -> Any:
    """An eigenmode result with its modes in the first mesh's order (ADR 0143).

    ``modes`` maps each mode index read on the first mesh to the index of
    the mode followed from it on ``result``'s mesh: that mode moves to the
    index; the other modes keep the solver's order in the remaining slots.
    """
    n = len(np.asarray(result.omega))
    order = _alignment(n, modes)
    if order == list(range(n)):
        return result
    take = np.asarray(order)

    def permute(x: Any) -> Any:
        return None if x is None else jnp.asarray(x)[take]

    changes: dict[str, Any] = {
        name: permute(getattr(result, name)) for name in _MODE_AXIS
    }
    for name in _MODE_MAPS:
        changes[name] = {k: permute(v) for k, v in getattr(result, name).items()}
    changes["e_fields"] = (
        tuple(result.e_fields[i] for i in order) if result.e_fields else ()
    )
    return dataclasses.replace(result, **changes)


def _alignment(n: int, modes: Mapping[int, int]) -> list[int]:
    """Solver index at each aligned position."""
    order: list[int | None] = [None] * n
    for start, index in modes.items():
        order[start] = index
    rest: Iterator[int] = iter(i for i in range(n) if i not in set(modes.values()))
    return [next(rest) if i is None else i for i in order]
