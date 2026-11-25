"""Reference implementations of the adaptive sweep's snapshot steps.

The full-matrices SVD choice of the next sample and the one-column-at-a-time
basis extension: the plain forms the thin SVD and the block extension are
tested against.
"""

import numpy as np

from gfem._sweep.adaptive import DROP, MRI_RANK


def reference_next_sample(z, chosen, snapshots, size):
    """The full-matrices SVD choice: ``(sample index, rank-deficient)``."""
    columns = []
    for i, c in zip(chosen, snapshots, strict=True):
        padded = np.zeros((size, c.shape[1]), complex)
        padded[: c.shape[0]] = c
        flat = padded.reshape(-1)
        columns.append(np.concatenate([flat, 1j * z[i] * flat]))
    u = np.column_stack(columns)
    _, sigma, vh = np.linalg.svd(u)
    sigma = np.concatenate([sigma, np.zeros(len(chosen) - len(sigma))])
    m = len(chosen) - 1
    while m > 0 and sigma[m] < MRI_RANK * sigma[0]:
        m -= 1
    deficient = m < len(chosen) - 1
    q = vh[m].conj()
    zs = z[chosen]
    candidates = np.setdiff1d(np.arange(len(z)), chosen)
    denominator = np.abs((q[None, :] / (z[candidates, None] - zs[None, :])).sum(axis=1))
    return int(candidates[np.argmin(denominator)]), deficient


def per_column_extend(model, snapshot):
    """The one-column-at-a-time ``ReducedModel.extend``,"""
    columns = np.stack([snapshot.real, snapshot.imag], axis=2).reshape(
        snapshot.shape[0], -1
    )
    old = model.basis
    basis = old
    for column in columns.T:
        norm = np.linalg.norm(column)
        if norm == 0:
            continue
        w = column.copy()
        for _ in range(2):
            w -= basis @ (basis.T @ w)
        remaining = np.linalg.norm(w)
        if remaining <= DROP * norm:
            continue
        basis = np.column_stack([basis, w / remaining])
    new = basis[:, old.shape[1] :]
    if not new.shape[1]:
        return 0
    for k, reduced in enumerate(model.operators):
        product = np.asarray(model.system.apply(k, new))
        top = old.T @ product
        corner = new.T @ product
        corner = 0.5 * (corner + corner.T)
        model.operators[k] = np.block([[reduced, top], [top.T, corner]])
    model.excitation = [
        np.vstack([r, new.T @ t.vectors])
        for r, t in zip(model.excitation, model.system.excitation, strict=True)
    ]
    model.outputs = [
        np.hstack([r, t.project(new)])
        for r, t in zip(model.outputs, model.system.outputs, strict=True)
    ]
    model.corrections = [
        np.vstack([r, new.T @ t.vectors])
        for r, t in zip(model.corrections, model.system.corrections, strict=True)
    ]
    model.basis = basis
    return new.shape[1]
