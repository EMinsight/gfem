r"""Diagonal maps of a pixel array in the eigenbasis of its second differences.

The second-difference Laplacian of a pixel array with reflecting (Neumann)
edges is diagonal in the orthonormal type-II cosine transform; with periodic
(wrapped) edges, in the discrete Fourier transform. A map that is a function
of that Laplacian (a screened-Poisson solve, a heat kernel) is then a gain
per mode: transform, multiply, transform back. The transforms act on separate
axes, so they commute, and an axis may reflect while the other wraps.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import fft


def laplacian_eigenvalues(
    shape: tuple[int, int], pitch: float, wrap: tuple[bool, bool]
) -> tuple[np.ndarray, np.ndarray]:
    r"""The eigenvalues of the negative second difference along each array axis.

    Along an axis of :math:`n` pixels of side :math:`h`, mode :math:`k` has
    :math:`\lambda_k = \left(\frac{2}{h} \sin \frac{\pi k}{2n}\right)^2` with
    reflecting edges and
    :math:`\lambda_k = \left(\frac{2}{h} \sin \frac{\pi k}{n}\right)^2` with
    wrapped ones; the eigenvalue of mode :math:`(k_y, k_x)` of the array is
    their sum.

    Args:
        shape: The array's shape, ``(ny, nx)``.
        pitch: Pixel side (m).
        wrap: Per array axis ``(y, x)``, whether its edges are one seam.

    Returns:
        The eigenvalues along y, ``(ny,)``, and along x, ``(nx,)`` (1/m²).
    """
    ky, kx = (
        (2 / pitch * np.sin(np.pi * np.arange(n) / (n if w else 2 * n))) ** 2
        for n, w in zip(shape, wrap, strict=True)
    )
    return ky, kx


def diagonal_map(
    gain: jax.Array, wrap: tuple[bool, bool]
) -> Callable[[jax.Array], jax.Array]:
    """The linear map that scales each mode of a pixel array by its gain.

    Args:
        gain: The gain of each mode, of the array's shape ``(ny, nx)``, in the
            order of `laplacian_eigenvalues`. It must be even in the
            wavenumber along a wrapped axis, so the result is real.
        wrap: Per array axis ``(y, x)``, whether its edges are one seam.

    Returns:
        The map, real arrays of shape ``(ny, nx)`` to the same. It is
        symmetric, so it is its own adjoint.
    """
    if not any(wrap):

        def apply(rho: jax.Array) -> jax.Array:
            return fft.idctn(gain * fft.dctn(rho, type=2, norm="ortho"), norm="ortho")

        return apply
    cosine = tuple(a for a in (0, 1) if not wrap[a])
    fourier = tuple(a for a in (0, 1) if wrap[a])

    def apply_wrapped(rho: jax.Array) -> jax.Array:
        if cosine:
            rho = fft.dctn(rho, type=2, norm="ortho", axes=cosine)
        rho = jnp.fft.ifftn(gain * jnp.fft.fftn(rho, axes=fourier), axes=fourier).real
        if cosine:
            rho = fft.idctn(rho, type=2, norm="ortho", axes=cosine)
        return rho

    return apply_wrapped


def function_of_laplacian(
    shape: tuple[int, int],
    pitch: float,
    periodic: tuple[bool, bool],
    gain: Callable[[np.ndarray], np.ndarray],
) -> Callable[[jax.Array], jax.Array]:
    r"""The map :math:`g(-\Delta_h)` of a pixel grid's arrays.

    Args:
        shape: The grid's shape, ``(ny, nx)``.
        pitch: Pixel side (m).
        periodic: Per grid axis ``(x, y)``, whether its edges are one seam.
        gain: The function :math:`g`, applied to the eigenvalues of
            `laplacian_eigenvalues` summed per mode, ``(ny, nx)`` (1/m²).

    Returns:
        The map of `diagonal_map`.
    """
    wrap = (periodic[1], periodic[0])  # array axes (y, x)
    ky, kx = laplacian_eigenvalues(shape, pitch, wrap)
    return diagonal_map(jnp.asarray(gain(ky[:, None] + kx[None, :])), wrap)
