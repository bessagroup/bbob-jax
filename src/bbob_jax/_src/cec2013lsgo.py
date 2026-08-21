"""JAX implementations of the CEC 2013 Large-Scale Global Optimization suite.

The 15 CEC 2013 LSGO functions (F1-F15) reimplemented in JAX:
differentiable, JIT-able and vmap-able, faithful to the reference NumPy
implementation.

Ported to JAX from MetaBox (BSD 3-Clause, (c) 2023 MetaEvolution Lab),
``MetaEvo/MetaBox@5565a28``, path
``src/environment/problem/SOO/CEC2013LSGO/cec2013lsgo_numpy.py``, which is
itself a copy of Daniel Molina's ``cec2013lsgo`` reference implementation.
Original benchmark: Li, Tang, Omidvar, Yang & Qin, *Benchmark Functions for
the CEC 2013 Special Session and Competition on Large-Scale Global
Optimization* (2013). Reference constants: see
``cec2013lsgo_data/PROVENANCE.md``.

Fixed-instance suite
--------------------
Unlike the CEC 2005 / 2017 suites, LSGO parameters are **fixed official
constants** loaded from ``cec2013lsgo_data/F{i}.npz`` — they are neither
sampled from a key nor resizable to an arbitrary ``ndim``. Each function
has its own native dimension (1000-D, or 905-D for the overlapping F13/F14).
The maker (:func:`bbob_jax._src.factories.make_cec2013lsgo`) therefore
validates ``ndim`` against the native dimension and ignores ``key``; there
is no ``deterministic`` variant. See ``CONTEXT.md``.

Categories (Li et al. 2013)
---------------------------
1. Fully separable            : F1, F2, F3
2. Partially separable        : F4-F7 (7 rotated + 1 separable remainder),
                                F8-F11 (20 rotated, no remainder)
3. Overlapping                : F12 (intrinsic), F13 (conforming),
                                F14 (conflicting)
4. Fully non-separable        : F15

Every function has ``f(x*) = 0`` (no bias). The optimum is attained at the
shift ``xopt`` for all functions **except F14**, whose conflicting
overlapping subcomponents cannot be simultaneously zeroed: for F14, 0 is a
true lower bound but is not attained at any point.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import importlib.resources
from collections.abc import Callable

# Third-party
import jax
import jax.numpy as jnp
import numpy as np

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = [
    "Martin van der Schelling",
    "MetaEvolution Lab",
    "Daniel Molina",
]
__status__ = "Stable"
# =============================================================================

_OVERLAP = 5  # F13/F14 subcomponent overlap width
_FULL_DIM = 1000
_OVERLAP_DIM = 905  # F13/F14 native dimension (1000 - 19 * _OVERLAP)

# Native dimensionality per function id.
NATIVE_DIM: dict[int, int] = {i: _FULL_DIM for i in range(1, 16)}
NATIVE_DIM[13] = _OVERLAP_DIM
NATIVE_DIM[14] = _OVERLAP_DIM

# Search-space bounds per function id (Li et al. 2013, Table).
BOUNDS: dict[int, tuple[float, float]] = {
    1: (-100.0, 100.0),
    2: (-5.0, 5.0),
    3: (-32.0, 32.0),
    4: (-100.0, 100.0),
    5: (-5.0, 5.0),
    6: (-32.0, 32.0),
    7: (-100.0, 100.0),
    8: (-100.0, 100.0),
    9: (-5.0, 5.0),
    10: (-32.0, 32.0),
    11: (-100.0, 100.0),
    12: (-100.0, 100.0),
    13: (-100.0, 100.0),
    14: (-100.0, 100.0),
    15: (-100.0, 100.0),
}

# Category tag per function id (see module docstring).
CATEGORY: dict[int, str] = {
    1: "separable",
    2: "separable",
    3: "separable",
    4: "partially_separable",
    5: "partially_separable",
    6: "partially_separable",
    7: "partially_separable",
    8: "partially_separable",
    9: "partially_separable",
    10: "partially_separable",
    11: "partially_separable",
    12: "overlapping",
    13: "overlapping",
    14: "overlapping",
    15: "non_separable",
}


#                                              Base transfer functions (blocks)
# =============================================================================
# Each acts on the (possibly rotated/transformed) subcomponent vector; the
# index-dependent ones (elliptic) use the LOCAL width, which is automatic
# because they are called on the subcomponent-sized vector.


def _sphere(z: jax.Array) -> jax.Array:
    """Sphere function, ``sum(z**2)``."""
    return jnp.sum(z**2)


def _elliptic(z: jax.Array) -> jax.Array:
    """Ill-conditioned elliptic function (condition number 1e6)."""
    nx = z.shape[-1]
    i = jnp.arange(nx)
    return jnp.sum(10.0 ** (6.0 * i / (nx - 1)) * z**2)


def _rastrigin(z: jax.Array) -> jax.Array:
    """Rastrigin function, ``sum(z**2 - 10 cos(2 pi z) + 10)``."""
    return jnp.sum(z**2 - 10.0 * jnp.cos(2.0 * jnp.pi * z) + 10.0)


def _ackley(z: jax.Array) -> jax.Array:
    """Ackley function (with the ``+20 + e`` constant, so ``f(0) = 0``)."""
    nx = z.shape[-1]
    sum1 = jnp.sum(z**2)
    sum2 = jnp.sum(jnp.cos(2.0 * jnp.pi * z))
    return (
        -20.0 * jnp.exp(-0.2 * jnp.sqrt(sum1 / nx))
        - jnp.exp(sum2 / nx)
        + 20.0
        + jnp.e
    )


def _schwefel(z: jax.Array) -> jax.Array:
    """Schwefel 1.2 function, ``sum(cumsum(z)**2)`` (order-sensitive)."""
    return jnp.sum(jnp.cumsum(z) ** 2)


def _rosenbrock(z: jax.Array) -> jax.Array:
    """Rosenbrock function over the shifted vector."""
    x0 = z[:-1]
    x1 = z[1:]
    t = x0**2 - x1
    return jnp.sum(100.0 * t**2 + (x0 - 1.0) ** 2)


#                                           Irregularity / asymmetry transforms
# =============================================================================
# Faithful to the reference; guarded with ``jnp.where`` so the forward value
# matches NumPy exactly and gradients stay finite on the search domain.


def _t_osz(z: jax.Array) -> jax.Array:
    """T_osz smooth log/sine deformation (elementwise, no index dependence)."""
    nonzero = z != 0.0
    z_safe = jnp.where(nonzero, jnp.abs(z), 1.0)
    hat = jnp.where(nonzero, jnp.log(z_safe), 0.0)
    c1 = jnp.where(z > 0.0, 10.0, 5.5)
    c2 = jnp.where(z > 0.0, 7.9, 3.1)
    return jnp.sign(z) * jnp.exp(
        hat + 0.049 * (jnp.sin(c1 * hat) + jnp.sin(c2 * hat))
    )


def _t_asy(z: jax.Array, beta: float = 0.2) -> jax.Array:
    """T_asy asymmetry (index-dependent; only strictly positive entries)."""
    d = z.shape[-1]
    i = jnp.arange(d)
    pos = z > 0.0
    base = jnp.where(pos, z, 1.0)
    exponent = 1.0 + beta * (i / (d - 1)) * jnp.sqrt(base)
    return jnp.where(pos, base**exponent, z)


def _lambda(z: jax.Array, alpha: float = 10.0) -> jax.Array:
    """Lambda^alpha diagonal ill-conditioning."""
    d = z.shape[-1]
    exponents = 0.5 * jnp.arange(d) / (d - 1)
    return z * (alpha**exponents)


# Transform pipelines (composed left-to-right in application order).
def _osz(z: jax.Array) -> jax.Array:
    """osz only."""
    return _t_osz(z)


def _osz_asy(z: jax.Array) -> jax.Array:
    """osz then asy."""
    return _t_asy(_t_osz(z))


def _osz_asy_lambda(z: jax.Array) -> jax.Array:
    """osz then asy then Lambda."""
    return _lambda(_t_asy(_t_osz(z)))


#                                             Function evaluators (f1-f15)
# =============================================================================
# Fully-separable / intrinsic functions take only the shift; the
# partially-separable and overlapping functions share one block-loop
# evaluator, differing only in the (statically precomputed) block indices,
# per-block offsets and rotations bound by the maker.


def f1(x: jax.Array, *, xopt: jax.Array) -> jax.Array:
    """F1 - shifted elliptic (fully separable)."""
    return _elliptic(_osz(x - xopt))


def f2(x: jax.Array, *, xopt: jax.Array) -> jax.Array:
    """F2 - shifted Rastrigin (fully separable)."""
    return _rastrigin(_osz_asy_lambda(x - xopt))


def f3(x: jax.Array, *, xopt: jax.Array) -> jax.Array:
    """F3 - shifted Ackley (fully separable)."""
    return _ackley(_osz_asy_lambda(x - xopt))


def f12(x: jax.Array, *, xopt: jax.Array) -> jax.Array:
    """F12 - shifted Rosenbrock (intrinsic overlap)."""
    return _rosenbrock(x - xopt)


def f15(x: jax.Array, *, xopt: jax.Array) -> jax.Array:
    """F15 - shifted Schwefel 1.2 (fully non-separable)."""
    return _schwefel(_osz_asy(x - xopt))


def _make_block_evaluator(
    sub_transfer: Callable[[jax.Array], jax.Array],
    sub_transforms: Callable[[jax.Array], jax.Array],
    rem_transfer: Callable[[jax.Array], jax.Array],
    rem_transforms: Callable[[jax.Array], jax.Array],
) -> Callable[..., jax.Array]:
    """Build a block-loop evaluator for a partially-separable function.

    The returned function is called (after the maker binds its arrays) as
    ``fn(x)`` and evaluates
    ``sum_i w_i * sub_transfer(sub_transforms(R_i @ (x[idx_i] - off_i)))``
    plus, when ``rem_idx`` is not ``None``, a weight-1 separable remainder.

    Parameters
    ----------
    sub_transfer, sub_transforms : Callable
        Transfer function and transform pipeline applied per rotated block.
    rem_transfer, rem_transforms : Callable
        Transfer function and transform pipeline for the separable
        remainder (used only by F4-F7; ignored when ``rem_idx`` is None).

    Returns
    -------
    Callable
        Evaluator with signature ``fn(x, *, sub_idx, sub_off, sub_rot,
        weights, rem_idx, rem_off)``.
    """

    def evaluate(
        x: jax.Array,
        *,
        sub_idx: tuple[jax.Array, ...],
        sub_off: tuple[jax.Array, ...],
        sub_rot: tuple[jax.Array, ...],
        weights: jax.Array,
        rem_idx: jax.Array | None,
        rem_off: jax.Array | None,
    ) -> jax.Array:
        total = jnp.asarray(0.0, dtype=x.dtype)
        for i in range(len(sub_idx)):
            z = sub_rot[i] @ (x[sub_idx[i]] - sub_off[i])
            z = sub_transforms(z)
            total = total + weights[i] * sub_transfer(z)
        if rem_idx is not None:
            assert rem_off is not None  # paired with rem_idx
            z = rem_transforms(x[rem_idx] - rem_off)
            total = total + rem_transfer(z)
        return total

    return evaluate


# F4/F8 elliptic; F5/F9 Rastrigin; F6/F10 Ackley; F7/F11/F13/F14 Schwefel.
# F7 uses a *sphere* separable remainder (osz+asy, no Lambda) - a code-wins
# detail; F4-F6 remainders re-apply the block pipeline.
f4 = _make_block_evaluator(_elliptic, _osz, _elliptic, _osz)
f5 = _make_block_evaluator(
    _rastrigin, _osz_asy_lambda, _rastrigin, _osz_asy_lambda
)
f6 = _make_block_evaluator(_ackley, _osz_asy_lambda, _ackley, _osz_asy_lambda)
f7 = _make_block_evaluator(_schwefel, _osz_asy, _sphere, _osz_asy)
f8 = _make_block_evaluator(_elliptic, _osz, _elliptic, _osz)
f9 = _make_block_evaluator(
    _rastrigin, _osz_asy_lambda, _rastrigin, _osz_asy_lambda
)
f10 = _make_block_evaluator(_ackley, _osz_asy_lambda, _ackley, _osz_asy_lambda)
f11 = _make_block_evaluator(_schwefel, _osz_asy, _sphere, _osz_asy)
f13 = _make_block_evaluator(_schwefel, _osz_asy, _sphere, _osz_asy)
f14 = _make_block_evaluator(_schwefel, _osz_asy, _sphere, _osz_asy)

_FUNCS: dict[int, Callable] = {
    1: f1,
    2: f2,
    3: f3,
    4: f4,
    5: f5,
    6: f6,
    7: f7,
    8: f8,
    9: f9,
    10: f10,
    11: f11,
    12: f12,
    13: f13,
    14: f14,
    15: f15,
}


#                                          Data loading + instance build
# =============================================================================


def _load_raw(function_id: int) -> dict[str, np.ndarray]:
    """Load the vendored ``.npz`` constants for one function (NumPy)."""
    ref = (
        importlib.resources.files("bbob_jax._src.cec2013lsgo_data")
        / f"F{function_id}.npz"
    )
    with importlib.resources.as_file(ref) as path:
        with np.load(path) as data:
            return {key: data[key] for key in data.files}


def _rotation_for(raw: dict[str, np.ndarray], sub_dim: int) -> np.ndarray:
    """Pick the rotation matrix matching a subcomponent width (25/50/100)."""
    return raw[f"R{sub_dim}"]


def _block_indices(
    perm: np.ndarray, sizes: np.ndarray, overlap: int
) -> list[np.ndarray]:
    """Slice the permutation into (possibly overlapping) block index arrays.

    ``overlap == 0`` gives the contiguous non-overlapping blocks of F4-F11;
    ``overlap == 5`` gives the conforming/conflicting windows of F13/F14
    (``start = c - i*overlap``, ``end = c + s_i - i*overlap``).
    """
    blocks = []
    c = 0
    for i, s_i in enumerate(sizes):
        start = c - i * overlap
        end = c + s_i - i * overlap
        blocks.append(perm[start:end])
        c += s_i
    return blocks


def lsgo_instance(
    function_id: int,
) -> tuple[Callable, dict, int]:
    """Build a ready-to-bind LSGO instance for one function.

    Loads the vendored constants, converts them to JAX arrays (at the
    configured precision), constructs the block indices / offsets /
    rotations, and returns everything the maker needs. The global optimum
    location is not returned here; :func:`problem` reconstructs it from the
    bound keywords via the spec's ``x_opt_from`` resolver.

    Parameters
    ----------
    function_id : int
        LSGO function id in ``1..15``.

    Returns
    -------
    fn : Callable
        The (unbound) evaluator from ``_FUNCS``.
    kwargs : dict
        Keyword arguments to bind into a ``Partial`` (e.g. ``xopt`` for the
        separable functions; ``sub_idx``/``sub_off``/``sub_rot``/``weights``
        /``rem_idx``/``rem_off`` for the block functions).
    native_dim : int
        The only ``ndim`` this function accepts (1000, or 905 for F13/F14).
    """
    raw = _load_raw(function_id)
    native_dim = NATIVE_DIM[function_id]
    fn = _FUNCS[function_id]

    # Fully-separable / intrinsic-overlap / non-separable: shift only.
    if function_id in (1, 2, 3, 12, 15):
        xopt = jnp.asarray(raw["xopt"])
        return fn, {"xopt": xopt}, native_dim

    # Block functions (F4-F11, F13, F14).
    sizes = raw["s"].astype(int)
    perm = raw["p"].astype(int) - 1  # stored 1-indexed; de-index here
    overlap = _OVERLAP if function_id in (13, 14) else 0
    blocks = _block_indices(perm, sizes, overlap)

    if function_id == 14:
        # Conflicting overlap: each block subtracts its OWN optimum, split
        # from the 1000-value xopt into per-block chunks; no global shift.
        chunks = []
        c_o = 0
        for s_i in sizes:
            chunks.append(raw["xopt"][c_o : c_o + s_i])
            c_o += s_i
        offsets = chunks
    else:
        # Global shift: each block subtracts xopt at its own indices.
        offsets = [raw["xopt"][idx] for idx in blocks]

    sub_idx = tuple(jnp.asarray(idx) for idx in blocks)
    sub_off = tuple(jnp.asarray(off) for off in offsets)
    sub_rot = tuple(jnp.asarray(_rotation_for(raw, int(s_i))) for s_i in sizes)
    weights = jnp.asarray(raw["w"])

    # Separable remainder only for F4-F7 (indices past the blocks).
    if function_id in (4, 5, 6, 7):
        c = int(sizes.sum())
        rem = perm[c:_FULL_DIM]
        rem_idx = jnp.asarray(rem)
        rem_off = jnp.asarray(raw["xopt"][rem])
    else:
        rem_idx = None
        rem_off = None

    kwargs = {
        "sub_idx": sub_idx,
        "sub_off": sub_off,
        "sub_rot": sub_rot,
        "weights": weights,
        "rem_idx": rem_idx,
        "rem_off": rem_off,
    }
    return fn, kwargs, native_dim
