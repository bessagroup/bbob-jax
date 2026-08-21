"""Tests for the CEC 2013 LSGO suite (F1-F15).

Validates the JAX port against reference regression pins
(``data/cec2013lsgo_golden.npz``, generated from MetaBox's NumPy reference
by ``scripts/crosscheck_cec2013lsgo.py``; re-confirm against the C-backed
``dmolina`` oracle with that script), plus the structural contract every
suite must satisfy: scalar output, JIT / vmap / grad, ``fn(x_opt) == f_opt``
(F14 excepted - see below), fixed-instance ``ndim`` validation, NaN
propagation and registry/tags/bounds key consistency.

LSGO is evaluated in float64: at 1000-D the elliptic 1e6 conditioning and
Schwefel 1.2 cumulative sums lose too many digits in float32 to compare
meaningfully against the reference.
"""

#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path

# Third-party
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

# Local
import bbob_jax as B

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

FIDS = list(range(1, 16))
NAMES = [f"cec2013lsgo_f{fid}" for fid in FIDS]
GOLDEN = np.load(Path(__file__).parent / "data" / "cec2013lsgo_golden.npz")


def _ndim(fid: int) -> int:
    """Native dimension of function ``fid`` (905 for the overlapping F13/F14)."""
    return 905 if fid in (13, 14) else 1000


@pytest.mark.parametrize("fid", FIDS)
def test_matches_reference_golden(fid):
    """Port reproduces the reference values within float64 tolerance."""
    xs = GOLDEN[f"x_{fid}"]
    f_ref = GOLDEN[f"f_{fid}"]
    with jax.enable_x64(True):
        fn, _ = B.cec2013lsgo_registry[NAMES[fid - 1]](
            ndim=_ndim(fid), key=jr.key(0)
        )
        f_jax = np.array([float(fn(jnp.asarray(x))) for x in xs])
    assert np.allclose(f_jax, f_ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("fid", FIDS)
def test_scalar_jit_vmap_grad(fid):
    """Output is scalar and JIT / vmap / grad all succeed with finite grads."""
    ndim = _ndim(fid)
    fn, _ = B.cec2013lsgo_registry[NAMES[fid - 1]](ndim=ndim, key=jr.key(0))
    lo, hi = B.cec2013lsgo_bounds[NAMES[fid - 1]]
    x = jr.uniform(jr.key(fid), (ndim,), minval=lo, maxval=hi)

    assert jnp.asarray(fn(x)).shape == ()
    assert jnp.isfinite(jax.jit(fn)(x))
    batch = jax.vmap(fn)(
        jr.uniform(jr.key(fid + 1), (4, ndim), minval=lo, maxval=hi)
    )
    assert batch.shape == (4,)
    grad = jax.grad(fn)(x)
    assert jnp.all(jnp.isfinite(grad))


@pytest.mark.parametrize("fid", FIDS)
def test_fn_at_x_opt(fid):
    """``fn(x_opt) == f_opt == 0`` for every function except F14.

    F14's conflicting-overlap subcomponents cannot be simultaneously zeroed,
    so its 0 optimum is a true lower bound that is never attained; there
    ``fn(x_opt) > 0`` by construction.
    """
    with jax.enable_x64(True):
        p = B.problem(NAMES[fid - 1], ndim=_ndim(fid), key=jr.key(0))
        value = float(p.fn(p.x_opt))
    assert float(p.f_opt) == 0.0
    if fid == 14:
        assert value > 0.0
    else:
        assert value == pytest.approx(0.0, abs=1e-6)


def test_ndim_validation_raises():
    """The fixed-instance maker rejects any non-native dimension."""
    with pytest.raises(ValueError, match="fixed-instance"):
        B.cec2013lsgo_registry["cec2013lsgo_f1"](ndim=500, key=jr.key(0))
    with pytest.raises(ValueError, match="fixed-instance"):
        B.cec2013lsgo_registry["cec2013lsgo_f13"](ndim=1000, key=jr.key(0))


@pytest.mark.parametrize("fid", FIDS)
def test_nan_propagation(fid):
    """NaN in the input propagates to a NaN output."""
    ndim = _ndim(fid)
    fn, _ = B.cec2013lsgo_registry[NAMES[fid - 1]](ndim=ndim, key=jr.key(0))
    x = jnp.zeros(ndim).at[0].set(jnp.nan)
    assert jnp.isnan(fn(x))


def test_registry_tags_bounds_consistency():
    """The three derived views share exactly the 15 LSGO names."""
    assert set(B.cec2013lsgo_registry) == set(NAMES)
    assert set(B.cec2013lsgo_function_characteristics) == set(NAMES)
    assert set(B.cec2013lsgo_bounds) == set(NAMES)


@pytest.mark.parametrize("fid", FIDS)
def test_tag_schema(fid):
    """Exactly one category flag is set; ``rotated`` matches the block set."""
    tags = B.cec2013lsgo_function_characteristics[NAMES[fid - 1]]
    categories = (
        "separable",
        "partially_separable",
        "overlapping",
        "non_separable",
    )
    assert sum(tags[c] for c in categories) == 1
    assert tags["rotated"] == (fid in (4, 5, 6, 7, 8, 9, 10, 11, 13, 14))
