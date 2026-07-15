"""BBOB-noisy noise models: Gaussian, uniform and Cauchy.

The three stochastic transforms of the BBOB-noisy suite
(`f101`-`f130`), replicating the legacy COCO reference
(``benchmarkshelper.c``: ``FGauss``, ``FUniform``, ``FCauchy``).
Each model disturbs the *residual* above the optimum — the base
function value before the boundary penalty and ``f_opt`` offset
are added — and applies the noise gate: residuals below ``TOL``
are returned undisturbed, everything else gets the disturbed
value plus ``1.01 * TOL``. This guarantees ``fn(x_opt, key) ==
f_opt`` exactly.

The current COCO revival (``transform_obj_*_noise.c``) drops the
gate and uses a linear boundary penalty; the legacy code and the
published definition agree with each other, so legacy semantics
are kept (code-wins rule, see ``docs/adr/0004``).
"""

#                                                                       Modules
# =============================================================================

# Third-party
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

TOL = 1e-8
"""Noise gate threshold: residuals below this stay undisturbed."""


def _gate(f_true: jax.Array, f_val: jax.Array) -> jax.Array:
    """Apply the noise gate (final adjustment) to a disturbed value.

    Parameters
    ----------
    f_true : jax.Array
        Undisturbed residual (base value above the optimum).
    f_val : jax.Array
        Disturbed residual, before the ``1.01 * TOL`` shift.

    Returns
    -------
    jax.Array
        ``f_true`` where ``f_true < TOL``, else
        ``f_val + 1.01 * TOL``.
    """
    return jnp.where(f_true < TOL, f_true, f_val + 1.01 * TOL)


def _gated_residual(f_true: jax.Array) -> jax.Array:
    """Clamp the residual used inside noise formulas to ``TOL``.

    Where ``f_true < TOL`` the disturbed branch of the gate is
    unselected, but ``jnp.where`` still differentiates through
    it; clamping keeps that dead branch free of ``0 ** x`` /
    division blow-ups that would poison gradients with NaNs.
    """
    return jnp.maximum(f_true, TOL)


def gauss_noise(
    f_true: jax.Array, key: PRNGKeyArray, beta: float
) -> jax.Array:
    """Multiplicative Gaussian (log-normal) noise.

    ``f * exp(beta * N(0, 1))``, gated. Moderate severity uses
    ``beta = 0.01``, severe uses ``beta = 1``.

    Parameters
    ----------
    f_true : jax.Array
        Undisturbed residual above the optimum.
    key : PRNGKeyArray
        JAX random key (one normal draw).
    beta : float
        Noise strength.

    Returns
    -------
    jax.Array
        Disturbed residual.
    """
    f_val = _gated_residual(f_true) * jnp.exp(beta * jr.normal(key, shape=()))
    return _gate(f_true, f_val)


def uniform_noise(
    f_true: jax.Array, key: PRNGKeyArray, alpha: float, beta: float
) -> jax.Array:
    """Multiplicative uniform noise.

    ``U1**beta * f * max(1, (1e9 / (f + 1e-99)) ** (alpha * U2))``,
    gated. Moderate severity uses ``alpha = 0.01 * (0.49 + 1/D)``
    and ``beta = 0.01``; severe uses ``alpha = 0.49 + 1/D`` and
    ``beta = 1``.

    Parameters
    ----------
    f_true : jax.Array
        Undisturbed residual above the optimum.
    key : PRNGKeyArray
        JAX random key (two uniform draws).
    alpha : float
        Exponent scale of the amplification factor.
    beta : float
        Exponent of the attenuation factor.

    Returns
    -------
    jax.Array
        Disturbed residual.
    """
    key1, key2 = jr.split(key)
    u1 = jr.uniform(key1, shape=())
    u2 = jr.uniform(key2, shape=())
    f_safe = _gated_residual(f_true)
    amplification = jnp.maximum(1.0, (1e9 / (f_safe + 1e-99)) ** (alpha * u2))
    f_val = u1**beta * f_safe * amplification
    return _gate(f_true, f_val)


def cauchy_noise(
    f_true: jax.Array, key: PRNGKeyArray, alpha: float, p: float
) -> jax.Array:
    """Additive, seldom-triggered Cauchy noise.

    ``f + alpha * max(0, 1e3 + I_{U < p} * N1 / |N2 + 1e-199|)``,
    gated. The ratio of two independent normals is standard
    Cauchy distributed, so with probability ``p`` a heavy-tailed
    outlier is added on top of the constant ``alpha * 1e3``
    offset. Moderate severity uses ``alpha = 0.01, p = 0.05``;
    severe uses ``alpha = 1, p = 0.2``.

    Parameters
    ----------
    f_true : jax.Array
        Undisturbed residual above the optimum.
    key : PRNGKeyArray
        JAX random key (two normal draws, one uniform draw).
    alpha : float
        Noise scale.
    p : float
        Probability of drawing the Cauchy outlier.

    Returns
    -------
    jax.Array
        Disturbed residual.
    """
    key1, key2, key3 = jr.split(key, 3)
    n1 = jr.normal(key1, shape=())
    n2 = jr.normal(key2, shape=())
    u = jr.uniform(key3, shape=())
    cauchy = n1 / jnp.abs(n2 + 1e-199)
    indicator = (u < p).astype(cauchy.dtype)
    f_val = _gated_residual(f_true) + alpha * jnp.maximum(
        0.0, 1e3 + indicator * cauchy
    )
    return _gate(f_true, f_val)
