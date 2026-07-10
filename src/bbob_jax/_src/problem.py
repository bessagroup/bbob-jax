"""Problem: one lookup for everything a benchmark consumer needs.

The registries return a bare ``(fn, f_opt)`` tuple; the
optimum location, bounds, tags and noise arity historically
had to be fetched separately (or dug out of the Partial's
keywords). :func:`problem` resolves all of them in one call
from the :class:`~bbob_jax._src.spec.FunctionSpec` table.
"""

#                                                                       Modules
# =============================================================================

# Standard
from collections.abc import Callable
from typing import NamedTuple

# Third-party
import jax
from jaxtyping import PRNGKeyArray

# Local
from bbob_jax._src.factories import _partial_keywords
from bbob_jax._src.spec import SPEC_BY_NAME

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class Problem(NamedTuple):
    """A fully-constructed benchmark problem instance.

    Attributes
    ----------
    name : str
        Registry key of the function.
    fn : Callable
        The benchmark function with all parameters bound.
        Called as ``fn(x)``, or ``fn(x, key)`` when ``noisy``
        is True.
    x_opt : jax.Array
        Location of the global minimum. For composition
        functions this is the first component's optimum; for
        ``rosenbrock_rotated`` and ``griewank_rosenbrock_f8f2``
        it can fall outside ``bounds`` for some rotations.
        For deterministic CEC 2005 compositions (F15-F25) the
        composition weighting is degenerate and ``fn(x_opt)``
        does not reach ``f_opt``.
    f_opt : jax.Array
        Global minimum value.
    bounds : tuple[float, float]
        Search-space box bounds.
    tags : dict[str, bool]
        Function characteristics (suite-specific schema).
    noisy : bool
        Whether ``fn`` takes a PRNG key as second argument.
    """

    name: str
    fn: Callable[..., jax.Array]
    x_opt: jax.Array
    f_opt: jax.Array
    bounds: tuple[float, float]
    tags: dict[str, bool]
    noisy: bool


def problem(
    name: str,
    ndim: int,
    key: PRNGKeyArray | None = None,
    deterministic: bool = False,
) -> Problem:
    """Construct a benchmark problem instance by name.

    Parameters
    ----------
    name : str
        Function name from either suite (e.g. ``"rastrigin"``
        or ``"f15"``).
    ndim : int
        Number of input dimensions.
    key : PRNGKeyArray or None, optional
        JAX random key for parameter generation. Required when
        ``deterministic`` is False; ignored otherwise.
    deterministic : bool, optional
        When True, construct the deterministic instance (zero
        shift, identity rotations, zero ``f_opt``).

    Returns
    -------
    Problem
        The constructed problem instance.

    Raises
    ------
    KeyError
        If ``name`` is not a known benchmark function.

    Examples
    --------
    >>> import jax.random as jr
    >>> p = problem("sphere", ndim=2, key=jr.key(0))
    >>> value = p.fn(p.x_opt)  # == p.f_opt
    """
    spec = SPEC_BY_NAME[name]
    fn, f_opt = spec.maker(ndim=ndim, key=key, deterministic=deterministic)
    kw = _partial_keywords(fn)
    x_opt = spec.x_opt_from(kw, ndim)
    return Problem(
        name=spec.name,
        fn=fn,
        x_opt=x_opt,
        f_opt=f_opt,
        bounds=spec.bounds,
        tags=dict(spec.tags),
        noisy=spec.tags.get("noise", False),
    )
