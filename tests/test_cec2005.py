"""Tests for CEC 2005 benchmark functions.

These tests validate JAX compatibility (jit, vmap, grad) and basic sanity
checks. They do NOT validate that results match official CEC 2005 data files —
see docs/superpowers/specs/2026-03-17-cec2005-jax-design.md for the rationale.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from bbob_jax import cec2005_registry, cec2005_registry_original
from bbob_jax._src.cec2005_tags import cec2005_function_characteristics
from bbob_jax._src.utils import _create_mesh

_NOISY = {
    name
    for name, chars in cec2005_function_characteristics.items()
    if chars.get("noise", False)
}

pytest_cec2005 = [
    pytest.param(name, fn, id=f"cec2005_registry::{name}")
    for name, fn in cec2005_registry.items()
]
pytest_cec2005_original = [
    pytest.param(name, fn, id=f"cec2005_registry_original::{name}")
    for name, fn in cec2005_registry_original.items()
]
all_cec2005 = pytest_cec2005 + pytest_cec2005_original

dimensions = [2, 3, 5, 10, 20]

COMPOSITION_NAMES = [f"f{i}" for i in range(15, 26)]


def _condition_number(matrix: jax.Array) -> jax.Array:
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    return singular_values[0] / singular_values[-1]


@pytest.mark.parametrize("name,fn", all_cec2005)
@pytest.mark.parametrize("dim", dimensions)
def test_function_output(name, fn, dim):
    key = jr.key(0)
    x = jr.uniform(key, shape=(dim,), minval=-100.0, maxval=100.0)
    try:
        fn_func, _ = fn(ndim=dim, key=key)
        y = fn_func(x, jr.key(42)) if name in _NOISY else fn_func(x)
    except NotImplementedError:
        pytest.skip(f"Function {name} not yet implemented")
    except Exception as e:
        pytest.fail(f"Function {name} raised an exception: {e}")
    assert jnp.isfinite(y), f"{name} returned non-finite value: {y}"
    assert jnp.ndim(y) == 0, f"{name} did not return scalar: {y.shape}"


@pytest.mark.parametrize("name,fn", all_cec2005)
@pytest.mark.parametrize("dim", dimensions)
def test_function_output_jit(name, fn, dim):
    key = jr.key(0)
    x = jr.uniform(key, shape=(dim,), minval=-100.0, maxval=100.0)
    try:
        fn_func, _ = fn(ndim=dim, key=key)
        if name in _NOISY:
            y = jax.jit(fn_func)(x, jr.key(42))
        else:
            y = jax.jit(fn_func)(x)
    except NotImplementedError:
        pytest.skip(f"Function {name} JIT not yet implemented")
    except Exception as e:
        pytest.fail(f"Function {name} JIT raised an exception: {e}")
    assert jnp.isfinite(y), f"{name} JIT returned non-finite: {y}"
    assert jnp.ndim(y) == 0


@pytest.mark.parametrize("name,fn", pytest_cec2005)
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("seed", [1, 2])
def test_function_vmap(name, fn, dim, seed):
    key = jr.key(seed)
    try:
        fn_func, _ = fn(ndim=dim, key=key)
        if name in _NOISY:
            eval_fn = lambda x: fn_func(x, jr.key(42))
        else:
            eval_fn = fn_func
        _, _, Z = _create_mesh(eval_fn, bounds=(-100.0, 100.0), px=50)
    except NotImplementedError:
        pytest.skip(f"Function {name} vmap not yet implemented")
    except Exception as e:
        pytest.fail(f"Function {name} vmap raised an exception: {e}")
    assert jnp.all(jnp.isfinite(Z)), f"{name} vmap returned non-finite values"


@pytest.mark.parametrize("name,fn", pytest_cec2005)
@pytest.mark.parametrize("dim", dimensions)
@pytest.mark.parametrize("seed", [1, 2])
def test_function_grad(name, fn, dim, seed):
    key = jr.key(seed)
    key_x, key_fn = jr.split(key)
    x = jr.uniform(key_x, shape=(10, dim), minval=-100.0, maxval=100.0)
    try:
        fn_func, _ = fn(ndim=dim, key=key_fn)
        if name in _NOISY:
            grad_fn = jax.grad(lambda x: fn_func(x, jr.key(42)))
        else:
            grad_fn = jax.grad(fn_func)
        grad_value = jax.vmap(grad_fn)(x)
    except NotImplementedError:
        pytest.skip(f"Function {name} grad not yet implemented")
    except Exception as e:
        pytest.fail(f"Function {name} grad raised an exception: {e}")
    assert grad_value.shape == x.shape
    assert jnp.all(jnp.isfinite(grad_value)), (
        f"{name} grad returned non-finite"
    )


@pytest.mark.parametrize("name", COMPOSITION_NAMES)
@pytest.mark.parametrize("dim", [2, 5, 10])
def test_composition_sanity(name, dim):
    """With deterministic registry, f(zeros) is finite and gradient is finite.

    The gradient magnitude check uses a loose bound to accommodate the
    Weierstrass component functions (F18-F25), whose float32 gradient at
    exactly z=0 is numerically large due to accumulated floating-point error
    in sin(pi * 3^k) for k >= 11.  Mathematically sin(pi * integer) = 0, but
    this does not hold in float32 for 3^k > 2^23 (approx. k >= 15). Combined
    with the height-normalisation lambda (which can be large when the reference
    value is near-zero), the gradient norm at the deterministic origin can
    reach ~2.4e9.  The check below uses 5e9 as an absolute upper bound — about
    2x above the empirically observed maximum — which is ~2000x tighter than
    the previous dim * 1e10 threshold while still covering all 25 functions
    across dims [2, 5, 10].  Functions without Weierstrass components (F15-F17)
    always have gradient norms of exactly 0 at the origin.
    """
    fn_factory = cec2005_registry_original[name]
    fn_func, _ = fn_factory(ndim=dim, key=jr.key(0))
    x_test = jnp.zeros(dim)
    try:
        if name in _NOISY:
            result = fn_func(x_test, jr.key(42))
            assert jnp.isfinite(result), f"{name} sanity: non-finite at zeros"
            grad = jax.grad(lambda x: fn_func(x, jr.key(42)))(x_test)
        else:
            result = fn_func(x_test)
            assert jnp.isfinite(result), f"{name} sanity: non-finite at zeros"
            grad = jax.grad(fn_func)(x_test)
        grad_norm = jnp.linalg.norm(grad)
        assert jnp.isfinite(grad_norm), f"{name} sanity: non-finite gradient"
        # 5e9 is ~2x above the empirically observed max (~2.4e9 for F21-F23 at
        # dim=2) caused by float32 precision loss in the Weierstrass components.
        assert grad_norm < 5e9, (
            f"{name} sanity: grad norm {grad_norm:.2e} >= 5e9"
        )
    except NotImplementedError:
        pytest.skip(f"{name} not yet implemented")
    except Exception as e:
        pytest.fail(f"{name} sanity raised an exception: {e}")


def test_f5_factory_uses_integer_nonsingular_matrix():
    fn_func, _ = cec2005_registry["f5"](ndim=5, key=jr.key(0))
    a = fn_func.keywords["R"]
    assert jnp.all(a == jnp.round(a))
    assert jnp.all(a >= -500)
    assert jnp.all(a <= 500)
    assert not jnp.isclose(jnp.linalg.det(a), 0.0)


def test_f8_factory_places_odd_1_based_coordinates_on_boundary():
    fn_func, _ = cec2005_registry["f8"](ndim=6, key=jr.key(0))
    x_opt = fn_func.keywords["x_opt"]
    assert jnp.all(x_opt[::2] == -32.0)


def test_f18_factory_sets_last_component_optimum_to_zero():
    fn_func, _ = cec2005_registry["f18"](ndim=5, key=jr.key(0))
    x_opt = fn_func.keywords["x_opt"]
    assert jnp.all(x_opt[9] == 0.0)


def test_f20_factory_clamps_even_1_based_coordinates_only():
    fn_func, _ = cec2005_registry["f20"](ndim=6, key=jr.key(0))
    x_opt = fn_func.keywords["x_opt"]
    assert jnp.all(x_opt[0, 1::2] == 5.0)
    assert jnp.any(x_opt[0, ::2] != 5.0)


def test_f25_optima_are_not_constrained_to_initialization_interval():
    fn_func, _ = cec2005_registry["f25"](ndim=5, key=jr.key(0))
    x_opt = fn_func.keywords["x_opt"]
    assert jnp.any((x_opt < 2.0) | (x_opt > 5.0))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("f8", 100.0),
        ("f10", 2.0),
        ("f11", 5.0),
        ("f14", 3.0),
    ],
)
def test_single_function_condition_numbers(name, expected):
    fn_func, _ = cec2005_registry[name](ndim=8, key=jr.key(0))
    condition_number = _condition_number(fn_func.keywords["R"])
    assert jnp.isclose(condition_number, expected, rtol=5e-2)


def test_f7_matrix_has_condition_number_three_after_scalar_scaling():
    fn_func, _ = cec2005_registry["f7"](ndim=8, key=jr.key(0))
    condition_number = _condition_number(fn_func.keywords["R"])
    assert jnp.isclose(condition_number, 3.0, rtol=5e-2)


def test_f16_component_condition_numbers_match_paper():
    fn_func, _ = cec2005_registry["f16"](ndim=8, key=jr.key(0))
    conds = jax.vmap(_condition_number)(fn_func.keywords["R"])
    assert jnp.all(jnp.isclose(conds, 2.0, rtol=5e-2))


def test_f18_component_condition_numbers_match_paper():
    fn_func, _ = cec2005_registry["f18"](ndim=8, key=jr.key(0))
    conds = jax.vmap(_condition_number)(fn_func.keywords["R"])
    expected = jnp.array(
        [2, 3, 2, 3, 2, 3, 20, 30, 200, 300], dtype=jnp.float32
    )
    assert jnp.all(jnp.isclose(conds, expected, rtol=5e-2))


def test_f22_component_condition_numbers_match_paper():
    fn_func, _ = cec2005_registry["f22"](ndim=8, key=jr.key(0))
    conds = jax.vmap(_condition_number)(fn_func.keywords["R"])
    expected = jnp.array(
        [10, 20, 50, 100, 200, 1000, 2000, 3000, 4000, 5000],
        dtype=jnp.float32,
    )
    assert jnp.all(jnp.isclose(conds, expected, rtol=8e-2))


def test_f24_component_condition_numbers_match_paper():
    fn_func, _ = cec2005_registry["f24"](ndim=8, key=jr.key(0))
    conds = jax.vmap(_condition_number)(fn_func.keywords["R"])
    expected = jnp.array(
        [100, 50, 30, 10, 5, 5, 4, 3, 2, 2], dtype=jnp.float32
    )
    assert jnp.all(jnp.isclose(conds, expected, rtol=5e-2))


def test_noise_metadata_marks_stochastic_functions():
    assert cec2005_function_characteristics["f4"]["noise"]
    assert cec2005_function_characteristics["f17"]["noise"]
    assert cec2005_function_characteristics["f24"]["noise"]
    assert cec2005_function_characteristics["f25"]["noise"]
    assert not cec2005_function_characteristics["f1"]["noise"]
    assert not cec2005_function_characteristics["f16"]["noise"]


def test_structure_modified_metadata():
    assert cec2005_function_characteristics["f23"]["structure_modified"]
    assert cec2005_function_characteristics["f24"]["structure_modified"]
    assert cec2005_function_characteristics["f25"]["structure_modified"]


_STRUCTURE_MODIFIED = {
    name
    for name, chars in cec2005_function_characteristics.items()
    if chars.get("structure_modified", False)
}


@pytest.mark.parametrize("name,fn", pytest_cec2005)
@pytest.mark.parametrize("dim", dimensions)
def test_global_minimum_at_xopt(name, fn, dim):
    """Verify that fn(x_opt) == f_opt for all CEC 2005 functions.

    Uses the randomized registry only. The deterministic registry places all
    composition component optima at zero, making the weighting degenerate
    (uniform weights → result ≈ weighted average of biases, not f_opt).
    """
    key = jr.key(0)
    fn_func, f_opt = fn(ndim=dim, key=key)
    x_opt = fn_func.keywords["x_opt"]

    if name in COMPOSITION_NAMES:
        x_opt = x_opt[0]

    if name in _NOISY:
        result = fn_func(x_opt, jr.key(42))
    else:
        result = fn_func(x_opt)

    if name in _STRUCTURE_MODIFIED:
        # Soft rounding shifts the effective minimum slightly
        atol = 5e-2
    elif name in COMPOSITION_NAMES:
        atol = 1e-2
    else:
        atol = 1e-8

    assert jnp.isclose(result, f_opt, atol=atol), (
        f"{name} dim={dim}: fn(x_opt)={result}, f_opt={f_opt}, "
        f"diff={result - f_opt}"
    )


@pytest.mark.parametrize("name,fn", pytest_cec2005_original)
@pytest.mark.parametrize("dim", dimensions)
def test_global_minimum_at_xopt_deterministic(name, fn, dim):
    """Verify fn(x_opt) == f_opt for single-component CEC 2005 functions.

    Composition functions (F15-F25) are skipped because the deterministic
    registry places all component optima at zero, making the composition
    weighting degenerate.
    """
    if name in COMPOSITION_NAMES:
        pytest.skip("Deterministic registry not meaningful for compositions")

    fn_func, f_opt = fn(ndim=dim, key=jr.key(0))
    x_opt = fn_func.keywords["x_opt"]

    if name in _NOISY:
        result = fn_func(x_opt, jr.key(42))
    else:
        result = fn_func(x_opt)

    assert jnp.isclose(result, f_opt, atol=1e-8), (
        f"{name} dim={dim}: fn(x_opt)={result}, f_opt={f_opt}, "
        f"diff={result - f_opt}"
    )
