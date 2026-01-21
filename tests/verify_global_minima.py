import jax
import jax.numpy as jnp
import jax.random as jr
from bbob_jax import registry
from bbob_jax._src.utils import bernoulli_vector


def verify_minima():
    key = jr.key(42)
    ndim = 2

    functions = registry.keys()

    print(f"{'Function':<30} | {'Diff':<15} | {'Status':<10}")
    print("-" * 65)

    for name in functions:
        key, subkey = jr.split(key)

        try:
            # New API: returns (func_instance, f_opt_returned)
            func_instance, f_opt_returned = registry[name](
                ndim=ndim, key=subkey
            )

            keywords = func_instance.keywords

            # Theoretical optimum parameters passed to the function
            x_opt_param = keywords["x_opt"]

            # Determine the *actual* input x where the minimum occurs.
            true_input_x = x_opt_param

            if name == "linear_slope":
                # F5: x_opt is internally generated as 5 * bernoulli_vector(ndim, key)
                # key is jr.fold_in(key, Q[0,0]). Q is keywords['Q']
                Q = keywords["Q"]
                k = jr.key(0)
                k = jr.fold_in(k, Q[0, 0])
                # x_opt (internal) = 5 * bernoulli_vector(ndim, k)
                x_opt_internal = 5 * bernoulli_vector(ndim, k)

                # Logic: z = where(cond, x, x_opt).
                # result = sum(5.|s| - s.z).
                # To minimize, we want s.z to be max. s = sign(x_opt)*10^...
                # Maximize sign(x_opt)*z.
                # If z = x_opt, sign(x_opt)*x_opt = |x_opt| = 5. Positive.
                # So best z is x_opt.
                # cond: x_opt*x < 25. If x=x_opt, x_opt^2 = 25. 25 < 25 is False.
                # So if x=x_opt, cond is false, z = x_opt.
                true_input_x = x_opt_internal

            elif name == "rosenbrock_rotated":
                # F9: z = zmax * (x @ R) + 0.5. Optimum at z=1.
                # x = (0.5/zmax) * 1_vec @ R.T
                R = keywords["R"]
                zmax = jnp.maximum(1.0, jnp.sqrt(ndim) / 8.0)
                true_input_x = ((0.5 / zmax) * jnp.ones(ndim)) @ R.T

            elif name == "lunacek_bi_rastrigin":
                # F24: Internally generated x_opt.
                Q = keywords["Q"]
                k = jr.key(0)
                k = jr.fold_in(k, Q[0, 0])
                mu0 = 2.5
                # x_opt (internal) = (mu0 / 2.0) * bernoulli_vector(ndim, k)
                x_opt_internal = (mu0 / 2.0) * bernoulli_vector(ndim, k)
                true_input_x = x_opt_internal

            elif name == "schwefel_xsinx":
                # F20: x_opt = 4.209687... / 2 * ones (derived from key)
                # But wait, bbob.py says:
                # key = fold_in(key, Q[0,0])
                # ones = bernoulli(ndim, key)
                # x_opt = 4.2096874633 / 2 * ones
                Q = keywords["Q"]
                k = jr.key(0)
                k = jr.fold_in(k, Q[0, 0])
                ones = bernoulli_vector(ndim, k)
                x_opt_internal = 4.2096874633 / 2.0 * ones

                # Logic: z_hat = ... z= ...
                # Global minimum is at x_opt_internal
                true_input_x = x_opt_internal

            elif name == "gallagher_21_peaks" or name == "gallagher_101_peaks":
                # F21/F22: x_opt is generated internally using key derived from Q.
                Q = keywords["Q"]
                k = jr.key(0)
                k = jr.fold_in(k, Q[0, 0])
                # key1, key2 = split(k). x_opt info is in y[0].
                # x_opt (param) seems unused for location?
                # bbob.py: x_opt = jr.uniform(k, ...)
                # y = y.at[0].set(x_opt).
                # optimum is at y[0].
                # But wait, does the passed 'x_opt' param match the internal one?
                # make_randomized uses 'key1' to generate x_opt.
                # gallagher uses 'key' (from Q) to generate x_opt.
                # Q is generated from 'key2' in make_randomized.
                # So Q -> key -> x_opt (internal).
                # Passed x_opt -> key1 -> x_opt.
                # They are likely different!
                # So we need to reproduce internal generation.
                k = jr.key(0)
                k = jr.fold_in(k, Q[0, 0])
                # bbob.py:
                # key1, key2 = jr.split(key) Wait, bbob line 988: key1, key2 = jr.split(key)
                # x_opt = jr.uniform(key, ...) -> reusing `key` after split? line 1000 uses `key`!
                # line 988 split key into k1, k2. line 1000 uses 'key'. 'key' is still folded value.
                # So x_opt = uniform(key...).
                x_opt_internal = jr.uniform(
                    k, shape=(ndim,), minval=-4.0, maxval=4.0
                )
                if name == "gallagher_21_peaks":
                    x_opt_internal = jr.uniform(
                        k, shape=(ndim,), minval=-3.92, maxval=3.92
                    )

                true_input_x = x_opt_internal

            # Evaluate
            val = func_instance(true_input_x)

            # Check 1: Does val match f_opt_returned?
            diff = jnp.abs(val - f_opt_returned)

            is_pass = diff < 1e-3
            status = "PASS" if is_pass else "FAIL"

            print(f"{name:<30} | {float(diff):<15.2e} | {status:<10}")

        except Exception as e:
            print(f"{name:<30} | {'ERROR':<15} | {str(e)}")


if __name__ == "__main__":
    verify_minima()
