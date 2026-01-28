import jax
import jax.numpy as jnp
import jax.random as jr
from bbob_jax._src.registry import registry_original, registry


def verify_minima():
    dims = [2, 3, 5, 10, 20, 40, 50, 100]
    print(f"Verifying global minima for dimensions: {dims}")
    print("=" * 80)

    failures = []

    # 1. Deterministic Verification
    print("\n[Part 1] Deterministic Functions (x_opt=0, f_opt=0)")
    print("-" * 60)
    for ndim in dims:
        print(f"Checking dimension {ndim}...")
        for name, factory in sorted(registry_original.items()):
            try:
                fn_instance, f_opt = factory(ndim=ndim)
                # x_opt should be zeros
                x_test = jnp.zeros(ndim)
                val = fn_instance(x_test)

                is_val_zero = jnp.isclose(val, 0.0, atol=1e-5)

                if not is_val_zero:
                    print(
                        f"[FAIL] {name:<30} (D={ndim}) val={val:.2e} (expected 0.0)"
                    )
                    failures.append((f"{name}_det_d{ndim}", val))
            except Exception as e:
                print(f"[ERROR] {name:<30} (D={ndim}) {e}")
                failures.append((f"{name}_det_d{ndim}", str(e)))

    # 2. Randomized Verification
    print("\n[Part 2] Randomized Functions (x_opt=random, f_opt=random)")
    print("-" * 60)

    key_root = jr.key(42)

    for ndim in dims:
        print(f"Checking dimension {ndim}...")
        key_dim = jr.fold_in(key_root, ndim)

        for name, factory in sorted(registry.items()):
            try:
                # Factory signature for randomized: (ndim, key)
                # But registry entries are Partial(make_randomized, fn=...)
                # so we call them with (ndim=ndim, key=key)

                # Use explicit enumeration for stable seeds that fit in uint32
                seed_offset = abs(hash(name)) % (2**32 - 1)
                key_fn = jr.fold_in(key_dim, seed_offset)

                fn_instance, f_opt_expected = factory(ndim=ndim, key=key_fn)

                # Extract x_opt from partial keywords
                if "x_opt" not in fn_instance.keywords:
                    # Should not happen for our make_randomized setup
                    raise ValueError(f"Could not find x_opt in {name}")

                x_opt = fn_instance.keywords["x_opt"]

                # Evaluate
                val = fn_instance(x_opt)

                # Check absolute difference
                diff = jnp.abs(val - f_opt_expected)
                # Tolerance might need to be slightly looser for high dimensions / complex functions
                is_close = jnp.allclose(val, f_opt_expected, atol=1e-4)

                if not is_close:
                    print(
                        f"[FAIL] {name:<30} (D={ndim}) diff={diff:.2e} (val={val:.2e}, exp={f_opt_expected:.2e})"
                    )
                    failures.append((f"{name}_rnd_d{ndim}", diff))

            except Exception as e:
                print(f"[ERROR] {name:<30} (D={ndim}) {e}")
                failures.append((f"{name}_rnd_d{ndim}", str(e)))

    print("=" * 80)
    if failures:
        print(f"FAILED: {len(failures)} checks failed.")
        for name, val in failures:
            print(f"  - {name}: {val}")
        exit(1)
    else:
        print("SUCCESS: All verification checks passed.")
        exit(0)


if __name__ == "__main__":
    verify_minima()
