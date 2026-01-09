import jax
import jax.numpy as jnp
import pandas as pd
from bbob_jax._src.bbob import (
    sphere,
    ellipsoid_seperable,
    rastrigin_seperable,
    skew_rastrigin_bueche,
    linear_slope,
    attractive_sector,
    step_ellipsoid,
    rosenbrock,
    rosenbrock_rotated,
    ellipsoid,
    discuss,
    bent_cigar,
    sharp_ridge,
    sum_of_different_powers,
    rastrigin,
    weierstrass,
    schaffer_f7_condition_10,
    schaffer_f7_condition_1000,
    griewank_rosenbrock_f8f2,
    schwefel_xsinx,
    gallagher_101_peaks,
    gallagher_21_peaks,
    katsuura,
    lunacek_bi_rastrigin,
)


def verify_minima():
    functions = {
        "F1: Sphere": sphere,
        "F2: Ellipsoid Separable": ellipsoid_seperable,
        "F3: Rastrigin Separable": rastrigin_seperable,
        "F4: Skew Rastrigin Bueche": skew_rastrigin_bueche,
        "F5: Linear Slope": linear_slope,
        "F6: Attractive Sector": attractive_sector,
        "F7: Step Ellipsoid": step_ellipsoid,
        "F8: Rosenbrock": rosenbrock,
        "F9: Rosenbrock Rotated": rosenbrock_rotated,
        "F10: Ellipsoid": ellipsoid,
        "F11: Discus": discuss,
        "F12: Bent Cigar": bent_cigar,
        "F13: Sharp Ridge": sharp_ridge,
        "F14: Sum of Different Powers": sum_of_different_powers,
        "F15: Rastrigin": rastrigin,
        "F16: Weierstrass": weierstrass,
        "F17: Schaffer F7 Condition 10": schaffer_f7_condition_10,
        "F18: Schaffer F7 Condition 1000": schaffer_f7_condition_1000,
        "F19: Griewank Rosenbrock F8F2": griewank_rosenbrock_f8f2,
        "F20: Schwefel x*sin(x)": schwefel_xsinx,
        "F21: Gallagher 101 Peaks": gallagher_101_peaks,
        "F22: Gallagher 21 Peaks": gallagher_21_peaks,
        "F23: Katsuura": katsuura,
        "F24: Lunacek bi-Rastrigin": lunacek_bi_rastrigin,
    }

    ndim = 5
    x = jnp.zeros(ndim)
    x_opt = jnp.zeros(ndim)
    f_opt = 0.0
    R = jnp.eye(ndim)
    Q = jnp.eye(ndim)

    results = []

    print(f"{'Function':<30} | {'Value at 0':<15} | {'Status':<10}")
    print("-" * 65)

    special_signatures = {"F13: Sharp Ridge", "F24: Lunacek bi-Rastrigin"}

    ignored_functions = {
        "F5: Linear Slope",
        "F9: Rosenbrock Rotated",
        "F19: Griewank Rosenbrock F8F2",
        "F20: Schwefel x*sin(x)",
        "F21: Gallagher 101 Peaks",
        "F22: Gallagher 21 Peaks",
        "F24: Lunacek bi-Rastrigin",
    }

    for name, func in functions.items():
        if name in ignored_functions:
            continue

        try:
            if name in special_signatures:
                # (x, x_opt, R, Q, f_opt)
                val = func(x, x_opt, R, Q, f_opt)
            else:
                # (x, x_opt, f_opt, R, Q)
                val = func(x, x_opt, f_opt, R, Q)

            val_scalar = float(val)
            is_zero = jnp.allclose(val, 0.0, atol=1e-6)
            status = "PASS" if is_zero else "FAIL"

            results.append(
                {"Function": name, "Value": val_scalar, "Status": status}
            )

            print(f"{name:<30} | {val_scalar:<15.6f} | {status:<10}")
        except Exception as e:
            results.append(
                {"Function": name, "Value": str(e), "Status": "ERROR"}
            )
            print(f"{name:<30} | {'ERROR':<15} | {str(e)}")


if __name__ == "__main__":
    verify_minima()
