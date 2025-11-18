# Getting Started

This page shows how to use the BBOB functions via the registries, pick deterministic or randomized problem instances, and work with plotting helpers.

- Prerequisite: install the package (see [Installation](installation.md)).
- API reference: see [API Reference](api.md).

## Quick Usage

```python
import jax
import jax.numpy as jnp
import jax.random as jr
from bbob_jax import registry

# Choose a function by name; dimensionality is taken from x's shape
fn = registry["sphere"]
key = jr.key(0)

x = jnp.zeros((2,))  # 2D input
val = fn(x, key=key)
print(float(val))
```

- `registry[<name>]`: randomized instance (shift/rotation and offset) via PRNG `key`.
- The last dimension of `x` defines the problem dimension.

## Deterministic vs. Randomized

Two registries are available:

- `registry`: randomized instance; call as `fn(x, key=...)`.
- `registry_original`: deterministic (no shift/rotation/offset); call as `fn(x)`.

```python
from bbob_jax import registry, registry_original
import jax.numpy as jnp
import jax.random as jr

x = jnp.zeros((5,))

# Randomized instance (shift, rotations, and fopt applied)
val_rand = registry["rastrigin"](x, key=jr.key(42))

# Deterministic/original (no shift/rotation/fopt)
val_det = registry_original["rastrigin"](x)
```

Notes:
- Use the same `key` across many evaluations to keep the same instance.
- Use a different `key` to create a different randomized instance.

## Evaluate Many Points (jit/vmap)

You can JIT-compile functions and batch-evaluate points with `vmap`:

```python
import jax
import jax.numpy as jnp
import jax.random as jr
from bbob_jax import registry

fn = registry["rosenbrock"]
key = jr.key(0)

# Same randomized instance across all points: pass the same key
X = jnp.stack([
    jnp.array([1.0, 1.0]),
    jnp.array([0.5, -0.5]),
    jnp.array([2.0, 2.0]),
])

def eval_point(x):
    return fn(x, key=key)

batched = jax.vmap(eval_point)
compiled = jax.jit(batched)
vals = compiled(X)
print(vals)
```

If you want a different instance per evaluation (usually not desired within a single batch), you could pass different keys per row.

## List Available Functions

```python
from bbob_jax import registry
print(sorted(registry.keys()))
```

## Filter by Function Characteristics

`function_characteristics` provides simple tags (e.g., `separable`, `unimodal`).

```python
from bbob_jax import function_characteristics

separable = [
    name for name, tags in function_characteristics.items()
    if tags.get("separable")
]
print(separable)
```

## Plotting Utilities (optional)

Install plotting extras and use the helpers to visualize a function:

```bash
pip install "bbob-jax[plot]"
```

```python
import jax.random as jr
import matplotlib.pyplot as plt
from bbob_jax import registry
from bbob_jax.plotting import plot_2d

fn = registry["sphere"]
key = jr.key(0)

fig, ax = plt.subplots(figsize=(5, 4))
plot_2d(fn, key=key, bounds=(-5.0, 5.0), px=200, ax=ax, log_norm=False)
ax.set_title("Sphere (randomized instance)")
plt.show()
```

See the plotting API in [API Reference](api.md) under "Plotting Utilities".

## Advanced: Under-the-hood Signatures

Internally, the raw functions are defined as `fn(x, x_opt, R, Q)` and the registries handle building these parameters for you (plus adding an offset `fopt` to the randomized variant). Prefer using `registry`/`registry_original` from the public API rather than calling internal functions directly.
