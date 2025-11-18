# Installation

- Purpose: quick ways to install `bbob-jax` for users and contributors.
- Requirements: Python `>=3.10`.

### Quick Install (pip)

- Recommended for most users.

```bash
pip install bbob-jax
```

This installs the core package with its main dependencies:
- `jax`
- `jaxtyping`

To include plotting helpers as well:

```bash
pip install "bbob-jax[plot]"
```

### From Source

- Use this for local development or to contribute.
```bash
git clone https://github.com/bessagroup/bbob-jax
cd bbob-jax

# Editable install with extras you need
pip install -e ".[plot,tests,docs,dev]"
```
