# Stochastax

A Jax library for advanced stochastic analysis.

[![PyPI version](https://badge.fury.io/py/stochastax.svg)](https://badge.fury.io/py/quicksig)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Fast**: Built on JAX with JIT compilation for maximum performance
- **Flexible**: Supports both path signatures and log signatures
- **GPU Support**: Leverages JAX's GPU acceleration when available

## Installation

```bash
pip install quicksig
```

For GPU support (CUDA 12):
```bash
pip install quicksig[cuda]
uv sync --extra cuda
```

For development:
```bash
pip install quicksig[dev]
uv sync --extra dev
```

For development:
```bash
pip install quicksig[all]
uv sync --all-extras
```

## Quick Start

```python
import jax.numpy as jnp
from quicksig import get_signature, get_log_signature

# Create a simple 2D path
path = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])

# Compute path signature up to depth 3
signature = get_signature(path, depth=3)
print(f"Signature shape: {signature.shape}")

# Compute log signature
log_sig = get_log_signature(path, depth=3, log_signature_type="lyndon")
print(f"Log signature shape: {log_sig.shape}")
```

## Batch Processing

```python
# Process multiple paths at once
batch_paths = jnp.array([
    [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
    [[0.0, 0.0], [1.0, -1.0], [2.0, 0.0]]
])

# Compute signatures for all paths
batch_signatures = jax.vmap(get_signature, in_axes=(0, None, None))(batch_paths, 2, False)
```

## API Reference

### `get_signature(path, depth, stream=False)`

Compute the signature of a path or batch of paths.

**Parameters:**
- `path` (jax.Array): Input path(s) of shape `(length, dim)` for single path or `(batch, length, dim)` for batch
- `depth` (int): Maximum signature depth to compute
- `stream` (bool): Whether to compute streaming signatures

**Returns:**
- `jax.Array`: Flattened signature tensor

### `get_log_signature(path, depth, log_signature_type)`

Compute the log signature of a path or batch of paths.

**Parameters:**
- `path` (jax.Array): Input path(s) 
- `depth` (int): Maximum signature depth
- `log_signature_type` (Literal["expanded", "lyndon"]): Type of log signature computation

**Returns:**
- `jax.Array`: Flattened log signature tensor

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/stochastax.git
cd quicksig

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Requirements

- Python 3.12+
- JAX >= 0.6.0

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
