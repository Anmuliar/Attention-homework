"""Decode reference entry point.

The wrapper in :mod:`ops.ninetoothed.torch` concatenates the KV cache and
launches the prefill kernel. This module keeps the original import path and
leaves room for a cache-aware implementation as an optimization exercise.
"""

from .scaled_dot_product_attention_prefill import (
    application,
    arrangement,
    kernel,
)

__all__ = ["arrangement", "application", "kernel"]
