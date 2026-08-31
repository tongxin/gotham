"""Gotham GPU roofline simulator: Python front-end over the C++ core.

Only `data` is imported eagerly so catalog-only tooling (e.g. the static-site
export) works without the native library present.
"""

__version__ = "0.1.0"

from . import data  # noqa: F401

__all__ = ["data"]
