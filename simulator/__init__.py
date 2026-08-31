"""Gotham GPU roofline simulator: Python front-end over the C++ core."""

__version__ = "0.1.0"

from . import core, data  # noqa: F401

__all__ = ["core", "data"]
