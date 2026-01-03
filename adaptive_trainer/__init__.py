"""Adaptive Trainer package — top-level convenience exports.

Correct import path for the primary adapter is::

    from adaptive_trainer import AdaptiveTrainer

This module exposes `AdaptiveTrainer` and a package `__version__`.
"""

from .optimizer import AdaptiveTrainer

__version__ = "0.1.1"

__all__ = ["AdaptiveTrainer", "__version__"]
