"""SAGE Core v1.6 refactor skeleton.

SAGE = Self-organizing Adaptive Generative Ecosystem

This package is a small, stable core layer for future benchmarks.
Keep benchmark-specific logic outside this package.
"""

from .state import SAGEState, OrganResult
from .base import BaseOrgan, BaseRouter, BaseEnvironment, BaseMetric
from .engine import SAGEEngine

__all__ = [
    "SAGEState",
    "OrganResult",
    "BaseOrgan",
    "BaseRouter",
    "BaseEnvironment",
    "BaseMetric",
    "SAGEEngine",
]
