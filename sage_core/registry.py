from __future__ import annotations

from typing import Dict, Type

from .base import BaseEnvironment, BaseMetric, BaseOrgan, BaseRouter


class ComponentRegistry:
    """Small registry for future YAML/JSON config loading."""

    def __init__(self) -> None:
        self.organs: Dict[str, Type[BaseOrgan]] = {}
        self.routers: Dict[str, Type[BaseRouter]] = {}
        self.environments: Dict[str, Type[BaseEnvironment]] = {}
        self.metrics: Dict[str, Type[BaseMetric]] = {}

    def register_organ(self, cls: Type[BaseOrgan]) -> Type[BaseOrgan]:
        self.organs[cls.name] = cls
        return cls

    def register_router(self, cls: Type[BaseRouter]) -> Type[BaseRouter]:
        self.routers[cls.name] = cls
        return cls

    def register_environment(self, cls: Type[BaseEnvironment]) -> Type[BaseEnvironment]:
        self.environments[cls.name] = cls
        return cls

    def register_metric(self, cls: Type[BaseMetric]) -> Type[BaseMetric]:
        self.metrics[cls.name] = cls
        return cls
