from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Tuple

from .state import OrganResult, SAGEState


class BaseOrgan(ABC):
    """One specialized module in the SAGE ecosystem."""

    name: str

    def fit(self, support: List[Tuple[Any, int]]) -> None:
        """Optional per-episode fitting hook."""
        return None

    @abstractmethod
    def process(self, state: SAGEState, signal: Dict[str, Any]) -> OrganResult:
        """Produce an organ-level action/result from the current state and signal."""
        raise NotImplementedError


class BaseRouter(ABC):
    """Selects organs and aggregates organ outputs."""

    name: str

    @abstractmethod
    def route(
        self,
        state: SAGEState,
        signal: Dict[str, Any],
        organs: Mapping[str, BaseOrgan],
    ) -> List[str]:
        """Return organ names to activate."""
        raise NotImplementedError

    @abstractmethod
    def aggregate(
        self,
        state: SAGEState,
        signal: Dict[str, Any],
        outputs: Mapping[str, OrganResult],
    ) -> Dict[str, Any]:
        """Convert organ outputs into an environment action."""
        raise NotImplementedError


class BaseEnvironment(ABC):
    """Benchmark/task environment."""

    name: str

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Return the first signal/stimulus."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply action and return next_signal, reward, done, info."""
        raise NotImplementedError


class BaseMetric(ABC):
    """Evaluates a completed SAGE run."""

    name: str

    @abstractmethod
    def evaluate(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError
