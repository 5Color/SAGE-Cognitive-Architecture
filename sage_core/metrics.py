from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List

from .base import BaseMetric


class BasicRunMetric(BaseMetric):
    name = "BasicRunMetric"

    def evaluate(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not history:
            return {
                "avg_reward": 0.0,
                "accuracy": 0.0,
                "total_reward": 0.0,
                "organ_usage": {},
            }

        rewards = [float(item.get("reward", 0.0)) for item in history]
        correct_values = [
            1.0 if item.get("info", {}).get("correct", False) else 0.0
            for item in history
        ]

        usage: Dict[str, int] = {}
        for item in history:
            for organ in item.get("selected_organs", []):
                usage[organ] = usage.get(organ, 0) + 1

        return {
            "avg_reward": mean(rewards),
            "accuracy": mean(correct_values),
            "total_reward": sum(rewards),
            "organ_usage": usage,
        }
