from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .base import BaseEnvironment, BaseMetric, BaseOrgan, BaseRouter
from .state import OrganResult, SAGEState


class SAGEEngine:
    """Central control loop for SAGE.

    Benchmarks should define organs/router/environment/metric,
    then let this engine handle the repeated state transition loop.
    """

    def __init__(
        self,
        organs: Mapping[str, BaseOrgan],
        router: BaseRouter,
        env: BaseEnvironment,
        metric: BaseMetric,
        state: Optional[SAGEState] = None,
        energy_decay: float = 0.995,
        reward_gain: float = 0.08,
        organ_cost: float = 0.01,
        max_memory: int = 128,
    ) -> None:
        self.organs = dict(organs)
        self.router = router
        self.env = env
        self.metric = metric
        self.state = state or SAGEState()
        self.energy_decay = energy_decay
        self.reward_gain = reward_gain
        self.organ_cost = organ_cost
        self.max_memory = max_memory

        for organ_name in self.organs:
            self.state.organ_energy.setdefault(organ_name, 1.0)

    def run(self, max_steps: int = 64) -> Dict[str, Any]:
        signal = self.env.reset()
        history: List[Dict[str, Any]] = []
        done = False

        while not done and self.state.is_alive() and self.state.step < max_steps:
            selected = self.router.route(self.state, signal, self.organs)
            selected = [name for name in selected if name in self.organs]

            outputs: Dict[str, OrganResult] = {}
            for name in selected:
                outputs[name] = self.organs[name].process(self.state, signal)

            action = self.router.aggregate(self.state, signal, outputs)
            next_signal, reward, done, info = self.env.step(action)

            self._transition(
                signal=signal,
                selected=selected,
                outputs=outputs,
                action=action,
                reward=reward,
                info=info,
            )

            history.append(
                {
                    "state": self.state.copy_public(),
                    "signal": signal,
                    "selected_organs": selected,
                    "outputs": {
                        name: {
                            "action": result.action,
                            "confidence": result.confidence,
                            "evidence": result.evidence,
                        }
                        for name, result in outputs.items()
                    },
                    "action": action,
                    "reward": reward,
                    "info": info,
                }
            )

            signal = next_signal

        result = self.metric.evaluate(history)
        result["final_state"] = self.state.copy_public()
        result["steps"] = len(history)
        result["engine"] = {
            "energy_decay": self.energy_decay,
            "reward_gain": self.reward_gain,
            "organ_cost": self.organ_cost,
            "max_memory": self.max_memory,
        }
        return result

    def _transition(
        self,
        signal: Dict[str, Any],
        selected: List[str],
        outputs: Mapping[str, OrganResult],
        action: Dict[str, Any],
        reward: float,
        info: Dict[str, Any],
    ) -> None:
        """Single centralized state transition function.

        This is the place to evolve future SAGE homeostasis,
        memory compression, router feedback, and energy logic.
        """
        self.state.step += 1
        self.state.reward = float(reward)

        cost = self.organ_cost * len(selected)
        self.state.energy = max(
            0.0,
            min(10.0, self.state.energy * self.energy_decay + reward * self.reward_gain - cost),
        )

        for name in selected:
            old = self.state.organ_energy.get(name, 1.0)
            self.state.organ_energy[name] = max(0.0, min(10.0, old * 0.99 + reward * 0.05))

        self.state.router_feedback = {
            "selected": list(selected),
            "num_outputs": len(outputs),
            "last_action": action,
            "last_info": info,
        }

        self.state.remember(
            {
                "step": self.state.step,
                "signal": signal,
                "selected": list(selected),
                "action": action,
                "reward": reward,
            },
            max_memory=self.max_memory,
        )
