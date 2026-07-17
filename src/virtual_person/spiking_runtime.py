from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .drives import SatisfactionEvaluator
from .env import VirtualPersonEnv
from .spiking_mind import MindDecision, SpikingMind


@dataclass(slots=True)
class AutonomousStep:
    decision: MindDecision
    message: str
    reward: float
    terminated: bool
    truncated: bool
    observation: dict[str, Any]


class AutonomousSpikingAgent:
    """Runs a SpikingMind against the validated symbolic environment."""

    def __init__(
        self,
        mind: SpikingMind,
        environment: VirtualPersonEnv | None = None,
    ) -> None:
        self.mind = mind
        self.environment = environment or VirtualPersonEnv()
        self.reward_evaluator = SatisfactionEvaluator()
        self.observation, self.info = self.environment.reset()

    def step(self, *, deterministic: bool = True) -> AutonomousStep:
        if self.environment.sim is None:
            raise RuntimeError("Environment was not initialized")

        candidates = self.environment.valid_action_templates()
        recent_memories = [
            memory.summary
            for memory in reversed(
                self.environment.sim.agent.memory.recent(8)
            )
        ]
        hour = self.environment.sim.hour_of_day
        task_pending = not self.environment.sim.agent.daily_computer_task_done

        physical_before = self.environment.sim.agent.body.need_pressure()
        cognitive_before = self.mind.cognitive_drives.discomfort()

        decision = self.mind.choose_action(
            self.observation,
            candidates,
            memories=recent_memories,
            hour_of_day=hour,
            task_pending=task_pending,
            deterministic=deterministic,
        )
        result = self.environment.step(decision.action)

        self.mind.update_drives_from_result(
            elapsed_seconds=float(result.info.get("elapsed_seconds", 1.0)),
            action_ok=bool(result.info.get("ok", False)),
            reward=result.reward,
            action_kind=decision.action.kind,
        )

        physical_after = self.environment.sim.agent.body.need_pressure()
        cognitive_after = self.mind.cognitive_drives.discomfort()
        satisfaction = self.reward_evaluator.evaluate(
            physical_before=physical_before,
            physical_after=physical_after,
            cognitive_before=cognitive_before,
            cognitive_after=cognitive_after,
            elapsed_seconds=float(result.info.get("elapsed_seconds", 1.0)),
            action_ok=bool(result.info.get("ok", False)),
        )
        combined_reward = result.reward + satisfaction.reward
        self.observation = result.observation

        return AutonomousStep(
            decision=decision,
            message=str(result.info.get("message", "")),
            reward=combined_reward,
            terminated=result.terminated,
            truncated=result.truncated,
            observation=result.observation,
        )

    def run(self, steps: int = 100, *, deterministic: bool = True) -> list[AutonomousStep]:
        results: list[AutonomousStep] = []
        for _ in range(steps):
            result = self.step(deterministic=deterministic)
            results.append(result)
            if result.terminated or result.truncated:
                break
        return results
