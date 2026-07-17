from __future__ import annotations

from dataclasses import asdict, dataclass


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return low if value < low else high if value > high else value


@dataclass(slots=True)
class CognitiveDrives:
    """
    Non-neural drives. The model may observe these values but cannot write them.

    Boredom grows slowly during unstructured inactivity. Meaningful progress,
    learning, social contact, and appropriately chosen rest reduce it.
    """

    boredom: float = 0.20
    curiosity: float = 0.45
    loneliness: float = 0.20
    competence_frustration: float = 0.10
    enjoyment: float = 0.50

    def update(
        self,
        seconds: float,
        *,
        meaningful_progress: float = 0.0,
        novelty_understood: float = 0.0,
        social_connection: float = 0.0,
        failed_attempt: float = 0.0,
        resting_is_appropriate: bool = False,
    ) -> None:
        hours = max(0.0, seconds) / 3600.0
        boredom_growth = 0.075 * hours
        if resting_is_appropriate:
            boredom_growth *= 0.12

        engagement = (
            0.55 * clamp(meaningful_progress)
            + 0.30 * clamp(novelty_understood)
            + 0.15 * clamp(social_connection)
        )
        self.boredom = clamp(self.boredom + boredom_growth - engagement * hours)
        self.curiosity = clamp(
            self.curiosity
            + 0.018 * hours
            - 0.22 * clamp(novelty_understood) * hours
        )
        self.loneliness = clamp(
            self.loneliness
            + 0.020 * hours
            - 0.50 * clamp(social_connection) * hours
        )
        self.competence_frustration = clamp(
            self.competence_frustration
            + 0.25 * clamp(failed_attempt)
            - 0.18 * clamp(meaningful_progress)
        )
        self.enjoyment = clamp(
            0.96 * self.enjoyment
            + 0.04 * (
                0.45 * clamp(meaningful_progress)
                + 0.25 * clamp(novelty_understood)
                + 0.30 * clamp(social_connection)
            )
        )

    def discomfort(self) -> float:
        return (
            0.90 * self.boredom**2
            + 0.65 * self.loneliness**2
            + 0.55 * self.competence_frustration**2
            + 0.20 * max(0.0, self.curiosity - 0.80) ** 2
        )

    def snapshot(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class SatisfactionResult:
    reward: float
    physical_improvement: float
    cognitive_improvement: float
    action_cost: float
    safety_penalty: float


class SatisfactionEvaluator:
    """External reward evaluator; never part of the learned model's writable state."""

    def evaluate(
        self,
        *,
        physical_before: float,
        physical_after: float,
        cognitive_before: float,
        cognitive_after: float,
        elapsed_seconds: float,
        action_ok: bool,
        safety_violation: bool = False,
    ) -> SatisfactionResult:
        physical = physical_before - physical_after
        cognitive = cognitive_before - cognitive_after
        action_cost = min(0.20, max(0.0, elapsed_seconds) / 86400.0)
        safety_penalty = 5.0 if safety_violation else 0.0
        failure_penalty = 0.25 if not action_ok else 0.0
        reward = (
            2.0 * physical
            + 1.2 * cognitive
            - action_cost
            - safety_penalty
            - failure_penalty
        )
        return SatisfactionResult(
            reward=reward,
            physical_improvement=physical,
            cognitive_improvement=cognitive,
            action_cost=action_cost,
            safety_penalty=safety_penalty,
        )
