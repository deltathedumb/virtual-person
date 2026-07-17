from __future__ import annotations

from dataclasses import dataclass

from .body import BodyState
from .types import Goal, GoalKind
from .world import ApartmentWorld


@dataclass(slots=True)
class PlannerThresholds:
    bladder: float = 0.62
    thirst: float = 0.55
    hunger: float = 0.58
    fatigue: float = 0.82
    hygiene: float = 0.38
    social: float = 0.75


class UtilityPlanner:
    """Chooses one high-level goal from current pressures and responsibilities."""

    def __init__(self, thresholds: PlannerThresholds | None = None) -> None:
        self.thresholds = thresholds or PlannerThresholds()

    def choose_goal(
        self,
        body: BodyState,
        world: ApartmentWorld,
        daily_computer_task_done: bool,
        hour_of_day: float,
    ) -> Goal:
        options: list[Goal] = []

        if body.bladder >= self.thresholds.bladder:
            options.append(
                Goal(
                    GoalKind.USE_RESTROOM,
                    "bladder pressure is high",
                    body.bladder * 1.30,
                )
            )
        if body.thirst >= self.thresholds.thirst:
            options.append(Goal(GoalKind.DRINK, "thirst is high", body.thirst * 1.20))
        if body.hunger >= self.thresholds.hunger:
            options.append(Goal(GoalKind.EAT, "hunger is high", body.hunger * 1.10))
        if body.fatigue >= self.thresholds.fatigue or hour_of_day >= 23.0:
            options.append(
                Goal(GoalKind.SLEEP, "fatigue or bedtime", body.fatigue * 1.15)
            )
        if body.hygiene <= self.thresholds.hygiene:
            options.append(
                Goal(
                    GoalKind.SHOWER,
                    "hygiene is low",
                    (1.0 - body.hygiene) * 0.95,
                )
            )
        if not daily_computer_task_done and 8.0 <= hour_of_day <= 21.0:
            options.append(Goal(GoalKind.COMPUTER_TASK, "daily task is pending", 0.72))
        if body.social >= self.thresholds.social:
            options.append(Goal(GoalKind.SOCIALIZE, "social need is high", 0.45))

        if not options:
            return Goal(GoalKind.IDLE, "no urgent goal", 0.10)
        return max(options, key=lambda goal: goal.urgency)
