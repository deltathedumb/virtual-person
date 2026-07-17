from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .types import Action, ActionKind, ActionResult, GoalKind


@dataclass(slots=True)
class SkillPlan:
    name: str
    actions: list[Action]


class SkillLibrary:
    """Turns high-level goals into validated action sequences."""

    def plan(self, goal: GoalKind, current_room: str) -> SkillPlan:
        if goal is GoalKind.USE_RESTROOM:
            return SkillPlan("use_restroom", self._route(current_room, "bathroom") + [
                Action(ActionKind.USE, "toilet"),
            ])

        if goal is GoalKind.DRINK:
            return SkillPlan("drink_water", self._route(current_room, "kitchen") + [
                Action(ActionKind.OPEN, "fridge"),
                Action(ActionKind.PICK_UP, "water"),
                Action(ActionKind.DRINK, "water"),
                Action(ActionKind.CLOSE, "fridge"),
            ])

        if goal is GoalKind.EAT:
            return SkillPlan("cook_breakfast", self._route(current_room, "kitchen") + [
                Action(ActionKind.OPEN, "fridge"),
                Action(ActionKind.PICK_UP, "eggs"),
                Action(ActionKind.CLOSE, "fridge"),
                Action(ActionKind.PICK_UP, "pan"),
                Action(ActionKind.PICK_UP, "plate"),
                Action(ActionKind.USE, "stove"),
                Action(ActionKind.EAT, "eggs"),
                Action(ActionKind.USE, "kitchen_sink"),
            ])

        if goal is GoalKind.SHOWER:
            return SkillPlan("take_shower", self._route(current_room, "bathroom") + [
                Action(ActionKind.USE, "shower"),
            ])

        if goal is GoalKind.SLEEP:
            return SkillPlan("sleep", self._route(current_room, "bedroom") + [
                Action(ActionKind.SLEEP, "bed", value=8.0),
            ])

        if goal is GoalKind.COMPUTER_TASK:
            return SkillPlan("computer_task", self._route(current_room, "living_room") + [
                Action(ActionKind.USE, "computer", value="power_on"),
                Action(ActionKind.USE, "computer", value="launch:notes"),
                Action(ActionKind.TYPE_TEXT, value=(
                    "Daily report: I checked my needs, maintained the apartment, "
                    "and completed the assigned virtual computer task."
                )),
                Action(ActionKind.CLICK, "submit_task"),
            ])

        if goal is GoalKind.SOCIALIZE:
            return SkillPlan("self_reflection", [
                Action(
                    ActionKind.SPEAK,
                    value="I would like some social interaction, so I am reflecting aloud.",
                ),
                Action(ActionKind.WAIT, value=300),
            ])

        return SkillPlan("idle", [Action(ActionKind.WAIT, value=900)])

    @staticmethod
    def _route(start: str, destination: str) -> list[Action]:
        if start == destination:
            return []
        # The default apartment is a star centered on the hallway.
        if start == "hallway":
            return [Action(ActionKind.MOVE, destination)]
        if destination == "hallway":
            return [Action(ActionKind.MOVE, "hallway")]
        return [
            Action(ActionKind.MOVE, "hallway"),
            Action(ActionKind.MOVE, destination),
        ]
