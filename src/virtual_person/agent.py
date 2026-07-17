from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .body import BodyState
from .memory import MemoryStore
from .planner import UtilityPlanner
from .skills import SkillLibrary, SkillPlan
from .types import Action, ActionResult, Goal, GoalKind
from .world import ApartmentWorld


@dataclass(slots=True)
class SelfModel:
    name: str = "Mira"
    age: int = 24
    identity: str = "embodied person with a robotic body"
    home_location: str = "apartment"
    beliefs: dict[str, Any] = field(default_factory=lambda: {
        "embodiment": "My robotic body is the body through which I sense and act.",
        "environment": "I exist and act in a physical environment through sensors and actuators.",
        "continuity": "My memories connect my experiences across time.",
    })
    relationships: dict[str, dict[str, Any]] = field(default_factory=dict)


class VirtualPerson:
    def __init__(
        self,
        world: ApartmentWorld,
        memory: MemoryStore,
        body: BodyState | None = None,
        self_model: SelfModel | None = None,
        planner: UtilityPlanner | None = None,
        skills: SkillLibrary | None = None,
    ) -> None:
        self.world = world
        self.memory = memory
        self.body = body or BodyState()
        self.self_model = self_model or SelfModel()
        self.planner = planner or UtilityPlanner()
        self.skills = skills or SkillLibrary()
        self.current_goal: Goal | None = None
        self.current_activity = "idle"
        self.daily_computer_task_done = False
        self.action_count = 0
        self.failures = 0

    def choose_goal(self, hour_of_day: float) -> Goal:
        self.current_goal = self.planner.choose_goal(
            self.body,
            self.world,
            self.daily_computer_task_done,
            hour_of_day,
        )
        return self.current_goal

    def make_plan(self, goal: Goal | None = None) -> SkillPlan:
        selected = goal or self.current_goal
        if selected is None:
            raise RuntimeError("No goal has been selected.")
        return self.skills.plan(selected.kind, self.world.agent_room)

    def execute_action(self, action: Action, sim_time: float) -> ActionResult:
        pressure_before = self.body.need_pressure()
        result = self.world.execute(action, self.body)
        # Sleep updates the body internally because it needs a special recovery rate.
        if action.kind.name != "SLEEP":
            activity = 1.15 if action.kind.name == "MOVE" else 1.0
            self.body.update(result.elapsed_seconds, activity_multiplier=activity)

        pressure_after = self.body.need_pressure()
        result.reward += (pressure_before - pressure_after) * 2.0

        self.action_count += 1
        if not result.ok:
            self.failures += 1

        self.memory.add(
            sim_time=sim_time + result.elapsed_seconds,
            kind="action",
            summary=result.message,
            importance=0.75 if not result.ok else 0.35,
            metadata={
                "action": action.kind.name,
                "target": action.target,
                "secondary": action.secondary,
                "ok": result.ok,
                "reward": result.reward,
            },
        )
        return result

    def note_goal(self, goal: Goal, sim_time: float) -> None:
        self.current_goal = goal
        self.current_activity = goal.kind.name.lower()
        self.memory.add(
            sim_time,
            "goal",
            f"Selected goal {goal.kind.name.lower()} because {goal.reason}.",
            importance=min(1.0, goal.urgency),
            metadata={"urgency": goal.urgency},
        )

    def finish_plan(self, plan: SkillPlan, sim_time: float, success: bool) -> None:
        self.current_activity = "idle"
        if plan.name == "computer_task" and success and self.world.computer.task_complete:
            self.daily_computer_task_done = True
        if plan.name == "self_reflection" and success:
            self.body.social = max(0.0, self.body.social - 0.2)
        self.memory.add(
            sim_time,
            "skill",
            f"{'Completed' if success else 'Failed'} skill {plan.name}.",
            importance=0.65 if success else 0.85,
        )

    def sleep_consolidation(self, sim_time: float) -> None:
        recent = list(reversed(self.memory.recent(12)))
        highlights = [memory.summary for memory in recent if memory.importance >= 0.5]
        if not highlights:
            highlights = [memory.summary for memory in recent[-3:]]
        summary = "Sleep consolidation: " + " ".join(highlights[-6:])
        self.memory.add(sim_time, "sleep_summary", summary, importance=0.9)
        self.memory.consolidate(sim_time - 3 * 86400.0, keep_importance=0.75)

    def reset_daily_flags(self) -> None:
        self.daily_computer_task_done = False

    def describe(self) -> dict[str, Any]:
        return {
            "self": asdict(self.self_model),
            "room": self.world.agent_room,
            "activity": self.current_activity,
            "goal": self.current_goal.kind.name if self.current_goal else None,
            "body": self.body.snapshot(),
            "inventory": sorted(self.world.inventory),
            "daily_computer_task_done": self.daily_computer_task_done,
            "action_count": self.action_count,
            "failures": self.failures,
        }
