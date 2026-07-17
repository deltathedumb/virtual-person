from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .agent import VirtualPerson
from .memory import MemoryStore
from .types import ActionKind
from .world import ApartmentWorld


@dataclass(slots=True)
class SimulationEvent:
    sim_time: float
    day: int
    hour: float
    kind: str
    message: str
    ok: bool = True
    reward: float = 0.0


class VirtualPersonSimulation:
    def __init__(
        self,
        agent: VirtualPerson,
        seed: int = 0,
        start_hour: float = 8.0,
    ) -> None:
        self.agent = agent
        self.random = random.Random(seed)
        self.sim_time = start_hour * 3600.0
        self.events: list[SimulationEvent] = []
        self.day = 1
        self._last_day_index = 0

    @classmethod
    def default(
        cls,
        seed: int = 0,
        memory_path: str | Path = ":memory:",
        name: str = "Mira",
    ) -> "VirtualPersonSimulation":
        world = ApartmentWorld.default()
        memory = MemoryStore(memory_path)
        agent = VirtualPerson(world=world, memory=memory)
        agent.self_model.name = name
        return cls(agent=agent, seed=seed)

    @property
    def hour_of_day(self) -> float:
        return (self.sim_time % 86400.0) / 3600.0

    def step(self) -> list[SimulationEvent]:
        day_index = int(self.sim_time // 86400.0)
        if day_index != self._last_day_index:
            self._last_day_index = day_index
            self.day = day_index + 1
            self.agent.reset_daily_flags()

        goal = self.agent.choose_goal(self.hour_of_day)
        self.agent.note_goal(goal, self.sim_time)
        plan = self.agent.make_plan(goal)
        generated: list[SimulationEvent] = []

        success = True
        for action in plan.actions:
            result = self.agent.execute_action(action, self.sim_time)
            self.sim_time += result.elapsed_seconds
            event = SimulationEvent(
                sim_time=self.sim_time,
                day=self.day,
                hour=self.hour_of_day,
                kind=action.kind.name.lower(),
                message=result.message,
                ok=result.ok,
                reward=result.reward,
            )
            self.events.append(event)
            generated.append(event)
            if not result.ok:
                success = False
                break

        self.agent.finish_plan(plan, self.sim_time, success)
        if success and any(action.kind is ActionKind.SLEEP for action in plan.actions):
            self.agent.sleep_consolidation(self.sim_time)

        return generated

    def run(
        self,
        hours: float,
        max_decisions: int = 10000,
        on_event: Callable[[SimulationEvent], None] | None = None,
    ) -> list[SimulationEvent]:
        target = self.sim_time + max(0.0, hours) * 3600.0
        decisions = 0
        while self.sim_time < target and decisions < max_decisions:
            new_events = self.step()
            decisions += 1
            if on_event:
                for event in new_events:
                    on_event(event)
        return self.events

    def snapshot(self) -> dict[str, Any]:
        return {
            "sim_time": self.sim_time,
            "day": self.day,
            "hour": self.hour_of_day,
            "agent": self.agent.describe(),
            "world": self.agent.world.observation(self.agent.body),
            "recent_events": [
                {
                    "time": event.sim_time,
                    "kind": event.kind,
                    "message": event.message,
                    "ok": event.ok,
                    "reward": event.reward,
                }
                for event in self.events[-20:]
            ],
        }

    def close(self) -> None:
        self.agent.memory.close()

    def __enter__(self) -> "VirtualPersonSimulation":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
