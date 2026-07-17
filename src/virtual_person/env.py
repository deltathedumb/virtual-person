from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .simulation import VirtualPersonSimulation
from .types import Action, ActionKind


@dataclass(slots=True)
class StepOutput:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class VirtualPersonEnv:
    """
    Minimal Gym-like environment without requiring gymnasium.

    Supply Action objects directly. `valid_action_templates()` exposes the
    currently meaningful symbolic action choices.
    """

    def __init__(self, seed: int = 0, max_sim_hours: float = 24.0) -> None:
        self.seed = seed
        self.max_sim_seconds = max_sim_hours * 3600.0
        self.sim: VirtualPersonSimulation | None = None
        self.start_time = 0.0

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.sim is not None:
            self.sim.close()
        self.sim = VirtualPersonSimulation.default(seed=self.seed if seed is None else seed)
        self.start_time = self.sim.sim_time
        return self._observation(), {"valid_actions": self.valid_action_templates()}

    def step(self, action: Action) -> StepOutput:
        if self.sim is None:
            raise RuntimeError("Call reset() before step().")
        pressure_before = self.sim.agent.body.need_pressure()
        result = self.sim.agent.execute_action(action, self.sim.sim_time)
        self.sim.sim_time += result.elapsed_seconds
        pressure_after = self.sim.agent.body.need_pressure()

        reward = result.reward + (pressure_before - pressure_after)
        terminated = self.sim.agent.body.health <= 0.05
        truncated = (self.sim.sim_time - self.start_time) >= self.max_sim_seconds
        return StepOutput(
            observation=self._observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "message": result.message,
                "ok": result.ok,
                "elapsed_seconds": result.elapsed_seconds,
                "valid_actions": self.valid_action_templates(),
            },
        )

    def valid_action_templates(self) -> list[dict[str, Any]]:
        if self.sim is None:
            return []
        world = self.sim.agent.world
        visible = world.visible_objects()
        actions: list[dict[str, Any]] = []

        for room in sorted(world.rooms[world.agent_room].neighbors):
            actions.append({"kind": "MOVE", "target": room})

        for obj in visible:
            actions.append({"kind": "INSPECT", "target": obj.object_id})
            if obj.openable:
                actions.append({
                    "kind": "CLOSE" if obj.is_open else "OPEN",
                    "target": obj.object_id,
                })
            if obj.portable:
                actions.append({"kind": "PICK_UP", "target": obj.object_id})
            if obj.usable:
                actions.append({"kind": "USE", "target": obj.object_id})
            if obj.kind == "food":
                actions.append({"kind": "EAT", "target": obj.object_id})
            if obj.kind == "drink":
                actions.append({"kind": "DRINK", "target": obj.object_id})

        for item in sorted(world.inventory):
            actions.append({"kind": "PUT_DOWN", "target": item})

        actions.append({"kind": "WAIT", "value": 60.0})
        return actions

    def _observation(self) -> dict[str, Any]:
        if self.sim is None:
            raise RuntimeError("Environment is not initialized.")
        return self.sim.snapshot()
