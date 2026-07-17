from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .simulation import VirtualPersonSimulation
from .spike_training import build_behavior_record
from .types import GoalKind


PROCEDURAL_TEXTS = [
    """
    A person should choose actions using both immediate needs and longer-term
    consequences. Hunger can motivate preparing food, but cooking requires
    checking ingredients, using heat safely, turning appliances off, and cleaning
    afterward.
    """,
    """
    Meaningful rest is not failure. Sleeping when fatigued, waiting while food
    cooks, and pausing to inspect an uncertain situation can be better than
    performing random actions merely to avoid inactivity.
    """,
    """
    A computer user can open applications, identify text fields and buttons,
    type and edit text, save work, inspect errors, and ask before performing
    consequential actions. The exact interface must still be observed.
    """,
    """
    Boredom is reduced by meaningful engagement, progress, learning, recreation,
    or social connection. Repetitive motion with no purpose is not meaningful
    engagement.
    """,
    """
    When perception is uncertain, inspect, move to a better viewpoint, or ask a
    question. Do not treat a low-confidence guess as a confirmed fact.
    """,
]


def generate_bootstrap_corpus(
    output: str | Path,
    *,
    episodes: int = 20,
    seed: int = 0,
) -> int:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for text in PROCEDURAL_TEXTS:
        rows.append({
            "text": " ".join(text.split()),
            "state_features": [0.0] * 16,
            "action_target": -100,
            "value_target": 0.0,
        })

    for episode in range(episodes):
        sim = VirtualPersonSimulation.default(seed=seed + episode)
        rng = sim.random
        sim.agent.body.hunger = rng.uniform(0.15, 0.92)
        sim.agent.body.thirst = rng.uniform(0.10, 0.90)
        sim.agent.body.fatigue = rng.uniform(0.10, 0.95)
        sim.agent.body.bladder = rng.uniform(0.05, 0.90)
        sim.agent.body.hygiene = rng.uniform(0.20, 1.00)

        for _ in range(16):
            observation = sim.agent.world.observation(sim.agent.body)
            candidates = []
            for room in sorted(sim.agent.world.rooms[sim.agent.world.agent_room].neighbors):
                candidates.append({"kind": "MOVE", "target": room})
            for obj in sim.agent.world.visible_objects():
                candidates.append({"kind": "INSPECT", "target": obj.object_id})
                if obj.openable:
                    candidates.append({
                        "kind": "CLOSE" if obj.is_open else "OPEN",
                        "target": obj.object_id,
                    })
                if obj.portable:
                    candidates.append({"kind": "PICK_UP", "target": obj.object_id})
                if obj.usable:
                    candidates.append({"kind": "USE", "target": obj.object_id})
                if obj.kind == "food":
                    candidates.append({"kind": "EAT", "target": obj.object_id})
                if obj.kind == "drink":
                    candidates.append({"kind": "DRINK", "target": obj.object_id})
            for item in sorted(sim.agent.world.inventory):
                candidates.append({"kind": "PUT_DOWN", "target": item})
            candidates.append({"kind": "WAIT", "value": 60.0})
            candidates = candidates[:32]

            goal = sim.agent.choose_goal(sim.hour_of_day)
            plan = sim.agent.make_plan(goal)
            target_action = plan.actions[0]
            target_template = {
                "kind": target_action.kind.name,
                "target": target_action.target,
                "secondary": target_action.secondary,
                "value": target_action.value,
            }

            selected_index = next(
                (
                    index
                    for index, candidate in enumerate(candidates)
                    if all(
                        candidate.get(key) == target_template.get(key)
                        for key in ("kind", "target", "secondary", "value")
                    )
                ),
                -1,
            )
            if selected_index >= 0:
                body = sim.agent.body
                rows.append(build_behavior_record(
                    state_text=json.dumps(observation, separators=(",", ":"), sort_keys=True),
                    candidates=candidates,
                    selected_index=selected_index,
                    reward=goal.urgency,
                    state_features=[
                        body.hunger,
                        body.thirst,
                        body.fatigue,
                        body.bladder,
                        1.0 - body.hygiene,
                        1.0 - body.health,
                        body.social,
                        0.2,
                        0.45,
                        0.2,
                        0.1,
                        0.5,
                        0.5,
                        0.5,
                        0.0 if sim.agent.daily_computer_task_done else 1.0,
                        1.0,
                    ],
                ))

            sim.step()
        sim.close()

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
