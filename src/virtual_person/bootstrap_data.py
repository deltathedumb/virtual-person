from __future__ import annotations

import json
import random
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


_DRIVE_PHRASES = (
    ("hunger", "hunger"),
    ("thirst", "thirst"),
    ("fatigue", "fatigue"),
    ("bladder", "bladder pressure"),
)


def _pressure_word(value: float) -> str | None:
    if value >= 0.85:
        return "urgent"
    if value >= 0.60:
        return "high"
    if value >= 0.35:
        return "noticeable"
    return None


_ROOM_TEMPLATES = (
    "The person is in the {room}.",
    "Currently standing in the {room}.",
    "Right now, the person is located in the {room}.",
    "The scene takes place in the {room}.",
)

_NEIGHBOR_TEMPLATES = (
    "Reachable adjoining rooms: {rooms}.",
    "From here, {rooms} can be reached directly.",
    "The person could walk straight into {rooms}.",
    "Adjacent and reachable: {rooms}.",
)

_DRIVE_INTRO_TEMPLATES = (
    "Physical state: {notes}.",
    "Bodily needs right now: {notes}.",
    "Noticeable physical signals: {notes}.",
)

_DRIVE_NONE_TEMPLATES = (
    "Physical state: no pressing physical needs right now.",
    "No physical need is pressing at the moment.",
    "The body currently feels comfortable overall.",
)

_VISIBLE_TEMPLATES = (
    "Visible objects: {items}.",
    "Objects in view here: {items}.",
    "Looking around, the person can see: {items}.",
)

_VISIBLE_NONE_TEMPLATES = (
    "No objects are currently visible here.",
    "Nothing of note is visible in this spot.",
)

_CARRYING_TEMPLATES = (
    "Carrying: {items}.",
    "Currently holding: {items}.",
    "In hand right now: {items}.",
)


def _choose(rng: random.Random, options: tuple[str, ...]) -> str:
    return options[rng.randrange(len(options))]


def _describe_state(observation: dict[str, Any], rng: random.Random | None = None) -> str:
    """Render a simulator observation as natural-language prose.

    Kept free of raw JSON on purpose: earlier training used
    ``json.dumps(observation)`` directly as training text, and the model
    over-learned that literal JSON pattern, leaking fragments of it into
    unrelated generations. A first fix rendered plain English instead, but
    used one fixed sentence template per slot; the model then over-learned
    *that* fixed phrasing the same way, leaking phrases like "Reachable
    adjoining room" into unrelated generations. Each slot below now has
    several interchangeable phrasings, chosen at random per record, so no
    single literal sentence dominates the corpus.
    """
    rng = rng or random.Random()
    body = observation.get("body", {})
    room = observation.get("room", "an unknown room").replace("_", " ")
    neighbors = observation.get("neighbors", [])
    visible = observation.get("visible_objects", [])
    inventory = observation.get("inventory", [])
    dirty_dishes = observation.get("dirty_dishes", 0)

    parts = [_choose(rng, _ROOM_TEMPLATES).format(room=room)]

    if neighbors:
        joined = ", ".join(n.replace("_", " ") for n in neighbors)
        parts.append(_choose(rng, _NEIGHBOR_TEMPLATES).format(rooms=joined))

    drive_notes = []
    for key, label in _DRIVE_PHRASES:
        word = _pressure_word(float(body.get(key, 0.0)))
        if word:
            drive_notes.append(f"{label} is {word}")
    if drive_notes:
        parts.append(_choose(rng, _DRIVE_INTRO_TEMPLATES).format(notes="; ".join(drive_notes)))
    else:
        parts.append(_choose(rng, _DRIVE_NONE_TEMPLATES))

    if visible:
        described = []
        for obj in visible:
            name = obj["id"].replace("_", " ")
            note = ""
            if obj.get("openable"):
                note = " (open)" if obj.get("is_open") else " (closed)"
            described.append(name + note)
        parts.append(_choose(rng, _VISIBLE_TEMPLATES).format(items=", ".join(described)))
    else:
        parts.append(_choose(rng, _VISIBLE_NONE_TEMPLATES))

    if inventory:
        joined = ", ".join(i.replace("_", " ") for i in inventory)
        parts.append(_choose(rng, _CARRYING_TEMPLATES).format(items=joined))

    if dirty_dishes:
        parts.append(f"There are {dirty_dishes} dirty dish(es) waiting to be washed.")

    return " ".join(parts)


_CANDIDATE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "MOVE": ("go to the {t}", "walk to the {t}", "head into the {t}", "move toward the {t}"),
    "INSPECT": ("inspect the {t}", "take a closer look at the {t}", "examine the {t}"),
    "OPEN": ("open the {t}", "open up the {t}"),
    "CLOSE": ("close the {t}", "shut the {t}"),
    "PICK_UP": ("pick up the {t}", "grab the {t}", "take hold of the {t}"),
    "PUT_DOWN": ("put down the {t}", "set the {t} down"),
    "USE": ("use the {t}", "make use of the {t}"),
    "EAT": ("eat the {t}", "eat some of the {t}"),
    "DRINK": ("drink the {t}", "have a drink of the {t}"),
}


def _describe_candidate(candidate: dict[str, Any], rng: random.Random | None = None) -> str:
    """Render one action candidate as a short natural-language phrase."""
    rng = rng or random.Random()
    kind = str(candidate.get("kind", "")).upper()
    target = candidate.get("target")
    target_name = target.replace("_", " ") if isinstance(target, str) else target
    value = candidate.get("value")

    if kind == "WAIT":
        if isinstance(value, (int, float)):
            return _choose(rng, (
                f"wait for {value:.0f} seconds",
                f"pause for about {value:.0f} seconds",
                f"do nothing for {value:.0f} seconds",
            ))
        return "wait"

    templates = _CANDIDATE_TEMPLATES.get(kind)
    if templates is not None:
        return _choose(rng, templates).format(t=target_name)
    return f"{kind.lower()} {target_name}".strip()


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
                    state_text=_describe_state(observation, rng),
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
                    candidate_text=lambda candidate: _describe_candidate(candidate, rng),
                ))

            sim.step()
        sim.close()

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
