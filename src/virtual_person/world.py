from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .body import BodyState
from .computer import VirtualComputer
from .types import Action, ActionKind, ActionResult


@dataclass(slots=True)
class Room:
    name: str
    neighbors: set[str] = field(default_factory=set)


@dataclass(slots=True)
class WorldObject:
    object_id: str
    kind: str
    room: str
    portable: bool = False
    openable: bool = False
    is_open: bool = False
    usable: bool = False
    container: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)


class ApartmentWorld:
    def __init__(
        self,
        rooms: dict[str, Room],
        objects: dict[str, WorldObject],
        start_room: str,
        computer: VirtualComputer | None = None,
    ) -> None:
        self.rooms = rooms
        self.objects = objects
        self.agent_room = start_room
        self.inventory: set[str] = set()
        self.computer = computer or VirtualComputer()
        self.dirty_dishes = 0

    @classmethod
    def default(cls) -> "ApartmentWorld":
        room_names = ["bedroom", "hallway", "bathroom", "kitchen", "living_room"]
        rooms = {name: Room(name) for name in room_names}

        def connect(a: str, b: str) -> None:
            rooms[a].neighbors.add(b)
            rooms[b].neighbors.add(a)

        connect("bedroom", "hallway")
        connect("bathroom", "hallway")
        connect("kitchen", "hallway")
        connect("living_room", "hallway")

        objects = {
            "bed": WorldObject("bed", "bed", "bedroom", usable=True),
            "toilet": WorldObject("toilet", "toilet", "bathroom", usable=True),
            "shower": WorldObject("shower", "shower", "bathroom", usable=True),
            "bathroom_sink": WorldObject(
                "bathroom_sink", "sink", "bathroom", usable=True
            ),
            "fridge": WorldObject(
                "fridge", "refrigerator", "kitchen", openable=True, usable=True
            ),
            "eggs": WorldObject(
                "eggs",
                "food",
                "kitchen",
                portable=True,
                container="fridge",
                properties={
                    "name": "eggs",
                    "raw": True,
                    "cooked": False,
                    "calories": 310.0,
                    "hydration": 0.05,
                },
            ),
            "water": WorldObject(
                "water",
                "drink",
                "kitchen",
                portable=True,
                container="fridge",
                properties={"name": "water", "liters": 0.45},
            ),
            "pan": WorldObject("pan", "cookware", "kitchen", portable=True),
            "plate": WorldObject("plate", "dish", "kitchen", portable=True),
            "stove": WorldObject("stove", "stove", "kitchen", usable=True),
            "kitchen_sink": WorldObject(
                "kitchen_sink", "sink", "kitchen", usable=True
            ),
            "table": WorldObject("table", "table", "kitchen", usable=True),
            "computer": WorldObject(
                "computer", "computer", "living_room", usable=True
            ),
            "sofa": WorldObject("sofa", "seat", "living_room", usable=True),
        }
        return cls(rooms=rooms, objects=objects, start_room="bedroom")

    def execute(self, action: Action, body: BodyState) -> ActionResult:
        try:
            match action.kind:
                case ActionKind.MOVE:
                    return self._move(action.target)
                case ActionKind.INSPECT:
                    return self._inspect(action.target)
                case ActionKind.OPEN:
                    return self._open(action.target)
                case ActionKind.CLOSE:
                    return self._close(action.target)
                case ActionKind.PICK_UP:
                    return self._pick_up(action.target)
                case ActionKind.PUT_DOWN:
                    return self._put_down(action.target)
                case ActionKind.USE:
                    return self._use(action.target, action.secondary, action.value, body)
                case ActionKind.EAT:
                    return self._eat(action.target, body)
                case ActionKind.DRINK:
                    return self._drink(action.target, body)
                case ActionKind.SLEEP:
                    return self._sleep(action.value, body)
                case ActionKind.WAIT:
                    seconds = max(1.0, float(action.value or 60.0))
                    return ActionResult(True, f"Waited for {seconds:.0f} seconds.", seconds)
                case ActionKind.TYPE_TEXT:
                    message = self.computer.type_text(str(action.value or ""))
                    return ActionResult(True, message, 2.0 + len(str(action.value or "")) * 0.08)
                case ActionKind.PRESS_KEY:
                    message = self.computer.press_key(str(action.value or ""))
                    return ActionResult(True, message, 1.0)
                case ActionKind.MOVE_MOUSE:
                    x, y = action.value
                    message = self.computer.move_mouse(int(x), int(y))
                    return ActionResult(True, message, 0.5)
                case ActionKind.CLICK:
                    message = self.computer.click(action.target)
                    return ActionResult(True, message, 0.7)
                case ActionKind.SPEAK:
                    return ActionResult(True, f'Said: "{action.value}"', 2.0)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            return ActionResult(False, str(exc), 1.0, reward=-0.25)
        return ActionResult(False, f"Unsupported action: {action.kind}", 1.0, reward=-0.25)

    def visible_objects(self) -> list[WorldObject]:
        result: list[WorldObject] = []
        for obj in self.objects.values():
            if obj.room != self.agent_room:
                continue
            if obj.container:
                container = self.objects.get(obj.container)
                if container and not container.is_open:
                    continue
            result.append(obj)
        return result

    def observation(self, body: BodyState) -> dict[str, Any]:
        return {
            "room": self.agent_room,
            "neighbors": sorted(self.rooms[self.agent_room].neighbors),
            "visible_objects": [
                {
                    "id": obj.object_id,
                    "kind": obj.kind,
                    "portable": obj.portable,
                    "openable": obj.openable,
                    "is_open": obj.is_open,
                    "properties": dict(obj.properties),
                }
                for obj in self.visible_objects()
            ],
            "inventory": sorted(self.inventory),
            "dirty_dishes": self.dirty_dishes,
            "body": body.snapshot(),
            "computer": self.computer.observe()
            if self.agent_room == "living_room"
            else None,
        }

    def _move(self, target: str | None) -> ActionResult:
        if target not in self.rooms:
            return ActionResult(False, f"Unknown room: {target}", 1.0, -0.25)
        if target == self.agent_room:
            return ActionResult(True, f"Already in {target}.", 1.0, -0.01)
        if target not in self.rooms[self.agent_room].neighbors:
            return ActionResult(
                False,
                f"Cannot move directly from {self.agent_room} to {target}.",
                2.0,
                -0.2,
            )
        old = self.agent_room
        self.agent_room = target
        return ActionResult(True, f"Moved from {old} to {target}.", 25.0, 0.02)

    def _inspect(self, target: str | None) -> ActionResult:
        if target is None:
            visible = ", ".join(obj.object_id for obj in self.visible_objects()) or "nothing"
            return ActionResult(True, f"Visible: {visible}.", 3.0, data=self.observation(BodyState()))
        obj = self.objects.get(target)
        if not obj or not self._reachable(obj):
            return ActionResult(False, f"{target} is not reachable.", 2.0, -0.15)
        return ActionResult(
            True,
            f"Inspected {target}: {obj.kind}, properties={obj.properties}.",
            3.0,
            data=asdict(obj),
        )

    def _open(self, target: str | None) -> ActionResult:
        obj = self._require_reachable(target)
        if not obj.openable:
            return ActionResult(False, f"{target} cannot be opened.", 1.0, -0.1)
        if obj.is_open:
            return ActionResult(True, f"{target} is already open.", 1.0, -0.01)
        obj.is_open = True
        return ActionResult(True, f"Opened {target}.", 2.0, 0.02)

    def _close(self, target: str | None) -> ActionResult:
        obj = self._require_reachable(target)
        if not obj.openable:
            return ActionResult(False, f"{target} cannot be closed.", 1.0, -0.1)
        obj.is_open = False
        return ActionResult(True, f"Closed {target}.", 2.0, 0.01)

    def _pick_up(self, target: str | None) -> ActionResult:
        obj = self._require_reachable(target)
        if not obj.portable:
            return ActionResult(False, f"{target} is not portable.", 1.0, -0.1)
        if target in self.inventory:
            return ActionResult(True, f"Already holding {target}.", 1.0, -0.01)
        obj.container = None
        self.inventory.add(obj.object_id)
        return ActionResult(True, f"Picked up {target}.", 2.0, 0.03)

    def _put_down(self, target: str | None) -> ActionResult:
        if target not in self.inventory:
            return ActionResult(False, f"Not holding {target}.", 1.0, -0.1)
        self.inventory.remove(target)
        obj = self.objects[target]
        obj.room = self.agent_room
        obj.container = None
        return ActionResult(True, f"Put down {target}.", 2.0, 0.01)

    def _use(
        self,
        target: str | None,
        secondary: str | None,
        value: Any,
        body: BodyState,
    ) -> ActionResult:
        obj = self._require_reachable(target)

        if obj.kind == "toilet":
            body.relieve_bladder()
            return ActionResult(True, "Used the restroom.", 180.0, 1.0)

        if obj.kind == "shower":
            body.shower()
            return ActionResult(True, "Took a shower.", 600.0, 0.8)

        if obj.kind == "sink":
            if self.dirty_dishes:
                washed = self.dirty_dishes
                self.dirty_dishes = 0
                return ActionResult(True, f"Washed {washed} dirty dish(es).", 240.0, 0.6)
            return ActionResult(True, "Washed hands at the sink.", 45.0, 0.1)

        if obj.kind == "stove":
            required = {"eggs", "pan"}
            if not required.issubset(self.inventory):
                missing = ", ".join(sorted(required - self.inventory))
                return ActionResult(False, f"Cannot cook; missing {missing}.", 2.0, -0.2)
            eggs = self.objects["eggs"]
            if eggs.properties.get("cooked"):
                return ActionResult(True, "The eggs are already cooked.", 2.0, -0.01)
            eggs.properties["raw"] = False
            eggs.properties["cooked"] = True
            eggs.properties["name"] = "cooked eggs"
            return ActionResult(True, "Cooked the eggs in the pan.", 420.0, 0.9)

        if obj.kind == "computer":
            operation = str(value or "power_on")
            if operation == "power_on":
                message = self.computer.power_on()
            elif operation.startswith("launch:"):
                message = self.computer.launch(operation.split(":", 1)[1])
            else:
                return ActionResult(False, f"Unknown computer operation: {operation}", 1.0, -0.1)
            return ActionResult(True, message, 3.0, 0.1)

        if obj.kind == "seat":
            return ActionResult(True, f"Sat on {target}.", 10.0, 0.02)

        return ActionResult(False, f"No defined use for {target}.", 1.0, -0.1)

    def _eat(self, target: str | None, body: BodyState) -> ActionResult:
        obj = self._require_reachable(target)
        if obj.kind != "food":
            return ActionResult(False, f"{target} is not food.", 1.0, -0.15)
        if obj.properties.get("raw"):
            return ActionResult(False, f"{target} is still raw.", 1.0, -0.3)
        body.eat(
            float(obj.properties.get("calories", 0.0)),
            float(obj.properties.get("hydration", 0.0)),
        )
        self.inventory.discard(obj.object_id)
        obj.room = "consumed"
        self.dirty_dishes += 2 if "plate" in self.inventory else 1
        return ActionResult(True, f"Ate {obj.properties.get('name', target)}.", 600.0, 1.2)

    def _drink(self, target: str | None, body: BodyState) -> ActionResult:
        obj = self._require_reachable(target)
        if obj.kind != "drink":
            return ActionResult(False, f"{target} is not a drink.", 1.0, -0.15)
        body.drink(float(obj.properties.get("liters", 0.0)))
        self.inventory.discard(obj.object_id)
        obj.room = "consumed"
        return ActionResult(True, f"Drank {obj.properties.get('name', target)}.", 60.0, 0.9)

    def _sleep(self, value: Any, body: BodyState) -> ActionResult:
        bed = self.objects["bed"]
        if self.agent_room != bed.room:
            return ActionResult(False, "The bed is not reachable.", 1.0, -0.2)
        hours = max(0.25, min(12.0, float(value or 8.0)))
        body.begin_sleep()
        elapsed = hours * 3600.0
        body.update(elapsed, activity_multiplier=0.45)
        body.end_sleep()
        return ActionResult(True, f"Slept for {hours:.1f} hours.", elapsed, 1.0)

    def _reachable(self, obj: WorldObject) -> bool:
        if obj.object_id in self.inventory:
            return True
        if obj.room != self.agent_room:
            return False
        if obj.container:
            container = self.objects.get(obj.container)
            return bool(container and container.is_open)
        return True

    def _require_reachable(self, target: str | None) -> WorldObject:
        if target is None or target not in self.objects:
            raise KeyError(f"Unknown object: {target}")
        obj = self.objects[target]
        if not self._reachable(obj):
            raise RuntimeError(f"{target} is not reachable.")
        return obj
