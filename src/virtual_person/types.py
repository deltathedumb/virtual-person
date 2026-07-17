from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ActionKind(Enum):
    MOVE = auto()
    INSPECT = auto()
    OPEN = auto()
    CLOSE = auto()
    PICK_UP = auto()
    PUT_DOWN = auto()
    USE = auto()
    EAT = auto()
    DRINK = auto()
    SLEEP = auto()
    WAIT = auto()
    TYPE_TEXT = auto()
    PRESS_KEY = auto()
    MOVE_MOUSE = auto()
    CLICK = auto()
    SPEAK = auto()


@dataclass(slots=True, frozen=True)
class Action:
    kind: ActionKind
    target: str | None = None
    secondary: str | None = None
    value: Any = None


@dataclass(slots=True)
class ActionResult:
    ok: bool
    message: str
    elapsed_seconds: float = 0.0
    reward: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


class GoalKind(Enum):
    USE_RESTROOM = auto()
    DRINK = auto()
    EAT = auto()
    SLEEP = auto()
    SHOWER = auto()
    COMPUTER_TASK = auto()
    SOCIALIZE = auto()
    EXPLORE = auto()
    IDLE = auto()


@dataclass(slots=True, frozen=True)
class Goal:
    kind: GoalKind
    reason: str
    urgency: float
