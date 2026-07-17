from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return low if value < low else high if value > high else value


@dataclass(slots=True)
class BodyState:
    """Normalized values: 0 is low, 1 is high unless otherwise noted."""

    hunger: float = 0.25
    thirst: float = 0.20
    fatigue: float = 0.20
    bladder: float = 0.15
    hygiene: float = 0.90
    health: float = 1.00
    social: float = 0.25
    body_temperature_c: float = 36.8

    calories_available: float = 1900.0
    hydration: float = 0.90
    asleep: bool = False

    def update(self, seconds: float, activity_multiplier: float = 1.0) -> None:
        if seconds <= 0:
            return
        hours = seconds / 3600.0
        activity_multiplier = max(0.2, activity_multiplier)

        if self.asleep:
            self.fatigue = _clamp(self.fatigue - 0.13 * hours)
            activity_multiplier = 0.45
        else:
            self.fatigue = _clamp(self.fatigue + 0.035 * hours * activity_multiplier)

        self.hunger = _clamp(self.hunger + 0.030 * hours * activity_multiplier)
        self.thirst = _clamp(self.thirst + 0.050 * hours * activity_multiplier)
        self.bladder = _clamp(self.bladder + 0.030 * hours + (1.0 - self.hydration) * 0.005 * hours)
        self.hygiene = _clamp(self.hygiene - 0.010 * hours * activity_multiplier)
        self.social = _clamp(self.social + 0.018 * hours)

        self.calories_available = max(0.0, self.calories_available - 85.0 * hours * activity_multiplier)
        self.hydration = _clamp(self.hydration - 0.025 * hours * activity_multiplier)

        # Needs affect health only when seriously neglected.
        distress = (
            max(0.0, self.hunger - 0.92)
            + max(0.0, self.thirst - 0.90) * 1.5
            + max(0.0, self.bladder - 0.97)
            + max(0.0, self.fatigue - 0.97)
        )
        if distress:
            self.health = _clamp(self.health - distress * 0.01 * hours)

    def eat(self, calories: float, hydration: float = 0.0) -> None:
        calories = max(0.0, calories)
        self.calories_available += calories
        self.hunger = _clamp(self.hunger - calories / 900.0)
        if hydration:
            self.drink(hydration)

    def drink(self, liters: float) -> None:
        liters = max(0.0, liters)
        self.hydration = _clamp(self.hydration + liters / 1.5)
        self.thirst = _clamp(self.thirst - liters / 0.75)
        self.bladder = _clamp(self.bladder + liters / 2.0)

    def relieve_bladder(self) -> None:
        self.bladder = 0.05
        self.hygiene = _clamp(self.hygiene - 0.015)

    def shower(self) -> None:
        self.hygiene = 1.0

    def begin_sleep(self) -> None:
        self.asleep = True

    def end_sleep(self) -> None:
        self.asleep = False

    def snapshot(self) -> dict[str, float | bool]:
        return asdict(self)

    def need_pressure(self) -> float:
        """A compact stress score useful for reward shaping."""
        cleanliness_pressure = 1.0 - self.hygiene
        return (
            self.hunger**2
            + self.thirst**2
            + self.fatigue**2
            + self.bladder**2
            + cleanliness_pressure**2
            + self.social**2 * 0.25
        )
