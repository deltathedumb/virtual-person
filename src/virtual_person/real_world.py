from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterable

from .agent import VirtualPerson
from .types import Action, ActionKind, ActionResult


class RiskLevel(Enum):
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass(slots=True)
class SensorSnapshot:
    monotonic_time: float
    wall_time: float
    location: str | None = None
    battery: float | None = None
    camera_frames: dict[str, Any] = field(default_factory=dict)
    microphone_chunk: Any = None
    lidar: Any = None
    joint_positions: dict[str, float] = field(default_factory=dict)
    touch: dict[str, float] = field(default_factory=dict)
    temperature_c: float | None = None
    detected_objects: list[dict[str, Any]] = field(default_factory=list)
    computer_accessibility_tree: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SensorSuite(ABC):
    @abstractmethod
    def read(self) -> SensorSnapshot:
        """Return one synchronized observation of the physical environment."""


class ActuatorSuite(ABC):
    @abstractmethod
    def stop_all(self) -> None:
        """Immediately command all actuators to stop."""

    @abstractmethod
    def move_base(self, linear_mps: float, angular_rps: float, seconds: float) -> None:
        """Move a mobile base at bounded speed."""

    @abstractmethod
    def move_joint(self, joint: str, position: float, speed: float) -> None:
        """Move one joint using the hardware controller's closed-loop control."""

    @abstractmethod
    def gripper(self, hand: str, opening: float, force_limit: float) -> None:
        """Set gripper opening with a force limit."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Produce speech through the robot's speaker."""

    @abstractmethod
    def computer_input(self, operation: str, payload: dict[str, Any]) -> None:
        """Send an allowlisted mouse/keyboard operation to a sandboxed computer."""


@dataclass(slots=True)
class SafetyPolicy:
    max_linear_mps: float = 0.35
    max_angular_rps: float = 0.75
    max_joint_speed: float = 0.25
    max_grip_force: float = 12.0
    minimum_battery: float = 0.12
    require_human_approval_for: set[str] = field(default_factory=lambda: {
        "cook",
        "use_stove",
        "use_knife",
        "open_exterior_door",
        "handle_hot_object",
        "administer_substance",
    })
    forbidden_operations: set[str] = field(default_factory=lambda: {
        "disable_safety",
        "bypass_emergency_stop",
        "access_credentials",
        "unrestricted_shell",
        "financial_transaction",
        "weapon_operation",
    })


class ApprovalProvider(ABC):
    @abstractmethod
    def approve(self, operation: str, context: dict[str, Any]) -> bool:
        """Return True only when a human explicitly approves the operation."""


class DenyByDefaultApproval(ApprovalProvider):
    def approve(self, operation: str, context: dict[str, Any]) -> bool:
        return False


class EmergencyStop:
    def __init__(self) -> None:
        self._event = threading.Event()

    def trigger(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event.clear()

    @property
    def active(self) -> bool:
        return self._event.is_set()


@dataclass(slots=True)
class RealWorldConfig:
    control_hz: float = 5.0
    observation_timeout_seconds: float = 2.0
    dry_run: bool = True
    allow_computer_control: bool = False
    physical_location_name: str = "real_world"
    body_description: str = "robotic body with cameras, microphones, joints, and grippers"


class RealWorldRuntime:
    """
    Hardware-abstraction runtime for placing a VirtualPerson in the physical world.

    This runtime deliberately does not infer low-level robot control from arbitrary
    language. High-level actions must be translated into bounded, allowlisted
    actuator commands. Dangerous operations are denied or require human approval.
    """

    def __init__(
        self,
        agent: VirtualPerson,
        sensors: SensorSuite,
        actuators: ActuatorSuite,
        *,
        config: RealWorldConfig | None = None,
        safety: SafetyPolicy | None = None,
        approval: ApprovalProvider | None = None,
        emergency_stop: EmergencyStop | None = None,
    ) -> None:
        self.agent = agent
        self.sensors = sensors
        self.actuators = actuators
        self.config = config or RealWorldConfig()
        self.safety = safety or SafetyPolicy()
        self.approval = approval or DenyByDefaultApproval()
        self.emergency_stop = emergency_stop or EmergencyStop()
        self.running = False
        self.last_snapshot: SensorSnapshot | None = None

        self.agent.self_model.identity = "embodied person with a robotic body"
        self.agent.self_model.home_location = self.config.physical_location_name
        self.agent.self_model.beliefs.update({
            "embodiment": f"My body is a {self.config.body_description}.",
            "environment": "I perceive and act in the physical world through hardware.",
            "limitations": (
                "My sensors can be incomplete and my actuators are safety-limited. "
                "I must not claim certainty when perception is ambiguous."
            ),
        })

    def observe(self) -> SensorSnapshot:
        snapshot = self.sensors.read()
        if snapshot.battery is not None and snapshot.battery < self.safety.minimum_battery:
            self.emergency_stop.trigger()
            self.actuators.stop_all()
            raise RuntimeError("Battery below the configured safety threshold.")
        self.last_snapshot = snapshot
        return snapshot

    def execute(self, action: Action) -> ActionResult:
        if self.emergency_stop.active:
            self.actuators.stop_all()
            return ActionResult(False, "Emergency stop is active.", 0.0, -1.0)

        try:
            if action.kind is ActionKind.SPEAK:
                return self._speak(str(action.value or ""))

            if action.kind is ActionKind.MOVE:
                return self._move(action.value)

            if action.kind is ActionKind.MOVE_MOUSE:
                return self._computer("move_mouse", {"position": action.value})

            if action.kind is ActionKind.CLICK:
                return self._computer("click", {"target": action.target})

            if action.kind is ActionKind.TYPE_TEXT:
                return self._computer("type_text", {"text": str(action.value or "")})

            if action.kind is ActionKind.PRESS_KEY:
                return self._computer("press_key", {"key": str(action.value or "")})

            if action.kind is ActionKind.USE:
                operation = str(action.value or action.target or "")
                return self._approved_physical_operation(operation, action)

            if action.kind is ActionKind.WAIT:
                seconds = max(0.0, min(60.0, float(action.value or 1.0)))
                if not self.config.dry_run:
                    time.sleep(seconds)
                return ActionResult(True, f"Waited {seconds:.1f} seconds.", seconds)

            return ActionResult(
                False,
                f"Real-world action {action.kind.name} has no hardware skill binding.",
                0.0,
                -0.25,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            self.actuators.stop_all()
            return ActionResult(False, str(exc), 0.0, -0.5)

    def run_control_loop(
        self,
        policy: Callable[[SensorSnapshot], Action | None],
        *,
        max_steps: int | None = None,
    ) -> None:
        interval = 1.0 / max(0.1, self.config.control_hz)
        self.running = True
        steps = 0
        try:
            while self.running and not self.emergency_stop.active:
                started = time.monotonic()
                snapshot = self.observe()
                action = policy(snapshot)
                if action is not None:
                    self.execute(action)
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break
                remaining = interval - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            self.running = False
            self.actuators.stop_all()

    def stop(self) -> None:
        self.running = False
        self.actuators.stop_all()

    def _move(self, value: Any) -> ActionResult:
        if not isinstance(value, dict):
            raise TypeError(
                "Real-world MOVE requires {'linear_mps', 'angular_rps', 'seconds'}."
            )
        linear = float(value.get("linear_mps", 0.0))
        angular = float(value.get("angular_rps", 0.0))
        seconds = max(0.0, min(5.0, float(value.get("seconds", 0.0))))

        linear = max(-self.safety.max_linear_mps, min(self.safety.max_linear_mps, linear))
        angular = max(-self.safety.max_angular_rps, min(self.safety.max_angular_rps, angular))

        if not self.config.dry_run:
            self.actuators.move_base(linear, angular, seconds)
        return ActionResult(
            True,
            f"Commanded base motion linear={linear:.2f} m/s, "
            f"angular={angular:.2f} rad/s for {seconds:.2f}s.",
            seconds,
            0.0,
        )

    def _speak(self, text: str) -> ActionResult:
        text = text.strip()
        if not text:
            return ActionResult(False, "Speech text was empty.", 0.0, -0.05)
        if not self.config.dry_run:
            self.actuators.speak(text)
        return ActionResult(True, f'Spoke: "{text}"', len(text) / 12.0)

    def _computer(self, operation: str, payload: dict[str, Any]) -> ActionResult:
        if not self.config.allow_computer_control:
            return ActionResult(False, "Computer control is disabled.", 0.0, -0.2)
        if operation in self.safety.forbidden_operations:
            return ActionResult(False, f"Forbidden computer operation: {operation}", 0.0, -1.0)
        if not self.config.dry_run:
            self.actuators.computer_input(operation, payload)
        return ActionResult(True, f"Performed sandboxed computer operation {operation}.", 0.5)

    def _approved_physical_operation(self, operation: str, action: Action) -> ActionResult:
        if operation in self.safety.forbidden_operations:
            return ActionResult(False, f"Forbidden operation: {operation}", 0.0, -1.0)

        context = {
            "operation": operation,
            "target": action.target,
            "secondary": action.secondary,
            "value": action.value,
            "last_snapshot": self.last_snapshot.metadata if self.last_snapshot else {},
        }

        if operation in self.safety.require_human_approval_for:
            if not self.approval.approve(operation, context):
                return ActionResult(
                    False,
                    f"Human approval was not granted for {operation}.",
                    0.0,
                    -0.5,
                )

        # Hardware-specific skills are intentionally delegated to a robot adapter.
        # This scaffold reports success only in dry-run mode until a concrete,
        # tested skill binding is registered.
        if self.config.dry_run:
            return ActionResult(
                True,
                f"Dry-run approved physical operation: {operation}.",
                0.0,
                0.0,
            )
        return ActionResult(
            False,
            f"No concrete hardware skill is registered for {operation}.",
            0.0,
            -0.25,
        )


class MockSensors(SensorSuite):
    """Useful for testing the real-world runtime without hardware."""

    def __init__(self) -> None:
        self.location = "laboratory"
        self.battery = 1.0

    def read(self) -> SensorSnapshot:
        return SensorSnapshot(
            monotonic_time=time.monotonic(),
            wall_time=time.time(),
            location=self.location,
            battery=self.battery,
            detected_objects=[
                {"id": "table", "label": "table", "confidence": 0.99},
                {"id": "chair", "label": "chair", "confidence": 0.98},
            ],
            metadata={"source": "mock"},
        )


class MockActuators(ActuatorSuite):
    """Records commands instead of controlling hardware."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.stopped = False

    def stop_all(self) -> None:
        self.stopped = True
        self.commands.append(("stop_all", {}))

    def move_base(self, linear_mps: float, angular_rps: float, seconds: float) -> None:
        self.commands.append(("move_base", {
            "linear_mps": linear_mps,
            "angular_rps": angular_rps,
            "seconds": seconds,
        }))

    def move_joint(self, joint: str, position: float, speed: float) -> None:
        self.commands.append(("move_joint", {
            "joint": joint,
            "position": position,
            "speed": speed,
        }))

    def gripper(self, hand: str, opening: float, force_limit: float) -> None:
        self.commands.append(("gripper", {
            "hand": hand,
            "opening": opening,
            "force_limit": force_limit,
        }))

    def speak(self, text: str) -> None:
        self.commands.append(("speak", {"text": text}))

    def computer_input(self, operation: str, payload: dict[str, Any]) -> None:
        self.commands.append(("computer_input", {
            "operation": operation,
            "payload": payload,
        }))
