"""QEMU-backed virtual machine sensors and actuators.

Implements the existing `SensorSuite` / `ActuatorSuite` contract from
`real_world.py` against a real QEMU virtual machine, using QEMU's QMP
(QEMU Machine Protocol) control socket for host-side mouse, keyboard, and
screen capture. No hypervisor input is exposed directly to the learned
model: this module is only ever driven through `RealWorldRuntime`, which
still applies the same dry-run gating, forbidden-operation checks, and
human-approval requirements as any other actuator suite.

The VM is a disposable, isolated guest. Nothing here gives the model access
to this host's filesystem, credentials, or network beyond whatever the VM's
own virtual NIC is configured to reach (defaults to QEMU's user-mode NAT,
which cannot be reached from the LAN and cannot reach LAN-local host
services either).
"""
from __future__ import annotations

import json
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .real_world import ActuatorSuite, SensorSnapshot, SensorSuite

# A minimal mapping from ASCII characters to QEMU qcodes. QEMU's QMP key
# events want named "qcodes", not raw characters, so typed text has to be
# translated one character at a time. This covers ordinary printable ASCII;
# anything outside this set is skipped rather than guessed at.
_QCODE_SHIFT_MAP: dict[str, tuple[str, bool]] = {}
for _lower, _upper in zip("abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    _QCODE_SHIFT_MAP[_lower] = (_lower, False)
    _QCODE_SHIFT_MAP[_upper] = (_lower, True)
for _digit in "0123456789":
    _QCODE_SHIFT_MAP[_digit] = (_digit, False)
_QCODE_SHIFT_MAP.update({
    " ": ("spc", False),
    "\n": ("ret", False),
    "\t": ("tab", False),
    ".": ("dot", False),
    ",": ("comma", False),
    "-": ("minus", False),
    "/": ("slash", False),
    ";": ("semicolon", False),
    "'": ("apostrophe", False),
})

_KEY_NAME_MAP = {
    "enter": "ret",
    "return": "ret",
    "tab": "tab",
    "backspace": "backspace",
    "escape": "esc",
    "esc": "esc",
    "space": "spc",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
}


class QmpProtocolError(RuntimeError):
    pass


class QmpClient:
    """A small synchronous client for QEMU's QMP JSON control protocol."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, timeout: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket: socket.socket | None = None
        self._file = None

    def connect(self) -> None:
        self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._file = self._socket.makefile("rwb")
        greeting = self._read()
        if "QMP" not in greeting:
            raise QmpProtocolError(f"Unexpected QMP greeting: {greeting!r}")
        response = self.execute("qmp_capabilities")
        if "error" in response:
            raise QmpProtocolError(f"qmp_capabilities failed: {response['error']}")

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None
                self._file = None

    def _read(self) -> dict[str, Any]:
        if self._file is None:
            raise QmpProtocolError("Not connected.")
        line = self._file.readline()
        if not line:
            raise QmpProtocolError("QMP connection closed unexpectedly.")
        return json.loads(line)

    def execute(self, command: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._file is None:
            raise QmpProtocolError("Not connected.")
        payload: dict[str, Any] = {"execute": command}
        if arguments:
            payload["arguments"] = arguments
        self._file.write((json.dumps(payload) + "\n").encode("utf-8"))
        self._file.flush()
        # Skip asynchronous "event" messages the server may interleave
        # with the actual command reply.
        while True:
            response = self._read()
            if "event" not in response:
                return response


@dataclass(slots=True)
class QemuVmConfig:
    disk_path: str
    memory_mb: int = 2048
    cpu_count: int = 2
    qmp_host: str = "127.0.0.1"
    qmp_port: int = 14444
    extra_args: list[str] = field(default_factory=list)
    headless: bool = True
    # "usb-tablet" reports absolute coordinates, which QMP's "abs" mouse
    # events expect; a plain PS/2 mouse only understands relative deltas.
    input_device_args: list[str] = field(default_factory=lambda: [
        "-device", "usb-tablet",
        "-usb",
    ])


class QemuVirtualMachine:
    """Launches and owns a single QEMU process, with a QMP control connection."""

    def __init__(self, config: QemuVmConfig) -> None:
        self.config = config
        self._process: subprocess.Popen | None = None
        self.qmp = QmpClient(host=config.qmp_host, port=config.qmp_port)

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, *, boot_timeout_seconds: float = 15.0) -> None:
        if self.running:
            return
        args = [
            "qemu-system-x86_64",
            "-m", str(self.config.memory_mb),
            "-smp", str(self.config.cpu_count),
            "-drive", f"file={self.config.disk_path},format=qcow2",
            "-qmp", f"tcp:{self.config.qmp_host}:{self.config.qmp_port},server,nowait",
            *self.config.input_device_args,
            *self.config.extra_args,
        ]
        if self.config.headless:
            args += ["-display", "none"]
        self._process = subprocess.Popen(args)

        deadline = time.monotonic() + boot_timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self.qmp.connect()
                return
            except (ConnectionRefusedError, OSError, QmpProtocolError) as exc:
                last_error = exc
                time.sleep(0.3)
        raise RuntimeError(
            f"QEMU did not accept a QMP connection within {boot_timeout_seconds}s: {last_error}"
        )

    def stop(self, *, force_after_seconds: float = 10.0) -> None:
        if not self.running:
            return
        try:
            self.qmp.execute("quit")
        except (QmpProtocolError, OSError):
            pass
        self.qmp.close()
        assert self._process is not None
        try:
            self._process.wait(timeout=force_after_seconds)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None

    def __enter__(self) -> "QemuVirtualMachine":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()


class QemuSensors(SensorSuite):
    """Reads the VM's screen via QMP screendump. No other host access."""

    def __init__(self, vm: QemuVirtualMachine) -> None:
        self.vm = vm

    def read(self) -> SensorSnapshot:
        with tempfile.TemporaryDirectory() as tmp:
            screenshot_path = Path(tmp) / "frame.ppm"
            response = self.vm.qmp.execute(
                "screendump", {"filename": str(screenshot_path)}
            )
            if "error" in response:
                raise RuntimeError(f"screendump failed: {response['error']}")
            frame_bytes = screenshot_path.read_bytes() if screenshot_path.is_file() else b""

        return SensorSnapshot(
            monotonic_time=time.monotonic(),
            wall_time=time.time(),
            location="virtual_machine",
            camera_frames={"vm_screen": frame_bytes},
            metadata={"source": "qemu_qmp", "format": "ppm"},
        )


class QemuActuators(ActuatorSuite):
    """Sends mouse/keyboard input to the VM via QMP. Everything else is a no-op.

    A VM has no mobile base, joints, or gripper, so those methods only
    record that they were called; only `computer_input` and `stop_all` do
    anything real.
    """

    def __init__(self, vm: QemuVirtualMachine, *, screen_width: int = 1024, screen_height: int = 768) -> None:
        self.vm = vm
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.commands: list[tuple[str, dict[str, Any]]] = []

    def stop_all(self) -> None:
        self.commands.append(("stop_all", {}))
        try:
            self.vm.qmp.execute("stop")
        except QmpProtocolError:
            pass

    def move_base(self, linear_mps: float, angular_rps: float, seconds: float) -> None:
        self.commands.append(("move_base", {
            "linear_mps": linear_mps, "angular_rps": angular_rps, "seconds": seconds,
        }))

    def move_joint(self, joint: str, position: float, speed: float) -> None:
        self.commands.append(("move_joint", {"joint": joint, "position": position, "speed": speed}))

    def gripper(self, hand: str, opening: float, force_limit: float) -> None:
        self.commands.append(("gripper", {"hand": hand, "opening": opening, "force_limit": force_limit}))

    def speak(self, text: str) -> None:
        self.commands.append(("speak", {"text": text}))

    def computer_input(self, operation: str, payload: dict[str, Any]) -> None:
        self.commands.append(("computer_input", {"operation": operation, "payload": payload}))
        if operation == "move_mouse":
            self._move_mouse(payload.get("position"))
        elif operation == "click":
            self._click()
        elif operation == "type_text":
            self._type_text(str(payload.get("text", "")))
        elif operation == "press_key":
            self._press_key(str(payload.get("key", "")))
        else:
            raise ValueError(f"Unsupported VM computer operation: {operation}")

    # -- internals ------------------------------------------------------------

    def _abs_axis_value(self, pixel: int, extent: int) -> int:
        # QMP absolute axes are scaled to 0..32767 regardless of the guest's
        # actual resolution.
        pixel = max(0, min(extent - 1, int(pixel)))
        return int(pixel / max(1, extent - 1) * 32767)

    def _move_mouse(self, position: Any) -> None:
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            raise TypeError("move_mouse requires a (x, y) pixel position.")
        x, y = position
        events = [
            {"type": "abs", "data": {"axis": "x", "value": self._abs_axis_value(x, self.screen_width)}},
            {"type": "abs", "data": {"axis": "y", "value": self._abs_axis_value(y, self.screen_height)}},
        ]
        self.vm.qmp.execute("input-send-event", {"events": events})

    def _click(self) -> None:
        events_down = [{"type": "btn", "data": {"down": True, "button": "left"}}]
        events_up = [{"type": "btn", "data": {"down": False, "button": "left"}}]
        self.vm.qmp.execute("input-send-event", {"events": events_down})
        self.vm.qmp.execute("input-send-event", {"events": events_up})

    def _press_key(self, key: str) -> None:
        qcode = _KEY_NAME_MAP.get(key.lower())
        if qcode is None:
            raise ValueError(f"Unsupported key: {key!r}")
        self._send_qcode(qcode)

    def _type_text(self, text: str) -> None:
        for char in text:
            mapping = _QCODE_SHIFT_MAP.get(char)
            if mapping is None:
                continue
            qcode, needs_shift = mapping
            if needs_shift:
                self._send_qcode("shift", down_only=True)
            self._send_qcode(qcode)
            if needs_shift:
                self._send_qcode("shift", up_only=True)

    def _send_qcode(self, qcode: str, *, down_only: bool = False, up_only: bool = False) -> None:
        key_data = {"type": "qcode", "data": qcode}
        if not up_only:
            self.vm.qmp.execute(
                "input-send-event",
                {"events": [{"type": "key", "data": {"down": True, "key": key_data}}]},
            )
        if not down_only:
            self.vm.qmp.execute(
                "input-send-event",
                {"events": [{"type": "key", "data": {"down": False, "key": key_data}}]},
            )
