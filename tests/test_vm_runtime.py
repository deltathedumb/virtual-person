import shutil
import subprocess

import pytest

from virtual_person.real_world import RealWorldConfig, RealWorldRuntime
from virtual_person.simulation import VirtualPersonSimulation
from virtual_person.types import Action, ActionKind
from virtual_person.vm_runtime import QemuActuators, QemuSensors, QemuVirtualMachine, QemuVmConfig

_QEMU_AVAILABLE = shutil.which("qemu-system-x86_64") is not None
_QEMU_IMG_AVAILABLE = shutil.which("qemu-img") is not None

requires_qemu = pytest.mark.skipif(
    not (_QEMU_AVAILABLE and _QEMU_IMG_AVAILABLE),
    reason="qemu-system-x86_64 / qemu-img not found on PATH",
)


@pytest.fixture()
def blank_disk(tmp_path):
    disk_path = tmp_path / "test_disk.qcow2"
    subprocess.run(
        ["qemu-img", "create", "-f", "qcow2", str(disk_path), "256M"],
        check=True, capture_output=True,
    )
    return disk_path


@pytest.fixture()
def running_vm(blank_disk):
    config = QemuVmConfig(disk_path=str(blank_disk), qmp_port=15234, memory_mb=256)
    vm = QemuVirtualMachine(config)
    vm.start(boot_timeout_seconds=15)
    yield vm
    vm.stop()


@requires_qemu
def test_vm_starts_and_stops_cleanly(blank_disk):
    config = QemuVmConfig(disk_path=str(blank_disk), qmp_port=15235, memory_mb=256)
    vm = QemuVirtualMachine(config)
    assert vm.running is False

    vm.start(boot_timeout_seconds=15)
    assert vm.running is True

    vm.stop()
    assert vm.running is False


@requires_qemu
def test_vm_context_manager(blank_disk):
    config = QemuVmConfig(disk_path=str(blank_disk), qmp_port=15236, memory_mb=256)
    with QemuVirtualMachine(config) as vm:
        assert vm.running is True
    assert vm.running is False


@requires_qemu
def test_sensor_read_returns_real_screen_bytes(running_vm):
    sensors = QemuSensors(running_vm)
    snapshot = sensors.read()
    assert snapshot.location == "virtual_machine"
    assert len(snapshot.camera_frames["vm_screen"]) > 0


@requires_qemu
def test_actuator_mouse_keyboard_operations_do_not_raise(running_vm):
    actuators = QemuActuators(running_vm)
    actuators.computer_input("move_mouse", {"position": (100, 100)})
    actuators.computer_input("click", {})
    actuators.computer_input("type_text", {"text": "Hello"})
    actuators.computer_input("press_key", {"key": "enter"})
    assert len(actuators.commands) == 4


@requires_qemu
def test_actuator_rejects_unknown_operation(running_vm):
    actuators = QemuActuators(running_vm)
    with pytest.raises(ValueError):
        actuators.computer_input("launch_missiles", {})


@requires_qemu
def test_full_real_world_runtime_integration(running_vm):
    sim = VirtualPersonSimulation.default(seed=1)
    runtime = RealWorldRuntime(
        sim.agent,
        QemuSensors(running_vm),
        QemuActuators(running_vm),
        config=RealWorldConfig(dry_run=False, allow_computer_control=True),
    )

    observation = runtime.observe()
    assert len(observation.camera_frames["vm_screen"]) > 0

    result = runtime.execute(Action(ActionKind.MOVE_MOUSE, value=(200, 150)))
    assert result.ok is True

    result = runtime.execute(Action(ActionKind.USE, value="disable_safety"))
    assert result.ok is False
    assert "Forbidden" in result.message

    sim.close()


@requires_qemu
def test_computer_control_disabled_by_default(running_vm):
    sim = VirtualPersonSimulation.default(seed=1)
    runtime = RealWorldRuntime(
        sim.agent,
        QemuSensors(running_vm),
        QemuActuators(running_vm),
        config=RealWorldConfig(dry_run=False, allow_computer_control=False),
    )

    result = runtime.execute(Action(ActionKind.TYPE_TEXT, value="should not reach the VM"))
    assert result.ok is False
    assert "disabled" in result.message.lower()
    sim.close()
