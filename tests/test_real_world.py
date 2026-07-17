from virtual_person import (
    Action,
    ActionKind,
    MockActuators,
    MockSensors,
    RealWorldConfig,
    RealWorldRuntime,
    VirtualPersonSimulation,
)


def test_real_world_dry_run_move_is_bounded():
    sim = VirtualPersonSimulation.default(seed=5)
    actuators = MockActuators()
    runtime = RealWorldRuntime(
        sim.agent,
        MockSensors(),
        actuators,
        config=RealWorldConfig(dry_run=True),
    )
    result = runtime.execute(Action(
        ActionKind.MOVE,
        value={"linear_mps": 99.0, "angular_rps": 99.0, "seconds": 99.0},
    ))
    assert result.ok is True
    assert "0.35" in result.message
    assert "0.75" in result.message
    assert "5.00" in result.message
    sim.close()


def test_real_world_dangerous_action_denied_by_default():
    sim = VirtualPersonSimulation.default(seed=6)
    runtime = RealWorldRuntime(
        sim.agent,
        MockSensors(),
        MockActuators(),
        config=RealWorldConfig(dry_run=True),
    )
    result = runtime.execute(Action(ActionKind.USE, "stove", value="use_stove"))
    assert result.ok is False
    assert "approval" in result.message.lower()
    sim.close()


def test_emergency_stop_blocks_actions():
    sim = VirtualPersonSimulation.default(seed=7)
    runtime = RealWorldRuntime(
        sim.agent,
        MockSensors(),
        MockActuators(),
        config=RealWorldConfig(dry_run=True),
    )
    runtime.emergency_stop.trigger()
    result = runtime.execute(Action(ActionKind.SPEAK, value="test"))
    assert result.ok is False
    sim.close()
