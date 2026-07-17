from virtual_person import (
    Action,
    ActionKind,
    MockActuators,
    MockSensors,
    RealWorldConfig,
    RealWorldRuntime,
    VirtualPersonSimulation,
)


with VirtualPersonSimulation.default(name="Mira") as sim:
    sensors = MockSensors()
    actuators = MockActuators()

    runtime = RealWorldRuntime(
        agent=sim.agent,
        sensors=sensors,
        actuators=actuators,
        config=RealWorldConfig(
            dry_run=True,
            allow_computer_control=False,
            physical_location_name="laboratory",
        ),
    )

    observation = runtime.observe()
    print("Observed:", observation)

    print(runtime.execute(Action(
        ActionKind.MOVE,
        value={"linear_mps": 0.2, "angular_rps": 0.0, "seconds": 1.0},
    )))
    print(runtime.execute(Action(ActionKind.SPEAK, value="Hello from the physical world.")))
