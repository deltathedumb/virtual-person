from virtual_person import VirtualPersonSimulation


with VirtualPersonSimulation.default(
    seed=42,
    memory_path="mira-memory.sqlite3",
    name="Mira",
) as sim:
    sim.agent.self_model.beliefs["purpose"] = (
        "Maintain my home, learn skills, and complete assigned tasks."
    )
    sim.agent.body.hunger = 0.70
    sim.agent.body.thirst = 0.60
    sim.run(hours=12)

    for memory in reversed(sim.agent.memory.recent(10)):
        print(f"[{memory.kind}] {memory.summary}")
