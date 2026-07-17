from virtual_person.simulation import VirtualPersonSimulation
from virtual_person.types import Action, ActionKind


def test_restroom_skill_reduces_bladder():
    sim = VirtualPersonSimulation.default(seed=1)
    sim.agent.body.bladder = 0.90
    sim.step()
    assert sim.agent.body.bladder < 0.2
    assert sim.agent.world.agent_room == "bathroom"
    sim.close()


def test_breakfast_cooks_and_eats():
    sim = VirtualPersonSimulation.default(seed=2)
    sim.agent.body.hunger = 0.90
    sim.agent.body.thirst = 0.0
    sim.agent.body.bladder = 0.0
    sim.step()
    assert sim.agent.world.objects["eggs"].room == "consumed"
    assert sim.agent.body.hunger < 0.75
    assert sim.agent.world.dirty_dishes == 0
    sim.close()


def test_computer_task():
    sim = VirtualPersonSimulation.default(seed=3)
    sim.agent.body.hunger = 0.0
    sim.agent.body.thirst = 0.0
    sim.agent.body.bladder = 0.0
    sim.agent.body.fatigue = 0.0
    sim.agent.body.hygiene = 1.0
    sim.step()
    assert sim.agent.daily_computer_task_done is True
    assert sim.agent.world.computer.task_complete is True
    assert "Daily report" in sim.agent.world.computer.notes_text
    sim.close()


def test_invalid_direct_room_move_fails():
    sim = VirtualPersonSimulation.default(seed=4)
    result = sim.agent.execute_action(
        Action(ActionKind.MOVE, "kitchen"),
        sim.sim_time,
    )
    assert result.ok is False
    sim.close()
