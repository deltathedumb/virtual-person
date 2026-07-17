import pytest

from virtual_person.economy import BASIC_JOB_PAYOUT, MAX_TASK_PAYOUT, TaskEconomy, TaskStatus
from virtual_person.simulation import VirtualPersonSimulation
from virtual_person.types import Action, ActionKind


def test_assign_submit_review_flow():
    economy = TaskEconomy()
    task = economy.assign_task("Write a haiku about rain", sim_time=0.0)
    assert task.status == TaskStatus.PENDING
    assert economy.pending_task().task_id == task.task_id

    economy.submit_task(task.task_id, sim_time=10.0, note="Done.")
    assert economy.tasks[task.task_id].status == TaskStatus.SUBMITTED
    assert economy.pending_task() is None

    review = economy.review_task(task.task_id, score=0.8, sim_time=20.0)
    assert review.payout == pytest.approx(0.8 * MAX_TASK_PAYOUT)
    assert economy.balance == pytest.approx(review.payout)
    assert task.task_id not in economy.tasks
    assert economy.history[-1].review.score == 0.8


def test_low_score_pays_little_and_raises_frustration():
    economy = TaskEconomy()
    before_frustration = economy.cognitive_drives.competence_frustration
    task = economy.assign_task("Clean the kitchen")
    economy.submit_task(task.task_id)
    review = economy.review_task(task.task_id, score=0.1)
    assert review.payout == pytest.approx(0.1 * MAX_TASK_PAYOUT)
    assert economy.cognitive_drives.competence_frustration > before_frustration


def test_cannot_review_a_task_that_is_still_pending():
    economy = TaskEconomy()
    task = economy.assign_task("Do something")
    with pytest.raises(ValueError):
        economy.review_task(task.task_id, score=0.5)


def test_cannot_submit_unknown_task():
    economy = TaskEconomy()
    with pytest.raises(KeyError):
        economy.submit_task("task-999")


def test_basic_job_always_available_and_low_value():
    economy = TaskEconomy()
    payout = economy.do_basic_job()
    assert payout == BASIC_JOB_PAYOUT
    assert economy.balance == BASIC_JOB_PAYOUT
    assert payout < MAX_TASK_PAYOUT


def test_empty_task_description_rejected():
    economy = TaskEconomy()
    with pytest.raises(ValueError):
        economy.assign_task("   ")


def test_purchase_groceries_restocks_kitchen_and_deducts_balance():
    sim = VirtualPersonSimulation.default(seed=1)
    world = sim.agent.world
    body = sim.agent.body
    world.economy.balance = 20.0
    world.agent_room = "living_room"
    world.execute(Action(ActionKind.USE, "computer", value="power_on"), body)

    result = world.execute(Action(ActionKind.USE, "computer", value="purchase:groceries"), body)

    assert result.ok is True
    assert world.economy.balance == pytest.approx(12.0)
    assert world.objects["fridge"].is_open is True
    assert world.objects["eggs"].container == "fridge"
    assert world.objects["eggs"].properties["raw"] is True
    sim.close()


def test_purchase_rejected_when_balance_insufficient():
    sim = VirtualPersonSimulation.default(seed=1)
    world = sim.agent.world
    body = sim.agent.body
    world.economy.balance = 2.0
    world.agent_room = "living_room"
    world.execute(Action(ActionKind.USE, "computer", value="power_on"), body)

    result = world.execute(Action(ActionKind.USE, "computer", value="purchase:groceries"), body)

    assert result.ok is False
    assert world.economy.balance == pytest.approx(2.0)
    sim.close()


def test_purchase_hygiene_kit_raises_hygiene():
    sim = VirtualPersonSimulation.default(seed=1)
    world = sim.agent.world
    body = sim.agent.body
    body.hygiene = 0.5
    world.economy.balance = 10.0
    world.agent_room = "living_room"
    world.execute(Action(ActionKind.USE, "computer", value="power_on"), body)

    result = world.execute(Action(ActionKind.USE, "computer", value="purchase:hygiene_kit"), body)

    assert result.ok is True
    assert body.hygiene == pytest.approx(0.85)
    sim.close()


def test_basic_job_action_pays_and_updates_balance():
    sim = VirtualPersonSimulation.default(seed=1)
    world = sim.agent.world
    body = sim.agent.body
    world.agent_room = "living_room"
    world.execute(Action(ActionKind.USE, "computer", value="power_on"), body)

    result = world.execute(Action(ActionKind.USE, "computer", value="basic_job"), body)

    assert result.ok is True
    assert world.economy.balance == pytest.approx(BASIC_JOB_PAYOUT)
    sim.close()
