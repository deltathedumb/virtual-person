import pytest

pytest.importorskip("tkinter")

import tkinter as tk

from virtual_person.env import VirtualPersonEnv
from virtual_person.runtime_ui import RuntimeApp
from virtual_person.spiking import NodeLinkSpikeModel, SpikingModelConfig
from virtual_person.spiking_mind import SpikingMind
from virtual_person.spiking_runtime import AutonomousSpikingAgent


def _tk_available() -> bool:
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except tk.TclError:
        return False


requires_display = pytest.mark.skipif(not _tk_available(), reason="No display available for Tk")


@requires_display
def test_app_constructs_and_lays_out_without_a_checkpoint():
    root = tk.Tk()
    app = RuntimeApp(root)
    root.update_idletasks()
    assert app.agent_runtime is None
    assert str(app.start_button.cget("state")) == "disabled"
    root.destroy()


@requires_display
def test_full_task_lifecycle_through_the_ui():
    root = tk.Tk()
    app = RuntimeApp(root)

    config = SpikingModelConfig(hidden_size=16, layer_count=1, ticks_per_token=1)
    model = NodeLinkSpikeModel(config)
    mind = SpikingMind(model, device="cpu")
    app.agent_runtime = AutonomousSpikingAgent(mind, VirtualPersonEnv())
    app._render_all()

    app._on_step()
    assert app.last_decision_message

    app.new_task_entry.insert(0, "Write a short note")
    app._on_assign_task()
    assert app.pending_list.size() == 1

    economy = app._economy
    task_id = next(iter(economy.tasks))
    economy.submit_task(task_id, sim_time=0.0)
    app._refresh_task_lists()
    assert app.review_list.size() == 1

    app.review_list.selection_set(0)
    app.score_var.set(0.75)
    app._on_review_selected()

    assert economy.balance == pytest.approx(37.5)
    root.destroy()
