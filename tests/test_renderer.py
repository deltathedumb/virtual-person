from virtual_person.renderer import ROOM_LAYOUT, RendererConfig, Renderer, _severity
from virtual_person.simulation import VirtualPersonSimulation


def test_room_layout_matches_world_rooms():
    sim = VirtualPersonSimulation.default(seed=1)
    assert set(ROOM_LAYOUT) == set(sim.agent.world.rooms)
    sim.close()


def test_severity_thresholds():
    assert _severity(0.10) == "good"
    assert _severity(0.60) == "warn"
    assert _severity(0.85) == "bad"


def test_renderer_constructs_without_opening_a_window():
    sim = VirtualPersonSimulation.default(seed=1)
    renderer = Renderer(sim, fps=15, config=RendererConfig(fps=15))
    assert renderer.is_running is False
    assert renderer.fps == 15
    sim.close()


def test_renderer_rejects_non_positive_fps():
    sim = VirtualPersonSimulation.default(seed=1)
    try:
        Renderer(sim, fps=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    sim.close()
