import json
import tempfile
from pathlib import Path

from virtual_person.spike_training import TrainConfig, TrainingExample, train_model
from virtual_person.spiking import NodeLinkSpikeModel, SpikingModelConfig
from virtual_person.trainer_support import (
    MODEL_PROFILES,
    CorpusSource,
    TrainerProject,
    create_starter_pack,
    create_workspace,
    detect_hardware,
    estimate_model,
    filter_sources_for_stage,
    scan_sources,
)


def test_workspace_starter_pack_and_scan():
    with tempfile.TemporaryDirectory() as tmp:
        layout = create_workspace(tmp)
        assert layout["checkpoints"].is_dir()
        sources = create_starter_pack(tmp)
        total, by_path, by_category = scan_sources(sources)
        assert total.records >= 7
        assert total.malformed_records == 0
        assert by_category["English"].records > 0
        assert by_category["Behavior"].action_records > 0


def test_trainer_project_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        project = TrainerProject(
            workspace=tmp,
            sources=[CorpusSource(str(Path(tmp) / "data.txt"), "English")],
            model={"hidden_size": 32},
            training={"stage": "Stage 1 — English and vocabulary"},
        )
        path = project.save()
        loaded = TrainerProject.load(path)
        assert loaded.workspace == tmp
        assert loaded.sources[0].category == "English"
        assert loaded.model["hidden_size"] == 32


def test_model_estimate_and_profiles():
    profile = MODEL_PROFILES["Architecture smoke test"]
    config = SpikingModelConfig(
        hidden_size=profile.hidden_size,
        layer_count=profile.layers,
        ticks_per_token=profile.ticks,
    )
    estimate = estimate_model(
        config,
        batch_size=profile.batch_size,
        sequence_length=profile.sequence_length,
    )
    assert estimate.parameters > 0
    assert estimate.total_training_bytes_estimate > estimate.parameter_bytes_fp32


def test_curriculum_filter():
    sources = [
        CorpusSource("english.txt", "English"),
        CorpusSource("procedures.txt", "Procedures"),
        CorpusSource("behavior.jsonl", "Behavior"),
    ]
    stage1 = filter_sources_for_stage(sources, "Stage 1 — English and vocabulary")
    stage3 = filter_sources_for_stage(sources, "Stage 3 — Autonomous behavior and drives")
    assert [source.category for source in stage1] == ["English"]
    assert len(stage3) == 3


def test_training_progress_callback_and_cancellation():
    model = NodeLinkSpikeModel(
        SpikingModelConfig(
            hidden_size=8,
            layer_count=1,
            ticks_per_token=1,
            max_action_candidates=4,
        )
    )
    examples = [
        TrainingExample(
            "Hunger is high. Choose food.",
            [0.9] + [0.0] * 15,
            action_target=0,
            value_target=0.8,
        ),
        TrainingExample(
            "Thirst is high. Choose water.",
            [0.0, 0.9] + [0.0] * 14,
            action_target=1,
            value_target=0.8,
        ),
    ]
    rows = []
    history = train_model(
        model,
        examples,
        TrainConfig(
            sequence_length=16,
            batch_size=1,
            epochs=2,
            learning_rate=1e-3,
        ),
        progress_callback=lambda row: rows.append(row),
        stop_requested=lambda: len(rows) >= 1,
    )
    assert len(rows) == 1
    assert len(history) == 1
    assert 0.0 < rows[0]["progress"] <= 1.0


def test_hardware_detection_returns_profile_inputs():
    hardware = detect_hardware()
    assert hardware.cpu_count >= 1
    assert hardware.python
