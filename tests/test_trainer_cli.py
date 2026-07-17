from __future__ import annotations

import io
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from virtual_person.spike_training import (
    TrainConfig,
    TrainingExample,
    score_model,
    train_model,
)
from virtual_person.spiking import NodeLinkSpikeModel, SpikingModelConfig
from virtual_person.trainer_cli import main
from virtual_person.trainer_support import TrainerProject


def run_cli(arguments: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(arguments)
    return code, stdout.getvalue(), stderr.getvalue()


def test_cli_init_next_and_source_list():
    with tempfile.TemporaryDirectory() as tmp:
        code, output, error = run_cli(
            ["--workspace", tmp, "init", "--starter"]
        )
        assert code == 0, error
        assert "Starter corpus added" in output
        assert (Path(tmp) / "trainer_project.json").is_file()

        code, output, error = run_cli(
            ["--workspace", tmp, "source", "list", "--scan"]
        )
        assert code == 0, error
        assert "English" in output
        assert "Behavior" in output

        code, output, error = run_cli(["--workspace", tmp, "next"])
        assert code == 0, error
        assert "NEXT STEP" in output


def test_cli_profile_and_config():
    with tempfile.TemporaryDirectory() as tmp:
        assert run_cli(["--workspace", tmp, "init"])[0] == 0
        code, output, error = run_cli(
            [
                "--workspace",
                tmp,
                "profile",
                "apply",
                "Architecture smoke test",
            ]
        )
        assert code == 0, error
        project = TrainerProject.load(Path(tmp) / "trainer_project.json")
        assert project.model["hidden_size"] == 32

        code, output, error = run_cli(
            ["--workspace", tmp, "config", "set", "ticks", "2"]
        )
        assert code == 0, error
        project = TrainerProject.load(Path(tmp) / "trainer_project.json")
        assert project.model["ticks"] == 2


def test_cli_neuron_inspection():
    code, output, error = run_cli(
        [
            "neurons",
            "--hunger",
            "0.92",
            "--thirst",
            "0.40",
        ]
    )
    assert code == 0, error
    assert "hunger_urgent" in output
    assert "thirst_notice" in output
    assert "thirst_need" not in output


def test_score_model_reports_action_accuracy():
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
            text="Hunger high. Correct candidate 0.",
            state_features=[0.9] + [0.0] * 15,
            action_target=0,
            value_target=0.8,
        ),
        TrainingExample(
            text="Thirst high. Correct candidate 1.",
            state_features=[0.0, 0.9] + [0.0] * 14,
            action_target=1,
            value_target=0.8,
        ),
    ]
    train_model(
        model,
        examples,
        TrainConfig(
            sequence_length=16,
            batch_size=2,
            epochs=1,
            learning_rate=1e-3,
        ),
    )
    result = score_model(
        model,
        examples,
        sequence_length=16,
        batch_size=2,
    )
    assert result.examples == 2
    assert result.action_examples == 2
    assert result.action_accuracy is not None
    assert result.language_loss >= 0
    assert 0 <= result.spike_rate <= 1


def test_cli_validate_strict_succeeds_for_starter():
    with tempfile.TemporaryDirectory() as tmp:
        assert run_cli(["--workspace", tmp, "init", "--starter"])[0] == 0
        code, output, error = run_cli(
            ["--workspace", tmp, "validate", "--strict"]
        )
        assert code == 0, error
        assert "Total:" in output
