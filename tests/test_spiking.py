import tempfile
from pathlib import Path

import torch

from virtual_person import (
    ByteTokenizer,
    DedicatedDriveNeuronBank,
    NodeLinkSpikeCluster,
    NodeLinkSpikeModel,
    SpikingMind,
    SpikingModelConfig,
)
from virtual_person.drives import CognitiveDrives
from virtual_person.spike_training import (
    TrainConfig,
    TrainingExample,
    build_dictionary_corpus,
    train_model,
)


def test_byte_tokenizer_round_trip():
    tokenizer = ByteTokenizer()
    text = "English, café, and symbols: ✓"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_surrogate_gradient_reaches_links():
    cluster = NodeLinkSpikeCluster(3, 4, refractory_ticks=0)
    state = cluster.initial_state(2)
    current = torch.ones(2, 3, requires_grad=True)
    spikes, _ = cluster.step(current, state)
    loss = spikes.sum()
    loss.backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert cluster.input_links.weight.grad is not None


def test_spiking_model_shapes_and_sparse_rates():
    config = SpikingModelConfig(
        hidden_size=24,
        layer_count=2,
        ticks_per_token=2,
        state_feature_count=16,
        max_action_candidates=8,
    )
    model = NodeLinkSpikeModel(config)
    tokens = torch.randint(0, config.vocab_size, (3, 12))
    features = torch.zeros(3, 16)
    output = model(tokens, features)
    assert output.token_logits.shape == (3, 12, config.vocab_size)
    assert output.action_logits.shape == (3, 8)
    assert output.value.shape == (3,)
    assert output.spike_rates.shape == (3, 12, 24)
    assert 0.0 <= float(output.spike_rates.mean().detach()) <= 1.0


def test_tiny_training_step_is_finite():
    config = SpikingModelConfig(
        hidden_size=16,
        layer_count=1,
        ticks_per_token=1,
        max_action_candidates=4,
    )
    model = NodeLinkSpikeModel(config)
    examples = [
        TrainingExample(
            "Hunger is high. Candidate 0 prepares food. Choose candidate 0.",
            [0.8] + [0.0] * 15,
            action_target=0,
            value_target=0.8,
        ),
        TrainingExample(
            "Fatigue is high. Candidate 1 sleeps. Choose candidate 1.",
            [0.0, 0.0, 0.9] + [0.0] * 13,
            action_target=1,
            value_target=0.9,
        ),
    ]
    history = train_model(
        model,
        examples,
        TrainConfig(
            sequence_length=32,
            batch_size=2,
            epochs=1,
            learning_rate=1e-3,
            device="cpu",
        ),
    )
    assert len(history) == 1
    assert history[0]["loss"] == history[0]["loss"]


def test_spiking_mind_selects_only_valid_candidate():
    config = SpikingModelConfig(
        hidden_size=16,
        layer_count=1,
        ticks_per_token=1,
        max_action_candidates=8,
    )
    mind = SpikingMind(NodeLinkSpikeModel(config), max_context_tokens=64)
    candidates = [
        {"kind": "WAIT", "value": 1.0},
        {"kind": "MOVE", "target": "hallway"},
    ]
    decision = mind.choose_action(
        {"room": "bedroom", "body": {}, "visible_objects": []},
        candidates,
    )
    assert 0 <= decision.candidate_index < len(candidates)
    assert decision.action.kind.name in {"WAIT", "MOVE"}


def test_dictionary_builder():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "dictionary.jsonl"
        output = Path(tmp) / "corpus.jsonl"
        source.write_text(
            '{"word":"cup","definition":"a drinking container","part_of_speech":"noun"}\n',
            encoding="utf-8",
        )
        count = build_dictionary_corpus(source, output)
        assert count >= 4
        assert '"text"' in output.read_text(encoding="utf-8")


def test_cognitive_boredom_reacts_to_progress():
    drives = CognitiveDrives(boredom=0.5)
    before = drives.boredom
    drives.update(3600, meaningful_progress=1.0)
    assert drives.boredom < before



def test_dedicated_hunger_and_thirst_neurons_are_interpretable():
    bank = DedicatedDriveNeuronBank(feature_count=16)
    features = torch.zeros(1, 16)
    features[0, 0] = 0.90
    features[0, 1] = 0.40
    spikes = bank(features)
    active = set(bank.active_neurons(spikes))

    assert "hunger_notice" in active
    assert "hunger_need" in active
    assert "hunger_urgent" in active
    assert "thirst_notice" in active
    assert "thirst_need" not in active
    assert "thirst_urgent" not in active


def test_drive_neurons_feed_model_and_are_reported():
    config = SpikingModelConfig(
        hidden_size=16,
        layer_count=1,
        ticks_per_token=1,
        max_action_candidates=4,
    )
    model = NodeLinkSpikeModel(config)
    tokens = torch.randint(0, config.vocab_size, (1, 5))
    features = torch.zeros(1, 16)
    features[0, 1] = 0.95
    output = model(tokens, features)
    report = model.drive_activity_report(output.drive_spikes, minimum_rate=1e-9)

    assert output.drive_spikes.shape[-1] == model.drive_neurons.neuron_count
    assert report["thirst_notice"] == 1.0
    assert report["thirst_need"] == 1.0
    assert report["thirst_urgent"] == 1.0
    assert "hunger_notice" not in report


def test_mind_decision_exposes_named_drive_activity():
    config = SpikingModelConfig(
        hidden_size=16,
        layer_count=1,
        ticks_per_token=1,
        max_action_candidates=4,
    )
    mind = SpikingMind(NodeLinkSpikeModel(config), max_context_tokens=48)
    decision = mind.choose_action(
        {
            "room": "kitchen",
            "body": {"hunger": 0.91, "thirst": 0.10},
            "visible_objects": [],
        },
        [{"kind": "WAIT", "value": 1.0}],
    )
    assert "hunger_urgent" in decision.drive_activity
    assert "thirst_notice" not in decision.drive_activity
