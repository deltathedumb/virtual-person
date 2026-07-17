from pathlib import Path

from virtual_person.bootstrap_data import generate_bootstrap_corpus
from virtual_person.spike_training import TrainConfig, read_training_examples, train_model
from virtual_person.spiking import NodeLinkSpikeModel, SpikingModelConfig


corpus = Path("bootstrap.jsonl")
checkpoint = Path("spiking_mind.pt")

generate_bootstrap_corpus(corpus, episodes=10, seed=1)

config = SpikingModelConfig(
    hidden_size=128,
    layer_count=2,
    ticks_per_token=3,
)
model = NodeLinkSpikeModel(config)
examples = read_training_examples([corpus])

history = train_model(
    model,
    examples,
    TrainConfig(
        sequence_length=192,
        batch_size=4,
        epochs=2,
        learning_rate=3e-4,
    ),
    checkpoint_path=checkpoint,
)

print(model.architecture_summary())
print("Final training row:", history[-1])
print("Saved:", checkpoint)
