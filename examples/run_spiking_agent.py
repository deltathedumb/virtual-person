from virtual_person import (
    AutonomousSpikingAgent,
    NodeLinkSpikeModel,
    SpikingMind,
    SpikingModelConfig,
)


# An untrained model is useful only for verifying the runtime. Train a checkpoint
# with examples/train_spiking_mind.py before expecting useful behavior.
model = NodeLinkSpikeModel(
    SpikingModelConfig(
        hidden_size=128,
        layer_count=2,
        ticks_per_token=3,
    )
)
mind = SpikingMind(model)
agent = AutonomousSpikingAgent(mind)

for step in agent.run(steps=5):
    print(
        step.decision.action,
        "confidence=", round(step.decision.confidence, 3),
        "spike_rate=", round(step.decision.spike_rate, 4),
        "result=", step.message,
        "reward=", round(step.reward, 3),
    )
