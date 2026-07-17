import torch

from virtual_person import NodeLinkSpikeModel, SpikingModelConfig


model = NodeLinkSpikeModel(
    SpikingModelConfig(
        hidden_size=64,
        layer_count=2,
        ticks_per_token=2,
    )
)

# hunger, thirst, fatigue, bladder, hygiene discomfort, health distress,
# social need, boredom, curiosity, loneliness, competence frustration,
# enjoyment, time A, time B, task pending, bias
features = torch.tensor([[
    0.92,  # hunger: urgent
    0.68,  # thirst: need
    0.20,
    0.10,
    0.05,
    0.00,
    0.10,
    0.72,  # boredom: need
    0.40,
    0.15,
    0.10,
    0.55,
    0.50,
    0.50,
    1.00,  # task pending: urgent
    1.00,
]], dtype=torch.float32)

tokens = torch.tensor([[1, 76, 77, 2]], dtype=torch.long)
output = model(tokens, state_features=features)

print("Active dedicated neurons:")
for name, rate in model.drive_activity_report(
    output.drive_spikes,
    minimum_rate=1e-9,
).items():
    print(f"  {name:32} {rate:.1f}")
