# Node-Link-Spike Training Guide

## First objective: prove the pipeline

Do not begin with a giant model or a huge unsorted corpus.

1. Install the package with `python -m pip install -e .`
2. Run `python -m virtual_person.trainer_ui`
3. Create a workspace.
4. Create the starter pack.
5. Validate it.
6. Choose **Architecture smoke test**.
7. Train one epoch.
8. Load the resulting checkpoint in **Evaluate**.
9. Verify that hunger and thirst neurons fire at the expected thresholds.

The smoke-test model will not speak well. Its purpose is to prove that data can
flow through the tokenizer, spiking layers, losses, optimizer, checkpoint, and
evaluation system.

## Build the real corpus

Use legally obtained material that permits your intended training and distribution.

### English

Include natural English rather than only definitions:

- conversations
- stories
- descriptions of scenes and actions
- explanations
- questions and answers
- corrections and disagreements
- multi-paragraph context
- instructions written in varied styles

### Dictionary

Convert dictionaries through the UI. Dictionary-derived data should expand
vocabulary, not dominate the model's speaking style.

### Procedures

Teach the practical knowledge expected of a young adult:

- food preparation and food safety
- cleaning and laundry
- hygiene and routines
- computer and file use
- browsing and communication
- scheduling
- troubleshooting
- ordinary tool use
- asking for help

### Safety and judgment

Include both good and bad plans, with corrections:

- when to stop
- when to wait
- when to inspect
- when to ask
- when permission is required
- how to react to unexpected heat, smoke, spills, breakage, or uncertainty
- why random action is not a valid cure for boredom

### Behavior and drives

Behavior JSONL records should contain:

```json
{
  "text": "State and candidate actions, including the correct candidate.",
  "state_features": [0.9, 0.2, 0.1, 0.1, 0.0, 0.0, 0.1, 0.3, 0.5, 0.1, 0.1, 0.5, 0.5, 0.5, 0.0, 1.0],
  "action_target": 1,
  "value_target": 0.8
}
```

The first state features are hunger, thirst, fatigue, bladder pressure, hygiene
discomfort, health distress, social need, boredom, curiosity, loneliness,
competence frustration, enjoyment, two time features, pending task, and bias.

The physical and cognitive drive values are external. The neural model observes
them but cannot rewrite them.

## Curriculum

### Stage 1 — English and vocabulary

Use English and Dictionary sources. Train until:

- held-out language loss improves
- generations become less broken
- the model can use common definitions in context

Save as `stage1_language.pt`.

### Stage 2 — Practical knowledge and judgment

Enable **Continue from checkpoint** and select `stage1_language.pt`. Add Procedure
and Safety/Judgment sources. Save as `stage2_practical.pt`.

Test practical prompts and correction cases.

### Stage 3 — Autonomous behavior and drives

Continue from `stage2_practical.pt`. Add Behavior sources. Save as
`stage3_autonomous.pt`.

Test identical environmental states while changing hunger, thirst, fatigue, and
boredom. The named drive-neuron report should change, and trained action choices
should change sensibly.

## Evaluation discipline

Keep a fixed suite of prompts and states. Compare every checkpoint on the same
suite. Record:

- language loss
- action accuracy
- value error
- spike rate
- invalid-action frequency
- safety-judgment accuracy
- repetitive-action frequency
- simulator task completion

Never select a checkpoint from one impressive sample.

## Scaling warning

The current model is a research implementation. Its recurrent spiking loop is
unrolled across sequence length, ticks, and layers. Before attempting hundreds
of millions of parameters, implement fused kernels, truncated backpropagation,
distributed data loading, validation splits, and robust checkpoint recovery.
