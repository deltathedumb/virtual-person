# Node-Link-Spike CLI Training Guide

The CLI can be used interactively:

```bash
virtual-person-trainer wizard
```

or as scriptable subcommands.

## 1. Inspect hardware

```bash
virtual-person-trainer doctor
```

## 2. Create a workspace

```bash
virtual-person-trainer --workspace ./mind_training init --starter
virtual-person-trainer --workspace ./mind_training next
```

The starter corpus only verifies the training pipeline.

## 3. Add real education data

```bash
virtual-person-trainer --workspace ./mind_training \
  source add ./data/english --category English

virtual-person-trainer --workspace ./mind_training \
  source add ./data/procedures --category Procedures

virtual-person-trainer --workspace ./mind_training \
  source add ./data/safety --category "Safety/Judgment"

virtual-person-trainer --workspace ./mind_training \
  source add ./data/behavior --category Behavior
```

Convert a legally usable dictionary:

```bash
virtual-person-trainer --workspace ./mind_training \
  build dictionary ./data/dictionary.jsonl
```

Validate:

```bash
virtual-person-trainer --workspace ./mind_training validate --strict
```

## 4. Prove the architecture

```bash
virtual-person-trainer --workspace ./mind_training \
  profile apply "Architecture smoke test"

virtual-person-trainer --workspace ./mind_training \
  train --stage 1
```

Then inspect:

```bash
virtual-person-trainer --workspace ./mind_training \
  evaluate ./mind_training/checkpoints/stage1_language.pt \
  --prompt "Mira entered the kitchen because" \
  --hunger 0.8
```

A smoke-test model will not produce good English. It only proves the pipeline.

## 5. Train the intended curriculum

Choose a larger tested profile:

```bash
virtual-person-trainer --workspace ./mind_training \
  profile apply "Small GPU experiment"
```

Run stages separately:

```bash
virtual-person-trainer --workspace ./mind_training train --stage 1
virtual-person-trainer --workspace ./mind_training train --stage 2
virtual-person-trainer --workspace ./mind_training train --stage 3
```

Stages 2 and 3 automatically resume the previous stage's checkpoint when it
exists.

Or run all three:

```bash
virtual-person-trainer --workspace ./mind_training curriculum
```

## 6. Score checkpoints

```bash
virtual-person-trainer --workspace ./mind_training \
  score ./mind_training/checkpoints/stage3_autonomous.pt --stage 3
```

The scorer reports:

- byte-level language loss and perplexity
- action-selection accuracy
- reward/value mean squared error
- average hidden spike rate

## 7. Inspect dedicated neurons

```bash
virtual-person-trainer neurons \
  --hunger 0.92 \
  --thirst 0.40 \
  --boredom 0.70
```

At `hunger=0.92`, `hunger_notice`, `hunger_need`, and `hunger_urgent` fire.

## 8. Ask the CLI what to do

```bash
virtual-person-trainer --workspace ./mind_training next
```

It checks the corpus and existing checkpoints, then prints the exact next command.

## Automation

Every major command works without the wizard. Add `--yes` to training commands
to skip the confirmation prompt:

```bash
virtual-person-trainer --workspace ./mind_training \
  curriculum --yes --skip-existing
```

Press Ctrl+C during training to request cancellation. The trainer finishes the
current batch and preserves a checkpoint.
