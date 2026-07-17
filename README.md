# Virtual Person AI

A Python and PyTorch package for experimenting with an embodied autonomous
person inside symbolic simulations and hardware-adapter runtimes.

It includes:

- persistent identity and autobiographical memory
- hunger, thirst, fatigue, bladder, hygiene, health, and social needs
- a room graph, containers, appliances, food, water, beds, toilets, and showers
- validated low-level actions
- hierarchical skills such as drinking, using the restroom, cooking breakfast,
  washing dishes, showering, sleeping, and completing a computer task
- a utility planner
- a sandboxed virtual computer
- an RL-style environment API
- demonstration recording and multiprocessing episode generation
- a command-line demo and unit tests

This is an engineering scaffold, not a claim that the agent is biologically human
or conscious. The default self-model identifies it as an embodied person using a
robotic body.

## Install

```bash
python -m pip install -e .
```

## Run the household demo

```bash
virtual-person demo
```

or:

```bash
python -m virtual_person demo
```

## Run a longer simulation

```bash
virtual-person simulate --hours 24 --seed 7 --memory ./mira-memory.sqlite3
```

## Generate demonstration trajectories

```bash
virtual-person generate-data --episodes 100 --output trajectories.jsonl --workers 4
```

## Minimal API example

```python
from virtual_person import VirtualPersonSimulation

sim = VirtualPersonSimulation.default(seed=3)
sim.agent.body.fatigue = 0.92
sim.run(hours=8)

print(sim.agent.describe())
print(sim.agent.world.computer.observe())
```

## Design notes

The world is deliberately symbolic. This makes it fast enough to run many
headless instances. A renderer can consume `simulation.snapshot()` without
changing the reasoning and training layers.

Low-level actions are validated by the world. The planner cannot execute arbitrary
Python code. The virtual computer is also simulated and cannot access the host
machine.


## Real-world embodied mode

`virtual_person.real_world` provides hardware-independent interfaces for:

- synchronized cameras, microphones, lidar, touch, joints, battery, and object detections
- mobile-base, joint, gripper, speech, and sandboxed computer actuators
- speed and force limits
- a physical emergency stop
- deny-by-default human approval for cooking, hot objects, knives, exterior doors,
  and similar higher-risk operations
- a dry-run mode and mock hardware for development

```python
from virtual_person import (
    Action,
    ActionKind,
    MockActuators,
    MockSensors,
    RealWorldConfig,
    RealWorldRuntime,
    VirtualPersonSimulation,
)

with VirtualPersonSimulation.default(name="Mira") as sim:
    runtime = RealWorldRuntime(
        sim.agent,
        MockSensors(),
        MockActuators(),
        config=RealWorldConfig(dry_run=True),
    )

    print(runtime.observe())
    print(runtime.execute(Action(
        ActionKind.MOVE,
        value={"linear_mps": 0.2, "angular_rps": 0.0, "seconds": 1.0},
    )))
```

To connect actual hardware, subclass `SensorSuite` and `ActuatorSuite`. Keep motion
control, collision avoidance, force control, and emergency-stop handling in the
hardware controller rather than an LLM.

Eating, drinking, restroom use, and cooking require a physical robot designed for
those tasks. The framework can represent and plan them, but software alone cannot
give ordinary computer hardware those capabilities.


## Node-Link-Spike mind

Version 0.3.0 adds a trainable recurrent spiking model for the agent's language,
state understanding, value estimate, and action-candidate selection.

The implementation is hybrid:

- UTF-8 byte embeddings and final output heads use ordinary tensors.
- Persistent temporal state is stored in membrane potentials.
- Hidden nodes communicate through binary spikes.
- Input and recurrent matrices are vectorized bundles of weighted links.
- Binary spikes are trained with a surrogate gradient.
- Language prediction, candidate-action prediction, and reward/value prediction
  share the same spiking recurrent core.
- Physical and cognitive drives remain external Python state.

### Architecture

```text
English bytes + structured body/drive features
                    |
                    v
             input projection
                    |
                    v
       Node-Link-Spike recurrent clusters
       - membrane state
       - learned threshold
       - learned decay
       - recurrent weighted links
       - binary spikes
                    |
         +----------+----------+
         |                     |
         v                     v
  next-byte language head  candidate-action head
                               |
                               v
                         safety validator
                               |
                               v
                    simulation or robot adapter
```

### Build an initial corpus

```bash
virtual-person build-bootstrap-corpus \
  --episodes 100 \
  --output bootstrap.jsonl
```

Convert your own dictionary:

```bash
virtual-person build-dictionary-corpus \
  data/sample_dictionary.jsonl \
  --output dictionary_corpus.jsonl
```

The dictionary format is JSONL or CSV with fields such as:

```json
{
  "word": "cup",
  "part_of_speech": "noun",
  "definition": "a small open container commonly used for drinking",
  "example": "She poured water into the cup.",
  "synonyms": ["drinking vessel"],
  "antonyms": []
}
```

### Train

A small architecture test:

```bash
virtual-person train-spike \
  bootstrap.jsonl dictionary_corpus.jsonl \
  --hidden-size 128 \
  --layers 2 \
  --ticks 3 \
  --sequence-length 192 \
  --batch-size 4 \
  --epochs 2 \
  --output spiking_mind.pt
```

A more serious model can increase hidden size, layer count, data volume, sequence
length, and training time. The included bootstrap corpus proves the mechanics; it
is not enough to create adult-level English or judgment by itself.

### Reward and boredom

The model observes boredom, curiosity, loneliness, competence frustration, and
physical needs. It cannot edit them. Reward is calculated after validated actions
from the actual change in externally owned drive state.

Meaningful progress, understood novelty, social connection, and appropriate rest
can reduce boredom. Merely moving or clicking does not automatically count as
meaningful activity.

### Important boundary

An untrained Node-Link-Spike model produces random choices. The package now
contains the architecture, corpus builders, training loop, checkpoint support,
runtime adapter, and tests. Adult-level behavior still requires a large,
high-quality controlled corpus and extensive simulation training.


## Dedicated hunger, thirst, and drive neurons

Version 0.3.0 adds a fixed interoceptive neuron bank before the learned recurrent
layers. These neurons have stable names and meanings.

For every drive, the default bank contains three neurons:

```text
hunger_notice     threshold 0.35
hunger_need       threshold 0.60
hunger_urgent     threshold 0.85

thirst_notice     threshold 0.35
thirst_need       threshold 0.60
thirst_urgent     threshold 0.85
```

The same pattern exists for fatigue, bladder pressure, hygiene discomfort, health
distress, social need, boredom, curiosity, loneliness, competence frustration,
low enjoyment, and pending tasks.

At hunger `0.90`, all three hunger neurons fire. At thirst `0.40`, only
`thirst_notice` fires. This is an explicit population code, not a learned label.

The sensory neurons are fixed. Trainable Node-Link connections carry their spikes
into the recurrent model, allowing the model to learn that hunger should influence
food-related goals without being able to directly change hunger.

Inspect them with:

```bash
python examples/inspect_drive_neurons.py
```

Or inspect any model output:

```python
report = model.drive_activity_report(output.drive_spikes)
print(report)
```

A decision also exposes active named neurons:

```python
decision = mind.choose_action(observation, candidates)
print(decision.drive_activity)
```


## Guided desktop trainer

Version 0.4.0 includes a Tkinter training studio that explains each step and can:

- create a structured training workspace
- create a small starter pack
- add and categorize English, dictionary, procedure, safety, and behavior data
- convert dictionary CSV/JSONL files
- generate bootstrap behavior records
- validate corpora and report malformed records
- select model profiles and estimate memory use
- train in a three-stage curriculum
- stream loss and spike-rate metrics
- cancel while preserving a checkpoint
- resume weights and optimizer state
- load checkpoints, generate text, and inspect named hunger/thirst neurons

Install and launch:

```bash
python -m pip install -e .
python -m virtual_person.trainer_ui
```

On Windows, you can also run `run_trainer.bat` or `run_trainer.ps1`.

Headless diagnostics:

```bash
python -m virtual_person.trainer_ui --diagnose
```

Read `TRAINING_GUIDE.md` before creating a serious corpus.


## Full command-line trainer

Version 0.5.0 adds a complete scriptable CLI alongside the desktop UI.

Launch the interactive CLI wizard:

```bash
python -m virtual_person.trainer_cli wizard
```

After installation:

```bash
virtual-person-trainer wizard
```

The CLI provides:

- `doctor` — inspect hardware
- `init` — create a workspace and project
- `next` — tell you the exact next step
- `source` — add, list, remove, enable, or disable data
- `build` — create starter, dictionary, and behavior corpora
- `validate` — scan and validate all records
- `profile` and `config` — size the model
- `preflight` — inspect a run without starting
- `train` — train one curriculum stage
- `curriculum` — run all three stages in sequence
- `score` — measure language, action, value, and spike metrics
- `evaluate` — generate text and inspect active drive neurons
- `neurons` — inspect named neurons without loading a checkpoint
- `wizard` — interactive guided workflow

See `CLI_TRAINING_GUIDE.md` for complete commands.
