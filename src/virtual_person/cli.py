from __future__ import annotations

import argparse
import json
from pathlib import Path

from .simulation import SimulationEvent, VirtualPersonSimulation
from .training import generate_demonstrations
from .bootstrap_data import generate_bootstrap_corpus
from .spike_training import (
    TrainConfig,
    build_dictionary_corpus,
    read_training_examples,
    train_model,
)
from .spiking import NodeLinkSpikeModel, SpikingModelConfig


def _format_event(event: SimulationEvent) -> str:
    hour = int(event.hour)
    minute = int((event.hour - hour) * 60)
    status = "OK" if event.ok else "FAIL"
    return (
        f"Day {event.day:02d} {hour:02d}:{minute:02d} "
        f"[{status:4}] {event.kind:10} {event.message}"
    )


def run_demo(seed: int) -> int:
    sim = VirtualPersonSimulation.default(seed=seed)
    # Force a useful first-morning sequence.
    sim.agent.body.bladder = 0.78
    sim.agent.body.thirst = 0.68
    sim.agent.body.hunger = 0.74
    sim.agent.body.fatigue = 0.30
    sim.agent.body.hygiene = 0.62

    print(f"Starting virtual-person demo for {sim.agent.self_model.name}.\n")
    sim.run(hours=18, on_event=lambda event: print(_format_event(event)))
    print("\nFinal state:")
    print(json.dumps(sim.agent.describe(), indent=2))
    print("\nVirtual computer:")
    print(json.dumps(sim.agent.world.computer.observe(), indent=2))
    sim.close()
    return 0


def run_simulation(hours: float, seed: int, memory: str, name: str) -> int:
    sim = VirtualPersonSimulation.default(seed=seed, memory_path=memory, name=name)
    sim.run(hours=hours, on_event=lambda event: print(_format_event(event)))
    print(json.dumps(sim.snapshot(), indent=2))
    sim.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="virtual-person")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run the household demonstration.")
    demo.add_argument("--seed", type=int, default=0)

    simulate = subparsers.add_parser("simulate", help="Run a longer simulation.")
    simulate.add_argument("--hours", type=float, default=24.0)
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument("--memory", default=":memory:")
    simulate.add_argument("--name", default="Mira")

    data = subparsers.add_parser(
        "generate-data",
        help="Generate JSONL scripted demonstration trajectories.",
    )
    data.add_argument("--episodes", type=int, default=10)
    data.add_argument("--workers", type=int, default=1)
    data.add_argument("--hours", type=float, default=18.0)
    data.add_argument("--seed", type=int, default=0)
    data.add_argument("--output", type=Path, default=Path("trajectories.jsonl"))


    bootstrap = subparsers.add_parser(
        "build-bootstrap-corpus",
        help="Generate language and scripted action records for the spiking model.",
    )
    bootstrap.add_argument("--output", type=Path, default=Path("bootstrap.jsonl"))
    bootstrap.add_argument("--episodes", type=int, default=20)
    bootstrap.add_argument("--seed", type=int, default=0)

    dictionary = subparsers.add_parser(
        "build-dictionary-corpus",
        help="Convert a CSV or JSONL dictionary into contextual training records.",
    )
    dictionary.add_argument("source", type=Path)
    dictionary.add_argument("--output", type=Path, default=Path("dictionary_corpus.jsonl"))

    trainer_cli = subparsers.add_parser(
        "trainer-cli",
        help="Run the guided command-line Node-Link-Spike trainer.",
    )
    trainer_cli.add_argument("trainer_args", nargs=argparse.REMAINDER)

    trainer_ui = subparsers.add_parser(
        "trainer-ui",
        help="Open the guided Node-Link-Spike training studio.",
    )

    spike_train = subparsers.add_parser(
        "train-spike",
        help="Train a Node-Link-Spike language and action model.",
    )
    spike_train.add_argument("inputs", nargs="+", type=Path)
    spike_train.add_argument("--output", type=Path, default=Path("spiking_mind.pt"))
    spike_train.add_argument("--hidden-size", type=int, default=256)
    spike_train.add_argument("--layers", type=int, default=4)
    spike_train.add_argument("--ticks", type=int, default=4)
    spike_train.add_argument("--sequence-length", type=int, default=256)
    spike_train.add_argument("--batch-size", type=int, default=8)
    spike_train.add_argument("--epochs", type=int, default=1)
    spike_train.add_argument("--learning-rate", type=float, default=3e-4)
    spike_train.add_argument("--device", default="cpu")
    spike_train.add_argument("--seed", type=int, default=0)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        return run_demo(args.seed)
    if args.command == "simulate":
        return run_simulation(args.hours, args.seed, args.memory, args.name)
    if args.command == "generate-data":
        rows = generate_demonstrations(
            output=args.output,
            episodes=args.episodes,
            workers=args.workers,
            base_seed=args.seed,
            hours=args.hours,
        )
        print(f"Wrote {rows} trajectory rows to {args.output}.")
        return 0

    if args.command == "build-bootstrap-corpus":
        count = generate_bootstrap_corpus(
            args.output,
            episodes=args.episodes,
            seed=args.seed,
        )
        print(f"Wrote {count} bootstrap records to {args.output}.")
        return 0
    if args.command == "build-dictionary-corpus":
        count = build_dictionary_corpus(args.source, args.output)
        print(f"Wrote {count} dictionary-derived records to {args.output}.")
        return 0
    if args.command == "trainer-cli":
        from .trainer_cli import main as trainer_cli_main
        return trainer_cli_main(args.trainer_args)
    if args.command == "trainer-ui":
        from .trainer_ui import main as trainer_main
        return trainer_main([])
    if args.command == "train-spike":
        model_config = SpikingModelConfig(
            hidden_size=args.hidden_size,
            layer_count=args.layers,
            ticks_per_token=args.ticks,
        )
        model = NodeLinkSpikeModel(model_config)
        examples = read_training_examples(
            args.inputs,
            state_feature_count=model_config.state_feature_count,
        )
        history = train_model(
            model,
            examples,
            TrainConfig(
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                device=args.device,
                seed=args.seed,
            ),
            checkpoint_path=args.output,
        )
        print(json.dumps({
            "checkpoint": str(args.output),
            "examples": len(examples),
            "steps": len(history),
            "final": history[-1] if history else None,
            "architecture": model.architecture_summary(),
        }, indent=2))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")
