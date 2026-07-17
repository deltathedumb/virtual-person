from __future__ import annotations

import argparse
import json
import random
import shlex
import signal
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .bootstrap_data import generate_bootstrap_corpus
from .interoception import DedicatedDriveNeuronBank
from .spike_tokenizer import ByteTokenizer
from .spike_training import (
    TrainConfig,
    TrainingExample,
    build_dictionary_corpus,
    load_checkpoint,
    read_training_examples,
    score_model,
    train_model,
)
from .spiking import NodeLinkSpikeModel, SpikingModelConfig
from .trainer_support import (
    CURRICULUM_CATEGORIES,
    MODEL_PROFILES,
    SOURCE_CATEGORIES,
    CorpusSource,
    TrainerProject,
    create_starter_pack,
    create_workspace,
    detect_hardware,
    estimate_model,
    filter_sources_for_stage,
    format_bytes,
    recommended_profile,
    scan_sources,
)


STAGE_ALIASES = {
    "1": "Stage 1 — English and vocabulary",
    "stage1": "Stage 1 — English and vocabulary",
    "language": "Stage 1 — English and vocabulary",
    "2": "Stage 2 — Practical knowledge and judgment",
    "stage2": "Stage 2 — Practical knowledge and judgment",
    "practical": "Stage 2 — Practical knowledge and judgment",
    "3": "Stage 3 — Autonomous behavior and drives",
    "stage3": "Stage 3 — Autonomous behavior and drives",
    "autonomous": "Stage 3 — Autonomous behavior and drives",
    "all": "All selected data",
}

STAGE_FILES = {
    "Stage 1 — English and vocabulary": "stage1_language.pt",
    "Stage 2 — Practical knowledge and judgment": "stage2_practical.pt",
    "Stage 3 — Autonomous behavior and drives": "stage3_autonomous.pt",
    "All selected data": "all_data.pt",
}

PREVIOUS_STAGE = {
    "Stage 2 — Practical knowledge and judgment": "stage1_language.pt",
    "Stage 3 — Autonomous behavior and drives": "stage2_practical.pt",
}

MODEL_KEYS = {
    "hidden_size": int,
    "layers": int,
    "ticks": int,
    "sequence_length": int,
    "batch_size": int,
    "epochs": int,
    "learning_rate": float,
}

TRAINING_KEYS = {
    "device": str,
    "seed": int,
    "validation_fraction": float,
}


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _workspace_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _project_path(workspace: str | Path) -> Path:
    return _workspace_path(workspace) / "trainer_project.json"


def _default_model_settings(profile_name: str) -> dict[str, Any]:
    profile = MODEL_PROFILES[profile_name]
    return {
        "profile": profile_name,
        "hidden_size": profile.hidden_size,
        "layers": profile.layers,
        "ticks": profile.ticks,
        "sequence_length": profile.sequence_length,
        "batch_size": profile.batch_size,
        "epochs": profile.epochs,
        "learning_rate": profile.learning_rate,
    }


def _ensure_project(
    workspace: str | Path,
    *,
    create: bool = False,
) -> TrainerProject:
    root = _workspace_path(workspace)
    project_path = _project_path(root)
    if project_path.is_file():
        project = TrainerProject.load(project_path)
        project.workspace = str(root)
        return project
    if not create:
        raise FileNotFoundError(
            f"No trainer project exists at {project_path}. "
            f"Run: virtual-person-trainer --workspace {_quote(root)} init"
        )

    hardware = detect_hardware()
    profile = recommended_profile(hardware)
    create_workspace(root)
    project = TrainerProject(
        workspace=str(root),
        sources=[],
        model=_default_model_settings(profile),
        training={
            "device": hardware.suggested_device,
            "seed": 0,
            "validation_fraction": 0.05,
        },
    )
    project.save(project_path)
    return project


def _save_project(project: TrainerProject) -> Path:
    return project.save(_project_path(project.workspace))


def _resolve_stage(value: str) -> str:
    if value in CURRICULUM_CATEGORIES:
        return value
    key = value.strip().lower()
    try:
        return STAGE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown stage {value!r}. Use 1, 2, 3, all, or a full stage name."
        ) from exc


def _model_settings(project: TrainerProject) -> dict[str, Any]:
    hardware = detect_hardware()
    defaults = _default_model_settings(recommended_profile(hardware))
    return {**defaults, **project.model}


def _training_settings(project: TrainerProject) -> dict[str, Any]:
    hardware = detect_hardware()
    return {
        "device": hardware.suggested_device,
        "seed": 0,
        "validation_fraction": 0.05,
        **project.training,
    }


def _spiking_config(settings: dict[str, Any]) -> SpikingModelConfig:
    return SpikingModelConfig(
        hidden_size=max(4, int(settings["hidden_size"])),
        layer_count=max(1, int(settings["layers"])),
        ticks_per_token=max(1, int(settings["ticks"])),
    )


def _stage_sources(project: TrainerProject, stage: str) -> list[CorpusSource]:
    return filter_sources_for_stage(project.sources, stage)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _print_sources(project: TrainerProject, *, scan: bool) -> None:
    stats_by_path: dict[str, Any] = {}
    if scan:
        _total, stats_by_path, _by_category = scan_sources(project.sources)

    if not project.sources:
        print("No sources have been added.")
        return

    headers = ("#", "On", "Category", "Records", "Status", "Path")
    rows: list[tuple[str, ...]] = []
    for index, source in enumerate(project.sources):
        stats = stats_by_path.get(source.path)
        records = "—" if stats is None else f"{stats.records:,}"
        if stats is None:
            status = "not scanned"
        elif stats.missing_paths:
            status = "missing"
        elif stats.malformed_records:
            status = f"{stats.malformed_records} malformed"
        else:
            status = "valid"
        rows.append(
            (
                str(index),
                "yes" if source.enabled else "no",
                source.category,
                records,
                status,
                source.path,
            )
        )
    _print_table(headers, rows)


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))
    line = "  ".join(
        str(header).ljust(widths[index])
        for index, header in enumerate(headers)
    )
    print(line)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  ".join(
                str(value).ljust(widths[index])
                for index, value in enumerate(row)
            )
        )


def _preflight(
    project: TrainerProject,
    stage: str,
    *,
    output: Path,
    resume: Path | None,
    device: str,
    overrides: dict[str, Any] | None = None,
) -> tuple[list[tuple[str, str]], bool]:
    results: list[tuple[str, str]] = []
    okay = True
    workspace = _workspace_path(project.workspace)

    if workspace.is_dir():
        results.append(("OK", f"Workspace exists: {workspace}"))
    else:
        results.append(("FAIL", f"Workspace is missing: {workspace}"))
        okay = False

    sources = _stage_sources(project, stage)
    if sources:
        results.append(("OK", f"{len(sources)} source(s) match this curriculum stage."))
    else:
        results.append(("FAIL", "No enabled sources match this stage."))
        okay = False

    total, _by_path, by_category = scan_sources(sources)
    if total.records:
        results.append(
            (
                "OK",
                f"Corpus has {total.records:,} records and "
                f"{total.characters:,} characters.",
            )
        )
    else:
        results.append(("FAIL", "No readable records were found."))
        okay = False

    if total.missing_paths:
        results.append(("FAIL", f"{total.missing_paths} source path(s) are missing."))
        okay = False
    if total.malformed_records:
        results.append(
            ("FAIL", f"{total.malformed_records} malformed record(s) must be fixed.")
        )
        okay = False

    needed_categories = set(CURRICULUM_CATEGORIES[stage])
    present_categories = {
        category
        for category, stats in by_category.items()
        if stats.records > 0
    }
    missing_categories = needed_categories - present_categories
    if missing_categories:
        results.append(
            (
                "WARN",
                "The stage has no records in: "
                + ", ".join(sorted(missing_categories)),
            )
        )

    if total.records < 100:
        results.append(
            (
                "WARN",
                "Fewer than 100 records: this is only a pipeline/smoke test.",
            )
        )
    elif total.records < 10_000:
        results.append(
            (
                "WARN",
                "This remains a small research corpus, not adult-level training data.",
            )
        )

    hardware = detect_hardware()
    if device == "cuda" and not hardware.cuda_available:
        results.append(("FAIL", "CUDA selected, but PyTorch cannot access a CUDA GPU."))
        okay = False
    else:
        results.append(("OK", f"Training device: {device}"))

    settings = {**_model_settings(project), **(overrides or {})}
    config = _spiking_config(settings)
    estimate = estimate_model(
        config,
        batch_size=int(settings["batch_size"]),
        sequence_length=int(settings["sequence_length"]),
    )
    available = (
        hardware.gpu_memory_bytes if device == "cuda" else hardware.ram_bytes
    )
    results.append(
        (
            "INFO",
            f"Estimated parameters: {estimate.parameters:,}; "
            f"rough training memory: {format_bytes(estimate.total_training_bytes_estimate)}.",
        )
    )
    if available and estimate.total_training_bytes_estimate > available * 0.80:
        results.append(
            (
                "WARN",
                "The rough estimate exceeds 80% of detected memory. "
                "Lower batch, sequence, hidden size, ticks, or layers.",
            )
        )

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        probe = output.parent / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        results.append(("OK", f"Checkpoint directory is writable: {output.parent}"))
    except Exception as exc:
        results.append(("FAIL", f"Checkpoint directory is not writable: {exc}"))
        okay = False

    if resume is not None:
        if resume.is_file():
            results.append(("OK", f"Will resume from {resume}"))
        else:
            results.append(("FAIL", f"Resume checkpoint does not exist: {resume}"))
            okay = False
    else:
        results.append(("INFO", "The model will start from random weights."))

    return results, okay


def _display_preflight(results: Sequence[tuple[str, str]]) -> None:
    for status, message in results:
        print(f"[{status:4}] {message}")


def _checkpoint_for_stage(project: TrainerProject, stage: str) -> Path:
    return _workspace_path(project.workspace) / "checkpoints" / STAGE_FILES[stage]


def _auto_resume(project: TrainerProject, stage: str) -> Path | None:
    previous = PREVIOUS_STAGE.get(stage)
    if previous is None:
        return None
    candidate = _workspace_path(project.workspace) / "checkpoints" / previous
    return candidate if candidate.is_file() else None


def _split_examples(
    examples: Sequence[TrainingExample],
    validation_fraction: float,
    seed: int,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    rows = list(examples)
    if len(rows) < 20 or validation_fraction <= 0:
        return rows, []
    rng = random.Random(seed)
    rng.shuffle(rows)
    validation_count = max(1, int(len(rows) * min(0.5, validation_fraction)))
    return rows[validation_count:], rows[:validation_count]


class ProgressPrinter:
    def __init__(self, log_path: Path | None = None) -> None:
        self.started = time.monotonic()
        self.last_length = 0
        self.log_path = log_path
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, row: dict[str, float]) -> None:
        percent = row["progress"] * 100.0
        width = 24
        filled = int(width * row["progress"])
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.monotonic() - self.started
        message = (
            f"\r[{bar}] {percent:6.2f}% "
            f"epoch {int(row['epoch'])}/{int(row['epochs'])} "
            f"batch {int(row['batch'])}/{int(row['batches'])} "
            f"loss {row['loss']:.4f} "
            f"lang {row['language_loss']:.4f} "
            f"act {row['action_loss']:.4f} "
            f"value {row['value_loss']:.4f} "
            f"spike {row['spike_rate']:.4f} "
            f"elapsed {elapsed:.1f}s"
        )
        padding = max(0, self.last_length - len(message))
        print(message + " " * padding, end="", flush=True)
        self.last_length = len(message)

        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    def finish(self) -> None:
        if self.last_length:
            print()


def _run_training(
    project: TrainerProject,
    *,
    stage: str,
    output: Path,
    resume: Path | None,
    device: str,
    overrides: dict[str, Any],
    validation_fraction: float,
    seed: int,
    checkpoint_every: int,
    yes: bool,
) -> dict[str, Any]:
    results, okay = _preflight(
        project,
        stage,
        output=output,
        resume=resume,
        device=device,
        overrides=overrides,
    )
    _display_preflight(results)
    if not okay:
        raise RuntimeError("Preflight failed.")

    if not yes and sys.stdin.isatty():
        answer = input("\nBegin training? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            raise RuntimeError("Training cancelled before starting.")

    sources = _stage_sources(project, stage)
    print("\nLoading training records...")
    examples = read_training_examples([source.path for source in sources])
    train_rows, validation_rows = _split_examples(
        examples,
        validation_fraction,
        seed,
    )
    print(
        f"Loaded {len(examples):,} records: "
        f"{len(train_rows):,} train, {len(validation_rows):,} validation."
    )

    settings = {**_model_settings(project), **overrides}
    optimizer_state = None
    start_step = 0
    if resume is not None:
        model, payload = load_checkpoint(resume, device=device)
        optimizer_state = payload.get("optimizer")
        start_step = int(payload.get("step", 0))
        print("Loaded checkpoint architecture:")
        _print_json(model.architecture_summary())
    else:
        model = NodeLinkSpikeModel(_spiking_config(settings))
        print("Initialized architecture:")
        _print_json(model.architecture_summary())

    stop_event = threading.Event()
    previous_handler = signal.getsignal(signal.SIGINT)

    def handle_interrupt(_signum: int, _frame: Any) -> None:
        if not stop_event.is_set():
            print(
                "\nCancellation requested. The current batch will finish, "
                "then a checkpoint will be saved.",
                flush=True,
            )
            stop_event.set()

    signal.signal(signal.SIGINT, handle_interrupt)
    log_path = (
        _workspace_path(project.workspace)
        / "logs"
        / f"{output.stem}_training.jsonl"
    )
    progress = ProgressPrinter(log_path)

    try:
        history = train_model(
            model,
            train_rows,
            TrainConfig(
                sequence_length=max(8, int(settings["sequence_length"])),
                batch_size=max(1, int(settings["batch_size"])),
                epochs=max(1, int(settings["epochs"])),
                learning_rate=float(settings["learning_rate"]),
                device=device,
                seed=seed,
                num_workers=0,
                checkpoint_every_steps=max(0, checkpoint_every),
            ),
            checkpoint_path=output,
            progress_callback=progress,
            stop_requested=stop_event.is_set,
            optimizer_state=optimizer_state,
            start_step=start_step,
            checkpoint_metadata={
                "stage": stage,
                "sources": [asdict(source) for source in sources],
                "model_settings": settings,
                "validation_fraction": validation_fraction,
            },
        )
    finally:
        progress.finish()
        signal.signal(signal.SIGINT, previous_handler)

    metrics: dict[str, Any] | None = None
    if validation_rows:
        print("Scoring held-out validation records...")
        metrics = score_model(
            model,
            validation_rows,
            sequence_length=int(settings["sequence_length"]),
            batch_size=int(settings["batch_size"]),
            device=device,
        ).as_dict()
        _print_json(metrics)

    manifest = {
        "stage": stage,
        "checkpoint": str(output),
        "resume": str(resume) if resume else None,
        "cancelled": stop_event.is_set(),
        "train_records": len(train_rows),
        "validation_records": len(validation_rows),
        "final_training_row": history[-1] if history else None,
        "validation_metrics": metrics,
        "architecture": model.architecture_summary(),
        "settings": settings,
        "device": device,
        "seed": seed,
    }
    manifest_path = output.with_suffix(".run.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Checkpoint: {output}")
    print(f"Run manifest: {manifest_path}")
    if stop_event.is_set():
        print("Training stopped by request; the checkpoint was preserved.")
    else:
        print("Training completed.")
    return manifest


def _state_features(args: argparse.Namespace) -> list[float]:
    return [
        float(args.hunger),
        float(args.thirst),
        float(args.fatigue),
        float(args.bladder),
        float(args.hygiene_discomfort),
        float(args.health_distress),
        float(args.social_need),
        float(args.boredom),
        float(args.curiosity),
        float(args.loneliness),
        float(args.competence_frustration),
        float(args.enjoyment),
        0.5,
        0.5,
        1.0 if bool(args.task_pending) else 0.0,
        1.0,
    ]


def _add_drive_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hunger", type=float, default=0.20)
    parser.add_argument("--thirst", type=float, default=0.20)
    parser.add_argument("--fatigue", type=float, default=0.20)
    parser.add_argument("--bladder", type=float, default=0.10)
    parser.add_argument("--hygiene-discomfort", type=float, default=0.05)
    parser.add_argument("--health-distress", type=float, default=0.0)
    parser.add_argument("--social-need", type=float, default=0.10)
    parser.add_argument("--boredom", type=float, default=0.20)
    parser.add_argument("--curiosity", type=float, default=0.45)
    parser.add_argument("--loneliness", type=float, default=0.10)
    parser.add_argument("--competence-frustration", type=float, default=0.10)
    parser.add_argument("--enjoyment", type=float, default=0.50)
    parser.add_argument("--task-pending", action="store_true")


def command_doctor(args: argparse.Namespace) -> int:
    hardware = detect_hardware()
    payload = {
        "python": hardware.python,
        "platform": hardware.platform,
        "torch": hardware.torch_version,
        "cpu_count": hardware.cpu_count,
        "system_ram": format_bytes(hardware.ram_bytes),
        "cuda_available": hardware.cuda_available,
        "cuda_devices": hardware.cuda_device_count,
        "gpu": hardware.gpu_name,
        "gpu_memory": format_bytes(hardware.gpu_memory_bytes),
        "suggested_device": hardware.suggested_device,
        "suggested_profile": recommended_profile(hardware),
    }
    if args.json:
        _print_json(payload)
    else:
        for key, value in payload.items():
            print(f"{key.replace('_', ' ').title():22} {value}")
    return 0


def command_init(args: argparse.Namespace) -> int:
    project = _ensure_project(args.workspace, create=True)
    if args.profile:
        if args.profile not in MODEL_PROFILES:
            raise ValueError(f"Unknown profile: {args.profile}")
        project.model = _default_model_settings(args.profile)
    if args.device:
        project.training["device"] = args.device

    if args.starter:
        additions = create_starter_pack(project.workspace)
        existing = {str(Path(source.path).resolve()) for source in project.sources}
        for source in additions:
            if str(Path(source.path).resolve()) not in existing:
                project.sources.append(source)

    path = _save_project(project)
    print(f"Workspace: {project.workspace}")
    print(f"Project:   {path}")
    print(f"Profile:   {_model_settings(project)['profile']}")
    print(f"Device:    {_training_settings(project)['device']}")
    if args.starter:
        print("Starter corpus added.")
    print("\nNext:")
    print(
        f"  virtual-person-trainer --workspace {_quote(project.workspace)} next"
    )
    return 0


def command_next(args: argparse.Namespace) -> int:
    try:
        project = _ensure_project(args.workspace)
    except FileNotFoundError:
        workspace = _workspace_path(args.workspace)
        print("No project exists yet.")
        print(
            "Run:\n  "
            f"virtual-person-trainer --workspace {_quote(workspace)} init --starter"
        )
        return 0

    workspace = _workspace_path(project.workspace)
    print(f"Workspace: {workspace}")
    if not project.sources:
        print("\nNEXT STEP: add education data.")
        print(
            "For a pipeline test:\n  "
            f"virtual-person-trainer --workspace {_quote(workspace)} build starter"
        )
        print(
            "For real data:\n  "
            f"virtual-person-trainer --workspace {_quote(workspace)} "
            "source add <path> --category English"
        )
        return 0

    total, _by_path, by_category = scan_sources(project.sources)
    if total.missing_paths or total.malformed_records:
        print("\nNEXT STEP: fix corpus errors.")
        print(
            f"Missing paths: {total.missing_paths}; "
            f"malformed records: {total.malformed_records}."
        )
        print(
            "Run:\n  "
            f"virtual-person-trainer --workspace {_quote(workspace)} validate --strict"
        )
        return 0
    if total.records == 0:
        print("\nNEXT STEP: add readable .txt or .jsonl records.")
        return 0

    categories = {name for name, stats in by_category.items() if stats.records}
    print(
        f"Validated corpus: {total.records:,} records, "
        f"{total.characters:,} characters."
    )
    print("Categories: " + (", ".join(sorted(categories)) or "none"))

    checkpoints = workspace / "checkpoints"
    stage1 = checkpoints / "stage1_language.pt"
    stage2 = checkpoints / "stage2_practical.pt"
    stage3 = checkpoints / "stage3_autonomous.pt"

    if not stage1.is_file():
        print("\nNEXT STEP: train Stage 1 — English and vocabulary.")
        print(
            "First prove the pipeline with the smoke profile:\n  "
            f"virtual-person-trainer --workspace {_quote(workspace)} "
            "profile apply \"Architecture smoke test\"\n  "
            f"virtual-person-trainer --workspace {_quote(workspace)} "
            "train --stage 1"
        )
        if total.records < 100:
            print(
                "\nYour current corpus is only large enough to test the pipeline, "
                "not to create the intended adult-capable model."
            )
        return 0

    if not stage2.is_file():
        print("\nNEXT STEP: train Stage 2 from Stage 1.")
        print(
            "Run:\n  "
            f"virtual-person-trainer --workspace {_quote(workspace)} "
            "train --stage 2"
        )
        if not {"Procedures", "Safety/Judgment"}.issubset(categories):
            print(
                "Before doing that, add both Procedures and Safety/Judgment data."
            )
        return 0

    if not stage3.is_file():
        print("\nNEXT STEP: train Stage 3 from Stage 2.")
        print(
            "Run:\n  "
            f"virtual-person-trainer --workspace {_quote(workspace)} "
            "train --stage 3"
        )
        if "Behavior" not in categories:
            print("Before doing that, add structured Behavior JSONL data.")
        return 0

    print("\nAll three curriculum checkpoints exist.")
    print(
        "NEXT STEP: score and inspect the autonomous checkpoint:\n  "
        f"virtual-person-trainer --workspace {_quote(workspace)} "
        f"score {_quote(stage3)} --stage 3\n  "
        f"virtual-person-trainer --workspace {_quote(workspace)} "
        f"evaluate {_quote(stage3)} "
        '--prompt "Mira is hungry in a kitchen. She decides to" '
        "--hunger 0.92"
    )
    return 0


def command_guide(args: argparse.Namespace) -> int:
    topics = {
        "overview": (
            "Train in order: English/vocabulary → practical knowledge/judgment "
            "→ autonomous behavior/drives. Keep separate checkpoints."
        ),
        "data": (
            "Use natural English, contextual dictionary examples, practical "
            "procedures, safety corrections, and structured behavior records. "
            "A dictionary alone is insufficient."
        ),
        "stages": (
            "Stage 1 uses English+Dictionary. Stage 2 adds Procedures and "
            "Safety/Judgment. Stage 3 adds Behavior and drive-conditioned actions."
        ),
        "behavior": (
            "Behavior JSONL needs text, 16 state_features, action_target, and "
            "value_target. Hunger is feature 0, thirst 1, fatigue 2, bladder 3, "
            "boredom 7, and task_pending 14."
        ),
        "scaling": (
            "Start with a smoke test. The current implementation unrolls "
            "sequence × ticks × layers, so large runs require much more compute "
            "and eventually optimized kernels."
        ),
    }
    if args.topic == "all":
        for name, text in topics.items():
            print(f"{name.upper()}\n{'-' * len(name)}\n{text}\n")
    else:
        print(topics[args.topic])
    return 0


def command_source_add(args: argparse.Namespace) -> int:
    project = _ensure_project(args.workspace)
    if args.category not in SOURCE_CATEGORIES:
        raise ValueError(
            "Category must be one of: " + ", ".join(SOURCE_CATEGORIES)
        )
    existing = {str(Path(source.path).expanduser().resolve()) for source in project.sources}
    added = 0
    for raw in args.paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists() and not args.allow_missing:
            raise FileNotFoundError(path)
        normalized = str(path)
        if normalized not in existing:
            project.sources.append(
                CorpusSource(
                    path=normalized,
                    category=args.category,
                    enabled=not args.disabled,
                )
            )
            existing.add(normalized)
            added += 1
    _save_project(project)
    print(f"Added {added} source(s).")
    _print_sources(project, scan=False)
    return 0


def command_source_list(args: argparse.Namespace) -> int:
    project = _ensure_project(args.workspace)
    _print_sources(project, scan=args.scan)
    return 0


def _source_indices(project: TrainerProject, selectors: Sequence[str]) -> set[int]:
    indices: set[int] = set()
    for selector in selectors:
        try:
            index = int(selector)
        except ValueError:
            target = str(Path(selector).expanduser().resolve())
            matches = [
                index
                for index, source in enumerate(project.sources)
                if str(Path(source.path).expanduser().resolve()) == target
            ]
            if not matches:
                raise ValueError(f"No source matches {selector!r}")
            indices.update(matches)
        else:
            if not 0 <= index < len(project.sources):
                raise IndexError(index)
            indices.add(index)
    return indices


def command_source_remove(args: argparse.Namespace) -> int:
    project = _ensure_project(args.workspace)
    indices = _source_indices(project, args.selectors)
    removed = [source for index, source in enumerate(project.sources) if index in indices]
    project.sources = [
        source for index, source in enumerate(project.sources) if index not in indices
    ]
    _save_project(project)
    print(f"Removed {len(removed)} source(s).")
    return 0


def command_source_toggle(args: argparse.Namespace, enabled: bool) -> int:
    project = _ensure_project(args.workspace)
    indices = _source_indices(project, args.selectors)
    for index in indices:
        project.sources[index].enabled = enabled
    _save_project(project)
    print(f"{'Enabled' if enabled else 'Disabled'} {len(indices)} source(s).")
    return 0


def command_build_starter(args: argparse.Namespace) -> int:
    project = _ensure_project(args.workspace)
    additions = create_starter_pack(project.workspace)
    existing = {str(Path(source.path).resolve()) for source in project.sources}
    added = 0
    for source in additions:
        if str(Path(source.path).resolve()) not in existing:
            project.sources.append(source)
            added += 1
    _save_project(project)
    print(f"Starter pack created; {added} new source(s) registered.")
    print("This data is only for checking that the pipeline works.")
    return 0


def command_build_dictionary(args: argparse.Namespace) -> int:
    project = _ensure_project(args.workspace)
    source = Path(args.source).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _workspace_path(project.workspace)
        / "corpora"
        / f"{source.stem}_dictionary.jsonl"
    )
    count = build_dictionary_corpus(source, output)
    print(f"Wrote {count:,} contextual dictionary records to {output}.")
    if not args.no_add:
        if str(output) not in {source.path for source in project.sources}:
            project.sources.append(CorpusSource(str(output), "Dictionary"))
            _save_project(project)
            print("Registered the generated corpus as a Dictionary source.")
    return 0


def command_build_bootstrap(args: argparse.Namespace) -> int:
    project = _ensure_project(args.workspace)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _workspace_path(project.workspace)
        / "corpora"
        / "bootstrap_behavior.jsonl"
    )
    count = generate_bootstrap_corpus(
        output,
        episodes=max(1, args.episodes),
        seed=args.seed,
    )
    print(f"Wrote {count:,} bootstrap records to {output}.")
    if not args.no_add:
        if str(output) not in {source.path for source in project.sources}:
            project.sources.append(CorpusSource(str(output), "Behavior"))
            _save_project(project)
            print("Registered the generated corpus as a Behavior source.")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    project = _ensure_project(args.workspace)
    stage = _resolve_stage(args.stage) if args.stage else None
    sources = _stage_sources(project, stage) if stage else [
        source for source in project.sources if source.enabled
    ]
    total, by_path, by_category = scan_sources(sources)
    if args.json:
        _print_json(
            {
                "stage": stage,
                "total": asdict(total),
                "by_path": {path: asdict(stats) for path, stats in by_path.items()},
                "by_category": {
                    category: asdict(stats)
                    for category, stats in by_category.items()
                },
            }
        )
    else:
        headers = ("Category", "Files", "Records", "Characters", "Malformed")
        rows = [
            (
                category,
                f"{stats.files:,}",
                f"{stats.records:,}",
                f"{stats.characters:,}",
                f"{stats.malformed_records:,}",
            )
            for category, stats in sorted(by_category.items())
        ]
        _print_table(headers, rows)
        print(
            f"\nTotal: {total.files:,} files, {total.records:,} records, "
            f"{total.characters:,} characters, "
            f"{total.malformed_records:,} malformed, "
            f"{total.missing_paths:,} missing path(s)."
        )
    failed = total.records == 0 or total.malformed_records or total.missing_paths
    return 1 if args.strict and failed else 0


def command_profile_list(_args: argparse.Namespace) -> int:
    rows = []
    for name, profile in MODEL_PROFILES.items():
        rows.append(
            (
                name,
                str(profile.hidden_size),
                str(profile.layers),
                str(profile.ticks),
                str(profile.sequence_length),
                str(profile.batch_size),
                str(profile.epochs),
            )
        )
    _print_table(
        ("Profile", "Hidden", "Layers", "Ticks", "Sequence", "Batch", "Epochs"),
        rows,
    )
    return 0


def command_profile_show(args: argparse.Namespace) -> int:
    profile = MODEL_PROFILES[args.name]
    _print_json(asdict(profile))
    return 0


def command_profile_apply(args: argparse.Namespace) -> int:
    project = _ensure_project(args.workspace)
    if args.name not in MODEL_PROFILES:
        raise ValueError(f"Unknown profile: {args.name}")
    project.model = _default_model_settings(args.name)
    _save_project(project)
    print(f"Applied profile: {args.name}")
    _print_json(project.model)
    return 0


def command_config_show(args: argparse.Namespace) -> int:
    project = _ensure_project(args.workspace)
    settings = {
        "model": _model_settings(project),
        "training": _training_settings(project),
    }
    _print_json(settings)
    return 0


def command_config_set(args: argparse.Namespace) -> int:
    project = _ensure_project(args.workspace)
    key = args.key
    if key in MODEL_KEYS:
        value = MODEL_KEYS[key](args.value)
        project.model[key] = value
        project.model["profile"] = "custom"
    elif key in TRAINING_KEYS:
        value = TRAINING_KEYS[key](args.value)
        if key == "device" and value not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        project.training[key] = value
    else:
        raise ValueError(
            "Unknown setting. Model keys: "
            + ", ".join(MODEL_KEYS)
            + ". Training keys: "
            + ", ".join(TRAINING_KEYS)
        )
    _save_project(project)
    print(f"Set {key} = {value}")
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    project = _ensure_project(args.workspace)
    stage = _resolve_stage(args.stage)
    settings = _model_settings(project)
    device = args.device or _training_settings(project)["device"]
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _checkpoint_for_stage(project, stage)
    )
    resume = (
        Path(args.resume).expanduser().resolve()
        if args.resume
        else (_auto_resume(project, stage) if args.auto_resume else None)
    )
    results, okay = _preflight(
        project,
        stage,
        output=output,
        resume=resume,
        device=device,
        overrides=settings,
    )
    _display_preflight(results)
    return 0 if okay else 1


def _training_overrides(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key in MODEL_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            values[key] = value
    return values


def command_train(args: argparse.Namespace) -> int:
    project = _ensure_project(args.workspace)
    stage = _resolve_stage(args.stage)
    training = _training_settings(project)
    device = args.device or training["device"]
    seed = args.seed if args.seed is not None else int(training["seed"])
    validation_fraction = (
        args.validation_fraction
        if args.validation_fraction is not None
        else float(training["validation_fraction"])
    )
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _checkpoint_for_stage(project, stage)
    )
    if args.no_resume:
        resume = None
    elif args.resume:
        resume = Path(args.resume).expanduser().resolve()
    else:
        resume = _auto_resume(project, stage)

    manifest = _run_training(
        project,
        stage=stage,
        output=output,
        resume=resume,
        device=device,
        overrides=_training_overrides(args),
        validation_fraction=validation_fraction,
        seed=seed,
        checkpoint_every=args.checkpoint_every,
        yes=args.yes,
    )
    project.training.update(
        {
            "device": device,
            "seed": seed,
            "validation_fraction": validation_fraction,
            "last_stage": stage,
            "last_checkpoint": str(output),
        }
    )
    _save_project(project)
    return 130 if manifest["cancelled"] else 0


def command_curriculum(args: argparse.Namespace) -> int:
    project = _ensure_project(args.workspace)
    training = _training_settings(project)
    device = args.device or training["device"]
    seed = args.seed if args.seed is not None else int(training["seed"])
    validation_fraction = (
        args.validation_fraction
        if args.validation_fraction is not None
        else float(training["validation_fraction"])
    )
    resume: Path | None = (
        Path(args.resume).expanduser().resolve() if args.resume else None
    )
    stages = [
        "Stage 1 — English and vocabulary",
        "Stage 2 — Practical knowledge and judgment",
        "Stage 3 — Autonomous behavior and drives",
    ]

    manifests = []
    for index, stage in enumerate(stages, 1):
        output = _checkpoint_for_stage(project, stage)
        if args.skip_existing and output.is_file():
            print(f"\nSkipping existing checkpoint: {output}")
            resume = output
            continue
        print(f"\n{'=' * 72}\nCURRICULUM {index}/3: {stage}\n{'=' * 72}")
        manifest = _run_training(
            project,
            stage=stage,
            output=output,
            resume=resume,
            device=device,
            overrides=_training_overrides(args),
            validation_fraction=validation_fraction,
            seed=seed + index - 1,
            checkpoint_every=args.checkpoint_every,
            yes=args.yes,
        )
        manifests.append(manifest)
        resume = output
        if manifest["cancelled"]:
            break

    summary = _workspace_path(project.workspace) / "logs" / "curriculum_summary.json"
    summary.write_text(
        json.dumps(manifests, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nCurriculum summary: {summary}")
    return 0


def command_score(args: argparse.Namespace) -> int:
    project = _ensure_project(args.workspace)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    device = args.device or _training_settings(project)["device"]
    model, payload = load_checkpoint(checkpoint, device=device)

    if args.inputs:
        paths = [Path(path).expanduser().resolve() for path in args.inputs]
    else:
        stage = _resolve_stage(args.stage)
        paths = [
            Path(source.path)
            for source in _stage_sources(project, stage)
        ]
    examples = read_training_examples(paths)
    settings = _model_settings(project)
    result = score_model(
        model,
        examples,
        sequence_length=args.sequence_length or int(settings["sequence_length"]),
        batch_size=args.batch_size or int(settings["batch_size"]),
        device=device,
    )
    payload_out = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "sources": [str(path) for path in paths],
        "metrics": result.as_dict(),
    }
    if args.json:
        _print_json(payload_out)
    else:
        _print_json(payload_out["metrics"])
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    project = _ensure_project(args.workspace)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    device = args.device or _training_settings(project)["device"]
    model, payload = load_checkpoint(checkpoint, device=device)
    model.eval()
    tokenizer = ByteTokenizer()

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    elif args.prompt is not None:
        prompt = args.prompt
    elif sys.stdin.isatty():
        prompt = input("Prompt: ")
    else:
        prompt = sys.stdin.read()
    if not prompt.strip():
        raise ValueError("The prompt is empty")

    features = torch.tensor(
        [_state_features(args)],
        dtype=torch.float32,
        device=device,
    )
    ids = tokenizer.encode(prompt, add_eos=False)
    if len(ids) > args.max_context:
        ids = ids[-args.max_context :]
    tokens = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        inspection = model(tokens, state_features=features)
        generated = model.generate(
            tokens,
            state_features=features,
            max_new_tokens=args.tokens,
            eos_token_id=tokenizer.EOS,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    full_text = tokenizer.decode(generated[0].detach().cpu().tolist())
    continuation = full_text[len(tokenizer.decode(ids)) :]
    drive_report = model.drive_activity_report(
        inspection.drive_spikes,
        minimum_rate=1e-9,
    )

    if args.json:
        _print_json(
            {
                "checkpoint": str(checkpoint),
                "step": payload.get("step"),
                "prompt": prompt,
                "continuation": continuation,
                "active_drive_neurons": drive_report,
                "value_estimate": float(inspection.value[0].item()),
                "spike_rate": float(inspection.spike_rates.mean().item()),
            }
        )
    else:
        print("CONTINUATION")
        print("------------")
        print(continuation)
        print("\nACTIVE DEDICATED DRIVE NEURONS")
        print("------------------------------")
        if drive_report:
            for name, rate in drive_report.items():
                print(f"{name:36} {rate:.3f}")
        else:
            print("none")
        print(f"\nValue estimate: {float(inspection.value[0].item()):.5f}")
        print(f"Hidden spike rate: {float(inspection.spike_rates.mean().item()):.5f}")
    return 0


def command_neurons(args: argparse.Namespace) -> int:
    features = torch.tensor([_state_features(args)], dtype=torch.float32)
    bank = DedicatedDriveNeuronBank(feature_count=16)
    spikes = bank(features)
    report = bank.activity_report(spikes, minimum_rate=1e-9)
    if args.json:
        _print_json(report)
    else:
        print("Active dedicated drive neurons:")
        if not report:
            print("  none")
        for name, rate in report.items():
            print(f"  {name:36} {rate:.1f}")
    return 0


def _wizard_input(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    answer = input(prompt + suffix + ": ").strip()
    return answer or (default or "")


def command_wizard(args: argparse.Namespace) -> int:
    workspace = _workspace_path(args.workspace)
    if not _project_path(workspace).is_file():
        print("No project exists in this workspace.")
        create = _wizard_input("Create it now? y/n", "y").lower()
        if create in {"y", "yes"}:
            starter = _wizard_input(
                "Add the small pipeline-test starter corpus? y/n",
                "y",
            ).lower()
            argv = ["--workspace", str(workspace), "init"]
            if starter in {"y", "yes"}:
                argv.append("--starter")
            main(argv)
        else:
            return 0

    while True:
        print(
            "\n"
            "NODE-LINK-SPIKE TRAINER\n"
            "1. Tell me the next step\n"
            "2. List corpus sources\n"
            "3. Add a source\n"
            "4. Convert a dictionary\n"
            "5. Generate bootstrap behavior\n"
            "6. Validate corpus\n"
            "7. Choose a model profile\n"
            "8. Train one curriculum stage\n"
            "9. Run all three curriculum stages\n"
            "10. Score a checkpoint\n"
            "11. Generate text / inspect neurons\n"
            "12. Inspect drive neurons only\n"
            "0. Exit\n"
        )
        choice = _wizard_input("Choice", "1")
        base = ["--workspace", str(workspace)]

        try:
            if choice == "0":
                return 0
            if choice == "1":
                main(base + ["next"])
            elif choice == "2":
                main(base + ["source", "list", "--scan"])
            elif choice == "3":
                path = _wizard_input("File or folder path")
                print("Categories: " + ", ".join(SOURCE_CATEGORIES))
                category = _wizard_input("Category", "English")
                main(base + ["source", "add", path, "--category", category])
            elif choice == "4":
                path = _wizard_input("Dictionary CSV/JSONL path")
                main(base + ["build", "dictionary", path])
            elif choice == "5":
                episodes = _wizard_input("Bootstrap episodes", "100")
                main(base + ["build", "bootstrap", "--episodes", episodes])
            elif choice == "6":
                main(base + ["validate"])
            elif choice == "7":
                main(base + ["profile", "list"])
                name = _wizard_input("Exact profile name", "Architecture smoke test")
                main(base + ["profile", "apply", name])
            elif choice == "8":
                stage = _wizard_input("Stage: 1, 2, or 3", "1")
                main(base + ["train", "--stage", stage])
            elif choice == "9":
                confirm = _wizard_input(
                    "This can be a long run. Continue? y/n",
                    "n",
                ).lower()
                if confirm in {"y", "yes"}:
                    main(base + ["curriculum"])
            elif choice == "10":
                checkpoint = _wizard_input("Checkpoint path")
                stage = _wizard_input("Corpus stage used for scoring", "3")
                main(base + ["score", checkpoint, "--stage", stage])
            elif choice == "11":
                checkpoint = _wizard_input("Checkpoint path")
                prompt = _wizard_input(
                    "Prompt",
                    "Mira is hungry in a kitchen. She decides to",
                )
                hunger = _wizard_input("Hunger 0-1", "0.92")
                main(
                    base
                    + [
                        "evaluate",
                        checkpoint,
                        "--prompt",
                        prompt,
                        "--hunger",
                        hunger,
                    ]
                )
            elif choice == "12":
                hunger = _wizard_input("Hunger 0-1", "0.90")
                thirst = _wizard_input("Thirst 0-1", "0.40")
                boredom = _wizard_input("Boredom 0-1", "0.20")
                main(
                    base
                    + [
                        "neurons",
                        "--hunger",
                        hunger,
                        "--thirst",
                        thirst,
                        "--boredom",
                        boredom,
                    ]
                )
            else:
                print("Unknown choice.")
        except (Exception, KeyboardInterrupt) as exc:
            print(f"\nError: {exc}", file=sys.stderr)
    return 0


def _add_training_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-size", dest="hidden_size", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--ticks", type=int)
    parser.add_argument("--sequence-length", dest="sequence_length", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="virtual-person-trainer",
        description=(
            "Guided and scriptable trainer for the Node-Link-Spike virtual-person model."
        ),
    )
    parser.add_argument(
        "--workspace",
        default="training_workspace",
        help="Trainer workspace containing trainer_project.json.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Inspect Python, PyTorch, CPU, RAM, and GPU.")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=command_doctor)

    init = sub.add_parser("init", help="Create or load a training workspace.")
    init.add_argument("--starter", action="store_true")
    init.add_argument("--profile", choices=tuple(MODEL_PROFILES))
    init.add_argument("--device", choices=("cpu", "cuda"))
    init.set_defaults(func=command_init)

    next_cmd = sub.add_parser("next", help="Tell you the exact next training step.")
    next_cmd.set_defaults(func=command_next)

    guide = sub.add_parser("guide", help="Explain corpus and curriculum requirements.")
    guide.add_argument(
        "topic",
        nargs="?",
        default="overview",
        choices=("overview", "data", "stages", "behavior", "scaling", "all"),
    )
    guide.set_defaults(func=command_guide)

    wizard = sub.add_parser("wizard", help="Run an interactive command-line training wizard.")
    wizard.set_defaults(func=command_wizard)

    source = sub.add_parser("source", help="Manage corpus sources.")
    source_sub = source.add_subparsers(dest="source_command", required=True)

    source_add = source_sub.add_parser("add")
    source_add.add_argument("paths", nargs="+")
    source_add.add_argument("--category", required=True, choices=SOURCE_CATEGORIES)
    source_add.add_argument("--disabled", action="store_true")
    source_add.add_argument("--allow-missing", action="store_true")
    source_add.set_defaults(func=command_source_add)

    source_list = source_sub.add_parser("list")
    source_list.add_argument("--scan", action="store_true")
    source_list.set_defaults(func=command_source_list)

    source_remove = source_sub.add_parser("remove")
    source_remove.add_argument("selectors", nargs="+")
    source_remove.set_defaults(func=command_source_remove)

    source_enable = source_sub.add_parser("enable")
    source_enable.add_argument("selectors", nargs="+")
    source_enable.set_defaults(
        func=lambda args: command_source_toggle(args, True)
    )

    source_disable = source_sub.add_parser("disable")
    source_disable.add_argument("selectors", nargs="+")
    source_disable.set_defaults(
        func=lambda args: command_source_toggle(args, False)
    )

    build = sub.add_parser("build", help="Build starter, dictionary, or behavior corpora.")
    build_sub = build.add_subparsers(dest="build_command", required=True)

    starter = build_sub.add_parser("starter")
    starter.set_defaults(func=command_build_starter)

    dictionary = build_sub.add_parser("dictionary")
    dictionary.add_argument("source")
    dictionary.add_argument("--output")
    dictionary.add_argument("--no-add", action="store_true")
    dictionary.set_defaults(func=command_build_dictionary)

    bootstrap = build_sub.add_parser("bootstrap")
    bootstrap.add_argument("--episodes", type=int, default=100)
    bootstrap.add_argument("--seed", type=int, default=0)
    bootstrap.add_argument("--output")
    bootstrap.add_argument("--no-add", action="store_true")
    bootstrap.set_defaults(func=command_build_bootstrap)

    validate = sub.add_parser("validate", help="Validate corpus sources.")
    validate.add_argument("--stage")
    validate.add_argument("--strict", action="store_true")
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(func=command_validate)

    profile = sub.add_parser("profile", help="List, inspect, or apply model profiles.")
    profile_sub = profile.add_subparsers(dest="profile_command", required=True)

    profile_list = profile_sub.add_parser("list")
    profile_list.set_defaults(func=command_profile_list)

    profile_show = profile_sub.add_parser("show")
    profile_show.add_argument("name", choices=tuple(MODEL_PROFILES))
    profile_show.set_defaults(func=command_profile_show)

    profile_apply = profile_sub.add_parser("apply")
    profile_apply.add_argument("name", choices=tuple(MODEL_PROFILES))
    profile_apply.set_defaults(func=command_profile_apply)

    config = sub.add_parser("config", help="Show or change project settings.")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    config_show = config_sub.add_parser("show")
    config_show.set_defaults(func=command_config_show)
    config_set = config_sub.add_parser("set")
    config_set.add_argument("key")
    config_set.add_argument("value")
    config_set.set_defaults(func=command_config_set)

    preflight = sub.add_parser("preflight", help="Check a training run without starting it.")
    preflight.add_argument("--stage", required=True)
    preflight.add_argument("--output")
    preflight.add_argument("--resume")
    preflight.add_argument("--auto-resume", action="store_true")
    preflight.add_argument("--device", choices=("cpu", "cuda"))
    preflight.set_defaults(func=command_preflight)

    train = sub.add_parser("train", help="Train one curriculum stage.")
    train.add_argument("--stage", required=True)
    train.add_argument("--output")
    train.add_argument("--resume")
    train.add_argument("--no-resume", action="store_true")
    train.add_argument("--device", choices=("cpu", "cuda"))
    train.add_argument("--seed", type=int)
    train.add_argument("--validation-fraction", type=float)
    train.add_argument("--checkpoint-every", type=int, default=50)
    train.add_argument("--yes", action="store_true")
    _add_training_override_args(train)
    train.set_defaults(func=command_train)

    curriculum = sub.add_parser(
        "curriculum",
        help="Run Stage 1, Stage 2, and Stage 3 in order.",
    )
    curriculum.add_argument("--resume")
    curriculum.add_argument("--device", choices=("cpu", "cuda"))
    curriculum.add_argument("--seed", type=int)
    curriculum.add_argument("--validation-fraction", type=float)
    curriculum.add_argument("--checkpoint-every", type=int, default=50)
    curriculum.add_argument("--skip-existing", action="store_true")
    curriculum.add_argument("--yes", action="store_true")
    _add_training_override_args(curriculum)
    curriculum.set_defaults(func=command_curriculum)

    score = sub.add_parser("score", help="Score a checkpoint on corpus records.")
    score.add_argument("checkpoint")
    score.add_argument("--stage", default="3")
    score.add_argument("--inputs", nargs="*")
    score.add_argument("--device", choices=("cpu", "cuda"))
    score.add_argument("--sequence-length", type=int)
    score.add_argument("--batch-size", type=int)
    score.add_argument("--json", action="store_true")
    score.set_defaults(func=command_score)

    evaluate = sub.add_parser(
        "evaluate",
        help="Generate text and inspect drive-neuron activity.",
    )
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--prompt")
    evaluate.add_argument("--prompt-file")
    evaluate.add_argument("--tokens", type=int, default=100)
    evaluate.add_argument("--temperature", type=float, default=0.8)
    evaluate.add_argument("--top-k", type=int, default=40)
    evaluate.add_argument("--max-context", type=int, default=1024)
    evaluate.add_argument("--device", choices=("cpu", "cuda"))
    evaluate.add_argument("--json", action="store_true")
    _add_drive_args(evaluate)
    evaluate.set_defaults(func=command_evaluate)

    neurons = sub.add_parser(
        "neurons",
        help="Inspect dedicated hunger, thirst, boredom, and other drive neurons.",
    )
    neurons.add_argument("--json", action="store_true")
    _add_drive_args(neurons)
    neurons.set_defaults(func=command_neurons)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
