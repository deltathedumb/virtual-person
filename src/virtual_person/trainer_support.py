from __future__ import annotations

import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .spiking import SpikingModelConfig


SOURCE_CATEGORIES = (
    "English",
    "Dictionary",
    "Procedures",
    "Safety/Judgment",
    "Behavior",
)

CURRICULUM_CATEGORIES: dict[str, tuple[str, ...]] = {
    "Stage 1 — English and vocabulary": ("English", "Dictionary"),
    "Stage 2 — Practical knowledge and judgment": (
        "English",
        "Dictionary",
        "Procedures",
        "Safety/Judgment",
    ),
    "Stage 3 — Autonomous behavior and drives": SOURCE_CATEGORIES,
    "All selected data": SOURCE_CATEGORIES,
}


@dataclass(slots=True)
class CorpusSource:
    path: str
    category: str
    enabled: bool = True

    def normalized_path(self) -> Path:
        return Path(self.path).expanduser().resolve()


@dataclass(slots=True)
class CorpusStats:
    files: int = 0
    records: int = 0
    characters: int = 0
    language_records: int = 0
    action_records: int = 0
    value_records: int = 0
    malformed_records: int = 0
    missing_paths: int = 0

    def add(self, other: "CorpusStats") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    @property
    def valid(self) -> bool:
        return self.missing_paths == 0 and self.malformed_records == 0 and self.records > 0


@dataclass(slots=True, frozen=True)
class HardwareInfo:
    python: str
    platform: str
    torch_version: str
    cpu_count: int
    ram_bytes: int | None
    cuda_available: bool
    cuda_device_count: int
    gpu_name: str | None
    gpu_memory_bytes: int | None

    @property
    def suggested_device(self) -> str:
        return "cuda" if self.cuda_available else "cpu"


@dataclass(slots=True, frozen=True)
class ModelProfile:
    name: str
    description: str
    hidden_size: int
    layers: int
    ticks: int
    sequence_length: int
    batch_size: int
    epochs: int
    learning_rate: float


MODEL_PROFILES: dict[str, ModelProfile] = {
    "Architecture smoke test": ModelProfile(
        "Architecture smoke test",
        "Verifies that data, gradients, checkpoints, and the UI work. Not a useful mind.",
        hidden_size=32,
        layers=1,
        ticks=1,
        sequence_length=64,
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
    ),
    "CPU prototype": ModelProfile(
        "CPU prototype",
        "Small enough for ordinary development. Use this to prove your dataset format.",
        hidden_size=96,
        layers=2,
        ticks=2,
        sequence_length=160,
        batch_size=2,
        epochs=3,
        learning_rate=3e-4,
    ),
    "Small GPU experiment": ModelProfile(
        "Small GPU experiment",
        "A serious architecture experiment, but still far below adult-level language.",
        hidden_size=192,
        layers=4,
        ticks=3,
        sequence_length=256,
        batch_size=4,
        epochs=5,
        learning_rate=3e-4,
    ),
    "Larger research run": ModelProfile(
        "Larger research run",
        "Requires substantial compute and an optimized kernel before it is practical.",
        hidden_size=384,
        layers=8,
        ticks=4,
        sequence_length=512,
        batch_size=2,
        epochs=8,
        learning_rate=2e-4,
    ),
}


@dataclass(slots=True, frozen=True)
class ModelEstimate:
    parameters: int
    parameter_bytes_fp32: int
    optimizer_bytes_fp32: int
    activation_bytes_estimate: int
    total_training_bytes_estimate: int

    @property
    def parameter_millions(self) -> float:
        return self.parameters / 1_000_000.0


@dataclass(slots=True)
class TrainerProject:
    workspace: str
    sources: list[CorpusSource] = field(default_factory=list)
    model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path | None = None) -> Path:
        destination = Path(path) if path else Path(self.workspace) / "trainer_project.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "workspace": self.workspace,
                    "sources": [asdict(source) for source in self.sources],
                    "model": self.model,
                    "training": self.training,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "TrainerProject":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            workspace=str(payload["workspace"]),
            sources=[CorpusSource(**row) for row in payload.get("sources", [])],
            model=dict(payload.get("model", {})),
            training=dict(payload.get("training", {})),
        )


def format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024.0 or suffix == "TiB":
            return f"{amount:.2f} {suffix}"
        amount /= 1024.0
    return f"{amount:.2f} TiB"


def _system_ram_bytes() -> int | None:
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        if hasattr(os, "sysconf"):
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return pages * page_size
    except Exception:
        return None
    return None


def detect_hardware() -> HardwareInfo:
    cuda = bool(torch.cuda.is_available())
    gpu_name: str | None = None
    gpu_memory: int | None = None
    device_count = int(torch.cuda.device_count()) if cuda else 0
    if cuda and device_count:
        properties = torch.cuda.get_device_properties(0)
        gpu_name = str(properties.name)
        gpu_memory = int(properties.total_memory)
    return HardwareInfo(
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        torch_version=str(torch.__version__),
        cpu_count=os.cpu_count() or 1,
        ram_bytes=_system_ram_bytes(),
        cuda_available=cuda,
        cuda_device_count=device_count,
        gpu_name=gpu_name,
        gpu_memory_bytes=gpu_memory,
    )


def recommended_profile(hardware: HardwareInfo) -> str:
    if not hardware.cuda_available:
        return "CPU prototype"
    memory_gib = (hardware.gpu_memory_bytes or 0) / (1024**3)
    if memory_gib >= 12:
        return "Small GPU experiment"
    return "CPU prototype"


def estimate_model(
    config: SpikingModelConfig,
    *,
    batch_size: int,
    sequence_length: int,
) -> ModelEstimate:
    h = int(config.hidden_size)
    layers = int(config.layer_count)
    expansion = int(config.expansion)
    expanded = h * expansion
    drive_neurons = 39 if config.use_dedicated_drive_neurons else 0

    global_parameters = (
        config.vocab_size * h
        + config.state_feature_count * h
        + drive_neurons * h
        + h  # input norm
        + h  # output norm
        + config.max_action_candidates * h
        + config.max_action_candidates
    )
    value_hidden = max(1, h // 2)
    global_parameters += h * value_hidden + value_hidden + value_hidden + 1

    per_layer = (
        h * h  # temporal input links
        + h * h  # recurrent links
        + 4 * h  # decay, threshold, bias, reset
        + h * h  # spike projection
        + h  # pre norm
        + h  # channel norm
        + h * expanded + expanded  # channel input
        + h * expanded + expanded  # channel gate
        + expanded * h + h  # channel output
    )
    parameters = int(global_parameters + layers * per_layer)
    parameter_bytes = parameters * 4
    # FP32 AdamW: weight + gradient + two moment buffers.
    optimizer_bytes = parameters * 16

    # Conservative approximation for unrolled surrogate-gradient state.
    activation_elements = (
        max(1, batch_size)
        * max(1, sequence_length)
        * max(1, config.ticks_per_token)
        * max(1, layers)
        * max(1, h)
        * 10
    )
    activation_bytes = int(activation_elements * 4)
    total = parameter_bytes + optimizer_bytes + activation_bytes
    return ModelEstimate(
        parameters=parameters,
        parameter_bytes_fp32=parameter_bytes,
        optimizer_bytes_fp32=optimizer_bytes,
        activation_bytes_estimate=activation_bytes,
        total_training_bytes_estimate=total,
    )


def create_workspace(path: str | Path) -> dict[str, Path]:
    root = Path(path).expanduser().resolve()
    layout = {
        "root": root,
        "raw_english": root / "raw" / "english",
        "raw_dictionaries": root / "raw" / "dictionaries",
        "raw_procedures": root / "raw" / "procedures",
        "raw_behavior": root / "raw" / "behavior",
        "corpora": root / "corpora",
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
        "exports": root / "exports",
    }
    for directory in layout.values():
        directory.mkdir(parents=True, exist_ok=True)

    guide = root / "README_FIRST.txt"
    if not guide.exists():
        guide.write_text(
            "1. Add English, dictionary, procedure, safety, and behavior data.\n"
            "2. Validate the corpus in the Trainer UI.\n"
            "3. Run an Architecture smoke test.\n"
            "4. Train Stage 1, then continue its checkpoint through Stages 2 and 3.\n"
            "5. Evaluate after every stage and keep older checkpoints.\n",
            encoding="utf-8",
        )
    return layout


def create_starter_pack(workspace: str | Path) -> list[CorpusSource]:
    layout = create_workspace(workspace)

    english = layout["raw_english"] / "starter_english.txt"
    english.write_text(
        """
A person can describe what they perceive, ask questions, explain a decision, and
correct a misunderstanding. Pronouns refer to people or objects already mentioned.

Mira entered the kitchen because she was hungry. She checked the refrigerator
before deciding what to cook. The eggs were available, so she chose a simple meal.

Waiting can be part of an activity. A person may wait for water to boil while
remaining attentive to the stove. Meaningful rest is different from purposeless
inactivity.
""".strip(),
        encoding="utf-8",
    )

    procedures = layout["raw_procedures"] / "starter_procedures.txt"
    procedures.write_text(
        """
To prepare a simple meal, inspect the available food, select a safe recipe, gather
the necessary tools, cook while monitoring heat, turn appliances off, eat, and
clean the used dishes.

To use a computer, observe the current application, identify controls such as text
fields and buttons, perform the smallest useful action, inspect the result, save
important work, and ask before consequential operations.

To clean a room, collect trash, return objects to appropriate locations, wipe
surfaces using suitable materials, and stop if an unfamiliar chemical or hazard
is present.
""".strip(),
        encoding="utf-8",
    )

    safety = layout["raw_procedures"] / "starter_safety.txt"
    safety.write_text(
        """
If a pan begins smoking, turn off the heat if it is safe to do so and seek help.
Do not continue merely because the original timer has not finished.

When an object is uncertain, inspect it or ask. Do not drink, eat, heat, or apply
an unidentified substance.

A failed action is information. Stop repeated failed attempts, reconsider the
preconditions, and choose a safer alternative.
""".strip(),
        encoding="utf-8",
    )

    behavior = layout["raw_behavior"] / "starter_behavior.jsonl"
    records = [
        {
            "text": (
                "State: hunger is urgent; food is available; the kitchen is reachable. "
                "Candidates: 0 wait, 1 go to kitchen, 2 sleep. Correct candidate: 1."
            ),
            "state_features": [0.92, 0.20, 0.20, 0.10, 0.05, 0.0, 0.10, 0.20, 0.40, 0.10, 0.10, 0.50, 0.5, 0.5, 0.0, 1.0],
            "action_target": 1,
            "value_target": 0.9,
        },
        {
            "text": (
                "State: thirst is urgent; clean water is reachable. "
                "Candidates: 0 drink water, 1 open a random application, 2 wait. "
                "Correct candidate: 0."
            ),
            "state_features": [0.15, 0.93, 0.20, 0.10, 0.05, 0.0, 0.10, 0.20, 0.40, 0.10, 0.10, 0.50, 0.5, 0.5, 0.0, 1.0],
            "action_target": 0,
            "value_target": 0.95,
        },
        {
            "text": (
                "State: boredom is high; an unfinished programming project is available. "
                "Candidates: 0 click randomly, 1 continue the project, 2 walk in circles. "
                "Correct candidate: 1."
            ),
            "state_features": [0.15, 0.15, 0.20, 0.10, 0.05, 0.0, 0.10, 0.86, 0.55, 0.10, 0.10, 0.45, 0.5, 0.5, 1.0, 1.0],
            "action_target": 1,
            "value_target": 0.8,
        },
        {
            "text": (
                "State: fatigue is urgent and no emergency exists. "
                "Candidates: 0 begin a risky repair, 1 sleep, 2 drink repeatedly. "
                "Correct candidate: 1."
            ),
            "state_features": [0.20, 0.20, 0.91, 0.15, 0.10, 0.0, 0.10, 0.25, 0.30, 0.10, 0.10, 0.45, 0.1, 0.9, 0.0, 1.0],
            "action_target": 1,
            "value_target": 0.9,
        },
    ]
    behavior.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    return [
        CorpusSource(str(english), "English"),
        CorpusSource(str(procedures), "Procedures"),
        CorpusSource(str(safety), "Safety/Judgment"),
        CorpusSource(str(behavior), "Behavior"),
    ]


def _iter_source_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in {".txt", ".jsonl"}:
                yield child


def scan_source(source: CorpusSource) -> CorpusStats:
    path = source.normalized_path()
    stats = CorpusStats()
    if not path.exists():
        stats.missing_paths = 1
        return stats

    for file_path in _iter_source_files(path):
        stats.files += 1
        try:
            if file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8")
                paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
                stats.records += len(paragraphs)
                stats.language_records += len(paragraphs)
                stats.characters += len(text)
            elif file_path.suffix.lower() == ".jsonl":
                with file_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                            text = str(
                                item.get("text")
                                or item.get("prompt")
                                or item.get("content")
                                or ""
                            )
                            if not text:
                                stats.malformed_records += 1
                                continue
                            stats.records += 1
                            stats.characters += len(text)
                            action_target = int(item.get("action_target", -100))
                            if action_target == -100:
                                stats.language_records += 1
                            else:
                                stats.action_records += 1
                            if "value_target" in item:
                                stats.value_records += 1
                        except Exception:
                            stats.malformed_records += 1
        except (OSError, UnicodeError):
            stats.malformed_records += 1
    return stats


def scan_sources(
    sources: Sequence[CorpusSource],
) -> tuple[CorpusStats, dict[str, CorpusStats], dict[str, CorpusStats]]:
    total = CorpusStats()
    by_path: dict[str, CorpusStats] = {}
    by_category: dict[str, CorpusStats] = {}
    for source in sources:
        if not source.enabled:
            continue
        stats = scan_source(source)
        by_path[source.path] = stats
        category_stats = by_category.setdefault(source.category, CorpusStats())
        category_stats.add(stats)
        total.add(stats)
    return total, by_path, by_category


def filter_sources_for_stage(
    sources: Sequence[CorpusSource],
    stage: str,
) -> list[CorpusSource]:
    categories = CURRICULUM_CATEGORIES.get(stage, SOURCE_CATEGORIES)
    return [
        source
        for source in sources
        if source.enabled and source.category in categories
    ]
