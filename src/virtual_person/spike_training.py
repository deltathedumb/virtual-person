from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .spike_tokenizer import ByteTokenizer
from .spiking import NodeLinkSpikeModel, SpikingModelConfig


@dataclass(slots=True, frozen=True)
class TrainConfig:
    sequence_length: int = 256
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    language_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    value_loss_weight: float = 0.15
    spike_sparsity_weight: float = 0.01
    device: str = "cpu"
    seed: int = 0
    num_workers: int = 0
    checkpoint_every_steps: int = 0


@dataclass(slots=True)
class TrainingExample:
    text: str
    state_features: list[float]
    action_target: int = -100
    value_target: float = 0.0


class MixedTrainingDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        examples: Sequence[TrainingExample],
        *,
        tokenizer: ByteTokenizer,
        sequence_length: int,
        state_feature_count: int,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.state_feature_count = state_feature_count

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        example = self.examples[index]
        ids = self.tokenizer.encode(example.text)
        ids = ids[: self.sequence_length + 1]
        if len(ids) < 2:
            ids = [self.tokenizer.BOS, self.tokenizer.EOS]

        input_ids = ids[:-1]
        labels = ids[1:]
        padding = self.sequence_length - len(input_ids)
        if padding > 0:
            input_ids += [self.tokenizer.PAD] * padding
            labels += [-100] * padding
        else:
            input_ids = input_ids[: self.sequence_length]
            labels = labels[: self.sequence_length]

        features = list(example.state_features)
        if len(features) < self.state_feature_count:
            features.extend([0.0] * (self.state_feature_count - len(features)))
        features = features[: self.state_feature_count]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "state_features": torch.tensor(features, dtype=torch.float32),
            "action_target": torch.tensor(example.action_target, dtype=torch.long),
            "value_target": torch.tensor(example.value_target, dtype=torch.float32),
        }


def read_training_examples(
    paths: Sequence[str | Path],
    *,
    state_feature_count: int = 16,
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    zero_state = [0.0] * state_feature_count
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            child_paths = sorted(
                child for child in path.rglob("*")
                if child.suffix.lower() in {".txt", ".jsonl"}
            )
            examples.extend(
                read_training_examples(
                    child_paths,
                    state_feature_count=state_feature_count,
                )
            )
            continue

        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8")
            paragraphs = [
                paragraph.strip()
                for paragraph in text.split("\n\n")
                if paragraph.strip()
            ]
            examples.extend(
                TrainingExample(paragraph, list(zero_state))
                for paragraph in paragraphs
            )
            continue

        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    text = str(
                        item.get("text")
                        or item.get("prompt")
                        or item.get("content")
                        or ""
                    )
                    if not text:
                        continue
                    features = [
                        float(value)
                        for value in item.get("state_features", zero_state)
                    ]
                    examples.append(
                        TrainingExample(
                            text=text,
                            state_features=features,
                            action_target=int(item.get("action_target", -100)),
                            value_target=float(item.get("value_target", 0.0)),
                        )
                    )
            continue

        raise ValueError(f"Unsupported training file: {path}")
    return examples


def build_dictionary_corpus(
    source: str | Path,
    output: str | Path,
) -> int:
    """
    Convert CSV or JSONL dictionary entries into contextual English examples.

    Accepted fields:
      word, definition, part_of_speech, example, synonyms, antonyms
    """
    source = Path(source)
    output = Path(output)
    rows: list[dict[str, Any]] = []

    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    elif source.suffix.lower() == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    else:
        raise ValueError("Dictionary source must be .csv or .jsonl")

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for entry in rows:
            word = str(entry.get("word", "")).strip()
            definition = str(entry.get("definition", "")).strip()
            if not word or not definition:
                continue
            part = str(entry.get("part_of_speech", "word")).strip() or "word"
            example = str(entry.get("example", "")).strip()
            synonyms = entry.get("synonyms", [])
            antonyms = entry.get("antonyms", [])
            if isinstance(synonyms, str):
                synonyms = [item.strip() for item in synonyms.split(",") if item.strip()]
            if isinstance(antonyms, str):
                antonyms = [item.strip() for item in antonyms.split(",") if item.strip()]

            variants = [
                f'The {part} "{word}" means {definition}.',
                f'Question: What does "{word}" mean?\nAnswer: {definition}.',
                f'Use the word "{word}" correctly. Its meaning is: {definition}.',
                f'"{word}" is a {part}. A person can understand it as {definition}.',
            ]
            if example:
                variants.append(
                    f'Example using "{word}": {example}\n'
                    f'In this example, "{word}" relates to the meaning: {definition}.'
                )
            if synonyms:
                variants.append(
                    f'Words related in meaning to "{word}" include '
                    + ", ".join(map(str, synonyms))
                    + "."
                )
            if antonyms:
                variants.append(
                    f'Words contrasting with "{word}" include '
                    + ", ".join(map(str, antonyms))
                    + "."
                )

            for text in variants:
                handle.write(json.dumps({
                    "text": text,
                    "state_features": [0.0] * 16,
                    "action_target": -100,
                    "value_target": 0.0,
                }, ensure_ascii=False) + "\n")
                count += 1
    return count


def build_behavior_record(
    *,
    state_text: str,
    candidates: Sequence[dict[str, Any]],
    selected_index: int,
    reward: float,
    state_features: Sequence[float] | None = None,
) -> dict[str, Any]:
    lines = [
        "Current state:",
        state_text,
        "Available action candidates:",
    ]
    for index, candidate in enumerate(candidates):
        lines.append(
            f"{index}: {json.dumps(candidate, separators=(',', ':'), sort_keys=True)}"
        )
    lines.append(f"Correct candidate: {selected_index}")
    return {
        "text": "\n".join(lines),
        "state_features": list(state_features or [0.0] * 16),
        "action_target": int(selected_index),
        "value_target": float(reward),
    }


def save_checkpoint(
    path: str | Path,
    model: NodeLinkSpikeModel,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "config": asdict(model.config),
        "model": model.state_dict(),
        "step": int(step),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, Path(path))


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[NodeLinkSpikeModel, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = NodeLinkSpikeModel(SpikingModelConfig(**payload["config"]))
    model.load_state_dict(payload["model"], strict=False)
    model.to(device)
    return model, payload


def train_model(
    model: NodeLinkSpikeModel,
    examples: Sequence[TrainingExample],
    config: TrainConfig,
    *,
    checkpoint_path: str | Path | None = None,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    optimizer_state: dict[str, Any] | None = None,
    start_step: int = 0,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    if not examples:
        raise ValueError("No training examples were supplied")
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    device = torch.device(config.device)
    model.to(device)
    model.train()
    tokenizer = ByteTokenizer()
    dataset = MixedTrainingDataset(
        examples,
        tokenizer=tokenizer,
        sequence_length=config.sequence_length,
        state_feature_count=model.config.state_feature_count,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=max(0, int(config.num_workers)),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    history: list[dict[str, float]] = []
    global_step = int(start_step)
    batches_per_epoch = max(1, len(loader))
    total_batches = max(1, batches_per_epoch * config.epochs)
    completed_batches = 0
    stopped = False

    def checkpoint_meta(extra: dict[str, Any]) -> dict[str, Any]:
        return {**(checkpoint_metadata or {}), **extra}

    for epoch in range(config.epochs):
        for batch_index, batch in enumerate(loader):
            if stop_requested is not None and stop_requested():
                stopped = True
                break

            input_ids = batch["input_ids"].to(device, non_blocking=device.type == "cuda")
            labels = batch["labels"].to(device, non_blocking=device.type == "cuda")
            features = batch["state_features"].to(
                device,
                non_blocking=device.type == "cuda",
            )
            action_target = batch["action_target"].to(
                device,
                non_blocking=device.type == "cuda",
            )
            value_target = batch["value_target"].to(
                device,
                non_blocking=device.type == "cuda",
            )

            output = model(input_ids, state_features=features)
            language_loss = torch.nn.functional.cross_entropy(
                output.token_logits.reshape(-1, model.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )

            valid_actions = action_target != -100
            if valid_actions.any():
                action_loss = torch.nn.functional.cross_entropy(
                    output.action_logits[valid_actions],
                    action_target[valid_actions],
                )
            else:
                action_loss = torch.zeros((), device=device)

            value_loss = torch.nn.functional.mse_loss(
                output.value,
                value_target,
            )
            spike_sparsity_loss = output.spike_rates.mean()

            loss = (
                config.language_loss_weight * language_loss
                + config.action_loss_weight * action_loss
                + config.value_loss_weight * value_loss
                + config.spike_sparsity_weight * spike_sparsity_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip,
            )
            optimizer.step()

            completed_batches += 1
            row = {
                "epoch": float(epoch + 1),
                "epochs": float(config.epochs),
                "batch": float(batch_index + 1),
                "batches": float(batches_per_epoch),
                "step": float(global_step),
                "progress": float(completed_batches / total_batches),
                "loss": float(loss.detach().cpu()),
                "language_loss": float(language_loss.detach().cpu()),
                "action_loss": float(action_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "spike_rate": float(output.spike_rates.mean().detach().cpu()),
            }
            history.append(row)
            global_step += 1

            if progress_callback is not None:
                progress_callback(dict(row))

            if (
                checkpoint_path is not None
                and config.checkpoint_every_steps > 0
                and global_step % config.checkpoint_every_steps == 0
            ):
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer=optimizer,
                    step=global_step,
                    metadata=checkpoint_meta({
                        "history_tail": history[-20:],
                        "stopped": False,
                    }),
                )

        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer=optimizer,
                step=global_step,
                metadata=checkpoint_meta({
                    "history_tail": history[-20:],
                    "stopped": stopped,
                    "completed_epoch": epoch + (0 if stopped else 1),
                }),
            )
        if stopped:
            break

    return history



@dataclass(slots=True, frozen=True)
class ScoreResult:
    examples: int
    tokens: int
    language_loss: float
    byte_perplexity: float
    action_examples: int
    action_accuracy: float | None
    value_mse: float
    spike_rate: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def score_model(
    model: NodeLinkSpikeModel,
    examples: Sequence[TrainingExample],
    *,
    sequence_length: int = 256,
    batch_size: int = 8,
    device: str | torch.device = "cpu",
) -> ScoreResult:
    """Evaluate language, action, value, and spike metrics without training."""
    if not examples:
        raise ValueError("No examples were supplied for scoring")

    device = torch.device(device)
    model.to(device)
    model.eval()
    dataset = MixedTrainingDataset(
        examples,
        tokenizer=ByteTokenizer(),
        sequence_length=sequence_length,
        state_feature_count=model.config.state_feature_count,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    language_loss_sum = 0.0
    token_count = 0
    action_correct = 0
    action_count = 0
    value_error_sum = 0.0
    value_count = 0
    spike_sum = 0.0
    spike_elements = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        features = batch["state_features"].to(device)
        action_target = batch["action_target"].to(device)
        value_target = batch["value_target"].to(device)

        output = model(input_ids, state_features=features)

        flat_logits = output.token_logits.reshape(-1, model.config.vocab_size)
        flat_labels = labels.reshape(-1)
        valid_tokens = flat_labels != -100
        if valid_tokens.any():
            loss_sum = torch.nn.functional.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=-100,
                reduction="sum",
            )
            language_loss_sum += float(loss_sum.cpu())
            token_count += int(valid_tokens.sum().item())

        valid_actions = action_target != -100
        if valid_actions.any():
            predictions = output.action_logits[valid_actions].argmax(dim=-1)
            action_correct += int(
                (predictions == action_target[valid_actions]).sum().item()
            )
            action_count += int(valid_actions.sum().item())

        squared = (output.value - value_target) ** 2
        value_error_sum += float(squared.sum().cpu())
        value_count += int(squared.numel())

        spike_sum += float(output.spike_rates.sum().cpu())
        spike_elements += int(output.spike_rates.numel())

    language_loss = language_loss_sum / max(1, token_count)
    return ScoreResult(
        examples=len(examples),
        tokens=token_count,
        language_loss=language_loss,
        byte_perplexity=math.exp(min(20.0, language_loss)),
        action_examples=action_count,
        action_accuracy=(
            action_correct / action_count if action_count else None
        ),
        value_mse=value_error_sum / max(1, value_count),
        spike_rate=spike_sum / max(1, spike_elements),
    )
