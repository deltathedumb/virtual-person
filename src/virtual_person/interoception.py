from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass(slots=True, frozen=True)
class DriveSourceSpec:
    """Maps one named internal drive to a stable state-feature index."""

    name: str
    feature_index: int
    invert: bool = False


@dataclass(slots=True, frozen=True)
class DriveNeuronSpec:
    """
    An interpretable sensory neuron.

    A neuron fires whenever its named drive crosses its threshold. High values
    can activate several severity neurons at once, producing a population code.
    """

    neuron_id: int
    name: str
    source_name: str
    feature_index: int
    threshold: float
    invert: bool = False


DEFAULT_DRIVE_SOURCES: tuple[DriveSourceSpec, ...] = (
    DriveSourceSpec("hunger", 0),
    DriveSourceSpec("thirst", 1),
    DriveSourceSpec("fatigue", 2),
    DriveSourceSpec("bladder", 3),
    DriveSourceSpec("hygiene_discomfort", 4),
    DriveSourceSpec("health_distress", 5),
    DriveSourceSpec("social_need", 6),
    DriveSourceSpec("boredom", 7),
    DriveSourceSpec("curiosity", 8),
    DriveSourceSpec("loneliness", 9),
    DriveSourceSpec("competence_frustration", 10),
    DriveSourceSpec("low_enjoyment", 11, invert=True),
    DriveSourceSpec("task_pending", 14),
)

DEFAULT_LEVELS: tuple[tuple[str, float], ...] = (
    ("notice", 0.35),
    ("need", 0.60),
    ("urgent", 0.85),
)


class DedicatedDriveNeuronBank(nn.Module):
    """
    Fixed, named, inspectable input neurons for interoception.

    These neurons are intentionally not learned. The external body/drive system
    owns the values, and this bank converts them into spikes. Trainable links
    from these nodes into the recurrent model learn how the signal affects action.
    """

    def __init__(
        self,
        *,
        feature_count: int = 16,
        sources: Sequence[DriveSourceSpec] = DEFAULT_DRIVE_SOURCES,
        levels: Sequence[tuple[str, float]] = DEFAULT_LEVELS,
    ) -> None:
        super().__init__()
        if feature_count <= 0:
            raise ValueError("feature_count must be positive")
        if not sources:
            raise ValueError("At least one drive source is required")
        if not levels:
            raise ValueError("At least one drive-neuron level is required")

        self.feature_count = int(feature_count)
        self.sources = tuple(sources)
        self.levels = tuple((str(name), float(threshold)) for name, threshold in levels)

        specs: list[DriveNeuronSpec] = []
        for source in self.sources:
            if not 0 <= source.feature_index < self.feature_count:
                raise ValueError(
                    f"Drive source {source.name!r} uses invalid feature index "
                    f"{source.feature_index}"
                )
            for level_name, threshold in self.levels:
                if not 0.0 <= threshold <= 1.0:
                    raise ValueError("Drive thresholds must be between 0 and 1")
                specs.append(
                    DriveNeuronSpec(
                        neuron_id=len(specs),
                        name=f"{source.name}_{level_name}",
                        source_name=source.name,
                        feature_index=source.feature_index,
                        threshold=threshold,
                        invert=source.invert,
                    )
                )
        self.specs = tuple(specs)

        self.register_buffer(
            "source_indices",
            torch.tensor([spec.feature_index for spec in self.specs], dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "thresholds",
            torch.tensor([spec.threshold for spec in self.specs], dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "invert_mask",
            torch.tensor([spec.invert for spec in self.specs], dtype=torch.bool),
            persistent=True,
        )

    @property
    def neuron_count(self) -> int:
        return len(self.specs)

    @property
    def neuron_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim not in {2, 3}:
            raise ValueError("features must have shape [B,F] or [B,T,F]")
        if features.shape[-1] != self.feature_count:
            raise ValueError(
                f"Expected {self.feature_count} features, got {features.shape[-1]}"
            )

        selected = torch.index_select(features, -1, self.source_indices)
        selected = selected.clamp(0.0, 1.0)
        selected = torch.where(self.invert_mask, 1.0 - selected, selected)
        thresholds = self.thresholds.to(dtype=selected.dtype)
        return (selected >= thresholds).to(dtype=selected.dtype)

    def activity_report(
        self,
        spikes: Tensor,
        *,
        minimum_rate: float = 0.0,
    ) -> dict[str, float]:
        if spikes.shape[-1] != self.neuron_count:
            raise ValueError("Spike tensor does not match this neuron bank")
        if spikes.ndim == 1:
            rates = spikes
        else:
            rates = spikes.float().mean(dim=tuple(range(spikes.ndim - 1)))

        report: dict[str, float] = {}
        for name, value in zip(self.neuron_names, rates.detach().cpu().tolist()):
            rate = float(value)
            if rate >= minimum_rate:
                report[name] = rate
        return report

    def active_neurons(self, spikes: Tensor) -> list[str]:
        report = self.activity_report(spikes, minimum_rate=1e-9)
        return [name for name, rate in report.items() if rate > 0.0]

    def specs_as_dicts(self) -> list[dict[str, object]]:
        return [
            {
                "neuron_id": spec.neuron_id,
                "name": spec.name,
                "source_name": spec.source_name,
                "feature_index": spec.feature_index,
                "threshold": spec.threshold,
                "invert": spec.invert,
            }
            for spec in self.specs
        ]
