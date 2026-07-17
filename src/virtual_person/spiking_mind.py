from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from .drives import CognitiveDrives
from .spike_tokenizer import ByteTokenizer
from .spiking import NodeLinkSpikeModel, SpikingModelConfig
from .types import Action, ActionKind


@dataclass(slots=True, frozen=True)
class MindDecision:
    candidate_index: int
    action: Action
    confidence: float
    value_estimate: float
    prompt: str
    spike_rate: float
    drive_activity: dict[str, float]


class StateFeatureEncoder:
    """
    Stable numeric state vector used alongside text.

    Feature order:
      hunger, thirst, fatigue, bladder, 1-hygiene, 1-health, social,
      boredom, curiosity, loneliness, competence frustration, enjoyment,
      hour sine-like proxy, hour cosine-like proxy, task pending, bias
    """

    feature_count = 16

    def encode(
        self,
        observation: dict[str, Any],
        cognitive: CognitiveDrives,
        *,
        hour_of_day: float = 12.0,
        task_pending: bool = False,
    ) -> list[float]:
        body = (
            observation.get("body")
            or observation.get("agent", {}).get("body")
            or observation.get("world", {}).get("body")
            or {}
        )
        # No math dependency is needed for a rough periodic representation.
        phase = (hour_of_day % 24.0) / 24.0
        triangular = 1.0 - abs(phase * 2.0 - 1.0)
        inverse_triangular = 1.0 - triangular
        values = [
            float(body.get("hunger", 0.0)),
            float(body.get("thirst", 0.0)),
            float(body.get("fatigue", 0.0)),
            float(body.get("bladder", 0.0)),
            1.0 - float(body.get("hygiene", 1.0)),
            1.0 - float(body.get("health", 1.0)),
            float(body.get("social", 0.0)),
            cognitive.boredom,
            cognitive.curiosity,
            cognitive.loneliness,
            cognitive.competence_frustration,
            cognitive.enjoyment,
            triangular,
            inverse_triangular,
            1.0 if task_pending else 0.0,
            1.0,
        ]
        return [max(-1.0, min(1.0, value)) for value in values]


class PromptBuilder:
    def build(
        self,
        observation: dict[str, Any],
        candidates: Sequence[dict[str, Any]],
        memories: Sequence[str] = (),
    ) -> str:
        compact_observation = {
            "room": observation.get("room")
            or observation.get("agent", {}).get("room"),
            "body": observation.get("body")
            or observation.get("agent", {}).get("body"),
            "inventory": observation.get("inventory")
            or observation.get("agent", {}).get("inventory"),
            "visible_objects": observation.get("visible_objects")
            or observation.get("world", {}).get("visible_objects"),
            "dirty_dishes": observation.get("dirty_dishes")
            or observation.get("world", {}).get("dirty_dishes"),
        }
        lines = [
            "You are choosing one physically valid action.",
            "Current state:",
            json.dumps(compact_observation, separators=(",", ":"), sort_keys=True),
        ]
        if memories:
            lines.append("Relevant memories:")
            lines.extend(f"- {memory}" for memory in memories[-8:])
        lines.append("Available action candidates:")
        for index, candidate in enumerate(candidates):
            lines.append(
                f"{index}: {json.dumps(candidate, separators=(',', ':'), sort_keys=True)}"
            )
        lines.append("Choose the action that best improves long-term satisfaction.")
        return "\n".join(lines)


class ActionCodec:
    @staticmethod
    def from_template(template: dict[str, Any]) -> Action:
        kind_value = template.get("kind")
        if isinstance(kind_value, ActionKind):
            kind = kind_value
        else:
            kind = ActionKind[str(kind_value).upper()]
        return Action(
            kind=kind,
            target=template.get("target"),
            secondary=template.get("secondary"),
            value=template.get("value"),
        )


class SpikingMind:
    def __init__(
        self,
        model: NodeLinkSpikeModel,
        *,
        tokenizer: ByteTokenizer | None = None,
        device: str | torch.device = "cpu",
        max_context_tokens: int = 1024,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer or ByteTokenizer()
        self.device = torch.device(device)
        self.max_context_tokens = max_context_tokens
        self.features = StateFeatureEncoder()
        self.prompts = PromptBuilder()
        self.cognitive_drives = CognitiveDrives()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "SpikingMind":
        payload = torch.load(path, map_location=device, weights_only=False)
        config = SpikingModelConfig(**payload["config"])
        model = NodeLinkSpikeModel(config)
        # strict=False can migrate v0.2 checkpoints; new drive links start
        # initialized and should then be trained.
        model.load_state_dict(payload["model"], strict=False)
        return cls(model, device=device)

    @torch.no_grad()
    def choose_action(
        self,
        observation: dict[str, Any],
        candidates: Sequence[dict[str, Any]],
        *,
        memories: Sequence[str] = (),
        hour_of_day: float = 12.0,
        task_pending: bool = False,
        deterministic: bool = True,
    ) -> MindDecision:
        if not candidates:
            raise ValueError("At least one action candidate is required")
        if len(candidates) > self.model.config.max_action_candidates:
            candidates = candidates[: self.model.config.max_action_candidates]

        prompt = self.prompts.build(observation, candidates, memories)
        token_ids = self.tokenizer.encode(prompt)
        token_ids = token_ids[-self.max_context_tokens :]
        token_tensor = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )
        feature_values = self.features.encode(
            observation,
            self.cognitive_drives,
            hour_of_day=hour_of_day,
            task_pending=task_pending,
        )
        feature_tensor = torch.tensor(
            [feature_values],
            dtype=torch.float32,
            device=self.device,
        )
        output = self.model(token_tensor, state_features=feature_tensor)
        logits = output.action_logits[0, : len(candidates)]
        probabilities = torch.softmax(logits, dim=-1)

        if deterministic:
            index = int(torch.argmax(probabilities).item())
        else:
            index = int(torch.multinomial(probabilities, 1).item())

        return MindDecision(
            candidate_index=index,
            action=ActionCodec.from_template(candidates[index]),
            confidence=float(probabilities[index].item()),
            value_estimate=float(output.value[0].item()),
            prompt=prompt,
            spike_rate=float(output.spike_rates.mean().item()),
            drive_activity=self.model.drive_activity_report(
                output.drive_spikes,
                minimum_rate=1e-9,
            ),
        )

    def update_drives_from_result(
        self,
        *,
        elapsed_seconds: float,
        action_ok: bool,
        reward: float,
        action_kind: ActionKind,
    ) -> None:
        progress = max(0.0, min(1.0, reward))
        novelty = 0.15 if action_kind in {
            ActionKind.INSPECT,
            ActionKind.MOVE,
            ActionKind.USE,
        } else 0.0
        resting = action_kind in {ActionKind.SLEEP, ActionKind.WAIT}
        self.cognitive_drives.update(
            elapsed_seconds,
            meaningful_progress=progress,
            novelty_understood=novelty if action_ok else 0.0,
            failed_attempt=0.0 if action_ok else 1.0,
            resting_is_appropriate=resting,
        )
