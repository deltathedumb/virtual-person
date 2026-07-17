from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .interoception import DedicatedDriveNeuronBank


class SurrogateSpike(torch.autograd.Function):
    """
    Binary forward spike with a smooth triangular surrogate derivative.

    Forward:
        s = 1 when membrane >= threshold, otherwise 0

    Backward:
        ds/dx ~= max(0, 1 - |x| / width) / width
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, width: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.width = float(width)
        return (x >= 0).to(dtype=x.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        (x,) = ctx.saved_tensors
        width = max(1e-4, ctx.width)
        gradient = torch.clamp(1.0 - x.abs() / width, min=0.0) / width
        return grad_output * gradient, None


def spike(x: Tensor, width: float = 1.0) -> Tensor:
    return SurrogateSpike.apply(x, width)


@dataclass(slots=True, frozen=True)
class NodeSpec:
    node_id: int
    threshold: float
    decay: float
    bias: float


@dataclass(slots=True, frozen=True)
class LinkSpec:
    source: int
    destination: int
    weight: float
    delay: int = 0


@dataclass(slots=True)
class ClusterState:
    membrane: Tensor
    spikes: Tensor
    refractory: Tensor

    def detach(self) -> "ClusterState":
        return ClusterState(
            membrane=self.membrane.detach(),
            spikes=self.spikes.detach(),
            refractory=self.refractory.detach(),
        )


@dataclass(slots=True)
class ModelState:
    clusters: list[ClusterState]

    def detach(self) -> "ModelState":
        return ModelState([cluster.detach() for cluster in self.clusters])


class NodeLinkSpikeCluster(nn.Module):
    """
    A vectorized Node-Link-Spike cluster.

    Each hidden unit is a node. `input_links` and `recurrent_links` are weighted
    links. The cluster advances in discrete simulation ticks and emits binary
    spikes. Parameters are represented as tensors for performance, but can be
    exported as explicit NodeSpec/LinkSpec records.
    """

    def __init__(
        self,
        input_size: int,
        node_count: int,
        *,
        recurrent: bool = True,
        surrogate_width: float = 1.0,
        refractory_ticks: int = 1,
    ) -> None:
        super().__init__()
        if input_size <= 0 or node_count <= 0:
            raise ValueError("input_size and node_count must be positive")
        self.input_size = input_size
        self.node_count = node_count
        self.surrogate_width = float(surrogate_width)
        self.refractory_ticks = max(0, int(refractory_ticks))

        self.input_links = nn.Linear(input_size, node_count, bias=False)
        self.recurrent_links = (
            nn.Linear(node_count, node_count, bias=False) if recurrent else None
        )

        # Constrained at runtime:
        # decay = sigmoid(raw_decay), threshold = softplus(raw_threshold)+epsilon
        self.raw_decay = nn.Parameter(torch.full((node_count,), 1.5))
        self.raw_threshold = nn.Parameter(torch.full((node_count,), 0.5))
        self.bias = nn.Parameter(torch.zeros(node_count))
        self.reset_strength = nn.Parameter(torch.ones(node_count))

        nn.init.xavier_uniform_(self.input_links.weight)
        if self.recurrent_links is not None:
            nn.init.orthogonal_(self.recurrent_links.weight, gain=0.45)

    @property
    def decay(self) -> Tensor:
        return torch.sigmoid(self.raw_decay)

    @property
    def threshold(self) -> Tensor:
        return F.softplus(self.raw_threshold) + 1e-3

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ClusterState:
        reference = self.bias
        device = device or reference.device
        dtype = dtype or reference.dtype
        shape = (batch_size, self.node_count)
        return ClusterState(
            membrane=torch.zeros(shape, device=device, dtype=dtype),
            spikes=torch.zeros(shape, device=device, dtype=dtype),
            refractory=torch.zeros(shape, device=device, dtype=dtype),
        )

    def step(self, current: Tensor, state: ClusterState) -> tuple[Tensor, ClusterState]:
        if current.ndim != 2 or current.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected current [batch,{self.input_size}], got {tuple(current.shape)}"
            )

        recurrent_current: Tensor | float = 0.0
        if self.recurrent_links is not None:
            recurrent_current = self.recurrent_links(state.spikes)

        membrane = (
            self.decay * state.membrane
            + self.input_links(current)
            + recurrent_current
            + self.bias
        )

        available = (state.refractory <= 0).to(membrane.dtype)
        emitted = spike(
            membrane - self.threshold,
            self.surrogate_width,
        ) * available

        # Soft reset preserves residual charge and trains more smoothly than
        # replacing membrane with zero.
        membrane = membrane - emitted * self.threshold * self.reset_strength

        refractory = torch.clamp(state.refractory - 1.0, min=0.0)
        if self.refractory_ticks:
            refractory = torch.where(
                emitted > 0,
                torch.full_like(refractory, float(self.refractory_ticks)),
                refractory,
            )

        new_state = ClusterState(
            membrane=membrane,
            spikes=emitted,
            refractory=refractory,
        )
        return emitted, new_state

    @torch.no_grad()
    def export_nodes(self) -> list[NodeSpec]:
        return [
            NodeSpec(
                node_id=index,
                threshold=float(self.threshold[index].cpu()),
                decay=float(self.decay[index].cpu()),
                bias=float(self.bias[index].cpu()),
            )
            for index in range(self.node_count)
        ]

    @torch.no_grad()
    def export_recurrent_links(self, minimum_abs_weight: float = 0.0) -> list[LinkSpec]:
        if self.recurrent_links is None:
            return []
        weights = self.recurrent_links.weight.detach().cpu()
        links: list[LinkSpec] = []
        for destination in range(weights.shape[0]):
            for source in range(weights.shape[1]):
                weight = float(weights[destination, source])
                if abs(weight) >= minimum_abs_weight:
                    links.append(LinkSpec(source, destination, weight))
        return links


class SpikeBlock(nn.Module):
    """
    Hybrid recurrent block with a spiking temporal core.

    Dense projections are treated as bundles of links; the persistent state and
    nonlinear communication are carried by spikes.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        expansion: int = 2,
        surrogate_width: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        expanded = hidden_size * expansion

        self.pre_norm = nn.RMSNorm(hidden_size)
        self.temporal_cluster = NodeLinkSpikeCluster(
            hidden_size,
            hidden_size,
            recurrent=True,
            surrogate_width=surrogate_width,
        )
        self.spike_projection = nn.Linear(hidden_size, hidden_size, bias=False)

        self.channel_norm = nn.RMSNorm(hidden_size)
        self.channel_in = nn.Linear(hidden_size, expanded)
        self.channel_gate = nn.Linear(hidden_size, expanded)
        self.channel_out = nn.Linear(expanded, hidden_size)

    def initial_state(self, batch_size: int, reference: Tensor) -> ClusterState:
        return self.temporal_cluster.initial_state(
            batch_size,
            device=reference.device,
            dtype=reference.dtype,
        )

    def step(self, x: Tensor, state: ClusterState) -> tuple[Tensor, ClusterState]:
        normalized = self.pre_norm(x)
        temporal_spikes, state = self.temporal_cluster.step(normalized, state)
        x = x + self.spike_projection(temporal_spikes)

        channel_input = self.channel_norm(x)
        channel = F.silu(self.channel_in(channel_input))
        gate = torch.sigmoid(self.channel_gate(channel_input))
        x = x + self.channel_out(channel * gate)
        return x, state


@dataclass(slots=True, frozen=True)
class SpikingModelConfig:
    vocab_size: int = 260
    hidden_size: int = 256
    layer_count: int = 4
    expansion: int = 2
    ticks_per_token: int = 4
    max_action_candidates: int = 32
    state_feature_count: int = 16
    dropout: float = 0.05
    surrogate_width: float = 1.0
    use_dedicated_drive_neurons: bool = True
    drive_neuron_levels: tuple[tuple[str, float], ...] = (
        ("notice", 0.35),
        ("need", 0.60),
        ("urgent", 0.85),
    )


@dataclass(slots=True)
class SpikingOutput:
    token_logits: Tensor
    action_logits: Tensor
    value: Tensor
    state: ModelState
    hidden: Tensor
    spike_rates: Tensor
    drive_spikes: Tensor


class NodeLinkSpikeModel(nn.Module):
    """
    Recurrent language-and-action model.

    Inputs:
      token_ids:      [batch, sequence]
      state_features: [batch, sequence, feature_count] or [batch, feature_count]

    Outputs:
      next-token logits for language training
      candidate-action logits for autonomous behavior
      scalar value estimate for reward learning
    """

    def __init__(self, config: SpikingModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.state_projection = nn.Linear(
            config.state_feature_count,
            config.hidden_size,
            bias=False,
        )
        self.drive_neurons = (
            DedicatedDriveNeuronBank(
                feature_count=config.state_feature_count,
                levels=config.drive_neuron_levels,
            )
            if config.use_dedicated_drive_neurons
            else None
        )
        self.drive_links = (
            nn.Linear(
                self.drive_neurons.neuron_count,
                config.hidden_size,
                bias=False,
            )
            if self.drive_neurons is not None
            else None
        )
        if self.drive_links is not None:
            nn.init.xavier_uniform_(self.drive_links.weight, gain=0.35)
        self.input_norm = nn.RMSNorm(config.hidden_size)
        self.blocks = nn.ModuleList([
            SpikeBlock(
                config.hidden_size,
                expansion=config.expansion,
                surrogate_width=config.surrogate_width,
            )
            for _ in range(config.layer_count)
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.output_norm = nn.RMSNorm(config.hidden_size)
        self.language_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.action_head = nn.Linear(
            config.hidden_size,
            config.max_action_candidates,
        )
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_size // 2, 1),
        )

        # Weight tying reduces parameters and encourages a shared lexical space.
        self.language_head.weight = self.token_embedding.weight

    def initial_state(self, batch_size: int, reference: Tensor) -> ModelState:
        return ModelState([
            block.initial_state(batch_size, reference)
            for block in self.blocks
        ])

    def forward(
        self,
        token_ids: Tensor,
        state_features: Tensor | None = None,
        state: ModelState | None = None,
    ) -> SpikingOutput:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        batch, sequence = token_ids.shape
        x_tokens = self.token_embedding(token_ids)

        if state_features is None:
            state_features = torch.zeros(
                batch,
                sequence,
                self.config.state_feature_count,
                device=token_ids.device,
                dtype=x_tokens.dtype,
            )
        elif state_features.ndim == 2:
            state_features = state_features[:, None, :].expand(-1, sequence, -1)
        elif state_features.ndim != 3:
            raise ValueError("state_features must be [B,F] or [B,T,F]")

        if state_features.shape[:2] != (batch, sequence):
            raise ValueError("state feature batch/sequence dimensions do not match tokens")
        if state_features.shape[-1] != self.config.state_feature_count:
            raise ValueError(
                f"Expected {self.config.state_feature_count} state features"
            )

        state_features = state_features.to(x_tokens.dtype)
        analog_state = self.state_projection(state_features)
        if self.drive_neurons is not None and self.drive_links is not None:
            drive_spikes = self.drive_neurons(state_features)
            interoceptive_current = self.drive_links(drive_spikes)
        else:
            drive_spikes = torch.zeros(
                batch,
                sequence,
                0,
                device=token_ids.device,
                dtype=x_tokens.dtype,
            )
            interoceptive_current = torch.zeros_like(x_tokens)

        x_tokens = self.input_norm(
            x_tokens + analog_state + interoceptive_current
        )
        state = state or self.initial_state(batch, x_tokens)
        if len(state.clusters) != len(self.blocks):
            raise ValueError("ModelState does not match layer count")

        hidden_steps: list[Tensor] = []
        rates: list[Tensor] = []
        ticks = max(1, self.config.ticks_per_token)

        for token_index in range(sequence):
            x = x_tokens[:, token_index]
            rate_accumulator = torch.zeros(
                batch,
                self.config.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )
            for _ in range(ticks):
                # Splitting token current across ticks produces temporal integration.
                tick_x = x / ticks
                new_states: list[ClusterState] = []
                for layer_index, block in enumerate(self.blocks):
                    tick_x, next_state = block.step(
                        tick_x,
                        state.clusters[layer_index],
                    )
                    new_states.append(next_state)
                    rate_accumulator = rate_accumulator + next_state.spikes
                state = ModelState(new_states)
                x = tick_x
            hidden_steps.append(self.dropout(x))
            rates.append(rate_accumulator / (ticks * len(self.blocks)))

        hidden = torch.stack(hidden_steps, dim=1)
        spike_rates = torch.stack(rates, dim=1)
        normalized = self.output_norm(hidden)
        token_logits = self.language_head(normalized)
        final_hidden = normalized[:, -1]
        action_logits = self.action_head(final_hidden)
        value = self.value_head(final_hidden).squeeze(-1)
        return SpikingOutput(
            token_logits=token_logits,
            action_logits=action_logits,
            value=value,
            state=state,
            hidden=hidden,
            spike_rates=spike_rates,
            drive_spikes=drive_spikes,
        )

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        *,
        state_features: Tensor | None = None,
        max_new_tokens: int = 128,
        eos_token_id: int | None = 2,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> Tensor:
        if token_ids.shape[0] != 1:
            raise ValueError("generate currently supports batch size 1")
        generated = token_ids
        model_state: ModelState | None = None

        # Prime recurrent state with the prompt once.
        output = self.forward(generated, state_features=state_features)
        model_state = output.state

        for _ in range(max_new_tokens):
            logits = output.token_logits[:, -1] / max(1e-4, temperature)
            if top_k > 0:
                values, indices = torch.topk(logits, min(top_k, logits.shape[-1]))
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(1, indices, values)
                logits = filtered
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                break

            next_features = None
            if state_features is not None:
                if state_features.ndim == 2:
                    next_features = state_features
                else:
                    next_features = state_features[:, -1]
            output = self.forward(
                next_token,
                state_features=next_features,
                state=model_state,
            )
            model_state = output.state

        return generated

    def architecture_summary(self) -> dict[str, Any]:
        return {
            "type": "NodeLinkSpikeModel",
            "config": asdict(self.config),
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "node_count": (
                self.config.hidden_size * self.config.layer_count
                + (self.drive_neurons.neuron_count if self.drive_neurons else 0)
            ),
            "hidden_node_count": self.config.hidden_size * self.config.layer_count,
            "dedicated_drive_neuron_count": (
                self.drive_neurons.neuron_count if self.drive_neurons else 0
            ),
            "dedicated_drive_neurons": (
                list(self.drive_neurons.neuron_names)
                if self.drive_neurons is not None
                else []
            ),
            "recurrent_link_count": (
                self.config.hidden_size
                * self.config.hidden_size
                * self.config.layer_count
            ),
        }

    def drive_activity_report(
        self,
        drive_spikes: Tensor,
        *,
        minimum_rate: float = 0.0,
    ) -> dict[str, float]:
        if self.drive_neurons is None:
            return {}
        return self.drive_neurons.activity_report(
            drive_spikes,
            minimum_rate=minimum_rate,
        )
