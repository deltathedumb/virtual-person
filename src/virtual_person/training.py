from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .simulation import VirtualPersonSimulation


@dataclass(slots=True)
class DemonstrationStep:
    episode: int
    sim_time: float
    observation: dict[str, Any]
    event: dict[str, Any]


class DemonstrationRecorder:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append_episode(self, episode: int, rows: Iterable[DemonstrationStep]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(asdict(row), separators=(",", ":")) + "\n")


def run_scripted_episode(episode: int, seed: int, hours: float = 18.0) -> list[DemonstrationStep]:
    sim = VirtualPersonSimulation.default(seed=seed)
    # Randomized initial pressures improve data variety while remaining reproducible.
    rng = sim.random
    sim.agent.body.hunger = rng.uniform(0.45, 0.85)
    sim.agent.body.thirst = rng.uniform(0.35, 0.75)
    sim.agent.body.fatigue = rng.uniform(0.15, 0.45)
    sim.agent.body.bladder = rng.uniform(0.25, 0.75)
    sim.agent.body.hygiene = rng.uniform(0.35, 0.95)

    rows: list[DemonstrationStep] = []
    target = sim.sim_time + hours * 3600.0
    while sim.sim_time < target:
        observation = sim.snapshot()
        events = sim.step()
        for event in events:
            rows.append(
                DemonstrationStep(
                    episode=episode,
                    sim_time=event.sim_time,
                    observation=observation,
                    event={
                        "kind": event.kind,
                        "message": event.message,
                        "ok": event.ok,
                        "reward": event.reward,
                    },
                )
            )
    sim.close()
    return rows


def _worker(args: tuple[int, int, float]) -> tuple[int, list[DemonstrationStep]]:
    episode, seed, hours = args
    return episode, run_scripted_episode(episode, seed, hours)


def generate_demonstrations(
    output: str | Path,
    episodes: int,
    workers: int = 1,
    base_seed: int = 0,
    hours: float = 18.0,
) -> int:
    output = Path(output)
    if output.exists():
        output.unlink()
    recorder = DemonstrationRecorder(output)
    tasks = [(episode, base_seed + episode, hours) for episode in range(episodes)]
    total_rows = 0

    if workers <= 1:
        iterator = map(_worker, tasks)
        for episode, rows in iterator:
            recorder.append_episode(episode, rows)
            total_rows += len(rows)
        return total_rows

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for episode, rows in pool.imap_unordered(_worker, tasks, chunksize=1):
            recorder.append_episode(episode, rows)
            total_rows += len(rows)
    return total_rows
