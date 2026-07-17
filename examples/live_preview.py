"""Open a live 2D top-down preview window for a running simulation.

Usage: python examples/live_preview.py --fps 10 --step-seconds 60
"""
from __future__ import annotations

import argparse

from virtual_person import Renderer, VirtualPersonSimulation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--step-seconds", type=float, default=60.0)
    args = parser.parse_args()

    with VirtualPersonSimulation.default(seed=args.seed) as sim:
        renderer = Renderer(sim, fps=args.fps, step_seconds=args.step_seconds)
        renderer.run()  # blocks until the window is closed
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
