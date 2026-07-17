"""Dump a simulation snapshot to JSON for the 2D top-down renderer.

Usage: python examples/dump_snapshot.py --seed 3 --hours 2 --out snapshot.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from virtual_person import VirtualPersonSimulation

ROOM_LAYOUT = {
    "bedroom": {"x": 0, "y": 0, "w": 4, "h": 4},
    "bathroom": {"x": 4, "y": 0, "w": 3, "h": 4},
    "hallway": {"x": 0, "y": 4, "w": 7, "h": 2},
    "kitchen": {"x": 7, "y": 0, "w": 4, "h": 6},
    "living_room": {"x": 0, "y": 6, "w": 11, "h": 4},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--hours", type=float, default=0.0)
    parser.add_argument("--out", default="snapshot.json")
    args = parser.parse_args()

    with VirtualPersonSimulation.default(seed=args.seed) as sim:
        if args.hours > 0:
            sim.run(hours=args.hours)
        snap = sim.snapshot()

    all_objects = []
    for obj_id, obj in sim.agent.world.objects.items():
        all_objects.append({
            "id": obj.object_id,
            "kind": obj.kind,
            "room": obj.room,
            "portable": obj.portable,
            "openable": obj.openable,
            "is_open": obj.is_open,
            "container": obj.container,
        })

    payload = {
        "layout": ROOM_LAYOUT,
        "rooms": {name: sorted(room.neighbors) for name, room in sim.agent.world.rooms.items()},
        "objects": all_objects,
        "snapshot": snap,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote snapshot to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
