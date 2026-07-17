"""Run the fixed evaluation suite (eval_suite.json) against one checkpoint.

Usage: python mind_training/run_eval_suite.py <checkpoint.pt> [--stage N] [--out report.json]

Produces one consolidated JSON report instead of many separate CLI calls, so
every checkpoint can be compared on the identical fixed suite.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from virtual_person.spike_tokenizer import ByteTokenizer
from virtual_person.spike_training import load_checkpoint

DRIVE_DEFAULTS = {
    "hunger": 0.20,
    "thirst": 0.20,
    "fatigue": 0.20,
    "bladder": 0.10,
    "hygiene_discomfort": 0.05,
    "health_distress": 0.0,
    "social_need": 0.10,
    "boredom": 0.20,
    "curiosity": 0.45,
    "loneliness": 0.10,
    "competence_frustration": 0.10,
    "enjoyment": 0.50,
    "task_pending": False,
}


def state_features(drives: dict) -> list[float]:
    merged = dict(DRIVE_DEFAULTS)
    merged.update(drives)
    return [
        float(merged["hunger"]),
        float(merged["thirst"]),
        float(merged["fatigue"]),
        float(merged["bladder"]),
        float(merged["hygiene_discomfort"]),
        float(merged["health_distress"]),
        float(merged["social_need"]),
        float(merged["boredom"]),
        float(merged["curiosity"]),
        float(merged["loneliness"]),
        float(merged["competence_frustration"]),
        float(merged["enjoyment"]),
        0.5,
        0.5,
        1.0 if bool(merged["task_pending"]) else 0.0,
        1.0,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--suite", default=str(Path(__file__).resolve().parent / "eval_suite.json"))
    parser.add_argument("--out")
    parser.add_argument("--tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    model, payload = load_checkpoint(checkpoint_path, device=args.device)
    model.eval()
    tokenizer = ByteTokenizer()

    results = []
    with torch.no_grad():
        for case in suite["cases"]:
            features = torch.tensor([state_features(case.get("drives", {}))], dtype=torch.float32, device=args.device)
            ids = tokenizer.encode(case["prompt"], add_eos=False)
            tokens = torch.tensor([ids], dtype=torch.long, device=args.device)
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
            continuation = full_text[len(tokenizer.decode(ids)):]
            drive_report = model.drive_activity_report(inspection.drive_spikes, minimum_rate=1e-9)
            results.append({
                "id": case["id"],
                "category": case["category"],
                "prompt": case["prompt"],
                "drives": case.get("drives", {}),
                "continuation": continuation,
                "active_drive_neurons": drive_report,
                "value_estimate": float(inspection.value[0].item()),
                "spike_rate": float(inspection.spike_rates.mean().item()),
            })

    report = {
        "checkpoint": str(checkpoint_path),
        "step": payload.get("step"),
        "results": results,
    }
    out_path = Path(args.out) if args.out else checkpoint_path.with_suffix(".eval_suite.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} case results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
