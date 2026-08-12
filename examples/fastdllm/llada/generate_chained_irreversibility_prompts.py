"""
Generate synthetic chained workflow prompts with request-varying irreversibility.

Run from repo root:
  python3 examples/fastdllm/llada/generate_chained_irreversibility_prompts.py \
    --output_path examples/fastdllm/llada/chained_irreversibility_60.jsonl \
    --num_records 60 \
    --seed 42

Each record uses the same three-stage workflow:
  0. parse city/date/priority
  1. lookup airport/SLA from tables
  2. finalize a dispatch code

The `hard_stage` field controls where generated fields become irreversible in
the chained run. Stages before that point have oracle fallback in downstream
prompts; the hard stage does not. Each base request is emitted three times,
once per hard stage, so hard-stage effects are not confounded with request
difficulty.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


CITY_TO_AIRPORT = {
    "Seoul": "ICN",
    "Tokyo": "NRT",
    "Paris": "CDG",
    "Austin": "AUS",
    "London": "LHR",
    "Amsterdam": "AMS",
    "Madrid": "MAD",
    "Rome": "FCO",
    "Berlin": "BER",
    "Dubai": "DXB",
    "Boston": "BOS",
    "Toronto": "YYZ",
}
PRIORITY_TO_SLA = {
    "urgent": "S1",
    "standard": "S2",
    "economy": "S3",
}
REQUEST_TEMPLATES = [
    "Create a {priority} travel dispatch for {city} on {date}.",
    "Set up {priority} routing for the {city} visit dated {date}.",
    "Prepare the {date} {city} itinerary with {priority} handling.",
    "Open a {priority} trip ticket to {city} for {date}.",
]


def _date(index: int) -> str:
    year = 2026 + (index % 2)
    month = 1 + ((index * 5) % 12)
    day = 1 + ((index * 7) % 24)
    return f"{year:04d}-{month:02d}-{day:02d}"


def _base_record(base_index: int, rng: random.Random) -> dict[str, object]:
    cities = list(CITY_TO_AIRPORT)
    priorities = list(PRIORITY_TO_SLA)
    city = cities[base_index % len(cities)]
    priority = priorities[
        (base_index + rng.randrange(len(priorities))) % len(priorities)
    ]
    date = _date(base_index)
    airport = CITY_TO_AIRPORT[city]
    sla = PRIORITY_TO_SLA[priority]
    dispatch = f"{airport}-{date.replace('-', '')}-{sla}"
    request = REQUEST_TEMPLATES[base_index % len(REQUEST_TEMPLATES)].format(
        city=city,
        date=date,
        priority=priority,
    )
    return {
        "base_request_id": base_index,
        "request": request,
        "targets": {
            "city": city,
            "date": date,
            "priority": priority,
            "airport": airport,
            "sla": sla,
            "dispatch": dispatch,
        },
    }


def _record(
    base: dict[str, object],
    request_id: int,
    hard_stage: int,
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "base_request_id": base["base_request_id"],
        "hard_stage": hard_stage,
        "request": base["request"],
        "targets": base["targets"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_records", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    if args.num_records % 3 != 0:
        raise ValueError("--num_records must be divisible by 3 for paired hard stages")
    rows = []
    for base_index in range(args.num_records // 3):
        base = _base_record(base_index, rng)
        for hard_stage in range(3):
            rows.append(
                _record(
                    base=base,
                    request_id=len(rows),
                    hard_stage=hard_stage,
                )
            )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"Wrote {len(rows)} records to {output_path}")


if __name__ == "__main__":
    main()
