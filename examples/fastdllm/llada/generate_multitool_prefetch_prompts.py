"""
Generate a larger deterministic multi-tool prompt set.

Run from repo root:
  python3 examples/fastdllm/llada/generate_multitool_prefetch_prompts.py \
    --output_path examples/fastdllm/llada/multitool_prefetch_prompts_120.jsonl \
    --num_records 120 \
    --seed 42

The output schema matches test_multitool_prefetch_signals.py:
  {"prompt": "...", "calls": [{"tool": "...", "args": {...}}, ...]}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


CITY_AIRPORTS = [
    ("Seoul", "ICN"),
    ("Tokyo", "NRT"),
    ("Paris", "CDG"),
    ("Austin", "AUS"),
    ("London", "LHR"),
    ("Amsterdam", "AMS"),
    ("Madrid", "MAD"),
    ("Rome", "FCO"),
    ("Berlin", "BER"),
    ("Dubai", "DXB"),
    ("Boston", "BOS"),
    ("Toronto", "YYZ"),
    ("Singapore", "SIN"),
    ("Sydney", "SYD"),
    ("Zurich", "ZRH"),
    ("Chicago", "ORD"),
    ("New York", "JFK"),
    ("San Francisco", "SFO"),
    ("Seattle", "SEA"),
    ("Los Angeles", "LAX"),
]

ORIGINS = ["SFO", "SEA", "BOS", "ORD", "DFW", "MIA", "ATL", "IAD", "LAX", "JFK"]
COMPANIES = [
    "Sony",
    "Dell",
    "HSBC",
    "Telefonica",
    "SAP",
    "Emirates",
    "NVIDIA",
    "Stripe",
    "Toyota",
    "Airbus",
]
ROLES = ["director", "manager", "analyst", "partner", "recruiter", "coordinator", "engineer"]
CALENDARS = ["research", "sales", "product", "ops", "recruiting", "partner"]
KEYWORDS = ["meeting", "workshop", "call", "demo", "maintenance", "interview", "review"]
TASKS = [
    "Prepare for the {city} visit",
    "Plan the {city} trip",
    "Set up the {city} customer meeting",
    "Organize the {city} project stop",
    "Prepare the {city} partner update",
]


def _date(year: int, month: int, day: int) -> str:
    return f"{year:04d}-{month:02d}-{day:02d}"


def _record(index: int, rng: random.Random) -> dict[str, object]:
    city, airport = CITY_AIRPORTS[index % len(CITY_AIRPORTS)]
    origin = ORIGINS[(index * 3 + 1) % len(ORIGINS)]
    if origin == airport:
        origin = ORIGINS[(index * 3 + 2) % len(ORIGINS)]
    year = 2026 + (index % 2)
    month = 1 + ((index * 5) % 12)
    day = 1 + ((index * 7) % 24)
    weather_date = _date(year, month, day)
    flight_date = _date(year, month, max(day - 1, 1))
    end_date = _date(year, month, min(day + 2, 28))
    company = COMPANIES[index % len(COMPANIES)]
    role = ROLES[(index * 2) % len(ROLES)]
    calendar = CALENDARS[(index * 3) % len(CALENDARS)]
    keyword = KEYWORDS[(index * 5) % len(KEYWORDS)]

    candidate_calls = [
        {
            "tool": "weather",
            "args": {"location": city, "date": weather_date},
            "phrase": f"check weather in {city} on {weather_date}",
        },
        {
            "tool": "flight_search",
            "args": {"origin": origin, "destination": airport, "date": flight_date},
            "phrase": f"find flights from {origin} to {airport} on {flight_date}",
        },
        {
            "tool": "calendar_api",
            "args": {
                "calendar": calendar,
                "start_date": weather_date,
                "end_date": end_date,
                "keyword": keyword,
            },
            "phrase": (
                f"list {calendar} calendar events from {weather_date} to "
                f"{end_date} with keyword {keyword}"
            ),
        },
        {
            "tool": "crm_api",
            "args": {"company": company, "role": role, "city": city},
            "phrase": f"fetch CRM contacts at company {company} with role {role} in city {city}",
        },
    ]
    selected = rng.sample(candidate_calls, 3)
    task = TASKS[index % len(TASKS)].format(city=city)
    phrases = [str(call["phrase"]) for call in selected]
    prompt = f"{task}: {phrases[0]}, {phrases[1]}, and {phrases[2]}."
    return {
        "prompt": prompt,
        "calls": [
            {"tool": call["tool"], "args": call["args"]}
            for call in selected
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_records", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [_record(index, rng) for index in range(args.num_records)]
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"Wrote {len(rows)} records to {output_path}")


if __name__ == "__main__":
    main()
