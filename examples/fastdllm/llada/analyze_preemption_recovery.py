"""Analyze Fast LLaDA recovery timings and generate the decision-space report.

Run after ``benchmark_preemption_recovery.py``:

    python examples/fastdllm/llada/analyze_preemption_recovery.py \
      --input_dir artifacts/preemption_state/recovery_killtest
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any


ACTIVE_STRATEGIES = (
    "full_offload_pinned",
    "semantic_only",
    "drop_restart",
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _group(
    rows: list[dict[str, str]], keys: tuple[str, ...], value: str
) -> dict[tuple[str, ...], list[float]]:
    grouped: dict[tuple[str, ...], list[float]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(float(row[value]))
    return grouped


def _point_summaries(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    keys = ("request_id", "cache_mode", "block", "inner_step")
    points: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        points.setdefault(tuple(row[key] for key in keys), []).append(row)
    summaries = []
    for point_rows in points.values():
        first = point_rows[0]
        costs = {
            strategy: _median(
                [
                    float(row["total_recovery_ms"])
                    for row in point_rows
                    if row["strategy"] == strategy
                ]
            )
            for strategy in ACTIVE_STRATEGIES
        }
        ordered = sorted(costs.items(), key=lambda item: item[1])
        iteration_ms = float(first["median_iteration_ms"])
        margin_ms = ordered[1][1] - ordered[0][1]
        summaries.append(
            {
                "request_id": first["request_id"],
                "cache_mode": first["cache_mode"],
                "prompt_length": int(first["prompt_length"]),
                "generation_length": int(first["generation_length"]),
                "block": int(first["block"]),
                "inner_step": int(first["inner_step"]),
                "progress": float(first["progress"]),
                "cache_bytes": int(first["cache_bytes"]),
                "semantic_bytes": int(first["semantic_bytes"]),
                "iteration_ms": iteration_ms,
                "costs": costs,
                "winner": ordered[0][0],
                "margin_ms": margin_ms,
                "meaningful": margin_ms >= 0.5 * iteration_ms,
            }
        )
    return summaries


def _plot(input_dir: Path, points: list[dict[str, Any]], boundaries: list[dict[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        print(f"plotting skipped: {error}")
        return

    labels = {
        "full_offload_pinned": "Full offload",
        "semantic_only": "Semantic-only",
        "drop_restart": "Drop/restart",
    }
    colors = {
        "full_offload_pinned": "#0B6E75",
        "semantic_only": "#D9822B",
        "drop_restart": "#8C3B4A",
    }

    for cache_mode in ("prefix", "dual"):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        cache_points = sorted(
            [point for point in points if point["cache_mode"] == cache_mode],
            key=lambda point: (point["request_id"], point["progress"]),
        )
        for request_id in sorted({point["request_id"] for point in cache_points}):
            request_points = [
                point for point in cache_points if point["request_id"] == request_id
            ]
            for strategy in ACTIVE_STRATEGIES:
                ax.plot(
                    [point["progress"] for point in request_points],
                    [point["costs"][strategy] for point in request_points],
                    marker="o",
                    linewidth=1.5,
                    color=colors[strategy],
                    alpha=0.85,
                    label=f"{labels[strategy]} ({request_id})",
                )
        ax.set_xlabel("Generation progress")
        ax.set_ylabel("Median recovery overhead (ms)")
        ax.set_title(f"{cache_mode.title()} cache recovery cost")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(input_dir / f"recovery_cost_{cache_mode}.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for cache_mode, marker in (("prefix", "o"), ("dual", "s")):
        cache_points = sorted(
            [point for point in points if point["cache_mode"] == cache_mode],
            key=lambda point: point["progress"],
        )
        ax.scatter(
            [point["progress"] for point in cache_points],
            [point["cache_bytes"] / 1_000_000 for point in cache_points],
            label=cache_mode,
            marker=marker,
        )
    ax.set_xlabel("Generation progress")
    ax.set_ylabel("Cache state (MB)")
    ax.set_title("Cache size across preemption points")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(input_dir / "cache_size_vs_progress.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for strategy in ACTIVE_STRATEGIES:
        ax.scatter(
            [point["cache_bytes"] / 1_000_000 for point in points],
            [point["costs"][strategy] for point in points],
            label=labels[strategy],
            color=colors[strategy],
            alpha=0.75,
        )
    ax.set_xlabel("Cache state (MB)")
    ax.set_ylabel("Median recovery overhead (ms)")
    ax.set_title("Cache size versus recovery cost")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(input_dir / "cache_size_vs_recovery_cost.pdf")
    plt.close(fig)

    if boundaries:
        boundary_groups = _group(
            boundaries,
            ("request_id", "cache_mode", "block", "inner_step"),
            "total_boundary_cost_ms",
        )
        immediate_groups = _group(
            boundaries,
            ("request_id", "cache_mode", "block", "inner_step"),
            "immediate_recovery_ms",
        )
        keys = sorted(boundary_groups)
        x = list(range(len(keys)))
        fig, ax = plt.subplots(figsize=(max(7.2, len(keys) * 0.45), 4.4))
        ax.plot(x, [_median(immediate_groups[key]) for key in keys], "o-", label="Immediate")
        ax.plot(x, [_median(boundary_groups[key]) for key in keys], "s-", label="Wait for boundary")
        ax.set_xlabel("Inner-step checkpoint")
        ax.set_ylabel("Median additional cost (ms)")
        ax.set_title("Immediate preemption versus boundary deferral")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(input_dir / "boundary_vs_immediate.pdf")
        plt.close(fig)

    winner_index = {strategy: index for index, strategy in enumerate(ACTIVE_STRATEGIES)}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for cache_mode, marker in (("prefix", "o"), ("dual", "s")):
        cache_points = [point for point in points if point["cache_mode"] == cache_mode]
        ax.scatter(
            [point["progress"] for point in cache_points],
            [winner_index[point["winner"]] for point in cache_points],
            marker=marker,
            label=cache_mode,
            s=55,
        )
    ax.set_yticks(list(winner_index.values()), [labels[key] for key in winner_index])
    ax.set_xlabel("Generation progress")
    ax.set_ylabel("Measured winner")
    ax.set_title("Recovery strategy regime map")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(input_dir / "strategy_regime_map.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="artifacts/preemption_state/recovery_killtest",
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    rows = _rows(input_dir / "recovery_costs.csv")
    boundaries = (
        _rows(input_dir / "boundary_preemption.csv")
        if (input_dir / "boundary_preemption.csv").exists()
        else []
    )
    points = _point_summaries(rows)
    _plot(input_dir, points, boundaries)

    meaningful_points = [point for point in points if point["meaningful"]]
    meaningful_winners = {point["winner"] for point in meaningful_points}
    request_shapes = {
        (point["prompt_length"], point["generation_length"]) for point in points
    }
    winners_by_cache = {
        cache: sorted(
            {
                point["winner"]
                for point in meaningful_points
                if point["cache_mode"] == cache
            }
        )
        for cache in ("prefix", "dual")
    }

    boundary_point_groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in boundaries:
        key = (row["request_id"], row["cache_mode"], row["block"], row["inner_step"])
        boundary_point_groups.setdefault(key, []).append(row)
    boundary_winners = set()
    meaningful_boundary_winners = set()
    for group in boundary_point_groups.values():
        immediate = _median([float(row["immediate_recovery_ms"]) for row in group])
        boundary = _median([float(row["total_boundary_cost_ms"]) for row in group])
        winner = "immediate" if immediate <= boundary else "boundary"
        boundary_winners.add(winner)
        iteration = _median([float(row["median_iteration_ms"]) for row in group])
        if abs(immediate - boundary) >= 0.5 * iteration:
            meaningful_boundary_winners.add(winner)

    all_one_winner = len({point["winner"] for point in points}) == 1
    if (
        len(request_shapes) >= 2
        and (
            len(meaningful_winners) >= 2
            or len(meaningful_boundary_winners) >= 2
        )
    ):
        judgment = "STRONG GO"
    elif len(request_shapes) >= 3 and all_one_winner and boundary_winners <= {"immediate"}:
        judgment = "NO-GO"
    else:
        judgment = "CONDITIONAL GO"

    full_rows = [row for row in rows if row["strategy"] == "full_offload_pinned"]
    semantic_rows = [row for row in rows if row["strategy"] == "semantic_only"]
    restart_rows = [row for row in rows if row["strategy"] == "drop_restart"]
    strategy_groups = {
        strategy: _group(
            [row for row in rows if row["strategy"] == strategy],
            ("request_id", "cache_mode", "block", "inner_step"),
            "total_recovery_ms",
        )
        for strategy in ACTIVE_STRATEGIES
    }
    strategy_medians = {
        strategy: [_median(values) for values in groups.values()]
        for strategy, groups in strategy_groups.items()
    }
    strategy_p95s = {
        strategy: [_percentile(values, 0.95) for values in groups.values()]
        for strategy, groups in strategy_groups.items()
    }
    rebuild_groups = _group(
        semantic_rows,
        ("request_id", "cache_mode", "block", "inner_step"),
        "rebuild_ms",
    )
    rebuild_medians = [_median(values) for values in rebuild_groups.values()]
    raw_winners = {point["winner"] for point in points}
    nonmeaningful_winners = raw_winners - meaningful_winners

    full_medians = strategy_medians["full_offload_pinned"]
    full_p95s = strategy_p95s["full_offload_pinned"]
    semantic_medians = strategy_medians["semantic_only"]
    semantic_p95s = strategy_p95s["semantic_only"]
    restart_medians = strategy_medians["drop_restart"]
    restart_p95s = strategy_p95s["drop_restart"]

    lines = [
        "# dLLM Preemption Recovery Decision Space",
        "",
        "## CONFIRMED",
        "",
        f"- Matched recovery points: {len(points)} across {len(request_shapes)} request shape(s).",
        f"- Full-offload point-median recovery range: {min(full_medians):.2f}–{max(full_medians):.2f} ms; point p95 range: {min(full_p95s):.2f}–{max(full_p95s):.2f} ms.",
        f"- Semantic-only point-median total recovery range: {min(semantic_medians):.2f}–{max(semantic_medians):.2f} ms; cache-rebuild component: {min(rebuild_medians):.2f}–{max(rebuild_medians):.2f} ms.",
        f"- Semantic-only point p95 range: {min(semantic_p95s):.2f}–{max(semantic_p95s):.2f} ms.",
        f"- Drop/restart point-median recomputation range: {min(restart_medians):.2f}–{max(restart_medians):.2f} ms; point p95 range: {min(restart_p95s):.2f}–{max(restart_p95s):.2f} ms.",
        f"- Meaningful measured strategy winners: {', '.join(sorted(meaningful_winners)) or 'none'}.",
        f"- Prefix winners: {', '.join(winners_by_cache['prefix']) or 'none'}.",
        f"- Dual winners: {', '.join(winners_by_cache['dual']) or 'none'}.",
        f"- Boundary decisions observed: {', '.join(sorted(boundary_winners)) or 'none'}.",
        "",
        "## NEGATIVE RESULTS",
        "",
    ]
    if all_one_winner:
        lines.append(f"- One strategy won every sampled point: {points[0]['winner']}.")
    else:
        lines.append("- No single strategy won every sampled point.")
    if nonmeaningful_winners:
        lines.append(
            "- Non-meaningful raw winners (margin below half an iteration): "
            + ", ".join(sorted(nonmeaningful_winners))
            + "."
        )
    if boundary_winners <= {"immediate"}:
        lines.append("- Waiting for a block boundary never beat immediate recovery.")
    elif boundary_winners <= {"boundary"}:
        lines.append("- Waiting for a block boundary beat immediate recovery at every sampled inner step.")
    else:
        lines.append("- Boundary deferral was not uniformly beneficial or harmful.")
    lines.extend(
        [
            "",
            "## REMAINING UNKNOWNS",
            "",
            "- KEEP opportunity cost under real multi-request memory contention.",
            "- Generalization beyond the tested model and request shapes.",
            "",
            "## Explicit Answers",
            "",
            f"- Q1: Full-state offload point medians span {min(full_medians):.2f}–{max(full_medians):.2f} ms over measured cache sizes.",
            f"- Q2: Semantic-only total recovery point medians span {min(semantic_medians):.2f}–{max(semantic_medians):.2f} ms; measured cache-rebuild work spans {min(rebuild_medians):.2f}–{max(rebuild_medians):.2f} ms.",
            f"- Q3: Lost-progress recomputation point medians span {min(restart_medians):.2f}–{max(restart_medians):.2f} ms.",
            f"- Q4: Raw winners across progress are {', '.join(sorted(raw_winners))}; meaningful winners are {', '.join(sorted(meaningful_winners)) or 'none'}.",
            f"- Q5: Prefix winners are {winners_by_cache['prefix']}; dual winners are {winners_by_cache['dual']}.",
            f"- Q6: Boundary outcomes are {sorted(boundary_winners)}.",
            f"- Q7: Measurements cover {len(request_shapes)} request shape(s): {sorted(request_shapes)}.",
            f"- Q8: Overall measured decision-space judgment is {judgment}.",
            "",
            judgment,
        ]
    )
    (input_dir / "decision_space_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(input_dir / "decision_space_report.md")


if __name__ == "__main__":
    main()
