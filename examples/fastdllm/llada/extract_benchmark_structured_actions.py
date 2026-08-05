"""
Extract compact structured tool calls from public agent benchmark traces.

Run from repo root:
  python examples/fastdllm/llada/extract_benchmark_structured_actions.py \
    --input_path /path/to/tau_or_toolbench_records.jsonl \
    --output_path artifacts/action_completeness/benchmark_structured.jsonl \
    --max_records 100

Supported inputs:
  - OpenAI tool-calling wire format, as used by tau-bench-derived SFT data:
    {"messages": [..., {"tool_calls": [{"function": {"name": ..., "arguments": "..."}}]}], "tools": [...]}
  - Generic JSON/JSONL records with prompt/instruction/messages plus tool/tool_name/name
    and args/arguments/parameters fields.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SCALAR_TYPES = (str, int, float, bool)


def _iter_input_records(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        records = []
        for child in sorted(path.rglob("*")):
            if child.suffix.lower() in {".json", ".jsonl"}:
                records.extend(_iter_input_records(child))
        return records
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            for key in ("data", "records", "examples", "instances"):
                if isinstance(data.get(key), list):
                    return [row for row in data[key] if isinstance(row, dict)]
            return [data]
    raise ValueError(f"Unsupported input path: {path}")


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"query": stripped}
    return {}


def _flatten_args(args: dict[str, Any]) -> dict[str, str]:
    flattened = {}
    for key, value in args.items():
        if value is None:
            continue
        if isinstance(value, SCALAR_TYPES):
            flattened[str(key)] = str(value)
        elif isinstance(value, list) and all(isinstance(item, SCALAR_TYPES) for item in value):
            flattened[str(key)] = " ".join(str(item) for item in value)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, SCALAR_TYPES):
                    flattened[f"{key}.{sub_key}"] = str(sub_value)
    return flattened


def _arg_token_count(args: dict[str, str]) -> int:
    text = " ".join(f"{key} {value}" for key, value in args.items())
    return len(re.findall(r"[A-Za-z0-9_./:-]+", text))


def _is_compact_structured(args: dict[str, str], min_fields: int, max_fields: int, max_arg_tokens: int) -> bool:
    if not (min_fields <= len(args) <= max_fields):
        return False
    if _arg_token_count(args) > max_arg_tokens:
        return False
    query_like = {"query", "q", "search", "text", "prompt"}
    if len(args) == 1 and next(iter(args)).lower() in query_like:
        return False
    return True


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", part))
            for part in content
            if isinstance(part, (str, dict))
        )
    if content is None:
        return ""
    return str(content)


def _tool_call_text(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list):
        return ""
    parts = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function", {})
        name = function.get("name") or tool_call.get("name")
        arguments = function.get("arguments") or tool_call.get("arguments")
        if name:
            parts.append(f"{name}({arguments})")
    return "; ".join(parts)


def _conversation_context(
    messages: list[dict[str, Any]],
    before_index: int,
    *,
    context_turns: int,
    max_context_chars: int,
) -> str:
    selected = messages[max(0, before_index - context_turns) : before_index]
    lines = []
    for message in selected:
        role = message.get("role", "unknown")
        if role == "system":
            continue
        content = _content_to_text(message.get("content"))
        if role == "assistant" and message.get("tool_calls"):
            call_text = _tool_call_text(message.get("tool_calls"))
            content = f"{content} TOOL_CALLS: {call_text}".strip()
        if role == "tool":
            content = f"TOOL_RESULT: {content}"
        content = re.sub(r"\s+", " ", content).strip()
        if content:
            lines.append(f"{role.upper()}: {content}")
    text = "\n".join(lines)
    if len(text) > max_context_chars:
        text = text[-max_context_chars:]
    return text


def _tool_signatures(record: dict[str, Any]) -> list[str]:
    signatures = []
    tools = record.get("tools")
    if not isinstance(tools, list):
        return signatures
    for tool in tools:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = function.get("name")
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        required = parameters.get("required", []) if isinstance(parameters, dict) else []
        arg_names = list(properties) or list(required)
        if name:
            signature = f"{name}({', '.join(str(arg) for arg in arg_names)})"
            signatures.append(signature)
    return signatures


def _extract_openai_wire(
    record: dict[str, Any],
    *,
    context_turns: int,
    max_context_chars: int,
) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    extracted = []
    available_tools = _tool_signatures(record)
    for index, message in enumerate(messages):
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        prompt = _conversation_context(
            messages,
            index,
            context_turns=context_turns,
            max_context_chars=max_context_chars,
        )
        for tool_call in tool_calls:
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            name = function.get("name") or tool_call.get("name")
            args = _parse_arguments(function.get("arguments") or tool_call.get("arguments"))
            if name:
                extracted.append(
                    {
                        "prompt": prompt,
                        "tool": str(name),
                        "args": _flatten_args(args),
                        "available_tools": available_tools,
                        "source": "openai_wire",
                    }
                )
    return extracted


def _extract_generic(record: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = (
        record.get("prompt")
        or record.get("instruction")
        or record.get("query")
        or record.get("question")
        or record.get("input")
        or ""
    )
    name = (
        record.get("tool")
        or record.get("tool_name")
        or record.get("api_name")
        or record.get("name")
    )
    args = (
        record.get("args")
        or record.get("arguments")
        or record.get("parameters")
        or record.get("params")
    )
    if not prompt or not name or args is None:
        return []
    return [
        {
            "prompt": str(prompt),
            "tool": str(name),
            "args": _flatten_args(_parse_arguments(args)),
            "source": "generic",
        }
    ]


def _format_prompt(record: dict[str, Any], explicit_tool: bool) -> str:
    if explicit_tool:
        args = "; ".join(f"{key}={value}" for key, value in record["args"].items())
        return (
            f"Use the {record['tool']} tool for this request. "
            f"Required argument values include: {args}. "
            f"User request: {record['prompt']}"
        )
    return str(record["prompt"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_records", type=int, default=100)
    parser.add_argument("--min_fields", type=int, default=2)
    parser.add_argument("--max_fields", type=int, default=6)
    parser.add_argument("--max_arg_tokens", type=int, default=32)
    parser.add_argument("--context_turns", type=int, default=10)
    parser.add_argument("--max_context_chars", type=int, default=5000)
    parser.add_argument("--explicit_tool_prompt", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    records = _iter_input_records(input_path)

    output_rows = []
    seen = set()
    for record in records:
        candidates = _extract_openai_wire(
            record,
            context_turns=args.context_turns,
            max_context_chars=args.max_context_chars,
        ) or _extract_generic(record)
        for candidate in candidates:
            compact_args = {
                key: value
                for key, value in candidate["args"].items()
                if value and len(str(value)) <= 120
            }
            if not _is_compact_structured(
                compact_args,
                min_fields=args.min_fields,
                max_fields=args.max_fields,
                max_arg_tokens=args.max_arg_tokens,
            ):
                continue
            key = (
                candidate["prompt"],
                candidate["tool"],
                json.dumps(compact_args, sort_keys=True),
            )
            if key in seen:
                continue
            seen.add(key)
            output_rows.append(
                {
                    "prompt": _format_prompt(
                        {**candidate, "args": compact_args},
                        explicit_tool=args.explicit_tool_prompt,
                    ),
                    "tool": candidate["tool"],
                    "args": compact_args,
                    "available_tools": candidate.get("available_tools", []),
                    "source": candidate["source"],
                }
            )
            if len(output_rows) >= args.max_records:
                break
        if len(output_rows) >= args.max_records:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    print(f"Read records: {len(records)}")
    print(f"Wrote structured actions: {len(output_rows)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
