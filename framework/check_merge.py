"""
Merge relevance-check and factual-check outputs.

This stage does not call LLMs or KG models.  It only combines existing boolean
and numeric fields into final pass/fail fields that downstream QA or filtering
steps can consume.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import read_jsonl, write_jsonl


def parse_bool(value: Any) -> bool:
    """Parse common boolean representations used in JSONL outputs."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1"}
    return False


def parse_optional_float(value: Any) -> Optional[float]:
    """Return float(value), or None when the score is missing/invalid."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def merge_checks(
    line: dict,
    relevance_key: str = "local_check",
    factual_key: str = "factual_score",
    threshold: float = 10000.0,
    factual_output_key: str = "factual_check",
    output_key: str = "final_check",
    missing_factual_pass: bool = False,
) -> dict:
    """
    Combine relevance and factual checks.

    Relevance passes when ``line[relevance_key]`` is truthy.  Factual check
    passes when ``line[factual_key] <= threshold``; lower factual scores are
    better in this project.  If the factual score is missing, the default is to
    fail unless ``missing_factual_pass`` is set.
    """
    relevance_pass = parse_bool(line.get(relevance_key))
    factual_score = parse_optional_float(line.get(factual_key))
    factual_pass = missing_factual_pass if factual_score is None else factual_score <= threshold

    line[factual_output_key] = factual_pass
    line[output_key] = relevance_pass and factual_pass
    return line


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Know3RAG check merge stage")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--relevance-key", default="local_check")
    parser.add_argument("--factual-key", default="factual_score")
    parser.add_argument("--threshold", type=float, default=10000.0)
    parser.add_argument("--factual-output-key", default="factual_check")
    parser.add_argument("--output-key", default="final_check")
    parser.add_argument(
        "--missing-factual-pass",
        action="store_true",
        help="Treat missing factual scores as passing instead of failing",
    )
    parser.add_argument("--test", action="store_true", help="Process first 5 lines only")
    args = parser.parse_args()

    data = list(read_jsonl(args.input))
    if args.test:
        data = data[:5]

    results = []
    for i, line in enumerate(data):
        print(f"[check_merge] {i + 1}/{len(data)}", end="\r")
        results.append(
            merge_checks(
                line,
                relevance_key=args.relevance_key,
                factual_key=args.factual_key,
                threshold=args.threshold,
                factual_output_key=args.factual_output_key,
                output_key=args.output_key,
                missing_factual_pass=args.missing_factual_pass,
            )
        )

    print()
    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()