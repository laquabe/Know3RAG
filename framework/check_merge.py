"""
Merge relevance-check and factual-check outputs at the directory level.

This stage does not call LLMs or KG models.  It scans two directories:
one with relevance-check JSONL files, one with factual-check JSONL files.
Records are aligned by ``(id, normalized_passages)`` so that the relevance
boolean and the factual score for the same passage can be combined.

For each query id, passages with ``local_check == False`` are dropped.  The
remaining passages are sorted by factual score (lower is better; ``None``
goes last but may still fill up to ``top_k``).  The top ``top_k`` passages
are emitted as ``reference`` alongside the query id.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def normalize_passage(passage: Any) -> str:
    """Lightweight whitespace normalization for passage alignment."""
    if passage is None:
        return ""
    return " ".join(str(passage).split())


def list_jsonl_files(directory: str) -> List[str]:
    """Non-recursively list regular files in ``directory``."""
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")
    paths = []
    for name in sorted(os.listdir(directory)):
        full = os.path.join(directory, name)
        if os.path.isfile(full):
            paths.append(full)
    return paths


def iter_records(directory: str) -> Iterable[Dict[str, Any]]:
    """Yield JSONL records from every file in ``directory`` (non-recursive)."""
    for path in list_jsonl_files(directory):
        try:
            for rec in read_jsonl(path):
                yield rec
        except json.JSONDecodeError as e:
            print(f"[check_merge] skip {path}: {e}", file=sys.stderr)


def first_record(directory: str) -> Optional[Dict[str, Any]]:
    """Return the first JSONL record found in ``directory``, or None."""
    for rec in iter_records(directory):
        return rec
    return None


def compute_carry_over_keys(
    rel_sample: Optional[Dict[str, Any]],
    factual_sample: Optional[Dict[str, Any]],
    passage_key: str,
) -> List[str]:
    """Keys present in BOTH directories' records, minus the per-passage key.

    Order follows the rel-check record so the output column order is stable.
    """
    if not rel_sample or not factual_sample:
        return []
    factual_keys = set(factual_sample.keys())
    return [k for k in rel_sample.keys() if k in factual_keys and k != passage_key]


def build_factual_lookup(
    directory: str,
    id_key: str,
    passage_key: str,
    factual_key: str,
) -> Dict[Tuple[str, str], Optional[float]]:
    """Map ``(id, normalized_passage) -> factual_score`` from factual-check files."""
    lookup: Dict[Tuple[str, str], Optional[float]] = {}
    for rec in iter_records(directory):
        rid = rec.get(id_key)
        if rid is None:
            continue
        key = (str(rid), normalize_passage(rec.get(passage_key)))
        lookup[key] = parse_optional_float(rec.get(factual_key))
    return lookup


def merge_directories(
    rel_check_dir: str,
    factual_check_dir: str,
    top_k: int,
    id_key: str,
    passage_key: str,
    relevance_key: str,
    factual_key: str,
) -> List[Dict[str, Any]]:
    """Merge relevance + factual checks and pick top-k per query id."""
    factual_lookup = build_factual_lookup(
        factual_check_dir, id_key, passage_key, factual_key
    )

    carry_over_keys = compute_carry_over_keys(
        first_record(rel_check_dir),
        first_record(factual_check_dir),
        passage_key,
    )

    # Group candidates by query id, preserving first-seen order of ids.
    grouped: Dict[str, List[Tuple[Optional[float], str]]] = {}
    carry_over: Dict[str, Dict[str, Any]] = {}
    id_order: List[str] = []

    for rec in iter_records(rel_check_dir):
        rid = rec.get(id_key)
        if rid is None:
            continue
        rid = str(rid)
        if not parse_bool(rec.get(relevance_key)):
            continue
        passage = rec.get(passage_key)
        if passage is None:
            continue
        key = (rid, normalize_passage(passage))
        score = factual_lookup.get(key)
        if rid not in grouped:
            grouped[rid] = []
            id_order.append(rid)
            carry_over[rid] = {k: rec.get(k) for k in carry_over_keys}
        grouped[rid].append((score, str(passage)))

    results: List[Dict[str, Any]] = []
    for rid in id_order:
        candidates = grouped[rid]
        # Sort by score ascending; None goes last but is still eligible for top-k.
        candidates.sort(
            key=lambda sp: (sp[0] is None, sp[0] if sp[0] is not None else 0.0)
        )
        reference = [passage for _, passage in candidates[:top_k]]
        record: Dict[str, Any] = dict(carry_over[rid])
        # Ensure id is present even when it wasn't in the carry-over intersection.
        record.setdefault(id_key, rid)
        record["reference"] = reference
        results.append(record)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Know3RAG check-merge stage (directory mode)"
    )
    parser.add_argument("--rel-check-dir", required=True,
                        help="Directory of relevance-check JSONL files")
    parser.add_argument("--factual-check-dir", required=True,
                        help="Directory of factual-check JSONL files")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path (one record per query id)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--id-key", default="id")
    parser.add_argument("--passage-key", default="passages")
    parser.add_argument("--relevance-key", default="local_check")
    parser.add_argument("--factual-key", default="factual_score",
                        help="triple mode: factual_score; fast mode: fast_factual_score")
    args = parser.parse_args()

    results = merge_directories(
        rel_check_dir=args.rel_check_dir,
        factual_check_dir=args.factual_check_dir,
        top_k=args.top_k,
        id_key=args.id_key,
        passage_key=args.passage_key,
        relevance_key=args.relevance_key,
        factual_key=args.factual_key,
    )

    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
