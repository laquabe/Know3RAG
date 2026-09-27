"""Answer-only HotpotQA metrics for Know3RAG JSONL predictions.

Metric semantics: https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
Answer extraction is project-specific; gold answers never enter extraction.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import string
import sys

# Load the existing leaf helper without importing utils/__init__.py and models.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
from data_io import extract_open_answer


def normalize_answer(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def answer_metrics(prediction, gold):
    pred, target = normalize_answer(prediction), normalize_answer(gold)
    metrics = dict(em=float(pred == target), f1=0.0, prec=0.0, recall=0.0)
    special = {'yes', 'no', 'noanswer'}
    if pred != target and (pred in special or target in special):
        return metrics
    pred_tokens, gold_tokens = pred.split(), target.split()
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap:
        precision, recall = overlap / len(pred_tokens), overlap / len(gold_tokens)
        metrics.update(prec=precision, recall=recall,
                       f1=2 * precision * recall / (precision + recall))
    return metrics


def extract_prediction(raw, mode='auto'):
    if not isinstance(raw, str):
        raise ValueError('prediction must be a string')
    if mode == 'raw':
        return raw, 'raw'
    # Support fenced / embedded JSON answers without consulting the gold answer.
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', raw):
        try:
            obj, _ = decoder.raw_decode(raw[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            value = obj.get('answer', obj.get('Answer'))
            if isinstance(value, str):
                return value, 'json_answer'
    answer, clean = extract_open_answer(raw)
    return answer or '', 'prefix' if clean else 'fallback'


def read_records(path):
    with open(path, encoding='utf-8') as stream:
        text = stream.read()
    if text.lstrip().startswith('['):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError('input must contain a non-empty list of records')
    return rows


def index_records(rows, id_key):
    index = {}
    for row in rows:
        rid = row[id_key]
        if not isinstance(rid, (str, int)) or isinstance(rid, bool):
            raise ValueError('record IDs must be strings or integers')
        rid = str(rid)
        if rid in index:
            raise ValueError('duplicate ID: ' + rid)
        index[rid] = row
    return index


def evaluate(rows, answer_key='llm_response', gold_key='answer', id_key='id',
             gold_rows=None, gold_id_key='_id', extraction='auto'):
    predictions = index_records(rows, id_key)
    targets = index_records(gold_rows, gold_id_key) if gold_rows is not None else predictions
    totals = dict(em=0.0, f1=0.0, prec=0.0, recall=0.0)
    details, methods = [], Counter()
    missing = 0
    for rid, target in targets.items():
        gold = target[gold_key]
        if not isinstance(gold, str):
            raise ValueError(f'{rid}: gold answer must be a string (HotpotQA format)')
        row = predictions.get(rid)
        if row is None or answer_key not in row or row[answer_key] is None:
            raw, prediction, method = None, '', 'missing'
            scores = dict.fromkeys(totals, 0.0)
            missing += 1
        else:
            raw = row[answer_key]
            prediction, method = extract_prediction(raw, extraction)
            scores = answer_metrics(prediction, gold)
        methods[method] += 1
        for key in totals:
            totals[key] += scores[key]
        details.append(dict(id=rid, raw_prediction=raw, prediction=prediction,
                            gold=gold, extraction=method, **scores))
    count = len(targets)
    if not count:
        raise ValueError('no gold records to evaluate')
    summary = dict(count=count, missing_predictions=missing,
                   extra_predictions=len(predictions.keys() - targets.keys()),
                   extraction_counts=dict(methods),
                   **{key: value / count for key, value in totals.items()})
    return summary, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Prediction JSONL or JSON array')
    parser.add_argument('--answer-key', default='llm_response')
    parser.add_argument('--id-key', default='id')
    parser.add_argument('--gold-key', default='answer')
    parser.add_argument('--gold-file', help='Optional official gold JSON array; otherwise use input gold field')
    parser.add_argument('--gold-id-key', default='_id')
    parser.add_argument('--extraction', choices=['auto', 'raw'], default='auto')
    parser.add_argument('--output', help='Optional summary JSON path')
    parser.add_argument('--details', help='Optional per-question JSONL path')
    args = parser.parse_args()
    inputs = {Path(p).resolve() for p in (args.input, args.gold_file) if p}
    outputs = [Path(p).resolve() for p in (args.output, args.details) if p]
    if any(p in inputs for p in outputs) or len(set(outputs)) != len(outputs):
        parser.error('output paths must be distinct and must not overwrite inputs')
    try:
        summary, details = evaluate(
            read_records(args.input), args.answer_key, args.gold_key, args.id_key,
            read_records(args.gold_file) if args.gold_file else None,
            args.gold_id_key, args.extraction)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    summary.update(answer_key=args.answer_key, extraction=args.extraction)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.output:
        Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.details:
        with open(args.details, 'w', encoding='utf-8') as stream:
            for row in details:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
