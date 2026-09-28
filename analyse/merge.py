"""Merge cached answers for every theta0 × c combination (no model calls)."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean


def read_records(path, id_key):
    records = {}
    with open(path, encoding='utf-8') as stream:
        for number, text in enumerate(stream, 1):
            if not text.strip():
                continue
            row = json.loads(text)
            rid = row[id_key]
            if not isinstance(rid, (str, int)) or isinstance(rid, bool):
                raise ValueError(f'{path}:{number}: ID must be a string or integer')
            key = str(rid)
            if key in records:
                raise ValueError(f'{path}:{number}: duplicate ID {key}')
            if 'merge_info' in row:
                raise ValueError(f'{path}:{number}: merge_info already exists')
            records[key] = row
    if not records:
        raise ValueError(f'{path}: no records')
    return records


def relative_score(triples, value_key, reference_key):
    if not isinstance(triples, list):
        raise ValueError('triple scores must be a list')
    distances = []
    for triple in triples:
        refs = triple[reference_key]
        if not isinstance(refs, list):
            raise ValueError('reference scores must be a list')
        if not refs:
            continue
        value = triple[value_key]
        for item in [value, *refs]:
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
                raise ValueError('scores must be finite numbers')
        distance = abs(value - mean(refs))
        if not math.isfinite(distance):
            raise ValueError('relative score overflow')
        distances.append(distance)
    return mean(distances) if distances else None


def prepare(first, second, args):
    if first.keys() != second.keys():
        raise ValueError(f'ID sets differ: {len(first.keys() - second.keys())} only in turn0, '
                         f'{len(second.keys() - first.keys())} only in turn1')
    prepared = []
    for rid, row0 in first.items():
        row1 = second[rid]
        answers = {}
        for name, row, key in (
            ('turn0_old', row0, args.turn0_old_answer_key),
            ('turn0_new', row0, args.turn0_new_answer_key),
            ('turn1_old', row1, args.turn1_old_answer_key),
            ('turn1_new', row1, args.turn1_new_answer_key),
        ):
            answer = row[key]
            if not isinstance(answer, str):
                raise ValueError(f'{rid}: {key} must be a string')
            answers[name] = answer
        score0 = relative_score(row0[args.turn0_score_key], args.triple_value_key, args.reference_scores_key)
        score1 = relative_score(row1[args.turn1_score_key], args.triple_value_key, args.reference_scores_key)
        prepared.append((row0[args.turn0_id_key], row1[args.turn1_id_key], row1, answers, score0, score1))
    return prepared


def parameter_grid(theta_values, c_values):
    thetas, constants = list(dict.fromkeys(theta_values)), list(dict.fromkeys(c_values))
    if any(not math.isfinite(x) or x < 0 for x in thetas):
        raise ValueError('theta values must be finite and non-negative')
    if any(not math.isfinite(x) or x <= 0 for x in constants):
        raise ValueError('c values must be finite and positive')
    grid = []
    for theta in thetas:
        theta = 0.0 if theta == 0 else theta
        for c in constants:
            threshold1 = theta * (c / (1 + math.exp(1 - theta)))
            if not math.isfinite(threshold1):
                raise ValueError('threshold overflow')
            grid.append((theta, c, theta, threshold1))
    return grid


def choose(score0, score1, threshold0, threshold1):
    if score0 is not None and score0 < threshold0:
        return 'turn0_old'
    if score1 is not None and score1 < threshold1:
        return 'turn1_old'
    return 'turn1_new'


def number_label(value):
    return str(int(value)) if value.is_integer() else repr(value)


def run(args):
    grid = parameter_grid(args.theta_values, args.c_values)
    reserved = {'merge_info', args.turn1_id_key}
    if args.output_answer_key in reserved:
        raise ValueError('output answer key cannot overwrite merge_info or the output ID')
    first = read_records(args.turn0_input, args.turn0_id_key)
    second = read_records(args.turn1_input, args.turn1_id_key)
    prepared = prepare(first, second, args)
    directory = Path(args.output_dir)
    paths = [directory / f'theta_{number_label(theta)}__c_{number_label(c)}.jsonl'
             for theta, c, _, _ in grid]
    summary_path = directory / 'summary.json'
    inputs = {Path(args.turn0_input).resolve(), Path(args.turn1_input).resolve()}
    destinations = paths + [summary_path]
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise ValueError('output paths collide')
    for path in destinations:
        if path.resolve() in inputs:
            raise ValueError('output cannot overwrite input')
        if path.exists() and (not args.overwrite or not path.is_file()):
            raise ValueError(f'output exists: {path}; use --overwrite to replace files')
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    missing0 = sum(item[4] is None for item in prepared)
    missing1 = sum(item[5] is None for item in prepared)
    for (theta, c, threshold0, threshold1), path in zip(grid, paths):
        counts = Counter(dict(turn0_old=0, turn1_old=0, turn1_new=0))
        with path.open('w', encoding='utf-8') as stream:
            for id0, id1, base, answers, score0, score1 in prepared:
                source = choose(score0, score1, threshold0, threshold1)
                counts[source] += 1
                row = dict(base)
                row[args.output_answer_key] = answers[source]
                row['merge_info'] = dict(
                    source_ids=dict(turn0=id0, turn1=id1), candidates=answers,
                    score0=score0, score1=score1, theta0=theta, c=c,
                    threshold0=threshold0, threshold1=threshold1, selected_source=source)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        result = dict(theta0=theta, c=c, threshold0=threshold0, threshold1=threshold1,
                      output_file=str(path.resolve()), count=len(prepared), selected_counts=dict(counts),
                      missing_score0=missing0, missing_score1=missing1)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))
    summary = dict(config=vars(args), comparisons='strictly less than', results=results)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return results


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--turn0-input', '--input0', dest='turn0_input', required=True)
    parser.add_argument('--turn1-input', '--input1', dest='turn1_input', required=True)
    for turn in (0, 1):
        parser.add_argument(f'--turn{turn}-id-key', default='id')
    parser.add_argument('--turn0-old-answer-key', default='turn0_response')
    parser.add_argument('--turn0-new-answer-key', default='llm_response')
    parser.add_argument('--turn0-score-key', default='turn0_triple_score')
    parser.add_argument('--turn1-old-answer-key', default='old_llm_response')
    parser.add_argument('--turn1-new-answer-key', default='llm_response')
    parser.add_argument('--turn1-score-key', default='llm_triple_score')
    parser.add_argument('--triple-value-key', default='triple_score')
    parser.add_argument('--reference-scores-key', default='ref_score')
    parser.add_argument('--output-answer-key', default='llm_response')
    parser.add_argument('--theta-values', type=float, nargs='+', required=True)
    parser.add_argument('--c-values', type=float, nargs='+', default=[128.0])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
