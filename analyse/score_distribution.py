#!/usr/bin/env python3
"""Summarize relative KGE scores and strict upper-tail thresholds (Python 3.7+)."""
import argparse
from bisect import bisect_left, bisect_right
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
import sys


LEVELS = ('triple', 'record_mean', 'record_sum')
PERCENTAGES = (0, 25, 50, 75, 100)


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('scores must be finite numbers')
    return value


def record_distances(items, value_key='triple_score', reference_key='ref_score'):
    if not isinstance(items, list):
        raise ValueError('triple-score field must be a list')
    distances, no_reference = [], 0
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError('score item {} must be an object'.format(index))
        refs = item[reference_key]
        if not isinstance(refs, list):
            raise ValueError('reference scores must be a list')
        if not refs:
            no_reference += 1
            continue
        distance = abs(finite_number(item[value_key]) - mean([finite_number(r) for r in refs]))
        distances.append(finite_number(distance))
    return distances, no_reference


def quantile(sorted_values, fraction):
    """Linear interpolation, equivalent to the usual (n-1)*q percentile."""
    position = (len(sorted_values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def summarize(values):
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return {'count': 0, 'min': None, 'max': None, 'mean': None, 'std': None,
                'median': None, 'thresholds': []}
    thresholds = []
    for percentage in PERCENTAGES:
        # Nonnegative distances: -1 explicitly gives 100% under strict >,
        # including zero scores. Q0=min would not have that property.
        threshold = -1.0 if percentage == 100 else quantile(ordered, 1 - percentage / 100.0)
        lower, upper = bisect_left(ordered, threshold), bisect_right(ordered, threshold)
        thresholds.append({
            'target_above_percent': percentage, 'threshold': threshold,
            'count_above': n - upper, 'actual_above_percent': 100.0 * (n - upper) / n,
            'count_equal': upper - lower, 'count_below': lower,
        })
    return {'count': n, 'min': ordered[0], 'max': ordered[-1], 'mean': mean(ordered),
            'std': pstdev(ordered), 'median': quantile(ordered, 0.5), 'thresholds': thresholds}


def analyze_file(path, score_key='llm_triple_score', value_key='triple_score', reference_key='ref_score'):
    values = {level: [] for level in LEVELS}
    counts = {'records': 0, 'records_with_scores': 0, 'records_without_scores': 0,
              'triple_items': 0, 'valid_triples': 0, 'triples_without_references': 0}
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('record must be an object')
                distances, skipped = record_distances(row[score_key], value_key, reference_key)
                counts['records'] += 1
                counts['triple_items'] += len(row[score_key])
                counts['triples_without_references'] += skipped
                counts['valid_triples'] += len(distances)
                if distances:
                    values['triple'].extend(distances)
                    values['record_mean'].append(finite_number(mean(distances)))
                    values['record_sum'].append(finite_number(math.fsum(distances)))
                    counts['records_with_scores'] += 1
                else:
                    counts['records_without_scores'] += 1
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                raise ValueError('{}:{}: {}'.format(path, number, exc)) from exc
    return {'input': str(Path(path).resolve()), 'counts': counts,
            'distributions': {level: summarize(values[level]) for level in LEVELS}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True,
                        help='One or more rescored JSONL files; each is analyzed separately')
    parser.add_argument('--labels', nargs='+', help='Unique file/model labels, in the same order as inputs')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--score-key', default='llm_triple_score')
    parser.add_argument('--triple-value-key', default='triple_score')
    parser.add_argument('--reference-scores-key', default='ref_score')
    args = parser.parse_args(argv)
    labels = args.labels or [path.stem for path in args.input]
    if len(labels) != len(args.input) or len(set(labels)) != len(labels):
        parser.error('Provide one unique --labels value per input file')
    json_path, csv_path = args.output_dir / 'distribution.json', args.output_dir / 'thresholds.csv'
    if json_path.exists() or csv_path.exists():
        parser.error('Output reports already exist; select a new --output-dir')
    try:
        reports = []
        for label, path in zip(labels, args.input):
            print('Analyzing {}: {}'.format(label, path), file=sys.stderr, flush=True)
            report = analyze_file(path, args.score_key, args.triple_value_key, args.reference_scores_key)
            report['label'] = label
            reports.append(report)
        payload = {
            'formula': 'abs(triple_score - mean(ref_score))',
            'comparison': 'score > threshold',
            'quantile_method': 'linear Q(1 - target_above_percent/100); 100% uses -1 for nonnegative scores',
            'denominator': 'Valid scores only; empty/missing-reference scores excluded, not replaced by zero.',
            'aggregation': {'triple': 'one value per valid triple occurrence',
                            'record_mean': 'mean per record, matching current merge.py',
                            'record_sum': 'sum per record, matching paper Eq. (1)'},
            'score_key': args.score_key, 'files': reports,
        }
        # Complete input validation before creating output files.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with json_path.open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        fields = ['label', 'input', 'level', 'count', 'target_above_percent', 'threshold',
                  'actual_above_percent', 'count_above', 'count_equal', 'count_below']
        with csv_path.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for report in reports:
                print('\n{}: {}'.format(report['label'], json.dumps(report['counts'])))
                for level, distribution in report['distributions'].items():
                    print('  {} (n={})'.format(level, distribution['count']))
                    if not distribution['count']:
                        print('    No valid scores; no thresholds generated.')
                    for row in distribution['thresholds']:
                        writer.writerow(dict(label=report['label'], input=report['input'], level=level,
                                             count=distribution['count'], **row))
                        print('    above {:3d}%: threshold={:.17g}; actual={:.2f}%; equal={}'.format(
                            row['target_above_percent'], row['threshold'],
                            row['actual_above_percent'], row['count_equal']))
        print('\nSaved {} and {}'.format(json_path, csv_path))
        return 0
    except (OSError, ValueError, OverflowError) as exc:
        print('Error: {}'.format(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
