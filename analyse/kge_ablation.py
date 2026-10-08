#!/usr/bin/env python3
"""Select cached answers with direct or dynamic KGE thresholds; evaluate results."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import merge
import sensitivity


SOURCES = ('turn0_old', 'turn1_old', 'turn1_new')
SCOPES = ('full', 'subset')
DETAIL_KEY = 'kge_selection'


def record_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError('ID must be a string or integer')
    return str(value)


def read_rows(path, id_key):
    seen = set()
    with open(path, encoding='utf-8') as stream:
        for number, text in enumerate(stream, 1):
            if not text.strip():
                continue
            try:
                row = json.loads(text)
                if not isinstance(row, dict):
                    raise ValueError('record must be a JSON object')
                rid = record_id(row[id_key])
                if rid in seen:
                    raise ValueError('duplicate ID {}'.format(rid))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError('{}:{}: {}'.format(path, number, exc)) from exc
            seen.add(rid)
            yield number, rid, row
    if not seen:
        raise ValueError('{}: no records'.format(path))


def read_answers(path, id_key, old_key, new_key, keep_base=False):
    records = {}
    for number, rid, row in read_rows(path, id_key):
        try:
            if DETAIL_KEY in row:
                raise ValueError('{} already exists; use original answers'.format(DETAIL_KEY))
            old, new = row[old_key], row[new_key]
            if not isinstance(old, str) or not isinstance(new, str):
                raise ValueError('old and new answers must be strings')
            # Check copied fields before opening any output files.
            if keep_base:
                json.dumps(row, allow_nan=False)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError('{}:{}: {}'.format(path, number, exc)) from exc
        records[rid] = dict(id=row[id_key], old=old, new=new,
                            base=row if keep_base else None)
    return records


def read_scores(path, id_key, score_key, value_key, reference_key):
    scores = {}
    for number, rid, row in read_rows(path, id_key):
        try:
            score = merge.relative_score(row[score_key], value_key, reference_key)
            if score is not None and not math.isfinite(score):
                raise ValueError('relative score overflow')
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            raise ValueError('{}:{}: ID {}: {}'.format(path, number, rid, exc)) from exc
        scores[rid] = score
    return scores


def prepare(args):
    answers, scores = [], []
    for turn in (0, 1):
        prefix = 'turn{}'.format(turn)
        answers.append(read_answers(
            getattr(args, prefix + '_input'), getattr(args, prefix + '_id_key'),
            getattr(args, prefix + '_old_answer_key'),
            getattr(args, prefix + '_new_answer_key'), keep_base=turn == 1))
        scores.append(read_scores(
            getattr(args, prefix + '_scores'), getattr(args, prefix + '_scores_id_key'),
            getattr(args, prefix + '_score_key'), args.triple_value_key, args.reference_scores_key))
    ids = set(answers[0])
    for name, rows in [('turn1 answers', answers[1]), ('turn0 scores', scores[0]),
                       ('turn1 scores', scores[1])]:
        if set(rows) != ids:
            raise ValueError('{} ID set differs from turn0 answers: {} missing, {} extra'.format(
                name, len(ids - rows.keys()), len(rows.keys() - ids)))
    prepared = {}
    for rid, row0 in answers[0].items():
        row1 = answers[1][rid]
        prepared[rid] = dict(
            base=row1['base'], source_ids=dict(turn0=row0['id'], turn1=row1['id']),
            candidates=dict(turn0_old=row0['old'], turn0_new=row0['new'],
                            turn1_old=row1['old'], turn1_new=row1['new']),
            score0=scores[0][rid], score1=scores[1][rid])
    return prepared


def threshold_grid(values0, values1):
    if not values0 or not values1:
        raise ValueError('both threshold lists must be non-empty')
    values = []
    for group in (values0, values1):
        if any(isinstance(v, bool) or not math.isfinite(v) for v in group):
            raise ValueError('thresholds must be finite numbers')
        values.append(list(dict.fromkeys(0.0 if v == 0 else float(v) for v in group)))
    valid, skipped = [], []
    for t0 in values[0]:
        for t1 in values[1]:
            if t0 < t1:
                valid.append((t0, t1))
            else:
                skipped.append(dict(threshold0=t0, threshold1=t1,
                                    reason='threshold0 must be smaller than threshold1'))
    if not valid:
        raise ValueError('no valid threshold pairs: threshold0 must be smaller than threshold1')
    return valid, skipped


def experiment_grid(args):
    direct = args.threshold0_values is not None or args.threshold1_values is not None
    dynamic = args.theta_values is not None or args.c_values is not None
    if direct and dynamic:
        raise ValueError('Choose direct thresholds OR theta/c; do not mix both parameter modes')
    if direct:
        pairs, skipped = threshold_grid(args.threshold0_values, args.threshold1_values)
        return [dict(threshold_mode='direct', theta0=None, c=None, threshold0=t0, threshold1=t1)
                for t0, t1 in pairs], skipped
    if args.theta_values is None:
        raise ValueError('Supply both direct threshold lists, or --theta-values [--c-values]')
    grid, skipped = [], []
    for theta, c, t0, t1 in merge.parameter_grid(
            args.theta_values, args.c_values if args.c_values is not None else [128.0]):
        group = dict(threshold_mode='dynamic', theta0=float(theta), c=float(c),
                     threshold0=float(t0), threshold1=float(t1))
        if t0 < t1:
            grid.append(group)
        else:
            skipped.append(dict(group, reason='threshold0 must be smaller than threshold1'))
    if not grid:
        raise ValueError('no valid theta/c pairs: computed threshold0 must be smaller than threshold1')
    return grid, skipped


def group_label(group):
    if group['threshold_mode'] == 'dynamic':
        return 'theta_{}__c_{}'.format(merge.number_label(group['theta0']), merge.number_label(group['c']))
    return 'threshold0_{}__threshold1_{}'.format(
        merge.number_label(group['threshold0']), merge.number_label(group['threshold1']))


def random_fallback_choices(prepared, seed):
    """Fix a uniform draw per ID, independent of model, grid, or input order."""
    choices = {}
    for rid, item in prepared.items():
        if item['score0'] is None and item['score1'] is None:
            key = json.dumps([seed, rid], ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            generator = random.Random(hashlib.sha256(key).digest())
            choices[rid] = generator.choice(SOURCES)
    return choices


def subset_ids(prepared, path=None):
    if path:
        values = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(values, list):
            raise ValueError('subset IDs file must contain a JSON array')
        ids = [record_id(value) for value in values]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate subset IDs')
        missing = set(ids) - prepared.keys()
        if missing:
            raise ValueError('subset IDs missing from inputs: {}'.format(sorted(missing)[:5]))
        chosen = set(ids)
    else:
        chosen = {rid for rid, item in prepared.items()
                  if item['score0'] is not None or item['score1'] is not None}
    # Output order always follows turn0 answers, including externally fixed subsets.
    return [rid for rid in prepared if rid in chosen]


def scope_coverage(prepared, ids, gold_ids=None):
    id_set = set(ids)
    return dict(
        count=len(ids),
        missing_score0=sum(prepared[rid]['score0'] is None for rid in ids),
        missing_score1=sum(prepared[rid]['score1'] is None for rid in ids),
        both_without_scores=sum(prepared[rid]['score0'] is None and
                                prepared[rid]['score1'] is None for rid in ids),
        both_with_scores=sum(prepared[rid]['score0'] is not None and
                             prepared[rid]['score1'] is not None for rid in ids),
        gold_count=len(gold_ids) if gold_ids is not None else None,
        missing_predictions=len(gold_ids - id_set) if gold_ids is not None else None,
        extra_predictions=len(id_set - gold_ids) if gold_ids is not None else None)


def evaluate(args, script, scale, files, gold_path, empty=False):
    metrics = {key: None for key in sensitivity.METRICS}
    status, error, raw = 'empty_subset' if empty else 'failed', '', None
    if empty:
        files['log'].write_text('Empty subset: evaluation skipped.\n', encoding='utf-8')
    else:
        command = [args.eval_python, str(script), str(files['prediction']), str(gold_path)]
        if args.dataset == '2wiki':
            command.append(str(Path(args.alias_file).resolve()))
        try:
            process = subprocess.run(command, capture_output=True, text=True)
            files['log'].write_text('COMMAND ' + json.dumps(command) + '\nSTDOUT\n' + process.stdout
                                    + '\nSTDERR\n' + process.stderr, encoding='utf-8')
            if process.returncode:
                raise ValueError('evaluator exited with code {}; see {}'.format(
                    process.returncode, files['log']))
            raw = sensitivity.parse_metrics(process.stdout)
            metrics = {key: raw[key] * scale for key in sensitivity.METRICS}
            status = 'ok'
        except (ValueError, OSError) as exc:
            error = str(exc)
            if isinstance(exc, OSError):
                files['log'].write_text(error + '\n', encoding='utf-8')
    sensitivity.write_json(files['metrics'], dict(
        status=status, error=error, raw_metrics=raw,
        raw_unit='fraction' if scale == 100 else 'percent', answer_metrics_percent=metrics))
    return dict(status=status, error=error, **metrics)


def run(args):
    grid, skipped = experiment_grid(args)
    if bool(args.dataset) != bool(args.gold_file):
        raise ValueError('--dataset and --gold-file must be supplied together')
    if args.alias_file and not args.dataset:
        raise ValueError('--alias-file requires evaluation arguments')
    if args.output_answer_key in {args.turn1_id_key, DETAIL_KEY}:
        raise ValueError('output answer key cannot overwrite the output ID or ' + DETAIL_KEY)
    prepared = prepare(args)
    random_choices = random_fallback_choices(prepared, args.seed) if args.random_fallback else {}
    subset = subset_ids(prepared, args.subset_ids)
    subset_set = set(subset)
    members = dict(full=list(prepared), subset=subset)
    root = Path(args.output_dir).expanduser().resolve()
    subset_path, gold_path = root / 'subset_ids.json', root / 'subset_gold.json'
    summary_path, csv_path = root / 'summary.json', root / 'summary.csv'
    inputs = [args.turn0_input, args.turn1_input, args.turn0_scores, args.turn1_scores,
              args.subset_ids, args.gold_file, args.alias_file]
    outputs = [subset_path, summary_path, csv_path]
    gold_ids, subset_gold, extracted = None, None, None
    if args.dataset:
        script, extract, scale, gold_ids = sensitivity.preflight(args)
        missing = subset_set - gold_ids
        if missing:
            raise ValueError('subset IDs missing from gold: {}'.format(sorted(missing)[:5]))
        gold = json.loads(Path(args.gold_file).read_text(encoding='utf-8'))
        subset_gold = [row for row in gold if row['_id'] in subset_set]
        extracted = {rid: {source: extract({'response': item['candidates'][source]}, 'response')
                           for source in SOURCES} for rid, item in prepared.items()}
        inputs.extend([script, script.parent / 'phrase_ans.py'])
        outputs.append(gold_path)
    coverage = {scope: scope_coverage(prepared, ids, gold_ids if scope == 'full' else
                                    (subset_set if gold_ids is not None else None))
                for scope, ids in members.items()}
    plans = []
    for group in grid:
        label = group_label(group)
        if args.random_fallback:
            label += '__random_seed_{}'.format(args.seed)
        paths = {}
        for scope in SCOPES:
            paths[scope] = dict(selected=root / scope / (label + '.jsonl'))
            if args.dataset:
                for key, folder, suffix in [('prediction', 'predictions', '.json'),
                                            ('log', 'logs', '.log'), ('metrics', 'metrics', '.json')]:
                    paths[scope][key] = root / folder / scope / (label + suffix)
            outputs.extend(paths[scope].values())
        plans.append((group, paths))
    input_paths = {Path(path).expanduser().resolve() for path in inputs if path}
    if len({p.resolve() for p in outputs}) != len(outputs):
        raise ValueError('output paths collide')
    for path in outputs:
        if path.resolve() in input_paths:
            raise ValueError('output would overwrite input: {}'.format(path))
        if path.exists() and (not args.overwrite or not path.is_file()):
            raise ValueError('output exists: {}; use --overwrite to replace files'.format(path))
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    sensitivity.write_json(subset_path, subset)
    if args.dataset:
        sensitivity.write_json(gold_path, subset_gold)
    metadata = dict(
        config=vars(args), formula='mean(abs(triple_score - mean(ref_score)))',
        threshold_mode=grid[0]['threshold_mode'],
        dynamic_formula='threshold0=theta0; threshold1=theta0 * (c / (1 + exp(1 - theta0)))',
        random_fallback_enabled=args.random_fallback,
        seed=args.seed if args.random_fallback else None,
        random_rule='both scores are None; uniform choice among ' + ', '.join(SOURCES),
        comparison='score < threshold', metric_unit='percent',
        subset_rule='fixed_ids' if args.subset_ids else 'score0 is not None or score1 is not None',
        subset_ids_file=str(subset_path), coverage=coverage, skipped_pairs=skipped)
    if skipped:
        print('Skipped {} threshold pairs with threshold0 >= threshold1.'.format(len(skipped)), flush=True)
    results = []
    for index, (group, paths) in enumerate(plans, 1):
        t0, t1 = group['threshold0'], group['threshold1']
        counts = {scope: Counter({source: 0 for source in SOURCES}) for scope in SCOPES}
        predictions = {scope: {} for scope in SCOPES}
        with paths['full']['selected'].open('w', encoding='utf-8') as full_stream, \
                paths['subset']['selected'].open('w', encoding='utf-8') as subset_stream:
            for rid, item in prepared.items():
                used_random = rid in random_choices
                source = random_choices[rid] if used_random else merge.choose(
                    item['score0'], item['score1'], t0, t1)
                row = dict(item['base'])
                row[args.output_answer_key] = item['candidates'][source]
                row[DETAIL_KEY] = dict(
                    source_ids=item['source_ids'], candidates=item['candidates'],
                    score0=item['score0'], score1=item['score1'],
                    random_applied=used_random, random_seed=args.seed if args.random_fallback else None,
                    **group, selected_source=source)
                text = json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                full_stream.write(text)
                counts['full'][source] += 1
                if extracted is not None:
                    predictions['full'][rid] = extracted[rid][source]
                if rid in subset_set:
                    subset_stream.write(text)
                    counts['subset'][source] += 1
                    if extracted is not None:
                        predictions['subset'][rid] = extracted[rid][source]
        for scope in SCOPES:
            files = paths[scope]
            metrics = dict(status='not_evaluated', error='', **{key: None for key in sensitivity.METRICS})
            if args.dataset:
                sensitivity.write_json(files['prediction'], dict(answer=predictions[scope], sp={}, evidence={}))
                metrics = evaluate(args, script, scale, files,
                                   Path(args.gold_file).resolve() if scope == 'full' else gold_path,
                                   empty=scope == 'subset' and not subset)
            result = dict(
                dataset=args.dataset, **group, scope=scope, **coverage[scope],
                random_fallback_enabled=args.random_fallback,
                seed=args.seed if args.random_fallback else None,
                random_fallback_count=sum(rid in random_choices for rid in members[scope]),
                **{'selected_' + source: counts[scope][source] for source in SOURCES},
                output_file=str(files['selected']),
                prediction_file=str(files['prediction']) if args.dataset else None,
                log_file=str(files['log']) if args.dataset else None,
                metrics_file=str(files['metrics']) if args.dataset else None, **metrics)
            results.append(result)
            # Keep completed results if a later evaluation fails or is interrupted.
            sensitivity.write_json(summary_path, dict(metadata, results=results))
            with csv_path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(result))
                writer.writeheader()
                writer.writerows(results)
            print('[{}/{}] threshold0={:g}, threshold1={:g}, {}: n={}, {}; EM={}, F1={}'.format(
                index, len(plans), t0, t1, scope, result['count'], result['status'],
                result['em'], result['f1']), flush=True)
    return results


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for turn in (0, 1):
        prefix = '--turn{}'.format(turn)
        parser.add_argument(prefix + '-input', required=True, help='JSONL old/new answers')
        parser.add_argument(prefix + '-scores', required=True, help='JSONL rescored triples')
        parser.add_argument(prefix + '-id-key', default='id', help='Answer-file ID field')
        parser.add_argument(prefix + '-scores-id-key', default='id', help='Score-file ID field')
        parser.add_argument(prefix + '-old-answer-key', default='old_llm_response')
        parser.add_argument(prefix + '-new-answer-key', default='llm_response')
        parser.add_argument(prefix + '-score-key', default='llm_triple_score')
        parser.add_argument('--threshold{}-values'.format(turn), type=float, nargs='+')
    parser.add_argument('--theta-values', '--theta', type=float, nargs='+',
                        help='Dynamic mode: theta0 values; mutually exclusive with direct thresholds')
    parser.add_argument('--c-values', '--c', type=float, nargs='+',
                        help='Dynamic mode: c values (default: 128)')
    parser.add_argument('--random', dest='random_fallback', action='store_true',
                        help='Uniformly choose among 3 candidates only when BOTH scores are missing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for --random (default: 42); draws are stable per question ID')
    parser.add_argument('--triple-value-key', default='triple_score')
    parser.add_argument('--reference-scores-key', default='ref_score')
    parser.add_argument('--output-answer-key', default='llm_response')
    parser.add_argument('--subset-ids', help='Reuse a fixed JSON array of IDs instead of filtering scores')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dataset', choices=('hotpot', '2wiki'))
    parser.add_argument('--gold-file')
    parser.add_argument('--alias-file')
    parser.add_argument('--dataset-test-dir', default=str(sensitivity.ROOT / 'dataset_test'))
    parser.add_argument('--eval-python', default=sys.executable)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        results = run(args)
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        parser.error(str(exc))
    return 1 if any(row['status'] == 'failed' for row in results) else 0


if __name__ == '__main__':
    sys.exit(main())
