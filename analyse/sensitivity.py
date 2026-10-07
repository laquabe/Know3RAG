"""Select cached answers, extract with historical parsers, and evaluate a grid."""
import ast
from collections import Counter
import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import merge

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    'hotpot': ('hotpot', 'hotpot_evaluate_v1.py', 100.0),
    '2wiki': ('2wikimultihop', '2wikimultihop_evaluate_v1.1.py', 1.0),
    'popqa': ('popqa', '2wikimultihop_evaluate.py', 1.0),
}
METRICS = ('em', 'f1', 'prec', 'recall')


def load_extractor(path):
    spec = importlib.util.spec_from_file_location('historical_phrase', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.phrase_answer


def parse_metrics(output):
    # Evaluators may print missing-answer warnings before their final dictionary.
    for offset in reversed([i for i, char in enumerate(output) if char == '{']):
        try:
            metrics = ast.literal_eval(output[offset:].strip())
        except (ValueError, SyntaxError):
            continue
        if isinstance(metrics, dict) and all(k in metrics for k in METRICS):
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                       and math.isfinite(v) for v in metrics.values()):
                raise ValueError('evaluator returned invalid metrics')
            return metrics
    raise ValueError('evaluator output contains no valid metric dictionary')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def preflight(args):
    folder, filename, scale = DATASETS[args.dataset]
    script = Path(args.dataset_test_dir).expanduser().resolve() / folder / filename
    extractor_path = script.parent / 'phrase_ans.py'
    required = [Path(args.gold_file), script, extractor_path]
    if args.dataset == '2wiki':
        if not args.alias_file:
            raise ValueError('2wiki requires --alias-file (JSONL id_aliases file)')
        required.append(Path(args.alias_file))
    for path in required:
        if not path.is_file():
            raise ValueError(f'file not found: {path}')
    check = subprocess.run([args.eval_python, '-c', 'import ujson'], capture_output=True, text=True)
    if check.returncode:
        raise ValueError(f'evaluation Python needs ujson: {args.eval_python}\n{check.stderr}')
    gold = json.loads(Path(args.gold_file).read_text(encoding='utf-8'))
    if not isinstance(gold, list) or not gold:
        raise ValueError('gold must be a non-empty JSON array')
    ids = []
    for row in gold:
        # JSON object keys are strings; do not silently change official gold IDs.
        if not isinstance(row['_id'], str) or not isinstance(row['answer'], str):
            raise ValueError('gold _id and answer must be strings')
        if args.dataset == '2wiki':
            row['answer_id']
        ids.append(row['_id'])
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate gold IDs')
    if args.dataset == '2wiki':
        with open(args.alias_file, encoding='utf-8') as stream:
            for line in stream:
                alias = json.loads(line)
                alias['Q_id']
                if not isinstance(alias['aliases'], list) or not isinstance(alias['demonyms'], list):
                    raise ValueError('aliases and demonyms must be lists')
    return script, load_extractor(extractor_path), scale, set(ids)


def run(args):
    grid = merge.parameter_grid(args.theta_values, args.c_values)
    script, extract, scale, gold_ids = preflight(args)
    first = merge.read_records(args.turn0_input, args.turn0_id_key)
    second = merge.read_records(args.turn1_input, args.turn1_id_key)
    prepared = merge.prepare(first, second, args)
    if args.output_answer_key in {'id', 'prediction', 'selected_source', 'score0', 'score1'}:
        raise ValueError('output answer key collides with a details field')
    # Extract once per candidate with the unchanged historical function.
    extracted = [{key: extract({'response': value}, 'response')
                  for key, value in item[3].items()} for item in prepared]
    root = Path(args.output_dir).resolve()
    plans = []
    for theta, c, t0, t1 in grid:
        label = f'theta_{merge.number_label(theta)}__c_{merge.number_label(c)}'
        files = {kind: root / kind / (label + suffix) for kind, suffix in
                 [('predictions', '.json'), ('logs', '.log'), ('metrics', '.json')]}
        if args.save_details:
            files['details'] = root / 'details' / (label + '.jsonl')
        plans.append((theta, c, t0, t1, files))
    summary_path, csv_path = root / 'summary.json', root / 'summary.csv'
    outputs = [summary_path, csv_path] + [p for *_, files in plans for p in files.values()]
    inputs = {Path(p).resolve() for p in [args.turn0_input, args.turn1_input, args.gold_file,
              str(script), str(script.parent / 'phrase_ans.py'), args.alias_file] if p}
    if len({p.resolve() for p in outputs}) != len(outputs):
        raise ValueError('output paths collide')
    for path in outputs:
        if path.resolve() in inputs:
            raise ValueError(f'output would overwrite input: {path}')
        if path.exists() and (not args.overwrite or not path.is_file()):
            raise ValueError(f'output exists: {path}; use --overwrite to replace files')
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    coverage = dict(prediction_count=len(first), gold_count=len(gold_ids),
                    missing_predictions=len(gold_ids - first.keys()),
                    extra_predictions=len(first.keys() - gold_ids),
                    missing_score0=sum(item[4] is None for item in prepared),
                    missing_score1=sum(item[5] is None for item in prepared))
    results = []
    for theta, c, t0, t1, files in plans:
        counts = Counter(dict(turn0_old=0, turn1_old=0, turn1_new=0))
        predictions, details = {}, []
        for item, short in zip(prepared, extracted):
            id0, id1, _, answers, score0, score1 = item
            source = merge.choose(score0, score1, t0, t1)
            counts[source] += 1
            predictions[str(id0)] = short[source]
            if args.save_details:
                details.append(dict(id=str(id0), selected_source=source,
                                    prediction=short[source], score0=score0, score1=score1,
                                    **{args.output_answer_key: answers[source]}))
        write_json(files['predictions'], dict(answer=predictions, sp={}, evidence={}))
        if args.save_details:
            with files['details'].open('w', encoding='utf-8') as stream:
                for row in details:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        command = [args.eval_python, str(script), str(files['predictions']), str(Path(args.gold_file).resolve())]
        if args.dataset == '2wiki':
            command.append(str(Path(args.alias_file).resolve()))
        result = dict(dataset=args.dataset, theta0=theta, c=c, threshold0=t0, threshold1=t1,
                      **coverage, **{f'selected_{k}': v for k, v in counts.items()},
                      prediction_file=str(files['predictions']), log_file=str(files['logs']),
                      status='failed', error='', **{k: None for k in METRICS})
        try:
            process = subprocess.run(command, capture_output=True, text=True)
            files['logs'].write_text('COMMAND ' + json.dumps(command) + '\nSTDOUT\n' + process.stdout
                                     + '\nSTDERR\n' + process.stderr, encoding='utf-8')
            if process.returncode:
                raise ValueError(f'evaluator exited with code {process.returncode}; see log')
            raw = parse_metrics(process.stdout)
            result.update(status='ok', **{key: raw[key] * scale for key in METRICS})
            write_json(files['metrics'], dict(status='ok', raw_metrics=raw,
                       raw_unit='fraction' if scale == 100 else 'percent',
                       answer_metrics_percent={key: result[key] for key in METRICS}))
        except (ValueError, OSError) as exc:
            result['error'] = str(exc)
            if isinstance(exc, OSError):
                files['logs'].write_text(str(exc) + '\n', encoding='utf-8')
            write_json(files['metrics'], dict(status='failed', error=str(exc)))
        results.append(result)
        # Persist after each group so failures or interruption do not lose earlier metrics.
        write_json(summary_path, dict(config=vars(args), metric_unit='percent', results=results))
        with csv_path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result))
            writer.writeheader()
            writer.writerows(results)
        print(f'theta={theta:g}, c={c:g}: {result["status"]}; EM={result["em"]}, F1={result["f1"]}', flush=True)
    return results


def build_parser():
    parser = merge.build_parser()
    parser.description = __doc__
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--dataset-test-dir', default=str(ROOT / 'dataset_test'),
                        help='Directory containing hotpot/, 2wikimultihop/, popqa/; may be outside this project')
    parser.add_argument('--gold-file', required=True)
    parser.add_argument('--alias-file')
    parser.add_argument('--eval-python', default=sys.executable)
    parser.add_argument('--save-details', action='store_true')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        results = run(args)
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        parser.error(str(exc))
    return 1 if any(row['status'] != 'ok' for row in results) else 0


if __name__ == '__main__':
    sys.exit(main())
