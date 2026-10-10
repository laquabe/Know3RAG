#!/usr/bin/env python3
"""Evaluate final predictions or select cached answers by gold entity coverage."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import subprocess
import sys

import merge
import sensitivity


GROUPS = ('fully_covered', 'partially_covered', 'uncovered', 'mapping_incomplete')
SOURCES = ('turn0_old', 'turn1_old', 'turn1_new')
GOLD_SOURCES = dict(hotpot='supporting_facts titles mapped to Wikidata IDs',
                    **{'2wiki': 'entity_ids', 'popqa': 's_uri (subject only)'})


def json_rows(path):
    """Read JSON arrays or JSONL by content, regardless of the file extension."""
    with open(path, encoding='utf-8-sig') as stream:
        first = stream.read(1)
        while first and first.isspace():
            first = stream.read(1)
        stream.seek(0)
        if first == '[':
            try:
                rows = json.load(stream)
            except ValueError as exc:
                raise ValueError(f'{path}: {exc}') from exc
            entries = enumerate(rows, 1)
        else:
            entries = ((number, line) for number, line in enumerate(stream, 1) if line.strip())
        for number, value in entries:
            try:
                row = value if first == '[' else json.loads(value)
                if not isinstance(row, dict):
                    raise ValueError('record must be an object')
            except ValueError as exc:
                raise ValueError(f'{path}:{number}: {exc}') from exc
            yield number, row


def record_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or str(value) == '':
        raise ValueError('ID must be a non-empty string or integer')
    return str(value)


def read_index(path, id_key, tsv=False):
    records = {}
    if tsv:
        with open(path, encoding='utf-8-sig', newline='') as stream:
            entries = list(enumerate(csv.DictReader(stream, delimiter='\t'), 2))
    else:
        entries = json_rows(path)
    for number, row in entries:
        try:
            rid = record_id(row[id_key])
            if rid in records:
                raise ValueError(f'duplicate ID {rid}')
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f'{path}:{number}: {exc}') from exc
        records[rid] = row
    if not records:
        raise ValueError(f'{path}: no records')
    return records


def qid(value):
    """Null/empty IDs are unmapped; malformed non-empty IDs are input errors."""
    if value is None or value == '':
        return None
    if not isinstance(value, str):
        raise ValueError(f'Wikidata ID must be a string: {value!r}')
    match = re.fullmatch(r'(?:https?://www\.wikidata\.org/entity/)?(Q[0-9]+)', value.strip())
    if not match:
        raise ValueError(f'invalid Wikidata ID: {value!r}')
    return match.group(1)


def require_same_ids(expected, actual, label):
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    if missing or extra:
        raise ValueError(f'{label} ID set differs: {len(missing)} missing, {len(extra)} extra; '
                         f'missing examples={sorted(missing)[:5]}, extra examples={sorted(extra)[:5]}')


def input_mode(args):
    if args.prediction_file:
        if args.turn0_input or args.turn1_input or args.theta0 is not None or args.c is not None:
            raise ValueError('--prediction-file cannot be combined with two-turn inputs or --theta0/--c')
        return 'predictions'
    if not args.turn0_input or not args.turn1_input or args.theta0 is None:
        raise ValueError('provide --prediction-file OR both --turn0-input/--turn1-input and --theta0')
    if args.c is None:
        args.c = 128.0
    return 'two_turn'


def load_predictions(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON key/ID {key!r}')
            result[key] = value
        return result

    try:
        with open(path, encoding='utf-8-sig') as stream:
            data = json.load(stream, object_pairs_hook=unique_object)
        if not isinstance(data, dict) or not isinstance(data.get('answer'), dict) or not data['answer']:
            raise ValueError('prediction file must contain a non-empty "answer" object mapping IDs to strings')
        for rid, answer in data['answer'].items():
            record_id(rid)
            if not isinstance(answer, str):
                raise ValueError(f'ID {rid}: prediction must be a string')
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{path}: {exc}') from exc
    return data['answer']


def load_answers(args):
    turns = []
    for turn in (0, 1):
        prefix = f'turn{turn}'
        path = getattr(args, prefix + '_input')
        rows = read_index(path, getattr(args, prefix + '_id_key'))
        answers = {}
        for rid, row in rows.items():
            try:
                old = row[getattr(args, prefix + '_old_answer_key')]
                new = row[getattr(args, prefix + '_new_answer_key')]
                if not isinstance(old, str) or not isinstance(new, str):
                    raise ValueError('old and new answers must be strings')
                score = merge.relative_score(row[getattr(args, prefix + '_score_key')],
                                             args.triple_value_key, args.reference_scores_key)
                answers[rid] = dict(old=old, new=new, score=score)
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                raise ValueError(f'{path}: ID {rid}: {exc}') from exc
        turns.append(answers)
    require_same_ids(turns[0], turns[1], 'turn1 answers versus turn0')
    return turns


def load_gold(args):
    gold = read_index(args.gold_file, '_id')
    for rid, row in gold.items():
        if not isinstance(row['_id'], str) or not isinstance(row.get('answer'), str):
            raise ValueError(f'{args.gold_file}: ID {rid}: gold _id and answer must be strings')
        if args.dataset == '2wiki' and 'answer_id' not in row:
            raise ValueError(f'{args.gold_file}: ID {rid}: missing answer_id')
    return gold


def gold_entities(args, gold):
    mapping, popqa = {}, {}
    if args.dataset == 'hotpot':
        if not args.entity_map_file:
            raise ValueError('hotpot requires --entity-map-file')
        for name, row in read_index(args.entity_map_file, 'entity').items():
            try:
                mapping[name] = qid(row['wikidata_id'])
            except (ValueError, KeyError) as exc:
                raise ValueError(f'{args.entity_map_file}: entity {name}: {exc}') from exc
    if args.dataset == 'popqa':
        if not args.popqa_source_file:
            raise ValueError('popqa requires --popqa-source-file (TSV or JSON/JSONL)')
        popqa = read_index(args.popqa_source_file, 'id',
                           tsv=Path(args.popqa_source_file).suffix.lower() == '.tsv')
        missing = gold.keys() - popqa.keys()
        if missing:
            raise ValueError(f'PopQA source missing {len(missing)} gold IDs: {sorted(missing)[:5]}')
    result = {}
    for rid, row in gold.items():
        ids, unmapped = set(), []
        try:
            if args.dataset == 'hotpot':
                facts = row['supporting_facts']
                if not isinstance(facts, list) or any(
                        not isinstance(fact, list) or len(fact) < 2 or
                        not isinstance(fact[0], str) or not fact[0] for fact in facts):
                    raise ValueError('supporting_facts must contain [title, sentence_index] lists')
                for title in sorted({fact[0] for fact in facts}):
                    if mapping.get(title):
                        ids.add(mapping[title])
                    else:
                        unmapped.append(title)
            elif args.dataset == '2wiki':
                value = row.get('entity_ids')
                if value not in (None, ''):
                    if not isinstance(value, str):
                        raise ValueError('entity_ids must be an underscore-separated string')
                    for part in value.split('_'):
                        entity = qid(part)
                        if entity:
                            ids.add(entity)
                        else:
                            unmapped.append('entity_ids: empty component')
            else:
                entity = qid(popqa[rid]['s_uri'])
                if entity:
                    ids.add(entity)
                else:
                    unmapped.append('s_uri')
        except (ValueError, KeyError, TypeError) as exc:
            source = args.popqa_source_file if args.dataset == 'popqa' else args.gold_file
            raise ValueError(f'{source}: ID {rid}: {exc}') from exc
        result[rid] = dict(gold_entity_ids=sorted(ids), unmapped_gold_entities=unmapped,
                           gold_complete=bool(ids) and not unmapped)
    return result


def coverage_details(args, ids, gold_info):
    el = read_index(args.el_file, args.el_id_key)
    missing = set(ids) - el.keys()
    if missing:
        raise ValueError(f'EL file missing {len(missing)} IDs: {sorted(missing)[:5]}')
    result = {}
    for rid in ids:
        entities, missing_ids = set(), 0
        try:
            query = el[rid]['query_entity']
            if not isinstance(query, dict):
                raise ValueError('query_entity must be an object (use {} for no entities)')
            for item in query.values():
                if not isinstance(item, dict):
                    raise ValueError('query_entity entries must be objects')
                entity = qid(item.get('id'))
                if entity:
                    entities.add(entity)
                else:
                    missing_ids += 1
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f'{args.el_file}: ID {rid}: {exc}') from exc
        info = gold_info[rid]
        matched = entities.intersection(info['gold_entity_ids'])
        ratio = None
        if not info['gold_complete']:
            group = 'mapping_incomplete'
        else:
            ratio = len(matched) / len(info['gold_entity_ids'])
            group = 'fully_covered' if ratio == 1 else 'uncovered' if ratio == 0 else 'partially_covered'
        result[rid] = dict(**info, el_entity_ids=sorted(entities), matched_entity_ids=sorted(matched),
                           el_entities_without_id=missing_ids, coverage=ratio, group=group)
    return result, len(el.keys() - set(ids))


def evaluator_setup(args):
    folder, filename, scale = sensitivity.DATASETS[args.dataset]
    script = Path(args.dataset_test_dir).expanduser().resolve() / folder / filename
    extractor = script.parent / 'phrase_ans.py'
    for path in (script,) if args.prediction_file else (script, extractor):
        if not path.is_file():
            raise ValueError(f'file not found: {path}')
    if args.dataset == '2wiki':
        if not args.alias_file:
            raise ValueError('2wiki requires --alias-file')
        for rid, row in read_index(args.alias_file, 'Q_id').items():
            if any(not isinstance(row.get(key), list) or
                   any(not isinstance(value, str) for value in row[key])
                   for key in ('aliases', 'demonyms')):
                raise ValueError(f'{args.alias_file}: ID {rid}: aliases/demonyms must be string lists')
    process = subprocess.run([args.eval_python, '-c', 'import ujson'], capture_output=True, text=True)
    if process.returncode:
        raise ValueError(f'evaluation Python needs ujson: {args.eval_python}\n{process.stderr}')
    return script, None if args.prediction_file else sensitivity.load_extractor(extractor), scale


def evaluate(args, script, scale, files, empty):
    metrics = {key: None for key in sensitivity.METRICS}
    status, error, raw = 'empty_group' if empty else 'failed', '', None
    if empty:
        files['log'].write_text('Empty group: evaluation skipped.\n', encoding='utf-8')
    else:
        command = [args.eval_python, str(script), str(files['prediction']), str(files['gold'])]
        if args.dataset == '2wiki':
            command.append(str(Path(args.alias_file).resolve()))
        try:
            process = subprocess.run(command, capture_output=True, text=True)
            files['log'].write_text('COMMAND ' + json.dumps(command) + '\nSTDOUT\n' + process.stdout
                                    + '\nSTDERR\n' + process.stderr, encoding='utf-8')
            if process.returncode:
                raise ValueError(f'evaluator exited with code {process.returncode}; see log')
            raw = sensitivity.parse_metrics(process.stdout)
            metrics = {key: raw[key] * scale for key in sensitivity.METRICS}
            status = 'ok'
        except (ValueError, OSError) as exc:
            error = str(exc)
            if isinstance(exc, OSError):
                files['log'].write_text(error + '\n', encoding='utf-8')
    sensitivity.write_json(files['metrics'], dict(status=status, error=error, raw_metrics=raw,
        raw_unit='fraction' if scale == 100 else 'percent', answer_metrics_percent=metrics))
    return dict(status=status, error=error, **metrics)


def output_preflight(args, script):
    root = Path(args.output_dir).expanduser().resolve()
    files = {group: {kind: root / folder / (group + suffix) for kind, folder, suffix in (
        ('prediction', 'predictions', '.json'), ('gold', 'gold', '.json'),
        ('metrics', 'metrics', '.json'), ('log', 'logs', '.log'))}
        for group in ('overall', *GROUPS)}
    outputs = [root / name for name in ('summary.json', 'summary.csv', 'details.jsonl')]
    outputs += [path for paths in files.values() for path in paths.values()]
    inputs = [Path(path).resolve() for path in (args.prediction_file, args.turn0_input, args.turn1_input, args.el_file,
        args.gold_file, args.entity_map_file, args.popqa_source_file, args.alias_file,
        script, None if args.prediction_file else script.parent / 'phrase_ans.py') if path]
    resolved = [path.resolve() for path in outputs]
    if len(set(resolved)) != len(resolved):
        raise ValueError('output paths collide')
    for path in outputs:
        if path.resolve() in inputs or (path.exists() and any(path.samefile(src) for src in inputs)):
            raise ValueError(f'output would overwrite input: {path}')
        if path.exists() and (not args.overwrite or not path.is_file()):
            raise ValueError(f'output exists: {path}; use --overwrite to replace files')
    return root, files, outputs


def run(args):
    mode = input_mode(args)
    selecting = mode == 'two_turn'
    threshold0 = threshold1 = None
    if selecting:
        _, _, threshold0, threshold1 = merge.parameter_grid([args.theta0], [args.c])[0]
        first, second = load_answers(args)
        input_ids = first.keys()
    else:
        predictions = load_predictions(args.prediction_file)
        input_ids = predictions.keys()
    gold = load_gold(args)
    require_same_ids(input_ids, gold, 'gold versus answers')
    coverage, extra_el = coverage_details(args, input_ids, gold_entities(args, gold))
    script, extract, scale = evaluator_setup(args)
    root, files, outputs = output_preflight(args, script)
    details = {}
    groups = {name: [] for name in ('overall', *GROUPS)}
    for rid in input_ids:
        if selecting:
            row0, row1 = first[rid], second[rid]
            source = merge.choose(row0['score'], row1['score'], threshold0, threshold1)
            candidates = dict(turn0_old=row0['old'], turn0_new=row0['new'],
                              turn1_old=row1['old'], turn1_new=row1['new'])
            answer = candidates[source]
            detail = dict(selected_source=source, score0=row0['score'], score1=row1['score'],
                          llm_response=answer, prediction=extract({'response': answer}, 'response'))
        else:
            detail = dict(selected_source='prediction_file', score0=None, score1=None,
                          llm_response=None, prediction=predictions[rid])
        details[rid] = dict(id=rid, **detail, **coverage[rid])
        groups['overall'].append(rid)
        groups[coverage[rid]['group']].append(rid)
    total = len(input_ids)
    eligible = total - len(groups['mapping_incomplete'])
    summary = dict(config=vars(args), input_mode=mode, metric_unit='percent', coverage_unit='fraction',
                   threshold0=threshold0, threshold1=threshold1,
                   threshold_formula='theta_t = theta0 * (c / (1 + exp(1 - theta0))) ** t' if selecting else None,
                   score_formula='mean(abs(triple_score - mean(ref_score))) over valid triples' if selecting else None,
                   comparison='strictly less than' if selecting else None, gold_entity_source=GOLD_SOURCES[args.dataset],
                   coverage_definition='unique matched QIDs / unique gold QIDs; full gold mapping required',
                   total_count=total, eligible_count=eligible, extra_el_records=extra_el, results=[])
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    with (root / 'details.jsonl').open('w', encoding='utf-8') as stream:
        for row in details.values():
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    for group, ids in groups.items():
        paths = files[group]
        sensitivity.write_json(paths['gold'], [gold[rid] for rid in ids])
        sensitivity.write_json(paths['prediction'], dict(
            answer={rid: details[rid]['prediction'] for rid in ids}, sp={}, evidence={}))
        counts = Counter(details[rid]['selected_source'] for rid in ids)
        result = dict(dataset=args.dataset, input_mode=mode, group=group, count=len(ids), percent_of_total=100 * len(ids) / total,
                      percent_of_eligible=100 * len(ids) / eligible if eligible and group in GROUPS[:3] else None,
                      theta0=args.theta0, c=args.c, threshold0=threshold0, threshold1=threshold1,
                      **{f'selected_{name}': counts[name] if selecting else None for name in SOURCES},
                      selected_prediction_file=counts['prediction_file'] if not selecting else None,
                      missing_score0=sum(details[rid]['score0'] is None for rid in ids) if selecting else None,
                      missing_score1=sum(details[rid]['score1'] is None for rid in ids) if selecting else None,
                      **{kind + '_file': str(path) for kind, path in paths.items()},
                      **evaluate(args, script, scale, paths, not ids))
        summary['results'].append(result)
        # Persist each completed evaluation, including failures, for inspection.
        sensitivity.write_json(root / 'summary.json', summary)
        with (root / 'summary.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result))
            writer.writeheader()
            writer.writerows(summary['results'])
        print(f'{group}: n={len(ids)}, total={result["percent_of_total"]:.2f}%, '
              f'eligible={result["percent_of_eligible"]}, EM={result["em"]}, F1={result["f1"]} '
              f'({result["status"]})', flush=True)
    return summary['results']


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=sensitivity.DATASETS, required=True)
    parser.add_argument('--prediction-file', help='Final parsed JSON predictions: {"answer": {"id": "answer"}}; skip selection/extraction')
    for turn in (0, 1):
        prefix = f'--turn{turn}'
        parser.add_argument(prefix + '-input')
        parser.add_argument(prefix + '-id-key', default='id')
        parser.add_argument(prefix + '-old-answer-key', default='old_llm_response')
        parser.add_argument(prefix + '-new-answer-key', default='llm_response')
        parser.add_argument(prefix + '-score-key', default='llm_triple_score')
    parser.add_argument('--triple-value-key', default='triple_score')
    parser.add_argument('--reference-scores-key', default='ref_score')
    parser.add_argument('--el-file', required=True)
    parser.add_argument('--el-id-key', default='id')
    parser.add_argument('--gold-file', required=True)
    parser.add_argument('--entity-map-file', help='Hotpot supporting-title to Wikidata mapping')
    parser.add_argument('--popqa-source-file', help='Original PopQA TSV or JSON/JSONL containing id and s_uri')
    parser.add_argument('--alias-file', help='2Wiki evaluation aliases')
    parser.add_argument('--theta0', type=float, help='Required with two-turn inputs')
    parser.add_argument('--c', type=float, help='Two-turn threshold constant (default: 128)')
    parser.add_argument('--dataset-test-dir', default=str(sensitivity.ROOT / 'dataset_test'))
    parser.add_argument('--eval-python', default=sys.executable)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        results = run(args)
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        parser.error(str(exc))
    return int(any(row['status'] == 'failed' for row in results))


if __name__ == '__main__':
    sys.exit(main())
