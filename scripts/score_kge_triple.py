#!/usr/bin/env python3
"""Score Wikidata (head, relation, tail) IDs with a trained LibKGE checkpoint."""
import argparse
import contextlib
import json
import math
from pathlib import Path
import re
import sys


def validate_triple(values):
    if not isinstance(values, (list, tuple)) or len(values) != 3 or not all(
                                  isinstance(value, str) and re.fullmatch(pattern, value) for pattern, value in
                                  zip((r'Q\d+', r'P\d+', r'Q\d+'), values)):
        raise ValueError('Expected HEAD RELATION TAIL, e.g. Q73437 P1412 Q1860')
    return tuple(values)


def read_triples(args):
    triples = [validate_triple(t) for t in (args.triple or [])]
    if args.input:
        with args.input.open() as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                try:
                    triples.append(validate_triple(line.split()))
                except ValueError as exc:
                    raise ValueError('{}:{}: {}'.format(args.input, number, exc))
    return triples


def make_mapping(ids, pattern):
    result = {}
    for index, identifier in enumerate(ids):
        identifier = str(identifier)
        if not re.fullmatch(pattern, identifier):
            raise ValueError('Unexpected dataset ID: {!r}; check the model dataset.'.format(identifier))
        if identifier in result:
            raise ValueError('Duplicate dataset ID: ' + identifier)
        result[identifier] = index
    return result


def score_indexes(model, torch, indexes, device, direction):
    tensor = torch.tensor(indexes, dtype=torch.long, device=device)
    with torch.no_grad():
        scores = model.score_spo(tensor[:, 0], tensor[:, 1], tensor[:, 2],
                                 direction=direction).reshape(-1).cpu().tolist()
    if len(scores) != len(indexes):
        raise ValueError('Model returned an unexpected number of scores.')
    return scores


def dataset_references(model, head_indexes):
    """Collect same-head positives from all three indexed CoDEx splits."""
    references = {head: [] for head in head_indexes}
    for split in ('train', 'valid', 'test'):
        triples = model.dataset.split(split)
        for start in range(0, len(triples), 65536):
            for h, r, t in triples[start:start + 65536].tolist():
                if h in references:
                    references[h].append([h, r, t])
    return references


def build_score_context(model, heads=None):
    # Preserve the embedding row order supplied by the model's dataset.
    entities = make_mapping(model.dataset.entity_ids(), r'Q\d+')
    relations = make_mapping(model.dataset.relation_ids(), r'P\d+')
    if len(entities) != model.dataset.num_entities() or len(relations) != model.dataset.num_relations():
        raise ValueError('ID mapping sizes do not match checkpoint dataset dimensions.')
    references = dataset_references(model, set(entities.values()) if heads is None else
                                    {entities[h] for h in heads if h in entities})
    return {'entities': entities, 'relations': relations, 'references': references, 'scores': {}}


def score_triples(model, torch, triples, device, batch_size, direction='o', context=None):
    context = context if context is not None else build_score_context(model, {h for h, _, _ in triples})
    entities, relations = context['entities'], context['relations']
    references, reference_scores = context['references'], context['scores']
    model.eval()

    def get_reference_scores(head):
        key = (head, direction)
        if key not in reference_scores:
            scores = []
            refs = references[head]
            for start in range(0, len(refs), batch_size):
                scores.extend(score_indexes(model, torch, refs[start:start + batch_size], device, direction))
            if not all(math.isfinite(score) for score in scores):
                raise ValueError('Nonfinite reference score for head index {}'.format(head))
            reference_scores[key] = scores
        return reference_scores[key]

    for offset in range(0, len(triples), batch_size):
        rows, valid, indexes = [], [], []
        for triple in triples[offset:offset + batch_size]:
            h, r, t = triple
            missing = {key: value for key, value, mapping in
                       [('head', h, entities), ('relation', r, relations), ('tail', t, entities)]
                       if value not in mapping}
            row = {'triple_id': list(triple), 'status': 'unknown_id' if missing else 'ok',
                   'triple_score': None, 'ref_score': [], 'direction': direction}
            if missing:
                row['missing_ids'] = missing
            else:
                row['model_indices'] = [entities[h], relations[r], entities[t]]
                valid.append(len(rows))
                indexes.append(row['model_indices'])
            rows.append(row)
        if indexes:
            scores = score_indexes(model, torch, indexes, device, direction)
            for index, score in zip(valid, scores):
                if math.isfinite(score):
                    rows[index]['triple_score'] = score
                    rows[index]['ref_score'] = get_reference_scores(rows[index]['model_indices'][0])
                    if not rows[index]['ref_score']:
                        rows[index]['status'] = 'no_references'
                else:
                    rows[index]['status'] = 'nonfinite_score'
        yield from rows


def result_fields(row, details=False):
    if row['status'] == 'ok' and not details:
        return {key: row[key] for key in ('triple_id', 'triple_score', 'ref_score')}
    return row


def score_jsonl(model, torch, input_path, output_path, device, batch_size,
                direction='o', source_key='llm_triple_id', target_key='llm_triple_score',
                details=False):
    from tqdm import tqdm
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Input and output must be different files.')
    if output_path.exists():
        raise ValueError('Output already exists; choose a new file: ' + str(output_path))
    with input_path.open(encoding='utf-8') as stream:
        total = sum(bool(line.strip()) for line in stream)
    print('Indexing train/valid/test references (once)...', file=sys.stderr)
    with contextlib.redirect_stdout(sys.stderr):
        context = build_score_context(model)
    counts = {'records': 0, 'triples': 0, 'ok': 0, 'unknown_id': 0,
              'no_references': 0, 'nonfinite_score': 0}
    with input_path.open(encoding='utf-8') as source, output_path.open('x', encoding='utf-8') as target, \
            tqdm(total=total, desc='Scoring JSONL', unit='record', file=sys.stderr) as progress:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError('Each JSONL record must be an object')
                if source_key not in record or not isinstance(record[source_key], list):
                    raise ValueError('{} must be present and be a list'.format(source_key))
                triples = [validate_triple(item) for item in record[source_key]]
                with contextlib.redirect_stdout(sys.stderr):
                    rows = list(score_triples(model, torch, triples, device, batch_size, direction, context))
                record[target_key] = [result_fields(row, details) for row in rows
                                      if row['status'] != 'unknown_id']
                target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                target.flush()
            except Exception as exc:
                raise ValueError('{} line {}: {}. Output contains only completed prior records.'
                                 .format(input_path, number, exc)) from exc
            counts['records'] += 1
            counts['triples'] += len(rows)
            for row in rows:
                counts[row['status']] += 1
            progress.set_postfix(ok=counts['ok'], unknown=counts['unknown_id'], refresh=False)
            progress.update(1)
    print('Scoring summary: ' + json.dumps(counts), file=sys.stderr)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--libkge-root', type=Path, help='LibKGE checkout containing kge/__init__.py')
    parser.add_argument('--dataset-dir', type=Path,
                        help='Original preprocessed training dataset, with dataset.yaml and ID maps')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--direction', choices=['o', 's'], default='o',
                        help='o: score tail given head/relation (default); s: score head via inverse relation')
    parser.add_argument('--triple', nargs=3, action='append', metavar=('HEAD', 'RELATION', 'TAIL'))
    parser.add_argument('--input', type=Path, help='One QID PID QID triple per line, tab or space separated')
    parser.add_argument('--output', type=Path,
                        help='With --input: read full JSONL records and write replaced score fields here')
    parser.add_argument('--source-key', default='llm_triple_id')
    parser.add_argument('--target-key', default='llm_triple_score')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--details', action='store_true', help='Include status, internal indexes and direction on successful rows')
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    if not args.checkpoint.is_file():
        parser.error('Checkpoint not found: ' + str(args.checkpoint))
    if args.output:
        if not args.input or args.triple:
            parser.error('--output requires --input JSONL and cannot be combined with --triple')
        if not args.input.is_file():
            parser.error('Input file not found: ' + str(args.input))
        if args.output.exists() or args.input.resolve() == args.output.resolve():
            parser.error('Choose a new output file, different from input; existing files are not overwritten')
        if args.source_key == args.target_key:
            parser.error('--source-key and --target-key must differ')
    else:
        try:
            triples = read_triples(args)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        if not triples:
            parser.error('Provide --triple or a nonempty --input file')
    if args.libkge_root:
        library = args.libkge_root.expanduser().resolve()
        if not (library / 'kge' / '__init__.py').is_file():
            parser.error('--libkge-root must point to the LibKGE source checkout, not CoDEx raw data')
        sys.path.insert(0, str(library))
    try:
        # Keep JSONL results on stdout, library diagnostics on stderr.
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            from kge import Config, Dataset
            from kge.model import KgeModel
            from kge.util.io import load_checkpoint
            checkpoint = load_checkpoint(str(args.checkpoint.resolve()), device=args.device)
            dataset = None
            if args.dataset_dir:
                folder = args.dataset_dir.expanduser().resolve()
                if not (folder / 'dataset.yaml').is_file():
                    raise ValueError('--dataset-dir needs preprocessed dataset.yaml and ID maps; '
                                     'raw kge/data/triples files are insufficient.')
                config = Config.create_from(checkpoint)
                expected = str(config.get('dataset.name')).rstrip('/')
                dataset = Dataset.create(config, preload_data=False, folder=str(folder))
                if str(config.get('dataset.name')).rstrip('/') != expected:
                    raise ValueError('Selected dataset name does not match checkpoint: ' + expected)
            model = KgeModel.create_from(checkpoint, dataset=dataset)
            model.to(args.device)
            print('Loaded model={} dataset={} device={}'.format(
                model.config.get('model'), model.config.get('dataset.name'), args.device), file=sys.stderr)
        failed = False
        if args.output:
            counts = score_jsonl(model, torch, args.input, args.output, args.device, args.batch_size,
                                 args.direction, args.source_key, args.target_key, args.details)
            return 2 if counts['nonfinite_score'] else 0
        iterator = score_triples(model, torch, triples, args.device, args.batch_size, args.direction)
        while True:
            with contextlib.redirect_stdout(sys.stderr):
                row = next(iterator, None)
            if row is None:
                break
            failed = failed or row['status'] != 'ok'
            row = result_fields(row, args.details)
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
        return 2 if failed else 0
    except Exception as exc:
        print('{}: {}'.format(type(exc).__name__, exc), file=sys.stderr)
        if 'SQAGeneratorRun' in str(exc) or 'MappedAnnotationError' in type(exc).__name__:
            print('Legacy Ax / SQLAlchemy conflict: use SQLAlchemy==1.4.54 in the training environment.',
                  file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
