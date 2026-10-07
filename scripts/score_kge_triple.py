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
    if len(values) != 3 or not all(re.fullmatch(pattern, value) for pattern, value in
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


def score_triples(model, torch, triples, device, batch_size, direction='o'):
    # Preserve the embedding row order supplied by the model's dataset.
    entities = make_mapping(model.dataset.entity_ids(), r'Q\d+')
    relations = make_mapping(model.dataset.relation_ids(), r'P\d+')
    if len(entities) != model.dataset.num_entities() or len(relations) != model.dataset.num_relations():
        raise ValueError('ID mapping sizes do not match checkpoint dataset dimensions.')
    model.eval()
    for offset in range(0, len(triples), batch_size):
        rows, valid, indexes = [], [], []
        for triple in triples[offset:offset + batch_size]:
            h, r, t = triple
            missing = {key: value for key, value, mapping in
                       [('head', h, entities), ('relation', r, relations), ('tail', t, entities)]
                       if value not in mapping}
            row = {'triple_id': list(triple), 'status': 'unknown_id' if missing else 'ok',
                   'triple_score': None, 'direction': direction}
            if missing:
                row['missing_ids'] = missing
            else:
                row['model_indices'] = [entities[h], relations[r], entities[t]]
                valid.append(len(rows))
                indexes.append(row['model_indices'])
            rows.append(row)
        if indexes:
            tensor = torch.tensor(indexes, dtype=torch.long, device=device)
            with torch.no_grad():
                scores = model.score_spo(tensor[:, 0], tensor[:, 1], tensor[:, 2],
                                         direction=direction).reshape(-1).cpu().tolist()
            if len(scores) != len(valid):
                raise ValueError('Model returned an unexpected number of scores.')
            for index, score in zip(valid, scores):
                if math.isfinite(score):
                    rows[index]['triple_score'] = score
                else:
                    rows[index]['status'] = 'nonfinite_score'
        yield from rows


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
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    if not args.checkpoint.is_file():
        parser.error('Checkpoint not found: ' + str(args.checkpoint))
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
        iterator = score_triples(model, torch, triples, args.device, args.batch_size, args.direction)
        while True:
            with contextlib.redirect_stdout(sys.stderr):
                row = next(iterator, None)
            if row is None:
                break
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
            failed = failed or row['status'] != 'ok'
        return 2 if failed else 0
    except Exception as exc:
        print('{}: {}'.format(type(exc).__name__, exc), file=sys.stderr)
        if 'SQAGeneratorRun' in str(exc) or 'MappedAnnotationError' in type(exc).__name__:
            print('Legacy Ax / SQLAlchemy conflict: use SQLAlchemy==1.4.54 in the training environment.',
                  file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
