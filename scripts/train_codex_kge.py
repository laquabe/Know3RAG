#!/usr/bin/env python3
"""Train one CoDEx-M KGE model using the official classification configuration."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codex-root', type=Path, required=True,
                        help='CoDEx repository where libkge_setup.sh completed')
    parser.add_argument('--model', choices=['transe', 'conve', 'rescal'], default='transe')
    parser.add_argument('--device', default='cuda:0', help='cuda:0, cuda:1, or cpu')
    parser.add_argument('--output', type=Path, help='New run folder; required again to resume a custom folder')
    parser.add_argument('--epochs', type=int, help='Total epoch limit, including already completed epochs')
    parser.add_argument('--batch-size', type=int, help='Override the official training batch size')
    parser.add_argument('--resume', action='store_true', help='Resume the last checkpoint in the run folder')
    parser.add_argument('--dry-run', action='store_true', help='Check inputs and print command without training')
    args = parser.parse_args()
    for name in ('epochs', 'batch_size'):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error('--' + name.replace('_', '-') + ' must be positive')

    root = args.codex_root.expanduser().resolve()
    kge = root / 'kge'
    source = root / 'models' / 'triple-classification' / 'codex-m' / args.model / 'config.yaml'
    dataset = kge / 'data' / 'codex-m'
    output = (args.output.expanduser().resolve() if args.output else
              root / 'local-runs' / 'triple-classification' / 'codex-m' / args.model)
    required = [kge / 'kge' / 'cli.py', source, dataset / 'dataset.yaml']
    required += [dataset / (split + '.txt') for split in ('train', 'valid', 'test')]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            parser.error('Missing or empty file: {}. Complete libkge_setup.sh first.'.format(path))

    try:
        import yaml
    except ImportError:
        parser.error('Activate the Python environment used to install LibKGE (PyYAML is required).')
    with source.open() as stream:
        config = yaml.safe_load(stream)
    actual_model = config.get('model')
    if actual_model == 'reciprocal_relations_model':
        actual_model = config['reciprocal_relations_model']['base_model']['type']
    if actual_model != args.model:
        parser.error('Official configuration does not match --model: ' + str(actual_model))
    with (dataset / 'dataset.yaml').open() as stream:
        data_config = yaml.safe_load(stream)
    for entry in data_config.get('dataset', {}).get('files', {}).values():
        if isinstance(entry, dict) and 'filename' in entry:
            path = dataset / entry['filename']
            if not path.is_file() or path.stat().st_size == 0:
                parser.error('Preprocessed dataset file is missing or empty: ' + str(path))

    command = [sys.executable, '-m', 'kge.cli']
    if args.resume:
        if not (output / 'config.yaml').is_file() or not list(output.glob('checkpoint_*.pt')):
            parser.error('--resume requires config.yaml and a checkpoint in ' + str(output))
        manifest = output.parent / (output.name + '.launcher.json')
        if not manifest.is_file():
            parser.error('Missing launcher metadata; resume a run created by this script.')
        saved = json.loads(manifest.read_text())
        if saved['model'] != args.model or saved['codex_root'] != str(root):
            parser.error('Resume model/repository does not match the original run.')
        command += ['resume', str(output), '--checkpoint', 'last']
    else:
        if output.exists():
            parser.error('Output already exists. Use --resume or choose a new --output: ' + str(output))
        command += ['start', str(source), '--folder', str(output)]
    command += ['--job.type', 'train', '--job.device', args.device, '--dataset.name', 'codex-m']
    if args.epochs:
        command += ['--train.max_epochs', str(args.epochs)]
    if args.batch_size:
        command += ['--train.batch_size', str(args.batch_size)]

    print('Official configuration: {}'.format(source), flush=True)
    print('Run folder: {}'.format(output), flush=True)
    print('Command: {}'.format(' '.join(shlex.quote(x) for x in command)), flush=True)
    if args.dry_run:
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        manifest = output.parent / (output.name + '.launcher.json')
        manifest.write_text(json.dumps({'model': args.model, 'codex_root': str(root),
                                        'command': command}, indent=2) + '\n')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(kge) + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONUNBUFFERED'] = '1'
    log = output.parent / (output.name + '.console.log')
    print('Console log: {}'.format(log), flush=True)
    with log.open('a') as stream:
        stream.write('\n$ ' + ' '.join(shlex.quote(x) for x in command) + '\n')
        stream.flush()
        with subprocess.Popen(command, cwd=str(kge), env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    stream.write(line)
                    stream.flush()
                result = process.wait()
            except KeyboardInterrupt:
                # The terminal delivers SIGINT to the child too; let LibKGE stop.
                process.wait()
                return 130
    if result:
        print('Training failed (exit {}). See {}'.format(result, log), file=sys.stderr)
        return result
    best = output / 'checkpoint_best.pt'
    print('Training command completed. Best checkpoint: {}'.format(best) if best.is_file()
          else 'Training command completed, but no checkpoint_best.pt was produced; inspect the log.')
    return 0 if best.is_file() else 1


if __name__ == '__main__':
    sys.exit(main())
