#!/usr/bin/env python3
"""Train one CoDEx KGE model using an official task- and size-specific configuration."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def check_runtime(kge, env):
    """Check the real CLI import chain before creating any run artifacts."""
    probe = subprocess.run([sys.executable, '-c', 'import kge.cli'], cwd=str(kge),
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True)
    if probe.returncode:
        print(probe.stdout, file=sys.stderr, end='')
        if ('SQAGeneratorRun' in probe.stdout and
                ('MappedAnnotationError' in probe.stdout or 'Mapped[]' in probe.stdout)):
            print('\nDependency conflict: legacy Ax ORM code is incompatible with SQLAlchemy 2.x.\n'
                  'In this Python environment, run:\n  {} -m pip install "SQLAlchemy==1.4.54"\n'
                  'Then verify: {} -c "import kge.cli; print(\'LibKGE import OK\')"'
                  .format(shlex.quote(sys.executable), shlex.quote(sys.executable)), file=sys.stderr)
        else:
            print('\nLibKGE import check failed. Fix the traceback above in this Python environment.',
                  file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codex-root', type=Path, required=True,
                        help='CoDEx repository where libkge_setup.sh completed')
    parser.add_argument('--model', choices=['transe', 'conve', 'rescal'], default='transe')
    parser.add_argument('--task', choices=['triple-classification', 'link-prediction'],
                        default='triple-classification', help='Official experiment configuration to use')
    parser.add_argument('--size', choices=['s', 'm', 'l'], default='m',
                        help='CoDEx size (default: m); classification supports s/m, prediction supports s/m/l')
    parser.add_argument('--device', default='cuda:0', help='cuda:0, cuda:1, or cpu')
    parser.add_argument('--output', type=Path, help='New run folder; required again to resume a custom folder')
    parser.add_argument('--epochs', type=int, help='Total epoch limit, including already completed epochs')
    parser.add_argument('--batch-size', type=int, help='Override the official training batch size')
    parser.add_argument('--resume', action='store_true', help='Resume the last checkpoint in the run folder')
    parser.add_argument('--dry-run', action='store_true', help='Check inputs and print command without training')
    args = parser.parse_args()
    if args.task == 'triple-classification' and args.size == 'l':
        parser.error('CoDEx has no official triple-classification configuration for size l. '
                     'Choose --size s/m or --task link-prediction.')
    for name in ('epochs', 'batch_size'):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error('--' + name.replace('_', '-') + ' must be positive')

    root = args.codex_root.expanduser().resolve()
    kge = root / 'kge'
    dataset_name = 'codex-' + args.size
    source = root / 'models' / args.task / dataset_name / args.model / 'config.yaml'
    dataset = kge / 'data' / dataset_name
    output = (args.output.expanduser().resolve() if args.output else
              root / 'local-runs' / args.task / dataset_name / args.model)
    if not source.is_file() or source.stat().st_size == 0:
        parser.error('Official configuration is missing or empty: ' + str(source))
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
    configured_dataset = config.get('dataset', {}).get('name', config.get('dataset.name'))
    if str(configured_dataset).rstrip('/') != dataset_name:
        parser.error('Official configuration dataset does not match --size: ' + str(configured_dataset))
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
        # Runs created before --task was added were always classification runs.
        if saved.get('task', 'triple-classification') != args.task:
            parser.error('Resume task does not match the original run.')
        # Earlier launchers supported only CoDEx-M.
        if saved.get('size', 'm') != args.size:
            parser.error('Resume dataset size does not match the original run.')
        command += ['resume', str(output), '--checkpoint', 'last']
    else:
        if output.exists():
            parser.error('Output already exists. Use --resume or choose a new --output: ' + str(output))
        command += ['start', str(source), '--folder', str(output)]
    command += ['--job.type', 'train', '--job.device', args.device, '--dataset.name', dataset_name]
    if args.epochs:
        command += ['--train.max_epochs', str(args.epochs)]
    if args.batch_size:
        command += ['--train.batch_size', str(args.batch_size)]

    print('Official configuration: {}'.format(source), flush=True)
    print('Run folder: {}'.format(output), flush=True)
    print('Command: {}'.format(' '.join(shlex.quote(x) for x in command)), flush=True)
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env['PYTHONPATH'] = str(kge) + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONUNBUFFERED'] = '1'
    print('Checking LibKGE runtime imports...', flush=True)
    if not check_runtime(kge, env):
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        manifest = output.parent / (output.name + '.launcher.json')
        manifest.write_text(json.dumps({'model': args.model, 'task': args.task, 'size': args.size,
                                        'codex_root': str(root),
                                        'command': command}, indent=2) + '\n')
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
