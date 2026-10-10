import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analyse'))
import sensitivity as s


class SensitivityTest(unittest.TestCase):
    def test_parse_metrics(self):
        self.assertEqual(s.parse_metrics("missing answer x\n{'em': 1, 'f1': 1, 'prec': 1, 'recall': 1}")['em'], 1)
        self.assertEqual(s.parse_metrics('{"em": 100, "f1": 100, "prec": 100, "recall": 100}')['em'], 100)
        with self.assertRaises(ValueError):
            s.parse_metrics('not metrics')

    def fixture(self, root, dataset):
        def scores(value):
            return [{'triple_score': value, 'ref_score': [0]}]
        a, b, gold, aliases = [root / name for name in ('a.jsonl', 'b.jsonl', 'gold.json', 'aliases.jsonl')]
        a.write_text(json.dumps(dict(uid='1', old='Reasoning. The answer is wrong.',
                                    new='Not equal to second old.', score=scores(3))) + '\n')
        b.write_text(json.dumps(dict(qid='1', old='The answer is Paris.',
                                    new='The answer is London.', score=scores(3))) + '\n')
        alias_id = '100' if dataset == 'popqa' else 'Q1'
        gold.write_text(json.dumps([dict(_id='1', answer='City of Paris' if dataset in ('2wiki', 'popqa') else 'Paris', answer_id=alias_id),
                                    dict(_id='2', answer='missing', answer_id='Q2')]))
        aliases.write_text(json.dumps(dict(Q_id=alias_id, aliases=['Paris'], demonyms=[])) + '\n')
        return s.build_parser().parse_args([
            '--dataset', dataset, '--turn0-input', str(a), '--turn1-input', str(b),
            '--turn0-id-key', 'uid', '--turn1-id-key', 'qid',
            '--turn0-old-answer-key', 'old', '--turn0-new-answer-key', 'new', '--turn0-score-key', 'score',
            '--turn1-old-answer-key', 'old', '--turn1-new-answer-key', 'new', '--turn1-score-key', 'score',
            '--theta-values', '1', '4', '--c-values', '2', '128', '--gold-file', str(gold),
            '--alias-file', str(aliases), '--output-dir', str(root / 'out'), '--save-details',
            '--eval-python', os.environ.get('KNOW3RAG_EVAL_PYTHON', sys.executable)])

    def test_original_evaluators(self):
        python = os.environ.get('KNOW3RAG_EVAL_PYTHON', sys.executable)
        if subprocess.run([python, '-c', 'import ujson'], capture_output=True).returncode:
            self.skipTest('set KNOW3RAG_EVAL_PYTHON to a Python with ujson')
        for dataset in s.DATASETS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as td:
                args = self.fixture(Path(td), dataset)
                folder, name, _ = s.DATASETS[dataset]
                script = s.ROOT / 'dataset_test' / folder / name
                before = hashlib.sha256(script.read_bytes()).hexdigest()
                with contextlib.redirect_stdout(io.StringIO()):
                    results = s.run(args)
                self.assertEqual(len(results), 4)
                self.assertTrue(all(r['status'] == 'ok' for r in results))
                self.assertEqual(results[1]['em'], 50)
                self.assertEqual(results[1]['f1'], 50)
                self.assertEqual(results[1]['missing_predictions'], 1)
                self.assertEqual(results[0]['em'], 0)
                pred = json.loads(Path(results[1]['prediction_file']).read_text())
                self.assertEqual(pred, dict(answer={'1': 'Paris'}, sp={}, evidence={}))
                self.assertEqual(hashlib.sha256(script.read_bytes()).hexdigest(), before)
                self.assertTrue((Path(args.output_dir) / 'summary.csv').exists())
                if dataset == 'popqa':
                    summary = json.loads((Path(args.output_dir) / 'summary.json').read_text())
                    self.assertEqual(Path(summary['evaluator_script']).name, 'popqa.py')
                    self.assertTrue(summary['answer_aliases_enabled'])
                with self.assertRaises(ValueError):
                    s.run(args)

    def test_popqa_requires_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root, 'popqa')
            args.alias_file = None
            with self.assertRaisesRegex(ValueError, 'popqa requires --alias-file'):
                s.run(args)
            self.assertFalse((root / 'out').exists())

    def test_failed_group_continues(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.fixture(Path(td), 'hotpot')
            script = s.ROOT / 'dataset_test/hotpot/hotpot_evaluate_v1.py'
            extractor = s.load_extractor(script.parent / 'phrase_ans.py')
            good = subprocess.CompletedProcess([], 0, "{'em': 1, 'f1': 1, 'prec': 1, 'recall': 1}", '')
            bad = subprocess.CompletedProcess([], 1, '', 'failed')
            with patch.object(s, 'preflight', return_value=(script, extractor, 100, {'1', '2'})), \
                 patch.object(s.subprocess, 'run', side_effect=[bad, good, good, good]), \
                 contextlib.redirect_stdout(io.StringIO()):
                results = s.run(args)
            self.assertEqual(results[0]['status'], 'failed')
            self.assertIsNone(results[0]['em'])
            self.assertEqual(results[-1]['status'], 'ok')


if __name__ == '__main__':
    unittest.main()
