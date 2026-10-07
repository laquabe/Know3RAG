import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('analyse_merge', Path(__file__).resolve().parents[1] / 'analyse/merge.py')
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)


class MergeTest(unittest.TestCase):
    def test_selection_boundaries(self):
        self.assertEqual(merge.choose(1, 100, 2, 3), 'turn0_old')
        self.assertEqual(merge.choose(2, 2, 2, 3), 'turn1_old')
        self.assertEqual(merge.choose(2, 3, 2, 3), 'turn1_new')
        self.assertEqual(merge.choose(None, 1, 2, 3), 'turn1_old')
        self.assertEqual(merge.choose(None, None, 2, 3), 'turn1_new')

    def test_score(self):
        self.assertIsNone(merge.relative_score([], 'v', 'r'))
        triples = [{'v': 6, 'r': [4, 6]}, {'v': 8, 'r': [4, 6]}, {'r': []}]
        self.assertEqual(merge.relative_score(triples, 'v', 'r'), 2)
        with self.assertRaises(ValueError):
            merge.relative_score([{'v': float('nan'), 'r': [1]}], 'v', 'r')

    def test_grid(self):
        grid = merge.parameter_grid([1., 2., 1.], [2., 4., 128.])
        self.assertEqual(len(grid), 6)
        self.assertEqual([row[2] for row in grid[:3]], [1, 1, 1])
        self.assertEqual([row[3] for row in grid[:3]], [1, 2, 64])
        with self.assertRaises(ValueError):
            merge.parameter_grid([-1.], [128.])

    def test_custom_fields_grid_and_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first, second = root / 'a.jsonl', root / 'b.jsonl'
            scores = [{'v': 4, 'r': [1]}]
            rows0 = [dict(uid=i, old='zero', new='different', score=scores) for i in ['a', 'b']]
            rows1 = [dict(qid=i, before='one', after='two', triples=scores) for i in ['b', 'a']]
            def write(path, rows):
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            write(first, rows0)
            write(second, rows1)
            args = merge.build_parser().parse_args([
                '--input0', str(first), '--input1', str(second),
                '--turn0-id-key', 'uid', '--turn1-id-key', 'qid',
                '--turn0-old-answer-key', 'old', '--turn0-new-answer-key', 'new', '--turn0-score-key', 'score',
                '--turn1-old-answer-key', 'before', '--turn1-new-answer-key', 'after', '--turn1-score-key', 'triples',
                '--triple-value-key', 'v', '--reference-scores-key', 'r', '--output-answer-key', 'selected',
                '--theta-values', '1', '4', '--c-values', '2', '4', '128', '--output-dir', str(root / 'grid')])
            with contextlib.redirect_stdout(io.StringIO()):
                results = merge.run(args)
            self.assertEqual(len(results), 6)
            for result in results:
                records = [json.loads(line) for line in Path(result['output_file']).read_text().splitlines()]
                self.assertEqual([r['qid'] for r in records], ['a', 'b'])
                self.assertEqual(sum(result['selected_counts'].values()), 2)
                # Each grid output must equal a fresh independent single run.
                args.theta_values = [result['theta0']]
                args.c_values = [result['c']]
                args.output_dir = str(root / f"single{result['theta0']}_{result['c']}")
                with contextlib.redirect_stdout(io.StringIO()):
                    single = merge.run(args)
                self.assertEqual(Path(single[0]['output_file']).read_bytes(), Path(result['output_file']).read_bytes())
            with self.assertRaises(ValueError):
                merge.run(args)  # existing output
            write(second, rows1[:1])
            with self.assertRaises(ValueError):
                merge.run(args)  # mismatched IDs
            write(second, rows1 + rows1[:1])
            with self.assertRaises(ValueError):
                merge.run(args)  # duplicate IDs


if __name__ == '__main__':
    unittest.main()
