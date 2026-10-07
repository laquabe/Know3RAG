import importlib.util
import math
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    'legacy_utils', Path(__file__).resolve().parents[1] / 'code' / 'utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


class ResultMergeTest(unittest.TestCase):
    def test_threshold_schedule(self):
        for turn in (0, 1, 2):
            self.assertAlmostEqual(utils.dynamic_threshold(0.2, 128, turn),
                                   0.2 * (128 / (1 + math.exp(0.8))) ** turn)

    def test_invalid_parameters(self):
        for args in ((None, 128, 0), (-1, 128, 0), (1, 0, 0),
                     (1, 128, -1), (1, 128, None), (math.nan, 128, 0)):
            with self.assertRaises(ValueError):
                utils.dynamic_threshold(*args)

    def test_selection_uses_mean_and_ignores_local_check(self):
        # Distances are 1 and 3; mean is 2, not sum 4.
        for local_check in (True, False):
            for threshold, expected in ((1, 'new'), (2, 'new'), (3, 'old')):
                row = {'old': 'old', 'llm_response': 'new', 'local_check': local_check,
                       'scores': [{'triple_score': 6, 'ref_score': [4, 6]},
                                  {'triple_score': 8, 'ref_score': [4, 6]},
                                  {'triple_score': 99, 'ref_score': []}]}
                utils.merge_answer(row, 'old', 'llm_response', 'scores', threshold)
                self.assertEqual(row['llm_response'], expected)

    def test_no_valid_scores_selects_new_and_custom_output_preserves_answers(self):
        for triples in ([], [{'triple_score': 10, 'ref_score': []}]):
            row = {'old': 'old', 'new': 'new', 'scores': triples}
            utils.merge_answer(row, 'old', 'new', 'scores', 100, 'selected')
            self.assertEqual(row['selected'], 'new')
            self.assertEqual(row['old'], 'old')
            self.assertEqual(row['new'], 'new')


if __name__ == '__main__':
    unittest.main()
