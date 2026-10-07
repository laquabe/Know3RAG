import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'eval_hotpot', Path(__file__).resolve().parents[1] / 'scripts' / 'eval_hotpot.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


class EvaluationTest(unittest.TestCase):
    def test_answer_phrases(self):
        cases = {
            'Reasoning here.\nThe answer is yes.': 'yes',
            'THE ANSWER IS: 3.14.': '3.14',
            '**Final answer:** U.S.': 'U.S',
            'The best answer is: New York.': 'New York',
            'The answer is no.\nFinal answer: yes.': 'yes',
            '```json\n{"answer": "New York"}\n```': 'New York',
            'Paris': 'Paris',
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(ev.extract_prediction(raw)[0], expected)

    def test_official_metric_semantics(self):
        self.assertEqual(ev.answer_metrics('The New York!', 'new york')['em'], 1)
        self.assertEqual(ev.answer_metrics('yes indeed', 'yes')['f1'], 0)
        self.assertEqual(ev.answer_metrics('no', 'yes')['f1'], 0)
        self.assertEqual(ev.answer_metrics('', '')['em'], 1)
        self.assertEqual(ev.answer_metrics('', '')['f1'], 0)
        self.assertEqual(ev.answer_metrics('3.14', '3.15')['em'], 0)
        scores = ev.answer_metrics('red red blue', 'red blue')
        self.assertAlmostEqual(scores['prec'], 2 / 3)
        self.assertEqual(scores['recall'], 1)
        self.assertAlmostEqual(scores['f1'], 0.8)

    def test_missing_stays_in_denominator(self):
        summary, details = ev.evaluate(
            [{'id': '1', 'llm_response': 'The answer is yes.'},
             {'id': 'extra', 'llm_response': 'no'}],
            gold_rows=[{'_id': '1', 'answer': 'yes'}, {'_id': '2', 'answer': 'no'}])
        self.assertEqual(summary['em'], 0.5)
        self.assertEqual(summary['missing_predictions'], 1)
        self.assertEqual(summary['extra_predictions'], 1)
        self.assertEqual(details[1]['extraction'], 'missing')

    def test_no_gold_leakage_and_no_input_mutation(self):
        row = {'id': '1', 'llm_response': 'The answer is London.', 'answer': 'Paris'}
        summary, details = ev.evaluate([row])
        self.assertEqual(details[0]['prediction'], 'London')
        self.assertEqual(summary['em'], 0)
        self.assertEqual(row['llm_response'], 'The answer is London.')
        self.assertEqual(ev.extract_prediction('The answer is yes.', 'raw')[0],
                         'The answer is yes.')

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            ev.evaluate([{'id': '1'}, {'id': '1'}])


if __name__ == '__main__':
    unittest.main()
