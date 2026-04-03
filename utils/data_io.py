"""
Pure parsing and I/O helpers extracted from code/utils.py.
No project-internal imports — this is a leaf module.
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

MAX_SCORE = 10000

# Words that are too generic to be meaningful triple subjects/objects
forbidden_list = set([
    '--', 'I', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those',
    'anyone', 'everyone', 'someone', 'no one', 'nobody', 'somebody', 'everybody',
    'anything', 'something', 'everything', 'nothing',
    'the', 'a', 'an', 'one', 'the two', 'the other', 'other', 'another',
    'book', 'song', 'country', 'school', 'friend', 'pet', 'job', 'event',
    'restaurant', 'app', 'company', 'film', 'people', 'person',
    'language', 'city', 'family member', 'hobby', 'sport', 'project',
    'skill', 'neighborhood', 'website', 'community', 'judge', 'court',
])


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def json_decode(ans: str) -> Tuple[Any, bool]:
    """Regex-based JSON object extraction from LLM text output."""
    json_pattern = r'\{.*?\}'
    match = re.search(json_pattern, ans, re.DOTALL)
    if match:
        json_str = match.group()
        json_str = json_str.replace('\n', '')
        try:
            return json.loads(json_str), True
        except Exception:
            return json_str, False
    return ans, False


def triple_extraction_decode(ans: str) -> Tuple[Any, bool]:
    """
    Parses LLM triple extraction output.
    Tries to match a top-level JSON array first, then individual objects.
    Returns (triple_list_or_raw_str, success_flag).
    """
    # Try array match first
    array_pattern = r'\[.*?\]'
    match = re.match(array_pattern, ans, re.DOTALL)
    if match:
        json_str = match.group().replace('\n', '')
        try:
            return json.loads(json_str), True
        except Exception:
            pass

    # Fall back to individual objects
    obj_pattern = r'\{.*?\}'
    matches = re.findall(obj_pattern, ans, re.DOTALL)
    triple_list = []
    for m in matches:
        try:
            triple_list.append(json.loads(m))
        except Exception:
            continue

    if triple_list:
        return triple_list, True
    return ans, False


def triple_verification(list_raw: List[Dict]) -> List[Dict]:
    """
    Validates and filters extracted triples.
    Removes triples where subject or object is in the forbidden_list
    or is not a proper string.
    """
    list_new = []
    for t in list_raw:
        if isinstance(t, str):
            continue
        if not all(k in t for k in ('subject', 'predicate', 'object')):
            continue

        head_list = [t['subject']] if isinstance(t['subject'], str) else t.get('subject', [])
        tail_list = [t['object']] if isinstance(t['object'], str) else t.get('object', [])

        for s in head_list:
            if s.lower() in forbidden_list:
                continue
            for o in tail_list:
                if o.lower() in forbidden_list:
                    continue
                list_new.append({
                    'subject': s,
                    'predicate': t['predicate'],
                    'object': o,
                })
    return list_new


def local_check_str(response: str) -> bool:
    """
    Parses LLM local reliability check response.
    Looks for "The reliability of the passage is yes/no".
    """
    prefix = 'The reliability of the passage is'
    ans_index = response.find(prefix)
    if ans_index == -1:
        # Fallback: look for yes/no anywhere in response
        return 'yes' in response.lower()
    ans = response[ans_index:]
    return 'yes' in ans


def answer_phrase(pred: str) -> Tuple[Optional[str], bool]:
    """
    Parses LLM multiple-choice answer, looking for a letter (A/B/C/D).
    Returns (choice_letter_or_None, parsed_cleanly_flag).
    """
    possible_prefix = [
        "The best answer is ",
        "the best answer is ",
        "answer:",
        "answer is:",
        "answer is ",
    ]
    error_flag = True
    pred_ans = pred

    for prefix in possible_prefix:
        if prefix in pred.lower():
            idx = pred.lower().rfind(prefix)
            pred_ans = pred[idx + len(prefix):].strip()
            if pred_ans:
                error_flag = False
                break

    def _find_option(text: str, options=('a.', 'b.', 'c.', 'd.')):
        positions = {}
        for opt in options:
            match = re.search(re.escape(opt), text)
            if match:
                positions[opt] = match.start()
        if not positions:
            return None
        first = min(positions, key=positions.get)
        return first[0]  # return letter only

    if error_flag:
        pred_ans = pred.strip()
        choice = _find_option(pred_ans.lower(), ('a.', 'b.', 'c.', 'd.'))
        if choice is None:
            choice = _find_option(pred_ans.lower(), ('a', 'b', 'c', 'd'))
    else:
        choice = _find_option(pred_ans.lower(), ('a', 'b', 'c', 'd'))

    return choice, not error_flag


def score_feature(
    score_list: List[Dict],
    entity_num: int,
    entity_count: bool = True,
) -> Optional[float]:
    """
    Computes the relative KGE triple score feature.
    Each element of score_list: {triple_id, triple_score, ref_score: [...]}.
    Lower score = more factually consistent with KG.
    """
    feature_score_list = []
    for triple in score_list:
        if not triple.get('ref_score'):
            continue
        ref_avg = np.average(triple['ref_score'])
        score = np.abs(triple['triple_score'] - ref_avg)
        feature_score_list.append(score)

    if entity_count:
        if not feature_score_list:
            feature_score_list = [MAX_SCORE - entity_num]
    else:
        if not feature_score_list:
            return None

    return float(np.average(feature_score_list))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: str) -> Iterator[Dict]:
    """Yields parsed dicts from a JSONL file."""
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, records: List[Dict]) -> None:
    """Writes a list of dicts to a JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def read_json(path: str) -> Any:
    """Reads a standard JSON file."""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def read_data(dataset_name: str, file_path: str) -> Any:
    """
    Dispatch function for reading different dataset formats.
    Temporal_QA and Truthful_QA are stored as a single JSON file.
    Everything else is JSONL (opened as a file handle for streaming).
    """
    if dataset_name in ('Truthful_QA', 'Temporal_QA'):
        return read_json(file_path)
    # For JSONL datasets (hotpotQA, 2wikimultihopQA, PopQA, MMLU)
    # Return an open file handle so callers can stream it
    return open(file_path, encoding='utf-8')
