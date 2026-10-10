#!/usr/bin/env python3
"""Measure aligned-mention QID accuracy and gold supporting-entity recall offline."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import sys
import unicodedata

from el_coverage import gold_entities, qid, read_index


TIERS = ('strict', 'without_parenthetical')
OUTPUTS = ('summary.json', 'summary.csv', 'mentions.jsonl', 'questions.jsonl')
COUNT_KEYS = ('questions', 'entity_records', 'direct_records', 'direct_link_records',
              'no_offset_records', 'invalid_offset_records', 'missing_qid_records',
              'empty_questions', 'questions_with_direct_links', 'gold_qids',
              'gold_qids_without_names')


def normalize_name(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().replace('_', ' ').split())


def without_parenthetical(value):
    return re.sub(r'\s*\([^()]*\)\s*$', '', normalize_name(value)).strip()


def supporting_titles(row):
    facts = row['supporting_facts']
    if not isinstance(facts, list) or any(
            not isinstance(fact, list) or len(fact) < 2 or
            not isinstance(fact[0], str) or not fact[0].strip() for fact in facts):
        raise ValueError('supporting_facts must contain [non-empty title, sentence_index] lists')
    # Sorting would destroy 2Wiki's title-to-QID pairing.
    return list(dict.fromkeys(fact[0] for fact in facts))


def load_gold_info(args):
    # Only the question IDs and entity annotations are needed, not answer/answer_id.
    gold = read_index(args.gold_file, '_id')
    info = gold_entities(args, gold)
    mapping, source = {}, {}
    if args.dataset == 'hotpot':
        mapping = read_index(args.entity_map_file, 'entity')
    elif args.dataset == 'popqa':
        source = read_index(args.popqa_source_file, 'id',
                            tsv=Path(args.popqa_source_file).suffix.lower() == '.tsv')
    known_titles = {}
    for rid, row in gold.items():
        names = []
        try:
            if args.dataset == '2wiki':
                titles = supporting_titles(row)
                parts = row.get('entity_ids')
                ids = [qid(part) for part in parts.split('_')] if parts else []
                if len(titles) != len(ids):
                    raise ValueError('supporting title count differs from entity_ids count')
                for title, entity_id in zip(titles, ids):
                    if entity_id:
                        if title in known_titles and known_titles[title] != entity_id:
                            raise ValueError(f'inconsistent title-to-QID mapping for {title!r}: '
                                             f'{known_titles[title]} versus {entity_id}')
                        known_titles[title] = entity_id
                    names.append(dict(name=title, qid=entity_id))
            elif args.dataset == 'hotpot':
                for title in supporting_titles(row):
                    names.append(dict(name=title, qid=qid(mapping.get(title, {}).get('wikidata_id'))))
            else:
                subject_id = qid(source[rid]['s_uri'])
                seen = set()
                for field in ('subj', 's_wiki_title'):
                    name = source[rid].get(field)
                    if name is None or name == '':
                        continue
                    if not isinstance(name, str):
                        raise ValueError(f'{field} must be a string')
                    if name.strip() and name not in seen:
                        names.append(dict(name=name, qid=subject_id))
                        seen.add(name)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'{args.dataset} gold: ID {rid}: {exc}') from exc
        info[rid]['gold_names'] = names
    return gold, info


def name_indexes(names):
    indexes = {tier: defaultdict(set) for tier in TIERS}
    for item in names:
        if not item['qid']:
            continue
        full = normalize_name(item['name'])
        if full:
            indexes['strict'][full].add(item['qid'])
            indexes['without_parenthetical'][full].add(item['qid'])
        base = without_parenthetical(item['name'])
        if base:
            indexes['without_parenthetical'][base].add(item['qid'])
    return indexes


def offset_status(entry):
    if 'start' not in entry and 'end' not in entry:
        return 'no_offsets'
    start, end = entry.get('start'), entry.get('end')
    # spaCy stores token offsets, so do not slice the question as character offsets.
    if type(start) is int and type(end) is int and 0 <= start < end:
        return 'direct'
    return 'invalid_offsets'


def align(mention, predicted, offsets, gold_complete, index):
    reason = ('mapping_incomplete' if not gold_complete else
              offsets if offsets != 'direct' else
              'missing_predicted_qid' if predicted is None else None)
    candidates = sorted(index.get(normalize_name(mention), ())) if reason is None else []
    if reason is None:
        reason = ('unmatched' if not candidates else 'ambiguous' if len(candidates) > 1 else
                  'correct' if candidates[0] == predicted else 'incorrect')
    scored = reason in ('correct', 'incorrect')
    return dict(status=reason, candidate_gold_qids=candidates,
                gold_qid=candidates[0] if scored else None,
                correct=(reason == 'correct') if scored else None)


def percent(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def evaluate(args, gold, info, el):
    missing = gold.keys() - el.keys()
    if missing:
        raise ValueError(f'EL file missing {len(missing)} gold IDs: {sorted(missing)[:5]}')
    counts = {scope: Counter() for scope in ('overall', 'eligible', 'mapping_incomplete')}
    outcomes = {tier: Counter() for tier in TIERS}
    matched_questions = {tier: set() for tier in TIERS}
    hits = Counter()
    mentions, questions = [], []
    for rid, row in gold.items():
        g = info[rid]
        complete = g['gold_complete']
        indexes = name_indexes(g['gold_names'])
        query = el[rid].get('query_entity')
        if not isinstance(query, dict):
            raise ValueError(f'EL ID {rid}: query_entity must be an object (use {{}} for no entities)')
        direct_ids, all_ids = set(), set()
        local = Counter(questions=1, entity_records=len(query), empty_questions=int(not query),
                        gold_qids=len(g['gold_entity_ids']))
        named_ids = {item['qid'] for item in g['gold_names'] if item['qid']}
        local['gold_qids_without_names'] = len(set(g['gold_entity_ids']) - named_ids)
        for mention, entry in query.items():
            if not isinstance(entry, dict):
                raise ValueError(f'EL ID {rid}: mention {mention!r}: entry must be an object')
            try:
                predicted = qid(entry.get('id'))
            except ValueError as exc:
                raise ValueError(f'EL ID {rid}: mention {mention!r}: {exc}') from exc
            offsets = offset_status(entry)
            local[dict(direct='direct_records', no_offsets='no_offset_records',
                       invalid_offsets='invalid_offset_records')[offsets]] += 1
            local['missing_qid_records'] += predicted is None
            if predicted:
                all_ids.add(predicted)
                if offsets == 'direct':
                    direct_ids.add(predicted)
                    local['direct_link_records'] += 1
            detail = dict(id=rid, mention=mention, normalized_mention=normalize_name(mention),
                          predicted_qid=predicted, start=entry.get('start'), end=entry.get('end'),
                          offset_status=offsets, eligible=complete)
            for tier in TIERS:
                result = align(mention, predicted, offsets, complete, indexes[tier])
                detail[tier] = result
                if complete:
                    outcomes[tier][result['status']] += 1
                    if result['correct'] is not None:
                        matched_questions[tier].add(rid)
            mentions.append(detail)
        local['questions_with_direct_links'] = int(bool(direct_ids))
        counts['overall'].update(local)
        counts['eligible' if complete else 'mapping_incomplete'].update(local)
        gold_ids = set(g['gold_entity_ids'])
        direct_hits, expanded_hits = direct_ids & gold_ids, all_ids & gold_ids
        if complete:
            hits['direct'] += len(direct_hits)
            hits['expanded'] += len(expanded_hits)
        questions.append(dict(id=rid, question=row.get('question'), eligible=complete, **g,
                              direct_entity_ids=sorted(direct_ids), expanded_entity_ids=sorted(all_ids),
                              direct_hit_ids=sorted(direct_hits), expanded_hit_ids=sorted(expanded_hits),
                              direct_hit_count=len(direct_hits), expanded_hit_count=len(expanded_hits),
                              record_counts={key: local[key] for key in COUNT_KEYS}))

    eligible = counts['eligible']
    common = dict(dataset=args.dataset, total_questions=len(gold), eligible_questions=eligible['questions'],
                  excluded_questions=counts['mapping_incomplete']['questions'],
                  excluded_records=counts['mapping_incomplete']['entity_records'],
                  extra_el_records=len(el.keys() - gold.keys()),
                  direct_records=eligible['direct_records'],
                  direct_link_records=eligible['direct_link_records'])
    results = []
    for tier in TIERS:
        values = outcomes[tier]
        denominator = values['correct'] + values['incorrect']
        status = ('no_eligible_gold' if not eligible['questions'] else
                  'no_direct_links' if not eligible['direct_link_records'] else
                  'no_aligned_mentions' if not denominator else 'ok')
        results.append(dict(**common, metric='accuracy_' + tier, status=status,
                            numerator=values['correct'], denominator=denominator,
                            value_percent=percent(values['correct'], denominator) if status == 'ok' else None,
                            matched_questions=len(matched_questions[tier]),
                            matched_percent_of_direct=percent(denominator, eligible['direct_records']),
                            unmatched_records=values['unmatched'], ambiguous_records=values['ambiguous'],
                            missing_qid_records=values['missing_predicted_qid'],
                            no_offset_records=values['no_offsets'], invalid_offset_records=values['invalid_offsets']))
    for kind in ('direct', 'expanded'):
        status = ('no_eligible_gold' if not eligible['questions'] else
                  'no_direct_links' if kind == 'direct' and not eligible['direct_link_records'] else 'ok')
        results.append(dict(**common, metric='support_recall_' + kind, status=status,
                            numerator=hits[kind], denominator=eligible['gold_qids'],
                            value_percent=percent(hits[kind], eligible['gold_qids']) if status == 'ok' else None,
                            matched_questions=None, matched_percent_of_direct=None,
                            unmatched_records=None, ambiguous_records=None, missing_qid_records=None,
                            no_offset_records=None, invalid_offset_records=None))
    warnings = []
    if not eligible['direct_link_records']:
        warnings.append('No recoverable direct links in eligible questions; direct metrics are not evaluable.')
    if counts['mapping_incomplete']['questions']:
        warnings.append('Questions with incomplete gold QID mappings are excluded from all main metrics.')
    if eligible['gold_qids_without_names']:
        warnings.append('Some gold QIDs lack names: included in recall, unavailable for mention alignment.')
    summary = dict(config=vars(args), metric_unit='percent',
                   counts={scope: {key: value[key] for key in COUNT_KEYS} for scope, value in counts.items()},
                   definitions=dict(
                       accuracy='QID agreement on uniquely name-aligned cached direct-link records',
                       normalization='NFKC, casefold, underscores to spaces, collapse whitespace',
                       without_parenthetical='also match gold names with one trailing non-nested parenthetical removed',
                       direct='integer (not boolean) offsets satisfying 0 <= start < end, plus a valid QID',
                       recall='sum of per-question unique predicted/gold QID intersections / sum of gold QID counts',
                       eligibility='non-empty and complete gold QID mappings; identical scope for all main metrics',
                       limitation='Supporting-entity recall and aligned-subset accuracy are not standard mention-level precision/recall'),
                   warnings=warnings, results=results)
    return summary, mentions, questions


def output_preflight(args):
    root = Path(args.output_dir).expanduser().resolve()
    outputs = [root / name for name in OUTPUTS]
    inputs = [Path(value).expanduser().resolve() for value in (
        args.el_file, args.gold_file, args.entity_map_file, args.popqa_source_file) if value]
    resolved = [path.resolve() for path in outputs]
    if len(set(resolved)) != len(resolved):
        raise ValueError('output paths collide')
    for i, path in enumerate(outputs):
        if path.resolve() in inputs or (path.exists() and any(
                src.exists() and path.samefile(src) for src in inputs)):
            raise ValueError(f'output would overwrite input: {path}')
        if path.exists() and (not args.overwrite or not path.is_file()):
            raise ValueError(f'output exists: {path}; use --overwrite to replace files')
        if path.exists() and any(other.exists() and path.samefile(other) for other in outputs[:i]):
            raise ValueError('output paths collide')
    return root


def run(args):
    root = output_preflight(args)
    gold, info = load_gold_info(args)
    el = read_index(args.el_file, args.el_id_key)
    summary, mentions, questions = evaluate(args, gold, info, el)
    # Complete input validation before creating any output.
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in (('mentions.jsonl', mentions), ('questions.jsonl', questions)):
        with (root / name).open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    (root / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (root / 'summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary['results'][0]))
        writer.writeheader()
        writer.writerows(summary['results'])
    for row in summary['results']:
        value = 'N/A' if row['value_percent'] is None else f'{row["value_percent"]:.5f}%'
        print(f'{row["metric"]}: {row["numerator"]}/{row["denominator"]} = {value} ({row["status"]})')
    for warning in summary['warnings']:
        print('Warning: ' + warning)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('2wiki', 'hotpot', 'popqa'), required=True)
    parser.add_argument('--el-file', required=True)
    parser.add_argument('--gold-file', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--el-id-key', default='id')
    parser.add_argument('--entity-map-file', help='Required for HotpotQA: entity -> wikidata_id records')
    parser.add_argument('--popqa-source-file', help='Required for PopQA: TSV or JSON/JSONL with id, s_uri, subj/s_wiki_title')
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    sys.exit(main())
