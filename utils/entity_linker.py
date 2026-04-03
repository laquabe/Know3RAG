"""
Entity linking and KG relation mapping utilities extracted from code/KG_mapping.py.
Pure NLP functions — no file-batch loops, no argparse.
Models are lazy-loaded on first use.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from config import EntityLinkerConfig


class EntityLinker:
    """
    Wraps spaCy + entityLinker for entity linking and
    SentenceTransformer for relation mapping.
    All models are loaded lazily on first call.
    """

    def __init__(self, config: EntityLinkerConfig):
        self.config = config
        self._loaded = False

        # Set by _load()
        self._nlp = None
        self._sent_model = None
        self._r_dict: Dict = {}
        self._r_name_dict: Dict = {}
        self._tmp2wiki: Dict = {}
        self._r_des_embedding = None
        self._template_info: List = []
        self._template_dict: Dict = {}
        self._template_sentence_info: List = []
        self._template_sentence_dict: Dict = {}

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        import spacy
        from sentence_transformers import SentenceTransformer

        self._nlp = spacy.load(self.config.spacy_model)
        if 'entityLinker' not in self._nlp.pipe_names:
            self._nlp.add_pipe('entityLinker', last=True)

        self._sent_model = SentenceTransformer(self.config.sbert_model_path)

        if self.config.relation_file:
            self._load_kg_relations(self.config.relation_file)
        if self.config.relation_template_file:
            self._template_info, self._template_dict = self._read_relation_template(
                self.config.relation_template_file
            )
        if self.config.relation_sentence_template_file:
            self._template_sentence_info, self._template_sentence_dict = self._read_relation_template(
                self.config.relation_sentence_template_file
            )

        self._loaded = True

    def _load_kg_relations(self, relation_file: str) -> None:
        import json
        from sentence_transformers import util
        r_des_list = []
        with open(relation_file) as f:
            for line in f:
                line = json.loads(line.strip())
                if not line.get('labels'):
                    continue
                self._r_dict[line['wiki_id']] = line
                self._r_name_dict[line['labels']] = line['wiki_id']
                for ali in line.get('aliases', []):
                    self._r_name_dict[ali] = line['wiki_id']
                aliases = ';'.join(line.get('aliases', []))
                relation_des = '{} {}. {}.'.format(
                    line['labels'], line.get('descriptions', ''), aliases
                )
                self._tmp2wiki[len(r_des_list)] = line['wiki_id']
                r_des_list.append(relation_des)
        self._r_des_embedding = self._sent_model.encode(r_des_list)

    @staticmethod
    def _read_relation_template(template_file: str) -> Tuple[List, Dict]:
        import json
        template_info = []
        template_dict: Dict = {}
        with open(template_file) as f:
            for line in f:
                line = json.loads(line)
                template_info.append(line)
                r_list = template_dict.get(line['wiki_id'], [])
                r_list.append(line)
                template_dict[line['wiki_id']] = r_list
        return template_info, template_dict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def link_entities(
        self,
        sentence: str,
        add_description: bool = True,
        ner_filter: bool = False,
    ) -> Dict[str, Dict]:
        """
        Runs spaCy entityLinker on sentence.
        Returns {mention: {id, entity, start, end, description?}}.
        Wraps entity_linking_with_spacy() from KG_mapping.py.
        """
        self._load()
        doc = self._nlp(sentence)

        ner_list = [ent.text for ent in doc.ents]
        ner_str = ' '.join(ner_list)

        ent_dict: Dict[str, Dict] = {}
        for ent in list(doc._.linkedEntities):
            entity_name = ent.get_label()
            mention = ent.get_span().text
            if entity_name is None or len(mention) < 2:
                continue
            if ner_filter:
                if mention not in ner_list:
                    if mention not in ner_str or len(mention) <= 2:
                        continue
            else:
                if len(mention) <= 2:
                    continue

            entry: Dict = {
                'id': 'Q{}'.format(ent.get_id()),
                'entity': entity_name,
                'start': ent.get_span().start,
                'end': ent.get_span().end,
            }
            if add_description:
                entry['description'] = ent.get_description()
            ent_dict[mention] = entry

        return ent_dict

    def map_entities_for_triples(
        self,
        triple_list: List[Dict],
        entity_dict: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """
        For entities in triple_list not already in entity_dict,
        runs EL to find their Wikidata IDs.
        Wraps entity_mapping_for_line() + relation_mapping_for_line() combined.
        Returns updated entity_dict.
        """
        self._load()
        local_entity_set = set()
        for t in triple_list:
            local_entity_set.add(t.get('subject', ''))
            local_entity_set.add(t.get('object', ''))

        linked_keys = set(entity_dict.keys())
        for e in local_entity_set:
            if e in linked_keys or not e:
                continue
            el_result = self.link_entities(e, add_description=False, ner_filter=True)
            if el_result:
                ids = [v['id'] for v in el_result.values()]
                labels = [v['entity'] for v in el_result.values()]
                entity_dict[e] = {'id': ids, 'entity': labels}

        return entity_dict

    def map_relations_for_triples(self, triple_list: List[Dict]) -> Dict[str, str]:
        """
        Maps each triple's predicate text to a Wikidata relation ID using
        template sentence similarity (SentenceTransformer).
        Wraps relation_mapping_for_line() from KG_mapping.py.
        Returns {triple_str: wiki_relation_id}.
        """
        self._load()
        from sentence_transformers import util
        import torch

        result: Dict[str, str] = {}
        for t in triple_list:
            triple_str = '{} {} {}.'.format(t['subject'], t['predicate'], t['object'])
            if t['predicate'] in self._r_name_dict:
                result[triple_str] = self._r_name_dict[t['predicate']]
            elif self._template_sentence_info:
                triple_embed = self._sent_model.encode(triple_str)
                templates = [
                    r['sentence_template'].format_map(
                        {'subject': t['subject'], 'object': t['object']}
                    )
                    for r in self._template_sentence_info
                ]
                templ_embeds = self._sent_model.encode(templates)
                sim = util.pytorch_cos_sim(triple_embed, templ_embeds)
                best_idx = torch.argmax(sim, dim=-1).item()
                result[triple_str] = self._template_sentence_info[best_idx]['wiki_id']
        return result

    def map_triple_ids(
        self,
        triple_list: List[Dict],
        entity_id_mapping: Dict[str, Dict],
        relation_id_mapping: Dict[str, str],
    ) -> List[Tuple[str, str, str]]:
        """
        Converts text triples to (wiki_s_id, wiki_p_id, wiki_o_id) tuples.
        Wraps triple_mapping() from KG_mapping.py.
        """
        result: List[Tuple[str, str, str]] = []
        for t in triple_list:
            s_info = entity_id_mapping.get(t['subject'])
            o_info = entity_id_mapping.get(t['object'])
            if s_info is None or o_info is None:
                continue

            s_ids = s_info['id'] if isinstance(s_info['id'], list) else [s_info['id']]
            o_ids = o_info['id'] if isinstance(o_info['id'], list) else [o_info['id']]
            triple_str = '{} {} {}.'.format(t['subject'], t['predicate'], t['object'])
            p_id = relation_id_mapping.get(triple_str)
            if p_id is None:
                continue

            for s in s_ids:
                for o in o_ids:
                    if s != o:
                        result.append((s, p_id, o))
        return result

    def convert_question_to_relations(
        self,
        question: str,
        entity: str,
        entity_info: Dict,
        topk: int = 10,
        count_num: int = 3,
    ) -> Tuple[List[str], str]:
        """
        Ranks relation templates by similarity to the question, then finds
        the best local relation from entity claims.
        Wraps convert_question_to_triple() from KG_mapping.py.
        Returns (top_relation_ids, local_pred_relation_id).
        """
        self._load()
        from collections import Counter
        from sentence_transformers import util
        import torch

        if not self._template_info:
            return [], ''

        entity_questions = [t['template'].format(entity) for t in self._template_info]
        eq_embeds = self._sent_model.encode(entity_questions)
        q_embed = self._sent_model.encode(question)
        sim = util.pytorch_cos_sim(q_embed, eq_embeds)
        top_k_actual = min(topk, len(entity_questions))
        _, top_indices = torch.topk(sim, k=top_k_actual)
        top_indices = top_indices.tolist()[0]
        top_relation = [self._template_info[i]['wiki_id'] for i in top_indices]

        # Local relation from existing claims
        local_pred_r = ''
        claims = entity_info.get('claims', {})
        local_indices: List[int] = []
        local_mapping: List[str] = []
        for r_id in claims.keys():
            for temp in self._template_dict.get(r_id, []):
                local_indices.append(temp['template_id'])
                local_mapping.append(r_id)

        if local_indices:
            local_sim = sim[:, local_indices]
            max_idx = torch.argmax(local_sim).item()
            local_pred_r = local_mapping[max_idx]

        # Sort by frequency then first occurrence
        counts = Counter(top_relation)
        sorted_relations = sorted(counts.items(), key=lambda x: (-x[1], top_relation.index(x[0])))
        top_relations = [r for r, _ in sorted_relations][:count_num]

        return top_relations, local_pred_r

    def expand_entity_description(
        self,
        entity_info: Dict,
        tail_dict: Dict,
    ) -> Tuple[str, set]:
        """
        Enriches an entity's description string with KG triple info
        about its tail entities.
        Wraps update_head_entity() from KG_mapping.py.
        Returns (new_description, valid_tail_set).
        """
        self._load()
        des = entity_info.get('description', '') + '.'
        valid_tail_set = set()

        for h_id, r_id, t_id in entity_info.get('kg_triple_id', []):
            templates = self._template_sentence_dict.get(r_id, [])
            if not templates:
                continue
            tail_info = tail_dict.get(t_id, {})
            if tail_info.get('labels'):
                des += ' ' + templates[0]['sentence_template'].format_map(
                    {'subject': entity_info.get('entity', ''), 'object': tail_info['labels']}
                )
                valid_tail_set.add(t_id)

        return des.rstrip('.'), valid_tail_set
