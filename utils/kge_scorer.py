"""
KGE (Knowledge Graph Embedding) scorer wrapping code/score.py.
Uses ComplEx model on Wikidata5M via the libkge library.
Models are lazy-loaded on first use to avoid requiring GPU at import time.
"""
from __future__ import annotations
import json
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import KGEConfig
from utils.data_io import score_feature  # re-export for convenience


class KGEScorer:
    """
    Wraps the libkge ComplEx model for triple scoring and tail entity prediction.

    All heavy state (model checkpoint, ID mappings, reference triples) is loaded
    lazily via _load() on first use.
    """

    def __init__(self, config: KGEConfig):
        self.config = config
        self._loaded = False
        self._model = None
        self._e_kgc_id_dict: Dict[str, int] = {}
        self._r_kgc_id_dict: Dict[str, int] = {}
        self._ref_triple_dict: Dict[str, List[Tuple]] = {}

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from kge.model import KgeModel  # type: ignore
        from kge.util.io import load_checkpoint  # type: ignore

        # ID mappings
        self._e_kgc_id_dict, self._r_kgc_id_dict = self._read_id_dicts(
            self.config.entity_ids_path,
            self.config.relation_ids_path,
        )

        # KGE model
        checkpoint = load_checkpoint(self.config.checkpoint_path)
        self._model = KgeModel.create_from(checkpoint)

        # Reference triples (used for relative scoring)
        if self.config.dataset_path:
            self._ref_triple_dict = self._load_dataset(self.config.dataset_path)

        self._loaded = True

    @staticmethod
    def _read_id_dicts(entity_file: str, relation_file: str) -> Tuple[Dict, Dict]:
        e_dict: Dict[str, int] = {}
        r_dict: Dict[str, int] = {}
        with open(entity_file) as ef:
            for line in ef:
                obj = json.loads(line)
                e_dict[obj['wiki_id']] = obj['map_id']
        with open(relation_file) as rf:
            for line in rf:
                obj = json.loads(line)
                r_dict[obj['wiki_id']] = obj['map_id']
        return e_dict, r_dict

    @staticmethod
    def _load_dataset(dataset_path: str) -> Dict[str, List[Tuple]]:
        """Loads Wikidata5M train/valid/test triples keyed by head entity."""
        ref_dict: Dict[str, List[Tuple]] = {}
        for split in ('train', 'valid', 'test'):
            path = dataset_path.rstrip('/') + '/' + split + '.txt'
            try:
                with open(path) as f:
                    for line in f:
                        s, p, o = line.strip().split('\t')
                        ref_dict.setdefault(s, []).append((s, p, o))
            except FileNotFoundError:
                continue
        return ref_dict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_triples(
        self,
        triples: List[Tuple[str, str, str]],
        use_relation: bool = True,
    ) -> List[Dict]:
        """
        Scores a list of (wiki_s_id, wiki_p_id, wiki_o_id) triples.
        Returns list of {triple_id, triple_score, ref_score: [...]}.
        Equivalent to score.py process_by_line() logic.
        """
        self._load()
        import torch

        s_ids, p_ids, o_ids, valid_triples, valid_heads = [], [], [], [], set()

        for t in triples:
            s, p, o = t
            s_id = self._e_kgc_id_dict.get(s, -1)
            o_id = self._e_kgc_id_dict.get(o, -1)
            if s_id == -1 or o_id == -1:
                continue
            s_ids.append(int(s_id))
            o_ids.append(int(o_id))
            valid_triples.append(t)
            valid_heads.add(s)

            if use_relation:
                p_id = self._r_kgc_id_dict.get(p)
                if p_id is None:
                    s_ids.pop()
                    o_ids.pop()
                    valid_triples.pop()
                    valid_heads.discard(s)
                    continue
                p_ids.append(int(p_id))

        if not s_ids:
            return []

        s_t = torch.LongTensor(s_ids)
        p_t = torch.LongTensor(p_ids)
        o_t = torch.LongTensor(o_ids)

        if use_relation:
            scores = self._model.score_spo(s_t, p_t, o_t).tolist()
        else:
            scores_2d = self._model.score_so(s_t, o_t)
            scores_max, _ = torch.max(scores_2d, dim=1)
            scores = scores_max.tolist()

        # Reference scores per head entity
        ref_score_dict: Dict[str, List[float]] = {}
        for ss in valid_heads:
            ref_triples = self._ref_triple_dict.get(ss, [])
            if not ref_triples:
                ref_score_dict[ss] = []
                continue
            rs_ids = [int(self._e_kgc_id_dict[r[0]]) for r in ref_triples if r[0] in self._e_kgc_id_dict]
            rp_ids = [int(self._r_kgc_id_dict[r[1]]) for r in ref_triples if r[1] in self._r_kgc_id_dict]
            ro_ids = [int(self._e_kgc_id_dict[r[2]]) for r in ref_triples if r[2] in self._e_kgc_id_dict]
            min_len = min(len(rs_ids), len(rp_ids), len(ro_ids))
            if min_len == 0:
                ref_score_dict[ss] = []
                continue
            if use_relation:
                ref_sc = self._model.score_spo(
                    torch.LongTensor(rs_ids[:min_len]),
                    torch.LongTensor(rp_ids[:min_len]),
                    torch.LongTensor(ro_ids[:min_len]),
                ).tolist()
            else:
                ref_sc_2d = self._model.score_so(
                    torch.LongTensor(rs_ids[:min_len]),
                    torch.LongTensor(ro_ids[:min_len]),
                )
                ref_sc, _ = torch.max(ref_sc_2d, dim=1)
                ref_sc = ref_sc.tolist()
            ref_score_dict[ss] = ref_sc

        result = []
        for t, sc in zip(valid_triples, scores):
            result.append({
                'triple_id': list(t),
                'triple_score': sc,
                'ref_score': ref_score_dict.get(t[0], []),
            })
        return result

    def score_entity_pairs(
        self,
        entity_pairs: List[Tuple[str, str]],
    ) -> List[Dict]:
        """
        Scores a list of (wiki_head_id, wiki_tail_id) pairs without requiring
        an explicit relation. Both directions, (head, tail) and (tail, head),
        are scored with score_so(); the direction with the lower feature
        distance to its reference scores is kept.
        Returns list of {pair_id, pair_score, ref_score, direction, scored_pair_id}.
        """
        if not entity_pairs:
            return []

        forward_triples = [(h, '__NO_REL__', t) for h, t in entity_pairs]
        backward_triples = [(t, '__NO_REL__', h) for h, t in entity_pairs]
        forward_scores = self.score_triples(forward_triples, use_relation=False)
        backward_scores = self.score_triples(backward_triples, use_relation=False)

        def index_scores(items: List[Dict]) -> Dict[Tuple[str, str], Dict]:
            indexed: Dict[Tuple[str, str], Dict] = {}
            for item in items:
                triple_id = item.get('triple_id', [])
                if len(triple_id) != 3:
                    continue
                indexed[(triple_id[0], triple_id[2])] = item
            return indexed

        def feature_distance(item: Optional[Dict]) -> float:
            if not item:
                return float('inf')
            score = item.get('triple_score')
            if score is None:
                return float('inf')
            ref_score = item.get('ref_score', [])
            if ref_score:
                return float(abs(score - np.average(ref_score)))
            # User-facing fast factual scores are lower-is-better. If no
            # reference distribution is available, fall back to raw score.
            return float(score)

        forward_by_pair = index_scores(forward_scores)
        backward_by_pair = index_scores(backward_scores)

        result = []
        for h, t in entity_pairs:
            forward = forward_by_pair.get((h, t))
            backward = backward_by_pair.get((t, h))
            forward_dist = feature_distance(forward)
            backward_dist = feature_distance(backward)

            if forward_dist == float('inf') and backward_dist == float('inf'):
                continue

            if backward_dist < forward_dist:
                chosen = backward
                direction = 'backward'
                scored_pair_id = [t, h]
            else:
                chosen = forward
                direction = 'forward'
                scored_pair_id = [h, t]

            result.append({
                'pair_id': [h, t],
                'pair_score': chosen.get('triple_score'),
                'ref_score': chosen.get('ref_score', []),
                'direction': direction,
                'scored_pair_id': scored_pair_id,
            })
        return result

    def predict_tail(
        self,
        entity_info: Dict,
        max_ref_num: Optional[int] = None,
    ) -> Tuple[List[Tuple], Set[str]]:
        """
        Predicts tail entities for a head entity using KGE.
        entity_info must have: id, local_pred_r, claims, pred_relation_rank.
        Returns (ref_triple_id_list, tail_wiki_id_set).
        Equivalent to score.py kg_perd_tail().
        """
        self._load()
        import torch

        max_ref_num = max_ref_num or self.config.max_ref_num
        tail_set: Set[str] = set()
        ref_triple_id: List[Tuple] = []

        head_id = self._e_kgc_id_dict.get(entity_info.get('id', ''), -1)
        local_pred_r = entity_info.get('local_pred_r', '')

        if local_pred_r:
            local_r_id = self._r_kgc_id_dict.get(local_pred_r)
            local_cand_tail = entity_info.get('claims', {}).get(local_pred_r, [])

            if local_r_id is not None and local_cand_tail:
                if len(local_cand_tail) > max_ref_num and head_id != -1:
                    # Score candidates and keep top-k
                    tail_ids = [self._e_kgc_id_dict.get(t, -1) for t in local_cand_tail]
                    valid = [(t, tid) for t, tid in zip(local_cand_tail, tail_ids) if tid != -1]
                    if len(valid) >= max_ref_num:
                        ts = torch.LongTensor([v[1] for v in valid])
                        ss = torch.LongTensor([int(head_id)] * len(valid))
                        ps = torch.LongTensor([int(local_r_id)] * len(valid))
                        sc = self._model.score_spo(ss, ps, ts)
                        topk_idx = torch.topk(sc, k=min(max_ref_num, len(valid))).indices.tolist()
                        local_cand_tail = [valid[i][0] for i in topk_idx]
                    else:
                        local_cand_tail = local_cand_tail[:max_ref_num]
                else:
                    local_cand_tail = local_cand_tail[:max_ref_num]

                tail_set.update(local_cand_tail)
                ref_triple_id = [(entity_info['id'], local_pred_r, t) for t in local_cand_tail]

        pred_relation_rank = entity_info.get('pred_relation_rank', [])
        if local_pred_r in pred_relation_rank or head_id == -1 or not pred_relation_rank:
            return ref_triple_id, tail_set

        # Global KGE prediction over ranked relations
        r_ids = [self._r_kgc_id_dict.get(r) for r in pred_relation_rank]
        valid_pairs = [(r, rid) for r, rid in zip(pred_relation_rank, r_ids) if rid is not None]
        if not valid_pairs:
            return ref_triple_id, tail_set

        ss = torch.LongTensor([int(head_id)] * len(valid_pairs))
        ps = torch.LongTensor([int(v[1]) for v in valid_pairs])
        scores = self._model.score_sp(ss, ps)
        max_vals, max_idx = torch.max(scores, dim=-1)
        global_max = torch.argmax(max_vals).item()
        pred_id_tensor = self._model.dataset.entity_ids(max_idx[global_max])
        pred_wiki_id = pred_id_tensor.item() if hasattr(pred_id_tensor, 'item') else pred_id_tensor

        if pred_wiki_id not in tail_set:
            tail_set.add(pred_wiki_id)
            ref_triple_id.append((entity_info['id'], valid_pairs[global_max][0], pred_wiki_id))

        return ref_triple_id, tail_set
