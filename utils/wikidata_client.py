"""
Wikidata REST API client merging code/wikidata_query.py and code/tail_map.py.
Both files queried the same endpoint with multiprocessing; unified here.
"""
from __future__ import annotations
import multiprocessing
from multiprocessing import Pool, Manager
from typing import Dict, List, Optional, Set

import requests


class WikidataClient:
    """
    Thin wrapper around the Wikidata EntityData REST API.
    """
    BASE_URL = "https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"

    def __init__(self, timeout: int = 10, process_num: int = 5):
        self.timeout = timeout
        self.process_num = process_num

    def query_entity(self, entity_id: str) -> Dict:
        """
        Queries Wikidata for a single entity.
        Returns {labels, descriptions, aliases, claims} or {} on error.
        """
        url = self.BASE_URL.format(entity_id=entity_id)
        try:
            response = requests.get(url, timeout=self.timeout)
            data = response.json()
            entity_data = data['entities'][entity_id]
        except Exception:
            return {}

        try:
            labels = entity_data.get('labels', {}).get('en', {}).get('value', '')
        except Exception:
            labels = ''
        try:
            descriptions = entity_data.get('descriptions', {}).get('en', {}).get('value', '')
        except Exception:
            descriptions = ''
        try:
            aliases = [i['value'] for i in entity_data.get('aliases', {}).get('en', [])]
        except Exception:
            aliases = []

        # Extract claims: relation_id -> [tail_entity_ids]
        claims: Dict[str, List[str]] = {}
        try:
            for prop_id, claim_list in entity_data.get('claims', {}).items():
                tail_ids = []
                for claim in claim_list:
                    try:
                        snak = claim['mainsnak']
                        if snak['snaktype'] == 'value' and snak['datavalue']['type'] == 'wikibase-entityid':
                            tail_ids.append('Q{}'.format(snak['datavalue']['value']['numeric-id']))
                    except Exception:
                        continue
                if tail_ids:
                    claims[prop_id] = tail_ids
        except Exception:
            pass

        return {
            'labels': labels,
            'descriptions': descriptions,
            'aliases': aliases,
            'claims': claims,
        }

    # ------------------------------------------------------------------
    # Batch / enrichment helpers
    # ------------------------------------------------------------------

    def query_entities_batch(
        self,
        entity_ids: List[str],
        skip_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Dict]:
        """
        Queries multiple entities using a multiprocessing pool.
        Returns {wiki_id: entity_info}.
        """
        skip_ids = skip_ids or set()
        ids_to_query = [eid for eid in entity_ids if eid not in skip_ids]

        results: Dict[str, Dict] = {}
        if not ids_to_query:
            return results

        queue = Manager().Queue()
        with Pool(processes=self.process_num) as pool:
            args = [(eid, queue) for eid in ids_to_query]
            pool.starmap(_query_worker, args)

        while not queue.empty():
            eid, info = queue.get()
            results[eid] = info

        return results

    def enrich_query_entities(
        self,
        line: Dict,
        entity_key: str = "query_entity",
    ) -> Dict:
        """
        For a single data record, queries Wikidata claims for each entity
        in line[entity_key] that has an 'id' field.
        Equivalent to wikidata_query.py process_line().
        Mutates and returns line.
        """
        for ent_name, ent_info in line.get(entity_key, {}).items():
            eid = ent_info.get('id')
            if not eid:
                continue
            info = self.query_entity(eid)
            if info:
                ent_info['claims'] = info.get('claims', {})
                if not ent_info.get('description'):
                    ent_info['description'] = info.get('descriptions', '')
        return line

    def enrich_tail_map(self, tail_map_entries: List[Dict]) -> List[Dict]:
        """
        For a list of {wiki_id} dicts, fetches full entity info from Wikidata.
        Equivalent to tail_map.py process_line().
        Returns the enriched list.
        """
        enriched = []
        for entry in tail_map_entries:
            wiki_id = entry.get('wiki_id')
            if not wiki_id:
                enriched.append(entry)
                continue
            info = self.query_entity(wiki_id)
            if info:
                entry.update(info)
            enriched.append(entry)
        return enriched


def _query_worker(entity_id: str, queue) -> None:
    """Multiprocessing worker: queries one entity and puts result on queue."""
    client = WikidataClient()
    info = client.query_entity(entity_id)
    queue.put((entity_id, info))
