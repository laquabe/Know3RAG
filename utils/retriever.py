"""
Hybrid BM25 + Dense local retriever.
A local alternative to ColBERTv2 used in the paper's experiments.

Usage:
    from utils.retriever import Corpus, HybridRetriever, build_index_from_jsonl
    from config import RetrieverConfig

    cfg = RetrieverConfig(
        corpus_path="wiki_corpus.jsonl",
        index_dir="retriever_index/",
        sbert_model_path="all-mpnet-base-v2",
    )
    retriever = build_index_from_jsonl(cfg.corpus_path, cfg)
    results = retriever.retrieve("What is the capital of France?", top_k=5)
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import RetrieverConfig


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

class Corpus:
    """In-memory corpus. Each document: {id, text, title?}."""

    def __init__(self, documents: List[Dict]):
        self.documents = documents
        self.texts: List[str] = [d['text'] for d in documents]
        self.ids: List[str] = [str(d['id']) for d in documents]
        self.titles: List[str] = [d.get('title', '') for d in documents]

    @classmethod
    def from_jsonl(cls, path: str) -> 'Corpus':
        """Load corpus from a JSONL file where each line is {id, text, title?}."""
        import json
        docs = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        return cls(docs)

    def __len__(self) -> int:
        return len(self.documents)


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

class BM25Index:
    """Thin wrapper around rank_bm25.BM25Okapi."""

    def __init__(self, config: RetrieverConfig):
        self.config = config
        self._index = None

    def build(self, corpus: Corpus) -> None:
        """Tokenize and build BM25 index."""
        from rank_bm25 import BM25Okapi  # type: ignore
        tokenized = [text.lower().split() for text in corpus.texts]
        self._index = BM25Okapi(tokenized, k1=self.config.bm25_k1, b=self.config.bm25_b)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self._index, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            self._index = pickle.load(f)

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Returns list of (corpus_index, bm25_score) sorted descending."""
        if self._index is None:
            raise RuntimeError("BM25Index not built or loaded.")
        tokenized_query = query.lower().split()
        scores = self._index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Dense index
# ---------------------------------------------------------------------------

class DenseIndex:
    """Dense retrieval using SentenceTransformer + cosine similarity."""

    def __init__(self, config: RetrieverConfig):
        self.config = config
        self._model = None
        self._embeddings: Optional[np.ndarray] = None  # shape [N, D]

    def _load_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.config.sbert_model_path)

    def build(self, corpus: Corpus, batch_size: int = 64) -> None:
        """Encode all corpus texts and store as float32 numpy array."""
        self._load_model()
        self._embeddings = self._model.encode(
            corpus.texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine = dot product when normalized
        ).astype(np.float32)

    def save(self, path: str) -> None:
        np.save(path, self._embeddings)

    def load(self, path: str) -> None:
        self._embeddings = np.load(path)

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Returns (corpus_index, cosine_score) sorted descending."""
        if self._embeddings is None:
            raise RuntimeError("DenseIndex not built or loaded.")
        self._load_model()
        q_embed = self._model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        # dot product of normalized vectors = cosine similarity
        scores = (q_embed @ self._embeddings.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def search_batch(self, queries: List[str], top_k: int) -> List[List[Tuple[int, float]]]:
        """Batch encode all queries at once for efficiency."""
        if self._embeddings is None:
            raise RuntimeError("DenseIndex not built or loaded.")
        self._load_model()
        q_embeds = self._model.encode(
            queries, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        scores_matrix = q_embeds @ self._embeddings.T  # [Q, N]
        results = []
        for scores in scores_matrix:
            top_indices = np.argsort(scores)[::-1][:top_k]
            results.append([(int(i), float(scores[i])) for i in top_indices])
        return results


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Combines BM25 + dense retrieval with min-max normalised score fusion.

    combined_score = bm25_weight * norm(bm25) + dense_weight * norm(dense)
    """

    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.corpus: Optional[Corpus] = None
        self._bm25 = BM25Index(config)
        self._dense = DenseIndex(config)

    def index(self, corpus: Corpus) -> None:
        """Build both BM25 and dense indexes and save to config.index_dir."""
        self.corpus = corpus
        index_dir = Path(self.config.index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        self._bm25.build(corpus)
        self._dense.build(corpus)

        self._bm25.save(str(index_dir / 'bm25.pkl'))
        self._dense.save(str(index_dir / 'dense.npy'))

        # Save corpus texts for retrieval output
        import json
        with open(str(index_dir / 'corpus.jsonl'), 'w', encoding='utf-8') as f:
            for doc in corpus.documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    def load_index(self, index_dir: Optional[str] = None) -> None:
        """Load pre-built indexes from disk."""
        import json
        d = Path(index_dir or self.config.index_dir)
        self._bm25.load(str(d / 'bm25.pkl'))
        self._dense.load(str(d / 'dense.npy'))

        # Reload corpus for output
        corpus_path = d / 'corpus.jsonl'
        if corpus_path.exists():
            docs = []
            with open(str(corpus_path), encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        docs.append(json.loads(line))
            self.corpus = Corpus(docs)

    @staticmethod
    def _min_max_normalize(scores: List[Tuple[int, float]]) -> Dict[int, float]:
        """Min-max normalise a list of (index, score) pairs into [0, 1]."""
        if not scores:
            return {}
        vals = np.array([s for _, s in scores], dtype=np.float32)
        mn, mx = vals.min(), vals.max()
        if mx == mn:
            normed = np.ones_like(vals)
        else:
            normed = (vals - mn) / (mx - mn)
        return {idx: float(n) for (idx, _), n in zip(scores, normed)}

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Hybrid retrieval returning top_k documents as
        {id, text, title, score} dicts.
        """
        k = top_k or self.config.top_k
        fetch_k = k * 3  # fetch more candidates before fusion

        bm25_hits = self._bm25.search(query, fetch_k)
        dense_hits = self._dense.search(query, fetch_k)

        bm25_norm = self._min_max_normalize(bm25_hits)
        dense_norm = self._min_max_normalize(dense_hits)

        all_indices = set(bm25_norm.keys()) | set(dense_norm.keys())
        combined: Dict[int, float] = {}
        for idx in all_indices:
            combined[idx] = (
                self.config.bm25_weight * bm25_norm.get(idx, 0.0)
                + self.config.dense_weight * dense_norm.get(idx, 0.0)
            )

        top_indices = sorted(combined, key=combined.get, reverse=True)[:k]

        if self.corpus is None:
            return [{'id': idx, 'score': combined[idx]} for idx in top_indices]

        return [
            {
                'id': self.corpus.ids[idx],
                'text': self.corpus.texts[idx],
                'title': self.corpus.titles[idx],
                'score': combined[idx],
            }
            for idx in top_indices
            if idx < len(self.corpus)
        ]

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[List[Dict]]:
        """
        Batch retrieval: encodes all queries at once for dense retrieval,
        then runs BM25 per-query.
        """
        k = top_k or self.config.top_k
        fetch_k = k * 3

        dense_batch = self._dense.search_batch(queries, fetch_k)
        results = []
        for query, dense_hits in zip(queries, dense_batch):
            bm25_hits = self._bm25.search(query, fetch_k)
            bm25_norm = self._min_max_normalize(bm25_hits)
            dense_norm = self._min_max_normalize(dense_hits)
            all_indices = set(bm25_norm.keys()) | set(dense_norm.keys())
            combined: Dict[int, float] = {
                idx: (
                    self.config.bm25_weight * bm25_norm.get(idx, 0.0)
                    + self.config.dense_weight * dense_norm.get(idx, 0.0)
                )
                for idx in all_indices
            }
            top_indices = sorted(combined, key=combined.get, reverse=True)[:k]
            if self.corpus is None:
                results.append([{'id': idx, 'score': combined[idx]} for idx in top_indices])
            else:
                results.append([
                    {
                        'id': self.corpus.ids[idx],
                        'text': self.corpus.texts[idx],
                        'title': self.corpus.titles[idx],
                        'score': combined[idx],
                    }
                    for idx in top_indices
                    if idx < len(self.corpus)
                ])
        return results


def build_index_from_jsonl(corpus_path: str, config: RetrieverConfig) -> HybridRetriever:
    """
    Convenience: load corpus from JSONL, build both indexes, save, and return retriever.
    """
    corpus = Corpus.from_jsonl(corpus_path)
    retriever = HybridRetriever(config)
    retriever.index(corpus)
    return retriever
