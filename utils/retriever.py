"""
Local retriever with BM25, ColBERTv2, hybrid BM25+ColBERTv2, and legacy SBERT modes.

Usage:
    from utils.retriever import Corpus, HybridRetriever, build_index_from_jsonl
    from config import RetrieverConfig

    cfg = RetrieverConfig(
        corpus_path="wiki_corpus.jsonl",
        index_dir="retriever_index/",
        retrieval_mode="hybrid",
        colbert_checkpoint="colbert-ir/colbertv2.0",
    )
    retriever = build_index_from_jsonl(cfg.corpus_path, cfg)
    results = retriever.retrieve("What is the capital of France?", top_k=5)
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RetrieverConfig


def _progress(iterable, enabled: bool = True, **kwargs):
    """tqdm wrapper with graceful fallback when tqdm is unavailable/disabled."""
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def read_jsonl(path: str, show_progress: bool = True, desc: str = 'Loading JSONL') -> List[Dict]:
    """Read a JSONL file into a list of dictionaries."""
    rows: List[Dict] = []
    with open(path, encoding='utf-8') as f:
        for line in _progress(f, enabled=show_progress, desc=desc, unit='lines'):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict], show_progress: bool = True) -> None:
    """Write dictionaries to a JSONL file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in _progress(rows, enabled=show_progress, desc='Writing retrieval output', unit='lines'):
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def get_query_from_record(record: Dict, query_key: str = 'question') -> str:
    """Extract query text from a JSONL record with common fallback keys."""
    candidate_keys = [query_key]
    for key in ('question', 'query', 'Question'):
        if key not in candidate_keys:
            candidate_keys.append(key)
    for key in candidate_keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ''


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

class Corpus:
    """
    In-memory corpus. Each document: {id, text, title?}.

    `text` may be either a string or a list of string fragments. In the latter
    case, fragments are concatenated into one passage/document by default.
    """

    def __init__(self, documents: List[Dict], show_progress: bool = True):
        self.documents = documents
        self.show_progress = show_progress
        self.texts: List[str] = [
            self._normalize_text(d['text'])
            for d in _progress(
                documents,
                enabled=show_progress,
                desc='Normalizing corpus text',
                unit='docs',
            )
        ]
        self.ids: List[str] = [str(d['id']) for d in documents]
        self.titles: List[str] = [d.get('title', '') for d in documents]

    @staticmethod
    def _normalize_text(text) -> str:
        """Convert corpus text from str or list[str] to a single passage string."""
        if isinstance(text, list):
            return ''.join(str(fragment) for fragment in text).strip()
        return str(text).strip()

    @classmethod
    def from_jsonl(cls, path: str, show_progress: bool = True) -> 'Corpus':
        """Load corpus from a JSONL file where each line is {id, text, title?}."""
        docs = []
        with open(path, encoding='utf-8') as f:
            for line in _progress(f, enabled=show_progress, desc='Loading corpus', unit='lines'):
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        return cls(docs, show_progress=show_progress)

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
        tokenized = [
            text.lower().split()
            for text in _progress(
                corpus.texts,
                enabled=getattr(self.config, 'show_progress', True),
                desc='Tokenizing corpus for BM25',
                unit='docs',
            )
        ]
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
            show_progress_bar=getattr(self.config, 'show_progress', True),
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
# ColBERT index
# ---------------------------------------------------------------------------

class ColBERTIndex:
    """
    ColBERTv2 retrieval wrapper using Stanford ColBERT's Indexer/Searcher.

    The collection is materialized as a TSV file with internal corpus indices as
    pids, so ColBERT hits can be mapped back to Corpus.ids/texts/titles.
    """

    def __init__(self, config: RetrieverConfig):
        self.config = config
        self._searcher = None
        self._collection_path: Optional[Path] = None

    def _root(self) -> Path:
        return Path(self.config.colbert_root or (Path(self.config.index_dir) / 'colbert'))

    def _collection_file(self, index_dir: Optional[str] = None) -> Path:
        return Path(index_dir or self.config.index_dir) / 'collection.tsv'

    @staticmethod
    def _write_collection(corpus: Corpus, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            iterator = enumerate(corpus.texts)
            for idx, text in _progress(
                iterator,
                enabled=getattr(corpus, 'show_progress', True),
                desc='Writing ColBERT collection',
                total=len(corpus.texts),
                unit='docs',
            ):
                clean_text = text.replace('\t', ' ').replace('\n', ' ').strip()
                f.write(f"{idx}\t{clean_text}\n")

    @staticmethod
    def _require_colbert() -> Tuple[object, object, object, object]:
        try:
            from colbert import Indexer, Searcher  # type: ignore
            from colbert.infra import Run, RunConfig, ColBERTConfig  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ColBERT is required for retrieval_mode='colbert' or 'hybrid'. "
                "Please install Stanford ColBERT in your retrieval environment."
            ) from exc
        return Indexer, Searcher, Run, (RunConfig, ColBERTConfig)

    def build(self, corpus: Corpus) -> None:
        """Build a ColBERT index for the corpus."""
        Indexer, _Searcher, Run, cfg_classes = self._require_colbert()
        RunConfig, ColBERTConfig = cfg_classes

        index_dir = Path(self.config.index_dir)
        collection_path = self._collection_file(str(index_dir))
        self._write_collection(corpus, collection_path)
        self._collection_path = collection_path

        root = self._root()
        root.mkdir(parents=True, exist_ok=True)

        if getattr(self.config, 'show_progress', True):
            print(f"Building ColBERT index '{self.config.colbert_index_name}' from {collection_path} ...")
        with Run().context(RunConfig(nranks=1, experiment='know3rag', root=str(root))):
            colbert_cfg = ColBERTConfig(
                doc_maxlen=self.config.colbert_doc_maxlen,
                nbits=self.config.colbert_nbits,
                kmeans_niters=self.config.colbert_kmeans_niters,
            )
            indexer = Indexer(checkpoint=self.config.colbert_checkpoint, config=colbert_cfg)
            indexer.index(
                name=self.config.colbert_index_name,
                collection=str(collection_path),
                overwrite=True,
            )
        if getattr(self.config, 'show_progress', True):
            print(f"Finished ColBERT index '{self.config.colbert_index_name}'.")

    def load(self, index_dir: Optional[str] = None) -> None:
        """Initialise a ColBERT Searcher from an existing index."""
        _Indexer, Searcher, Run, cfg_classes = self._require_colbert()
        RunConfig, _ColBERTConfig = cfg_classes

        collection_path = self._collection_file(index_dir)
        if not collection_path.exists():
            raise FileNotFoundError(
                f"ColBERT collection file not found: {collection_path}. "
                "Please build the ColBERT index first."
            )
        self._collection_path = collection_path

        root = self._root()
        with Run().context(RunConfig(experiment='know3rag', root=str(root))):
            self._searcher = Searcher(
                index=self.config.colbert_index_name,
                collection=str(collection_path),
            )

    @staticmethod
    def _ranking_to_hits(ranking: object, top_k: int) -> List[Tuple[int, float]]:
        """Convert ColBERT Ranking/search return values to [(pid, score), ...]."""
        if isinstance(ranking, tuple) and len(ranking) >= 3:
            pids, _ranks, scores = ranking[:3]
            return [(int(pid), float(score)) for pid, score in zip(pids, scores)][:top_k]

        if hasattr(ranking, 'todict'):
            data = ranking.todict()
            hits = []
            for pid, value in data.items():
                if isinstance(value, dict):
                    score = value.get('score', value.get('Score', 0.0))
                elif isinstance(value, (list, tuple)) and value:
                    score = value[-1]
                else:
                    score = 0.0
                hits.append((int(pid), float(score)))
            return hits[:top_k]

        if isinstance(ranking, Iterable):
            hits = []
            for item in ranking:
                if isinstance(item, dict):
                    pid = item.get('pid', item.get('docid', item.get('id')))
                    score = item.get('score', item.get('Score', 0.0))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    pid, score = item[0], item[-1]
                else:
                    continue
                hits.append((int(pid), float(score)))
            return hits[:top_k]

        raise RuntimeError(f"Unsupported ColBERT ranking format: {type(ranking)!r}")

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self._searcher is None:
            self.load()
        ranking = self._searcher.search(query, k=top_k)
        return self._ranking_to_hits(ranking, top_k)

    def search_batch(self, queries: List[str], top_k: int) -> List[List[Tuple[int, float]]]:
        return [
            self.search(query, top_k)
            for query in _progress(
                queries,
                enabled=getattr(self.config, 'show_progress', True),
                desc='ColBERT batch retrieval',
                unit='queries',
            )
        ]


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Retriever supporting BM25, ColBERTv2, hybrid BM25+ColBERTv2, and legacy SBERT.

    In hybrid mode:
        combined_score = bm25_weight * norm(bm25) + dense_weight * norm(colbert)
    """

    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.corpus: Optional[Corpus] = None
        self._bm25 = BM25Index(config)
        self._dense = self._make_dense_index()

    def _mode(self) -> str:
        return getattr(self.config, 'retrieval_mode', 'hybrid').lower()

    def _uses_bm25(self) -> bool:
        return self._mode() in {'bm25', 'hybrid'}

    def _uses_dense(self) -> bool:
        return self._mode() in {'colbert', 'hybrid', 'sbert'}

    def _make_dense_index(self):
        mode = self._mode()
        if mode in {'colbert', 'hybrid'}:
            return ColBERTIndex(self.config)
        if mode == 'sbert':
            return DenseIndex(self.config)
        if mode == 'bm25':
            return None
        raise ValueError(
            "Unsupported retrieval_mode '{}'. Expected one of: bm25, colbert, hybrid, sbert.".format(mode)
        )

    def index(self, corpus: Corpus) -> None:
        """Build indexes required by config.retrieval_mode and save to config.index_dir."""
        self.corpus = corpus
        index_dir = Path(self.config.index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        if self._uses_bm25():
            self._bm25.build(corpus)
            self._bm25.save(str(index_dir / 'bm25.pkl'))

        if self._uses_dense():
            self._dense.build(corpus)
            if self._mode() == 'sbert':
                self._dense.save(str(index_dir / 'dense.npy'))

        # Save corpus texts for retrieval output
        with open(str(index_dir / 'corpus.jsonl'), 'w', encoding='utf-8') as f:
            for doc in corpus.documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    def load_index(self, index_dir: Optional[str] = None) -> None:
        """Load pre-built indexes from disk."""
        d = Path(index_dir or self.config.index_dir)
        if self._uses_bm25():
            self._bm25.load(str(d / 'bm25.pkl'))
        if self._uses_dense():
            if self._mode() == 'sbert':
                self._dense.load(str(d / 'dense.npy'))
            else:
                self._dense.load(str(d))

        # Reload corpus for output
        corpus_path = d / 'corpus.jsonl'
        if corpus_path.exists():
            docs = []
            with open(str(corpus_path), encoding='utf-8') as f:
                for line in _progress(
                    f,
                    enabled=getattr(self.config, 'show_progress', True),
                    desc='Loading indexed corpus',
                    unit='lines',
                ):
                    line = line.strip()
                    if line:
                        docs.append(json.loads(line))
            self.corpus = Corpus(docs, show_progress=getattr(self.config, 'show_progress', True))

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

        mode = self._mode()
        if mode == 'bm25':
            return self._format_results(self._bm25.search(query, k))
        if mode in {'colbert', 'sbert'}:
            return self._format_results(self._dense.search(query, k))

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

        return self._format_results([(idx, combined[idx]) for idx in top_indices])

    def _format_results(self, hits: List[Tuple[int, float]]) -> List[Dict]:
        if self.corpus is None:
            return [{'id': idx, 'score': score} for idx, score in hits]
        return [
            {
                'id': self.corpus.ids[idx],
                'text': self.corpus.texts[idx],
                'title': self.corpus.titles[idx],
                'score': score,
            }
            for idx, score in hits
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

        mode = self._mode()
        if mode == 'bm25':
            return [
                self._format_results(self._bm25.search(query, k))
                for query in _progress(
                    queries,
                    enabled=getattr(self.config, 'show_progress', True),
                    desc='BM25 batch retrieval',
                    unit='queries',
                )
            ]
        if mode in {'colbert', 'sbert'}:
            return [self._format_results(hits) for hits in self._dense.search_batch(queries, k)]

        dense_batch = self._dense.search_batch(queries, fetch_k)
        results = []
        for query, dense_hits in _progress(
            zip(queries, dense_batch),
            enabled=getattr(self.config, 'show_progress', True),
            desc='Hybrid batch retrieval',
            total=len(queries),
            unit='queries',
        ):
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
            results.append(self._format_results([(idx, combined[idx]) for idx in top_indices]))
        return results


def build_index_from_jsonl(corpus_path: str, config: RetrieverConfig) -> HybridRetriever:
    """
    Convenience: load corpus from JSONL, build both indexes, save, and return retriever.
    """
    corpus = Corpus.from_jsonl(corpus_path, show_progress=getattr(config, 'show_progress', True))
    retriever = HybridRetriever(config)
    retriever.index(corpus)
    return retriever


def _build_cli_config(args) -> RetrieverConfig:
    return RetrieverConfig(
        corpus_path=getattr(args, 'corpus', None),
        index_dir=args.index_dir,
        retrieval_mode=args.mode,
        sbert_model_path=getattr(args, 'sbert_model_path', ''),
        colbert_checkpoint=getattr(args, 'colbert_checkpoint', 'colbert-ir/colbertv2.0'),
        colbert_index_name=getattr(args, 'colbert_index_name', 'know3rag_colbert'),
        colbert_root=getattr(args, 'colbert_root', ''),
        colbert_doc_maxlen=getattr(args, 'colbert_doc_maxlen', 180),
        colbert_nbits=getattr(args, 'colbert_nbits', 2),
        colbert_kmeans_niters=getattr(args, 'colbert_kmeans_niters', 4),
        bm25_weight=getattr(args, 'bm25_weight', 0.5),
        dense_weight=getattr(args, 'dense_weight', 0.5),
        top_k=getattr(args, 'top_k', 5),
        show_progress=not getattr(args, 'no_progress', False),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Know3RAG local retriever")
    subparsers = parser.add_subparsers(dest='command', required=True)

    def add_common_args(p):
        p.add_argument('--mode', choices=['bm25', 'colbert', 'hybrid', 'sbert'], default='hybrid')
        p.add_argument(
            '--passage-level', action='store_true', default=True,
            help='Treat each JSONL record as one passage; list-valued text is concatenated. Enabled by default.'
        )
        p.add_argument('--no-progress', action='store_true', help='Disable tqdm progress bars and progress messages')
        p.add_argument('--index-dir', default='retriever_index/')
        p.add_argument('--top-k', type=int, default=5)
        p.add_argument('--bm25-weight', type=float, default=0.5)
        p.add_argument('--dense-weight', type=float, default=0.5)
        p.add_argument('--sbert-model-path', default='')
        p.add_argument('--colbert-checkpoint', default='colbert-ir/colbertv2.0')
        p.add_argument('--colbert-index-name', default='know3rag_colbert')
        p.add_argument('--colbert-root', default='')
        p.add_argument('--colbert-doc-maxlen', type=int, default=180)
        p.add_argument('--colbert-nbits', type=int, default=2)
        p.add_argument('--colbert-kmeans-niters', type=int, default=4)

    index_parser = subparsers.add_parser('index', help='Build retriever index')
    add_common_args(index_parser)
    index_parser.add_argument('--corpus', required=True, help='Corpus JSONL path: {id, text, title?}')

    retrieve_parser = subparsers.add_parser('retrieve', help='Retrieve from an existing index')
    add_common_args(retrieve_parser)
    query_group = retrieve_parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument('--query', help='Single query string')
    query_group.add_argument('--query-file', help='Input JSONL file containing queries')
    retrieve_parser.add_argument('--output', help='Output JSONL path for --query-file results')
    retrieve_parser.add_argument('--query-key', default='question', help='Field name used as query in --query-file')
    retrieve_parser.add_argument(
        '--output-key', default='retrieved_passages',
        help='Field name used to store retrieval results in --query-file output'
    )

    args = parser.parse_args()
    cfg = _build_cli_config(args)

    if args.command == 'index':
        retriever = build_index_from_jsonl(args.corpus, cfg)
        print(f"Built {cfg.retrieval_mode} index under {cfg.index_dir}")
        return

    retriever = HybridRetriever(cfg)
    retriever.load_index()

    if args.query_file:
        if not args.output:
            raise ValueError("--output is required when using --query-file")
        rows = read_jsonl(
            args.query_file,
            show_progress=cfg.show_progress,
            desc='Loading query file',
        )
        queries = [get_query_from_record(row, args.query_key) for row in rows]
        results_batch = retriever.retrieve_batch(queries, top_k=args.top_k)
        output_rows = []
        for row, query, results in zip(rows, queries, results_batch):
            new_row = dict(row)
            new_row[args.output_key] = results if query else []
            output_rows.append(new_row)
        write_jsonl(args.output, output_rows, show_progress=cfg.show_progress)
        print(f"Wrote {len(output_rows)} retrieval records to {args.output}")
        return

    results = retriever.retrieve(args.query, top_k=args.top_k)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
