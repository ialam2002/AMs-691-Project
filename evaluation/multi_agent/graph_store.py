from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None

from hotpot_evaluate import normalize_answer


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ENTITY_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9'-]{2,}(?:\s+[A-Z][A-Za-z0-9'-]{2,})*)\b")


@dataclass
class SentenceRecord:
    sent_id: str
    title: str
    sent_idx: int
    text: str
    source: str
    tokens: Set[str]
    entities: List[str]


class WikipediaGraphStore:
    """
    Maintains a lightweight entity graph + sentence index for GraphRAG retrieval.
    Persists snapshots under the configured cache directory for inspection.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self) -> None:
        self.sentences: Dict[str, SentenceRecord] = {}
        self.entity_to_sentences: Dict[str, Set[str]] = defaultdict(set)
        self.entity_graph = nx.Graph() if nx else None
        self.token_df: Dict[str, int] = defaultdict(int)
        self._dirty = False

    def ingest_page(self, title: str, text: str, source: str = "wikipedia") -> None:
        if not text:
            return
        sentences = self._split_sentences(text)
        for idx, sentence in enumerate(sentences):
            sent_id = f"{title}:{idx}"
            tokens = set(self._tokenize(sentence))
            if not tokens:
                continue
            entities = self.extract_entities(sentence)
            record = SentenceRecord(
                sent_id=sent_id,
                title=title,
                sent_idx=idx,
                text=sentence.strip(),
                source=source,
                tokens=tokens,
                entities=entities,
            )
            self.sentences[sent_id] = record
            for token in tokens:
                self.token_df[token] += 1
            for entity in entities or [title]:
                key = entity.lower()
                self.entity_to_sentences[key].add(sent_id)
            if self.entity_graph is not None:
                self._update_graph(entities or [title])
            self._dirty = True

    def ingest_dataset_context(self, context_entries: Sequence[Sequence]) -> None:
        for entry in context_entries:
            if len(entry) != 2:
                continue
            title, sentences = entry
            if not isinstance(title, str) or not isinstance(sentences, list):
                continue
            joined = " ".join(sentences)
            self.ingest_page(title, joined, source="dataset")

    def collect_snippets(
        self,
        query_texts: Sequence[str],
        entity_hints: Sequence[str],
        top_k: int,
        feedback: str = "",
    ) -> List[Dict[str, object]]:
        if not self.sentences:
            return []
        query_tokens = set()
        for text in query_texts:
            query_tokens |= set(self._tokenize(text))
        query_tokens |= set(self._tokenize(feedback))

        candidate_ids: Set[str] = set()
        normalized_hints = [hint.lower() for hint in entity_hints if hint]
        for hint in normalized_hints:
            candidate_ids |= set(self.entity_to_sentences.get(hint, []))
            for neighbor in self._neighbor_entities(hint):
                candidate_ids |= set(self.entity_to_sentences.get(neighbor, []))
        if not candidate_ids:
            candidate_ids = set(self.sentences.keys())

        scored: List[tuple[float, SentenceRecord]] = []
        doc_count = max(len(self.sentences), 1)
        for sent_id in candidate_ids:
            record = self.sentences[sent_id]
            overlap = 0.0
            for token in query_tokens:
                if token in record.tokens:
                    df = self.token_df.get(token, 1)
                    idf = math.log((1 + doc_count) / (1 + df)) + 1.0
                    overlap += idf
            if overlap == 0 and not normalized_hints:
                overlap = sum(1 for token in record.tokens if token in query_tokens)
            entity_bonus = sum(2.5 for ent in record.entities if ent.lower() in normalized_hints)
            length_penalty = math.log(len(record.tokens) + 1)
            source_bonus = 0.6 if record.source == "dataset" else 0.3
            score = overlap + entity_bonus + source_bonus + (1.0 / length_penalty)
            scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        unique: Dict[str, Dict[str, object]] = {}
        for score, record in scored:
            if len(unique) >= top_k:
                break
            payload = {
                "id": record.sent_id,
                "title": record.title,
                "sent_idx": record.sent_idx,
                "text": record.text,
                "source": record.source,
                "score": round(score, 4),
            }
            unique[record.sent_id] = payload
        return list(unique.values())

    def flush(self, label: Optional[str] = None) -> None:
        if not self._dirty:
            return
        snapshot = {
            "sentences": [
                {
                    "id": record.sent_id,
                    "title": record.title,
                    "sent_idx": record.sent_idx,
                    "text": record.text,
                    "source": record.source,
                    "entities": record.entities,
                }
                for record in self.sentences.values()
            ],
            "entity_index": {entity: sorted(list(sent_ids)) for entity, sent_ids in self.entity_to_sentences.items()},
            "edges": self._serialize_edges(),
        }
        graph_path = self.cache_dir / "graph_snapshot.json"
        with graph_path.open("w", encoding="utf-8") as fp:
            json.dump(snapshot, fp, ensure_ascii=False, indent=2)
        if label:
            labeled_path = self.cache_dir / f"{label}.json"
            with labeled_path.open("w", encoding="utf-8") as fp:
                json.dump(snapshot, fp, ensure_ascii=False, indent=2)
        self._dirty = False

    def _serialize_edges(self) -> List[Dict[str, object]]:
        edges: List[Dict[str, object]] = []
        if self.entity_graph is None:
            return edges
        for src, dst, data in self.entity_graph.edges(data=True):
            edges.append({"src": src, "dst": dst, "weight": data.get("weight", 1)})
        return edges

    def _split_sentences(self, text: str) -> List[str]:
        return [segment.strip() for segment in _SENTENCE_SPLIT.split(text) if segment.strip()]

    @staticmethod
    def extract_entities(sentence: str) -> List[str]:
        return list({match.group(1).strip() for match in _ENTITY_PATTERN.finditer(sentence)})

    def _tokenize(self, sentence: str) -> List[str]:
        normalized = normalize_answer(sentence)
        return [token for token in normalized.split() if token]

    def _neighbor_entities(self, entity: str) -> Iterable[str]:
        if self.entity_graph is None or entity not in self.entity_graph:
            return []
        reached = set()
        queue = [(entity, 0)]
        while queue:
            current, depth = queue.pop(0)
            if depth >= 2:
                continue
            for neighbor in self.entity_graph.neighbors(current):
                if neighbor in reached:
                    continue
                reached.add(neighbor)
                queue.append((neighbor, depth + 1))
        return reached

    def _update_graph(self, entities: Sequence[str]) -> None:
        if self.entity_graph is None:
            return
        lowered = [ent.lower() for ent in entities if ent]
        for i, src in enumerate(lowered):
            if not src:
                continue
            if src not in self.entity_graph:
                self.entity_graph.add_node(src)
            for dst in lowered[i + 1 :]:
                if not dst:
                    continue
                if dst not in self.entity_graph:
                    self.entity_graph.add_node(dst)
                if not self.entity_graph.has_edge(src, dst):
                    self.entity_graph.add_edge(src, dst, weight=1)
                else:
                    self.entity_graph[src][dst]["weight"] += 1
