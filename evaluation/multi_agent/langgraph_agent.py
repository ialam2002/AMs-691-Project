from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from langgraph.graph import END, StateGraph

try:
    from baseline_runner import OllamaClient
except ImportError:
    from evaluation.baseline_runner import OllamaClient

from .graph_store import WikipediaGraphStore
from .state import GraphState
from .wiki_connector import WikipediaConnector


class LangGraphAgent:
    """
    LangGraph multi-agent orchestrator powered by the shared OllamaClient so all
    model calls flow through the same llama3.1 runtime as the single-agent runner.
    """

    def __init__(
        self,
        client: OllamaClient,
        cache_dir: Path,
        retrieval_top_k: int = 8,
        max_reflection_loops: int = 2,
        wiki_search_limit: int = 3,
    ) -> None:
        self.client = client
        self.cache_dir = cache_dir
        self.retrieval_top_k = retrieval_top_k
        self.max_reflection_loops = max_reflection_loops
        self.wiki_search_limit = wiki_search_limit

        self.connector = WikipediaConnector(cache_dir=cache_dir / "pages")
        self.graph_store = WikipediaGraphStore(cache_dir=cache_dir / "graph")

        self.workflow = self._build_graph()

    def answer_hotpot(self, example: Dict[str, Any]) -> Dict[str, Any]:
        self.graph_store.reset()
        dataset_context = example.get("context", [])
        if dataset_context:
            self.graph_store.ingest_dataset_context(dataset_context)
        initial_state: GraphState = {
            "question": example.get("question", ""),
            "dataset_context": dataset_context,
            "sub_questions": [],
            "entity_hints": [],
            "wiki_pages": [],
            "context_snippets": [],
            "draft_answer": "",
            "supporting_facts": [],
            "feedback": "",
            "is_final": False,
            "iteration": 0,
            "processed_hints": [],
        }
        final_state: GraphState = self.workflow.invoke(initial_state)
        label_source = example.get("_id") or example.get("id") or example.get("question", "")
        self.graph_store.flush(self._slugify(label_source))
        selected = [
            (snippet.get("title", ""), snippet.get("sent_idx", -1), snippet.get("text", ""))
            for snippet in final_state.get("context_snippets", [])[: self.retrieval_top_k]
        ]
        return {
            "answer": final_state.get("draft_answer", ""),
            "supporting_facts": final_state.get("supporting_facts", []),
            "_selected": selected,
        }

    # --------------------------------------------------------------------- #
    # LangGraph construction
    # --------------------------------------------------------------------- #

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("wiki_search", self._wiki_search_node)
        graph.add_node("graph_retrieve", self._graph_retrieve_node)
        graph.add_node("synthesizer", self._synthesizer_node)
        graph.add_node("reflection", self._reflection_node)

        graph.set_entry_point("planner")
        graph.add_edge("planner", "wiki_search")
        graph.add_edge("wiki_search", "graph_retrieve")
        graph.add_edge("graph_retrieve", "synthesizer")
        graph.add_edge("synthesizer", "reflection")
        graph.add_conditional_edges("reflection", self._reflection_router, {"search": "wiki_search", "final": END})
        return graph.compile()

    # --------------------------------------------------------------------- #
    # Graph nodes
    # --------------------------------------------------------------------- #

    def _planner_node(self, state: GraphState) -> GraphState:
        if state.get("sub_questions") and state.get("entity_hints"):
            return state
        prompt = (
            "You are a planning specialist for multi-hop QA.\n"
            "Break the question into reasoning steps and Wikipedia entities to retrieve.\n"
            "Return JSON with keys:\n"
            '  "answer": short plan overview,\n'
            '  "sub_questions": ordered list of granular questions,\n'
            '  "entity_hints": list of entities or page titles to search.\n'
            f"Question: {state.get('question','')}\n"
        )
        parsed = self._invoke_model(prompt)
        sub_questions = self._coerce_str_list(parsed.get("sub_questions"))
        if not sub_questions:
            sub_questions = [state.get("question", "")]
        entity_hints = self._coerce_str_list(parsed.get("entity_hints"))
        if not entity_hints:
            entity_hints = list(sub_questions)
        updated = dict(state)
        updated["sub_questions"] = sub_questions
        updated["entity_hints"] = entity_hints
        return updated

    def _wiki_search_node(self, state: GraphState) -> GraphState:
        hints = state.get("entity_hints", [])[: self.wiki_search_limit]
        if not hints:
            fallback_text = " ".join(state.get("sub_questions", []) or [state.get("question", "")])
            fallback = WikipediaGraphStore.extract_entities(fallback_text)
            if fallback:
                hints = fallback[: self.wiki_search_limit]
                state = self._update_state_hints(state, fallback)
        if not hints:
            hints = self._keyword_fallback(state.get("question", ""))
        known_titles = {page.get("title", "").lower() for page in state.get("wiki_pages", [])}
        pages = list(state.get("wiki_pages", []))
        processed = set(h.lower() for h in state.get("processed_hints", []))
        new_processed = list(state.get("processed_hints", []))
        for hint in hints:
            if hint.lower() in processed:
                continue
            results = self.connector.search(hint, limit=2)
            for item in results:
                title = (item.get("title") or item.get("key") or "").strip()
                if not title or title.lower() in known_titles:
                    continue
                page = self.connector.fetch_page(title)
                if not page or not page.get("extract"):
                    continue
                self.graph_store.ingest_page(page["title"], page["extract"], source="wikipedia")
                record = {
                    "title": page["title"],
                    "pageid": page.get("pageid"),
                    "snippet": item.get("excerpt") if isinstance(item, dict) else "",
                }
                pages.append(record)
                known_titles.add(title.lower())
            processed.add(hint.lower())
            new_processed.append(hint)
        updated = dict(state)
        updated["wiki_pages"] = pages
        updated["processed_hints"] = new_processed
        updated["entity_hints"] = self._merge_hints(state.get("entity_hints", []), hints)
        return updated

    def _graph_retrieve_node(self, state: GraphState) -> GraphState:
        iteration = int(state.get("iteration", 0)) + 1
        query_texts: List[str] = [state.get("question", "")]
        query_texts.extend(state.get("sub_questions", []))
        snippets = self.graph_store.collect_snippets(
            query_texts=query_texts,
            entity_hints=state.get("entity_hints", []),
            top_k=self.retrieval_top_k,
            feedback=state.get("feedback", ""),
        )
        combined = {snip["id"]: snip for snip in state.get("context_snippets", [])}
        for snippet in snippets:
            combined[snippet["id"]] = snippet
        ordered = sorted(combined.values(), key=lambda item: item["score"], reverse=True)[: self.retrieval_top_k]
        updated = dict(state)
        updated["context_snippets"] = ordered
        updated["iteration"] = iteration
        updated["feedback"] = ""
        return updated

    def _synthesizer_node(self, state: GraphState) -> GraphState:
        snippets = state.get("context_snippets", [])
        if not snippets:
            return state
        lines = []
        for idx, snippet in enumerate(snippets, start=1):
            lines.append(
                f"[{idx}] ({snippet.get('source')}) {snippet.get('title')}#{snippet.get('sent_idx')}: {snippet.get('text')}"
            )
        context_block = "\n".join(lines)
        prompt = (
            "You are a careful HotpotQA assistant that must cite evidence.\n"
            "Use ONLY the provided snippets. Return JSON with keys:\n"
            '  "answer": concise response,\n'
            '  "supporting_facts": list of strings referencing snippet numbers and rationale.\n'
            f"\nQuestion: {state.get('question','')}\n"
            f"Context:\n{context_block}\n"
        )
        parsed = self._invoke_model(prompt)
        answer = parsed.get("answer") or ""
        supporting = parsed.get("supporting_facts") or []
        if isinstance(supporting, list):
            supporting_facts = [str(item) for item in supporting if str(item).strip()]
        else:
            supporting_facts = [str(supporting)]
        updated = dict(state)
        updated["draft_answer"] = answer.strip()
        updated["supporting_facts"] = supporting_facts
        return updated

    def _reflection_node(self, state: GraphState) -> GraphState:
        if state.get("iteration", 0) >= self.max_reflection_loops + 1:
            updated = dict(state)
            updated["is_final"] = True
            updated["feedback"] = ""
            return updated
        context_summary = "\n".join(snippet.get("text", "") for snippet in state.get("context_snippets", []))
        prompt = (
            "You are a strict QA auditor verifying that the answer is grounded in the evidence.\n"
            "Return JSON with keys:\n"
            '  "answer": repeat either FINAL or RETRY (for compatibility),\n'
            '  "status": same value (\"final\" or \"retry\"),\n'
            '  "feedback": if retry, describe what additional info is needed.\n'
            f"Question: {state.get('question','')}\n"
            f"Answer: {state.get('draft_answer','')}\n"
            f"Supporting facts: {state.get('supporting_facts',[])}\n"
            f"Evidence:\n{context_summary}\n"
        )
        parsed = self._invoke_model(prompt)
        status = parsed.get("status", "").strip().lower()
        feedback = parsed.get("feedback", "")
        updated = dict(state)
        updated["is_final"] = status == "final"
        updated["feedback"] = feedback if status != "final" else ""
        if not updated["is_final"]:
            updated["supporting_facts"] = state.get("supporting_facts", [])
            new_hints = WikipediaGraphStore.extract_entities(feedback or "")
            updated["entity_hints"] = self._merge_hints(state.get("entity_hints", []), new_hints)
        return updated

    def _reflection_router(self, state: GraphState) -> str:
        return "final" if state.get("is_final") else "search"

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _invoke_model(self, prompt: str) -> Dict[str, Any]:
        response = self.client.generate(prompt)
        if isinstance(response, dict):
            return response
        return {"answer": str(response)}

    def _coerce_str_list(self, value: Any) -> List[str]:
        items: List[str] = []
        if value is None:
            return items
        if isinstance(value, str):
            for part in value.replace("\r", "\n").split("\n"):
                part = part.strip()
                if part:
                    items.append(part)
            return items
        if isinstance(value, dict):
            for key in ("question", "step", "text", "value", "entity", "answer"):
                if key in value:
                    items.extend(self._coerce_str_list(value[key]))
            if not items:
                items.extend(self._coerce_str_list(" ".join(str(v) for v in value.values())))
            return items
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            for entry in value:
                items.extend(self._coerce_str_list(entry))
            return items
        text = str(value).strip()
        if text:
            items.append(text)
        return items

    def _merge_hints(self, existing: Sequence[str], new_hints: Sequence[str], max_items: int = 12) -> List[str]:
        merged: List[str] = []
        seen = set()
        for item in list(existing) + list(new_hints):
            if not item:
                continue
            normalized = item.strip()
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
            if len(merged) >= max_items:
                break
        return merged

    def _update_state_hints(self, state: GraphState, new_hints: Sequence[str]) -> GraphState:
        updated = dict(state)
        updated["entity_hints"] = self._merge_hints(state.get("entity_hints", []), new_hints)
        return updated

    def _slugify(self, text: str) -> str:
        if not text:
            return hashlib.sha1(b"default").hexdigest()[:10]
        slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
        if not slug:
            slug = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:10]
        return slug[:60]

    def _keyword_fallback(self, text: str) -> List[str]:
        tokens = [tok for tok in re.split(r"[^A-Za-z0-9]+", text or "") if len(tok) > 4]
        if not tokens:
            return []
        ranked = sorted(tokens, key=lambda t: (-len(t), t))
        return ranked[: self.wiki_search_limit]
