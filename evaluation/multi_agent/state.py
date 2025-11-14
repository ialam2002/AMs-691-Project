from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class GraphState(TypedDict, total=False):
    """
    Shared LangGraph state that flows between planner, retriever, synthesizer,
    and reflection agents.
    """

    question: str
    dataset_context: List[List[Any]]
    sub_questions: List[str]
    entity_hints: List[str]
    wiki_pages: List[Dict[str, Any]]
    context_snippets: List[Dict[str, Any]]
    draft_answer: str
    supporting_facts: List[str]
    feedback: str
    is_final: bool
    iteration: int
    processed_hints: List[str]
