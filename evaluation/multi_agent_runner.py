#!/usr/bin/env python3
"""
Multi-agent evaluation harness (skeleton) with planner → retriever →
synthesizer → verifier roles. Designed to be plug-compatible with the
baseline runner outputs and metrics for HotpotQA and SQuAD v2.0.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from hotpot_evaluate import exact_match_score, f1_score, normalize_answer

# Reuse the Ollama client from baseline runner via package import
try:
    from baseline_runner import OllamaClient, OllamaConfig
    from multi_agent import LangGraphAgent
except ImportError:  # When executed via `python -m evaluation.multi_agent_runner`
    from evaluation.baseline_runner import OllamaClient, OllamaConfig
    from evaluation.multi_agent import LangGraphAgent


# --------------------------------------------------------------------------- #
# Dataset loaders (duplicated to avoid cross-file coupling)
# --------------------------------------------------------------------------- #


def load_hotpotqa(path: Path, limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    for idx, example in enumerate(data):
        if limit is not None and idx >= limit:
            break
        yield example


def load_squad_v2(path: Path, limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    count = 0
    for article in payload.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                if limit is not None and count >= limit:
                    return
                count += 1
                yield {
                    "id": qa.get("id"),
                    "question": qa.get("question", ""),
                    "context": context,
                    "answers": [ans.get("text", "") for ans in qa.get("answers", [])],
                    "is_impossible": qa.get("is_impossible", False),
                }


# --------------------------------------------------------------------------- #
# Metrics (mirrors baseline to stay comparable)
# --------------------------------------------------------------------------- #


def _best_over_ground_truths(prediction: str, ground_truths: Sequence[str]) -> Tuple[float, float, float, float]:
    if not ground_truths:
        ground_truths = [""]
    best = (0.0, 0.0, 0.0, 0.0)
    for gold in ground_truths:
        em = float(exact_match_score(prediction, gold))
        f1, prec, recall = f1_score(prediction, gold)
        if f1 > best[0]:
            best = (f1, prec, recall, em)
    return best


def _rouge_l_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = _longest_common_subsequence(pred_tokens, gold_tokens)
    prec = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if prec + recall == 0:
        return 0.0
    return 2 * prec * recall / (prec + recall)


def _longest_common_subsequence(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> int:
    len_b = len(tokens_b)
    dp = [0] * (len_b + 1)
    for token_a in tokens_a:
        prev = 0
        for idx, token_b in enumerate(tokens_b, start=1):
            temp = dp[idx]
            if token_a == token_b:
                dp[idx] = prev + 1
            else:
                dp[idx] = max(dp[idx], dp[idx - 1])
            prev = temp
    return dp[-1]


def _bleu_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    weights = [0.25, 0.25, 0.25, 0.25]
    log_precision_sum = 0.0
    for n, weight in zip(range(1, 5), weights):
        p_n = _modified_precision(pred_tokens, gold_tokens, n)
        log_precision_sum += weight * (math.log(p_n) if p_n > 0 else -9999)
    brevity_penalty = _brevity_penalty(len(pred_tokens), len(gold_tokens))
    return brevity_penalty * math.exp(log_precision_sum)


def _modified_precision(pred_tokens: Sequence[str], gold_tokens: Sequence[str], n: int) -> float:
    from collections import Counter

    def ngram_counts(tokens: Sequence[str], n: int) -> Counter:
        counts = Counter()
        for i in range(len(tokens) - n + 1):
            counts[tuple(tokens[i : i + n])] += 1
        return counts

    pred_ngrams = ngram_counts(pred_tokens, n)
    gold_ngrams = ngram_counts(gold_tokens, n)
    overlap = sum((pred_ngrams & gold_ngrams).values())
    total = sum(pred_ngrams.values())
    return (overlap + 1.0) / (total + 1.0)


def _brevity_penalty(pred_len: int, gold_len: int) -> float:
    if pred_len == 0:
        return 0.0
    if pred_len > gold_len:
        return 1.0
    return math.exp(1 - (gold_len / pred_len))


def _supporting_fact_metrics(predicted: Sequence[str], gold: Sequence[Sequence[str]]) -> Tuple[float, float, float]:
    pred_norm = {normalize_answer(item) for item in predicted if isinstance(item, str)}
    gold_norm = {
        normalize_answer(f"{title} {sentence}") for title, sentence in gold if isinstance(title, str) and isinstance(sentence, str)
    }
    if not pred_norm and not gold_norm:
        return 1.0, 1.0, 1.0
    tp = len(pred_norm & gold_norm)
    fp = len(pred_norm - gold_norm)
    fn = len(gold_norm - pred_norm)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0, precision, recall
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


# --------------------------------------------------------------------------- #
# Multi-Agent Pipeline (skeleton)
# --------------------------------------------------------------------------- #


class MultiAgentPipeline:
    def __init__(
        self,
        client: OllamaClient,
        top_k_sentences: int = 8,
        wiki_cache_dir: Optional[Path] = None,
        reflection_loops: int = 2,
        wiki_search_limit: int = 3,
    ) -> None:
        self.client = client
        self.top_k = top_k_sentences
        cache_dir = wiki_cache_dir or Path("results/wiki_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.langgraph_agent = LangGraphAgent(
            client=client,
            cache_dir=cache_dir,
            retrieval_top_k=top_k_sentences,
            max_reflection_loops=reflection_loops,
            wiki_search_limit=wiki_search_limit,
        )

    def run_hotpot(self, example: Dict[str, Any]) -> Dict[str, Any]:
        return self.langgraph_agent.answer_hotpot(example)

    def run_squad(self, example: Dict[str, Any]) -> Dict[str, Any]:
        question = example.get("question", "")
        context = example.get("context", "")
        prompt = (
            "Answer strictly from the passage. If not answerable, respond 'noanswer'.\n"
            'Return JSON with key "answer".\n\n'
            f"Passage:\n{context}\n\nQuestion: {question}\n"
        )
        return self.client.generate(prompt)


# --------------------------------------------------------------------------- #
# Evaluation routines (mirroring baseline)
# --------------------------------------------------------------------------- #


def evaluate_hotpotqa(
    pipeline: MultiAgentPipeline,
    dataset: Iterable[Dict[str, Any]],
    predictions_dir: Path,
) -> Dict[str, Any]:
    metrics = {
        "count": 0,
        "em": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "rouge_l": 0.0,
        "bleu": 0.0,
        "supporting_fact_f1": 0.0,
        "supporting_fact_precision": 0.0,
        "supporting_fact_recall": 0.0,
    }
    prediction_records: Dict[str, Any] = {}

    for example in dataset:
        qid = example.get("_id")
        answer_gold = example.get("answer", "")

        result = pipeline.run_hotpot(example)
        predicted_answer = str(result.get("answer", "")).strip()
        supporting = result.get("supporting_facts", [])

        f1, prec, rec, em = _best_over_ground_truths(predicted_answer, [answer_gold])
        rouge = _rouge_l_score(predicted_answer, answer_gold)
        bleu = _bleu_score(predicted_answer, answer_gold)
        sp_f1, sp_prec, sp_rec = _supporting_fact_metrics(supporting, example.get("supporting_facts", []))

        metrics["count"] += 1
        metrics["em"] += em
        metrics["f1"] += f1
        metrics["precision"] += prec
        metrics["recall"] += rec
        metrics["rouge_l"] += rouge
        metrics["bleu"] += bleu
        metrics["supporting_fact_f1"] += sp_f1
        metrics["supporting_fact_precision"] += sp_prec
        metrics["supporting_fact_recall"] += sp_rec

        prediction_records[qid] = {
            "answer": predicted_answer,
            "supporting_facts": supporting,
            "selected_ctx": [f"{t}|{i}: {s}" for (t, i, s) in result.get("_selected", [])],
        }

    return _finalize_metrics(metrics, prediction_records, predictions_dir / "hotpotqa_predictions.json")


def evaluate_squad_v2(
    pipeline: MultiAgentPipeline,
    dataset: Iterable[Dict[str, Any]],
    predictions_dir: Path,
) -> Dict[str, Any]:
    metrics = {
        "count": 0,
        "em": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "rouge_l": 0.0,
        "bleu": 0.0,
    }
    prediction_records: Dict[str, Any] = {}

    for example in dataset:
        qid = example.get("id")
        answers = example.get("answers", [])
        if example.get("is_impossible", False) and not answers:
            answers = [""]

        result = pipeline.run_squad(example)
        predicted_answer = str(result.get("answer", "")).strip()

        f1, prec, rec, em = _best_over_ground_truths(predicted_answer, answers)
        rouge = max((_rouge_l_score(predicted_answer, g) for g in answers), default=0.0)
        bleu = max((_bleu_score(predicted_answer, g) for g in answers), default=0.0)

        metrics["count"] += 1
        metrics["em"] += em
        metrics["f1"] += f1
        metrics["precision"] += prec
        metrics["recall"] += rec
        metrics["rouge_l"] += rouge
        metrics["bleu"] += bleu

        prediction_records[qid] = {"answer": predicted_answer}

    return _finalize_metrics(metrics, prediction_records, predictions_dir / "squad_v2_predictions.json")


def _finalize_metrics(
    metrics: Dict[str, float],
    prediction_records: Dict[str, Any],
    prediction_path: Path,
) -> Dict[str, Any]:
    count = metrics.get("count", 0) or 1
    averaged = {key: value / count for key, value in metrics.items() if key != "count"}
    averaged["count"] = metrics.get("count", 0)

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as fp:
        json.dump(prediction_records, fp, indent=2)

    averaged["prediction_file"] = str(prediction_path)
    return averaged


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-agent (skeleton) QA evaluation.")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model identifier to use.")
    parser.add_argument("--ollama-host", default="127.0.0.1", help="Ollama server host.")
    parser.add_argument("--ollama-port", type=int, default=11434, help="Ollama server port.")
    parser.add_argument("--hotpot-path", type=Path, required=True, help="Path to HotpotQA dev JSON.")
    parser.add_argument("--squad-path", type=Path, required=True, help="Path to SQuAD v2.0 dev JSON.")
    parser.add_argument("--output", type=Path, default=Path("multi_agent_metrics.json"), help="Where to store aggregated scores.")
    parser.add_argument("--predictions-dir", type=Path, default=Path("predictions_multi"), help="Directory for per-example predictions.")
    parser.add_argument("--max-hotpot", type=int, default=None, help="Optional cap on number of HotpotQA examples.")
    parser.add_argument("--max-squad", type=int, default=None, help="Optional cap on number of SQuAD examples.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for Ollama.")
    parser.add_argument("--top-k", type=int, default=8, help="Top sentences to retrieve from GraphRAG.")
    parser.add_argument("--wiki-cache", type=Path, default=Path("results/wiki_cache"), help="Directory for cached Wikipedia responses.")
    parser.add_argument("--reflection-loops", type=int, default=2, help="Maximum number of reflection retries before stopping.")
    parser.add_argument("--wiki-search-limit", type=int, default=3, help="Number of entity hints to fan out to Wikipedia search.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Ollama request/response debug output.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = OllamaConfig(
        model=args.model,
        temperature=args.temperature,
        host=args.ollama_host,
        port=args.ollama_port,
        options={},
        verbose=getattr(args, "verbose", False),
    )
    client = OllamaClient(config)
    pipeline = MultiAgentPipeline(
        client,
        top_k_sentences=args.top_k,
        wiki_cache_dir=args.wiki_cache,
        reflection_loops=args.reflection_loops,
        wiki_search_limit=args.wiki_search_limit,
    )

    hotpot_examples = list(load_hotpotqa(args.hotpot_path, args.max_hotpot))
    squad_examples = list(load_squad_v2(args.squad_path, args.max_squad))

    hotpot_metrics = evaluate_hotpotqa(pipeline, hotpot_examples, args.predictions_dir)
    squad_metrics = evaluate_squad_v2(pipeline, squad_examples, args.predictions_dir)

    summary = {
        "model": args.model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_hotpot": args.max_hotpot,
            "max_squad": args.max_squad,
            "reflection_loops": args.reflection_loops,
            "wiki_search_limit": args.wiki_search_limit,
            "wiki_cache": str(args.wiki_cache),
        },
        "datasets": {
            "hotpotqa": hotpot_metrics,
            "squad_v2": squad_metrics,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"Saved aggregated metrics to {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
