#!/usr/bin/env python3
"""
Baseline single-agent evaluation harness for HotpotQA and SQuAD v2.0.

This script queries an Ollama-hosted model (default: llama3.1:8b) for each
example, computes the requested metrics, and stores aggregated scores in JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
 
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from hotpot_evaluate import exact_match_score, f1_score, normalize_answer


# --------------------------------------------------------------------------- #
# Ollama client
# --------------------------------------------------------------------------- #

@dataclass
class OllamaConfig:
    model: str = "llama3.1:8b"
    temperature: float = 0.3
    num_retries: int = 3
    request_timeout: int = 300  # seconds
    host: str = "127.0.0.1"
    port: int = 11434
    options: Dict[str, Any] = field(default_factory=dict)
    verbose: bool = False


class OllamaClient:
    def __init__(self, config: OllamaConfig) -> None:
        self.config = config
        # Lazy import to provide a clean error if the package is missing.
        try:
            import ollama  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "The 'ollama' Python package is required. Install with: pip install ollama"
            ) from exc

        # Prefer the official client to set a custom host/port.
        base_url = f"http://{self.config.host}:{self.config.port}"
        try:
            from ollama import Client  # type: ignore

            # Not all versions support a timeout arg; avoid passing unknown kwargs.
            self._client = Client(host=base_url)
            self._use_module_level = False
        except Exception:
            # Fall back to module-level functions (uses default host env/localhost).
            import ollama as _ollama  # type: ignore

            self._ollama_module = _ollama
            self._use_module_level = True

    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Invoke Ollama using the official Python library with the specified prompt
        and parse a JSON object from the model response. Falls back to raw text
        if JSON parsing fails.
        """
        request_kwargs = {
            "model": self.config.model,
            "prompt": prompt,
            "options": self._build_options(),
            "stream": False,
        }
        if self.config.verbose:
            print("[ollama debug] request payload:", json.dumps(request_kwargs, indent=2))

        last_error: Optional[str] = None
        for attempt in range(1, self.config.num_retries + 1):
            try:
                if getattr(self, "_use_module_level", False):
                    payload = self._ollama_module.generate(**request_kwargs)  # type: ignore[attr-defined]
                else:
                    payload = self._client.generate(**request_kwargs)  # type: ignore[attr-defined]
                if self.config.verbose:
                    preview = json.dumps({k: payload.get(k) for k in ("model", "created_at", "total_duration") if k in payload}, indent=2)
                    print(f"[ollama debug] response meta:\n{preview}")
            except Exception as exc:
                last_error = f"ollama generate error: {exc}"
                time.sleep(1.5 * attempt)
                continue

            text = payload.get("response", "")
            parsed = _parse_json_response(text)
            if parsed is not None:
                return parsed
            return {"answer": text, "supporting_facts": []}

        raise RuntimeError(f"Ollama generation failed after {self.config.num_retries} attempts: {last_error}")

    def _build_options(self) -> Dict[str, Any]:
        options = {"temperature": self.config.temperature}
        options.update(self.config.options)
        return options


JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    match = JSON_PATTERN.search(text)
    if not match:
        return None
    try:
        candidate = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    if isinstance(candidate, dict) and "answer" in candidate:
        candidate.setdefault("supporting_facts", [])
        if not isinstance(candidate["supporting_facts"], list):
            candidate["supporting_facts"] = []
        # Ensure answer is a string to avoid attribute errors downstream.
        if not isinstance(candidate.get("answer"), str):
            candidate["answer"] = str(candidate.get("answer", ""))
        return candidate
    return None


# --------------------------------------------------------------------------- #
# Datasets
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
# Metric helpers
# --------------------------------------------------------------------------- #

def _best_over_ground_truths(prediction: str, ground_truths: Sequence[str]) -> Tuple[float, float, float, float]:
    """
    Return best (f1, precision, recall, em) against multiple reference answers.
    """
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
        log_precision_sum += weight * math.log(p_n)
    brevity_penalty = _brevity_penalty(len(pred_tokens), len(gold_tokens))
    return brevity_penalty * math.exp(log_precision_sum)


def _modified_precision(pred_tokens: Sequence[str], gold_tokens: Sequence[str], n: int) -> float:
    pred_ngrams = _ngram_counts(pred_tokens, n)
    gold_ngrams = _ngram_counts(gold_tokens, n)
    overlap = sum((pred_ngrams & gold_ngrams).values())
    total = sum(pred_ngrams.values())
    # Add-one smoothing.
    return (overlap + 1.0) / (total + 1.0)


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    from collections import Counter

    counts = Counter()
    for idx in range(len(tokens) - n + 1):
        counts[tuple(tokens[idx : idx + n])] += 1
    return counts


def _brevity_penalty(pred_len: int, gold_len: int) -> float:
    if pred_len == 0:
        return 0.0
    if pred_len > gold_len:
        return 1.0
    return math.exp(1 - (gold_len / pred_len))


# --------------------------------------------------------------------------- #
# Lightweight response cache
# --------------------------------------------------------------------------- #


class ResponseCache:
    def __init__(self, root: Optional[Path]) -> None:
        self.root = root
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Optional[Path]:
        if self.root is None:
            return None
        return self.root / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path_for(key)
        if path is None or not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        path = self._path_for(key)
        if path is None:
            return
        try:
            with path.open("w", encoding="utf-8") as fp:
                json.dump(value, fp, ensure_ascii=False)
        except Exception:
            pass


def _cache_key(model: str, prompt: str, options: Dict[str, Any], sample_tag: Optional[str] = None) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "options": options or {},
        "sample": sample_tag or "",
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _generate_with_cache(client: "OllamaClient", prompt: str, cache: ResponseCache, sample_tag: Optional[str]) -> Dict[str, Any]:
    key = _cache_key(client.config.model, prompt, client._build_options(), sample_tag)
    cached = cache.get(key) if cache is not None else None
    if cached is not None:
        return cached
    resp = client.generate(prompt)
    if cache is not None:
        cache.set(key, resp)
    return resp


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
# Prompt construction
# --------------------------------------------------------------------------- #

HOTPot_PROMPT_TEMPLATE = """You are an expert at answering multi-hop questions using the supplied context.
Provide a concise answer using only the given information.

Return a JSON object with:
  "answer": string (use "noanswer" if the question cannot be answered)
  "supporting_facts": list of strings citing the key sentences from the context

Context:
{context}

Question: {question}
"""

SQUAD_PROMPT_TEMPLATE = """You answer questions strictly based on the provided passage.
If the question cannot be answered, respond with the string "noanswer".

Return JSON with a single key "answer".

Passage:
{context}

Question: {question}
"""


def _format_hotpot_context(context_entries: Sequence[Sequence[Any]]) -> str:
    blocks = []
    for entry in context_entries:
        if len(entry) != 2:
            continue
        title, sentences = entry
        if not isinstance(sentences, list):
            continue
        paragraph = " ".join(sentence.strip() for sentence in sentences if isinstance(sentence, str))
        blocks.append(f"{title}: {paragraph}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Evaluation routines
# --------------------------------------------------------------------------- #


def evaluate_hotpotqa(
    client: OllamaClient,
    dataset: Iterable[Dict[str, Any]],
    predictions_dir: Path,
    *,
    samples: int = 1,
    cache: Optional[ResponseCache] = None,
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
    prediction_records = {}

    for example in dataset:
        qid = example.get("_id")
        question = example.get("question", "")
        answer = example.get("answer", "")
        context_text = _format_hotpot_context(example.get("context", []))

        prompt = HOTPot_PROMPT_TEMPLATE.format(context=context_text, question=question)
        responses: List[Dict[str, Any]] = []
        vote_counter: Dict[str, int] = {}
        for i in range(max(1, int(samples))):
            tag = f"sample_{i+1}" if samples and samples > 1 else None
            resp = _generate_with_cache(client, prompt, cache or ResponseCache(None), tag)
            responses.append(resp)
            norm_ans = normalize_answer(str(resp.get("answer", "")).strip())
            vote_counter[norm_ans] = vote_counter.get(norm_ans, 0) + 1

        # Choose majority normalized answer; tie-break by shortest string
        if responses:
            best_norm = sorted(vote_counter.items(), key=lambda kv: (-kv[1], len(kv[0])))[0][0]
            chosen = next((r for r in responses if normalize_answer(str(r.get("answer", "")).strip()) == best_norm), responses[0])
        else:
            chosen = {"answer": "", "supporting_facts": []}
        predicted_answer = str(chosen.get("answer", "")).strip()
        supporting = chosen.get("supporting_facts", [])

        f1, prec, rec, em = _best_over_ground_truths(predicted_answer, [answer])
        rouge = _rouge_l_score(predicted_answer, answer)
        bleu = _bleu_score(predicted_answer, answer)
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

        record: Dict[str, Any] = {"answer": predicted_answer, "supporting_facts": supporting, "raw_prompt": prompt}
        if samples and samples > 1:
            record["votes"] = vote_counter
        prediction_records[qid] = record

    return _finalize_metrics(metrics, prediction_records, predictions_dir / "hotpotqa_predictions.json")


def evaluate_squad_v2(
    client: OllamaClient,
    dataset: Iterable[Dict[str, Any]],
    predictions_dir: Path,
    *,
    samples: int = 1,
    cache: Optional[ResponseCache] = None,
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
    prediction_records = {}

    for example in dataset:
        qid = example.get("id")
        question = example.get("question", "")
        context = example.get("context", "")
        answers = example.get("answers", [])
        if example.get("is_impossible", False) and not answers:
            answers = [""]

        prompt = SQUAD_PROMPT_TEMPLATE.format(context=context, question=question)
        responses: List[Dict[str, Any]] = []
        vote_counter: Dict[str, int] = {}
        for i in range(max(1, int(samples))):
            tag = f"sample_{i+1}" if samples and samples > 1 else None
            resp = _generate_with_cache(client, prompt, cache or ResponseCache(None), tag)
            responses.append(resp)
            norm_ans = normalize_answer(str(resp.get("answer", "")).strip())
            vote_counter[norm_ans] = vote_counter.get(norm_ans, 0) + 1

        if responses:
            best_norm = sorted(vote_counter.items(), key=lambda kv: (-kv[1], len(kv[0])))[0][0]
            chosen = next((r for r in responses if normalize_answer(str(r.get("answer", "")).strip()) == best_norm), responses[0])
        else:
            chosen = {"answer": ""}
        predicted_answer = str(chosen.get("answer", "")).strip()

        f1, prec, rec, em = _best_over_ground_truths(predicted_answer, answers)
        rouge = max(_rouge_l_score(predicted_answer, gold) for gold in answers) if answers else 0.0
        bleu = max(_bleu_score(predicted_answer, gold) for gold in answers) if answers else 0.0

        metrics["count"] += 1
        metrics["em"] += em
        metrics["f1"] += f1
        metrics["precision"] += prec
        metrics["recall"] += rec
        metrics["rouge_l"] += rouge
        metrics["bleu"] += bleu

        record: Dict[str, Any] = {"answer": predicted_answer, "raw_prompt": prompt}
        if samples and samples > 1:
            record["votes"] = vote_counter
        prediction_records[qid] = record

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
    parser = argparse.ArgumentParser(description="Run baseline Ollama QA evaluation.")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model identifier to use.")
    parser.add_argument("--ollama-host", default="127.0.0.1", help="Ollama server host.")
    parser.add_argument("--ollama-port", type=int, default=11434, help="Ollama server port.")
    parser.add_argument("--hotpot-path", type=Path, required=True, help="Path to HotpotQA dev JSON.")
    parser.add_argument("--squad-path", type=Path, required=True, help="Path to SQuAD v2.0 dev JSON.")
    parser.add_argument("--output", type=Path, default=Path("baseline_metrics.json"), help="Where to store aggregated scores.")
    parser.add_argument("--predictions-dir", type=Path, default=Path("predictions"), help="Directory for per-example predictions.")
    parser.add_argument("--max-hotpot", type=int, default=None, help="Optional cap on number of HotpotQA examples.")
    parser.add_argument("--max-squad", type=int, default=None, help="Optional cap on number of SQuAD examples.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for Ollama.")
    parser.add_argument("--samples", type=int, default=1, help="Self-consistency samples per example (majority vote).")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional directory to cache model responses.")
    parser.add_argument(
        "--ollama-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional Ollama generation option (repeatable). Example: --ollama-option num_gpu=1",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Ollama request/response debug output.")
    return parser.parse_args(argv)


def _parse_ollama_options(items: Sequence[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --ollama-option '{item}', expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --ollama-option '{item}', empty key")
        parsed[key] = _coerce_value(value.strip())
    return parsed


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    extra_options = _parse_ollama_options(args.ollama_option)
    config = OllamaConfig(
        model=args.model,
        temperature=args.temperature,
        host=args.ollama_host,
        port=args.ollama_port,
        options=extra_options,
        verbose=getattr(args, "verbose", False),
    )
    client = OllamaClient(config)
    cache = ResponseCache(args.cache_dir) if getattr(args, "cache_dir", None) else ResponseCache(None)

    hotpot_examples = list(load_hotpotqa(args.hotpot_path, args.max_hotpot))
    squad_examples = list(load_squad_v2(args.squad_path, args.max_squad))

    hotpot_metrics = evaluate_hotpotqa(client, hotpot_examples, args.predictions_dir, samples=args.samples, cache=cache)
    squad_metrics = evaluate_squad_v2(client, squad_examples, args.predictions_dir, samples=args.samples, cache=cache)

    summary = {
        "model": args.model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "temperature": args.temperature,
            "samples": args.samples,
            "cache_dir": str(args.cache_dir) if args.cache_dir else None,
            "max_hotpot": args.max_hotpot,
            "max_squad": args.max_squad,
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
