"""
AutoDrive RAG v2.0 — RAG Evaluation Metrics
Implements standard QA evaluation metrics following the SQuAD paper:
  - Exact Match (EM)
  - F1 Score (token-level)
  - Answer Recall
  - Retrieval metrics: Precision@K, Recall@K, NDCG@K, MRR
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Optional


def _normalize(text: str) -> str:
    """Lower-case, strip punctuation/articles/extra whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _get_tokens(text: str) -> list[str]:
    return _normalize(text).split()


# ── Answer-level metrics (SQuAD-style) ──────────────────────────────


def exact_match(prediction: str, reference: str) -> float:
    """Case-insensitive exact match. Returns 1.0 or 0.0."""
    return float(_normalize(prediction) == _normalize(reference))


def f1_score(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference."""
    pred_tokens = _get_tokens(prediction)
    ref_tokens = _get_tokens(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_recall(prediction: str, reference: str) -> float:
    """Fraction of reference tokens present in prediction."""
    pred_tokens = set(_get_tokens(prediction))
    ref_tokens = _get_tokens(reference)

    if not ref_tokens:
        return 1.0

    hits = sum(1 for t in ref_tokens if t in pred_tokens)
    return hits / len(ref_tokens)


def best_score_multi_ref(
    prediction: str,
    references: list[str],
    metric_fn,
) -> float:
    """When multiple reference answers exist, take the best score."""
    if not references:
        return 0.0
    return max(metric_fn(prediction, ref) for ref in references)


# ── Retrieval-level metrics ─────────────────────────────────────────


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 5,
) -> float:
    """Precision@K: fraction of top-K retrieved docs that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 5,
) -> float:
    """Recall@K: fraction of relevant docs found in top-K."""
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 5,
) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    import math

    top_k = retrieved_ids[:k]
    dcg = sum(
        (1.0 / math.log2(i + 2)) for i, doc_id in enumerate(top_k)
        if doc_id in relevant_ids
    )
    ideal = sorted(
        [1 if doc_id in relevant_ids else 0 for doc_id in top_k],
        reverse=True,
    )
    idcg = sum(
        (rel / math.log2(i + 2)) for i, rel in enumerate(ideal) if rel > 0
    )
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mean_reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """MRR: 1/rank of first relevant document."""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


# ── Batch evaluation ────────────────────────────────────────────────


class RAGMetrics:
    """
    Evaluate a RAG system's output against reference answers.

    Usage:
        metrics = RAGMetrics()
        results = metrics.evaluate_batch(predictions, references)
    """

    def evaluate_single(
        self,
        prediction: str,
        references: list[str],
    ) -> dict:
        """Evaluate a single prediction against one or more references."""
        # Handle semicolon-separated references
        all_refs = []
        for ref in references:
            all_refs.extend(r.strip() for r in ref.split(";") if r.strip())

        return {
            "exact_match": best_score_multi_ref(prediction, all_refs, exact_match),
            "f1": best_score_multi_ref(prediction, all_refs, f1_score),
            "recall": best_score_multi_ref(prediction, all_refs, answer_recall),
        }

    def evaluate_batch(
        self,
        predictions: dict[str, str],
        references: dict[str, str | list[str]],
    ) -> dict:
        """
        Evaluate all predictions against references.

        Args:
            predictions: {question_number: answer_string}
            references:  {question_number: reference_string_or_list}

        Returns:
            Dict with per-question scores and aggregate metrics.
        """
        per_question = {}
        em_scores = []
        f1_scores = []
        recall_scores = []

        for qid, pred in predictions.items():
            ref = references.get(str(qid), references.get(qid, ""))
            if isinstance(ref, str):
                ref_list = [ref]
            else:
                ref_list = ref

            scores = self.evaluate_single(pred, ref_list)
            per_question[qid] = scores
            em_scores.append(scores["exact_match"])
            f1_scores.append(scores["f1"])
            recall_scores.append(scores["recall"])

        n = len(em_scores) or 1
        return {
            "num_questions": len(predictions),
            "aggregate": {
                "exact_match": sum(em_scores) / n,
                "f1": sum(f1_scores) / n,
                "recall": sum(recall_scores) / n,
            },
            "per_question": per_question,
        }

    def evaluate_retrieval(
        self,
        retrieved_per_query: dict[str, list[str]],
        relevant_per_query: dict[str, set[str]],
        k: int = 5,
    ) -> dict:
        """Evaluate retrieval quality across queries."""
        p_scores, r_scores, ndcg_scores, mrr_scores = [], [], [], []

        for qid, retrieved in retrieved_per_query.items():
            relevant = relevant_per_query.get(qid, set())
            p_scores.append(precision_at_k(retrieved, relevant, k))
            r_scores.append(recall_at_k(retrieved, relevant, k))
            ndcg_scores.append(ndcg_at_k(retrieved, relevant, k))
            mrr_scores.append(mean_reciprocal_rank(retrieved, relevant))

        n = len(p_scores) or 1
        return {
            f"precision@{k}": sum(p_scores) / n,
            f"recall@{k}": sum(r_scores) / n,
            f"ndcg@{k}": sum(ndcg_scores) / n,
            "mrr": sum(mrr_scores) / n,
        }
