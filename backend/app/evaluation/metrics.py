"""
Information Retrieval Evaluation Metrics.
Implements:
- Precision@K
- Recall@K
- Average Precision (AP) and Mean Average Precision (MAP)
- Discounted Cumulative Gain (DCG@K) and Normalized DCG (nDCG@K)
- 11-Point Interpolated Precision-Recall Curve calculation
"""

import math
from typing import List, Set, Dict, Tuple, Optional


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    """
    Computes Precision@K: fraction of top-K retrieved documents that are relevant.
    """
    if k <= 0:
        return 0.0
    cutoff_retrieved = retrieved[:k]
    if not cutoff_retrieved:
        return 0.0
    num_rel_retrieved = sum(1 for doc in cutoff_retrieved if doc in relevant)
    return num_rel_retrieved / float(k)


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    """
    Computes Recall@K: fraction of total relevant documents retrieved in top-K.
    """
    if not relevant or k <= 0:
        return 0.0
    cutoff_retrieved = retrieved[:k]
    num_rel_retrieved = sum(1 for doc in cutoff_retrieved if doc in relevant)
    return num_rel_retrieved / float(len(relevant))


def average_precision(retrieved: List[str], relevant: Set[str], cutoff: int = 50) -> float:
    """
    Computes Average Precision (AP) for a single query:
    AP = (1 / |R|) * sum_{i=1}^N (P@i * rel(i))
    where rel(i) is 1 if document at rank i is relevant, 0 otherwise.
    """
    if not relevant or not retrieved:
        return 0.0

    retrieved_slice = retrieved[:cutoff]
    hits = 0
    sum_precisions = 0.0

    for rank, doc in enumerate(retrieved_slice, start=1):
        if doc in relevant:
            hits += 1
            sum_precisions += (hits / float(rank))

    # Normalize by the total number of relevant documents (or cutoff min)
    return sum_precisions / float(min(len(relevant), cutoff))


def dcg_at_k(retrieved: List[str], relevance_grades: Dict[str, float], k: int = 10) -> float:
    """
    Computes Discounted Cumulative Gain at rank K using standard log2 discounting:
    DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
    """
    if k <= 0 or not retrieved:
        return 0.0
    dcg = 0.0
    for rank, doc in enumerate(retrieved[:k], start=1):
        rel = float(relevance_grades.get(doc, 0.0))
        if rel > 0:
            dcg += (math.pow(2.0, rel) - 1.0) / math.log2(rank + 1.0)
    return dcg


def ndcg_at_k(retrieved: List[str], relevance_grades: Dict[str, float], k: int = 10) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at rank K:
    nDCG@K = DCG@K / IDCG@K
    """
    if k <= 0 or not relevance_grades:
        return 0.0

    actual_dcg = dcg_at_k(retrieved, relevance_grades, k)
    if actual_dcg == 0.0:
        return 0.0

    # Ideal ranking: documents sorted by true relevance descending
    sorted_ideal_rels = sorted(relevance_grades.values(), reverse=True)[:k]
    idcg = 0.0
    for rank, rel in enumerate(sorted_ideal_rels, start=1):
        if rel > 0:
            idcg += (math.pow(2.0, rel) - 1.0) / math.log2(rank + 1.0)

    if idcg == 0.0:
        return 0.0

    return actual_dcg / idcg


def compute_11_point_interpolated_pr(
    retrieved_list_per_query: List[List[str]],
    relevant_sets_per_query: List[Set[str]],
    cutoff: int = 50
) -> List[Dict[str, float]]:
    """
    Computes standard 11-point interpolated precision averaged across queries:
    Recall levels: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    For each query:
      P_interp(r) = max_{r' >= r} P(r')
    """
    recall_levels = [round(x * 0.1, 1) for x in range(11)]
    sum_precisions = [0.0] * 11
    valid_query_count = 0

    for retrieved, relevant in zip(retrieved_list_per_query, relevant_sets_per_query):
        if not relevant:
            continue
        valid_query_count += 1
        
        # Calculate precision and recall at every rank position
        precisions = []
        recalls = []
        hits = 0
        total_rel = len(relevant)

        for rank, doc in enumerate(retrieved[:cutoff], start=1):
            if doc in relevant:
                hits += 1
            precisions.append(hits / float(rank))
            recalls.append(hits / float(total_rel))

        # Interpolate at standard recall levels
        for idx, rec_target in enumerate(recall_levels):
            # Find max precision for all observed recall >= rec_target
            qualifying = [p for p, r in zip(precisions, recalls) if r >= rec_target]
            max_p = max(qualifying) if qualifying else 0.0
            sum_precisions[idx] += max_p

    if valid_query_count == 0:
        return [{"recall": r, "precision": 0.0} for r in recall_levels]

    return [
        {"recall": r, "precision": round(sum_precisions[i] / valid_query_count, 4)}
        for i, r in enumerate(recall_levels)
    ]
