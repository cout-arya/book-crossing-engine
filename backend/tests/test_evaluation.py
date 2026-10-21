"""
Unit tests for evaluation metrics with hand-verified mathematical values.
"""

import math
import pytest
from backend.app.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    average_precision,
    dcg_at_k,
    ndcg_at_k,
    compute_11_point_interpolated_pr
)


def test_precision_at_k():
    retrieved = ["D1", "D2", "D3", "D4", "D5"]
    relevant = {"D1", "D3", "D7"}

    # P@1 = 1 / 1 = 1.0
    assert precision_at_k(retrieved, relevant, k=1) == 1.0
    # P@2 = 1 / 2 = 0.5
    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    # P@3 = 2 / 3 = 0.6666...
    assert pytest.approx(precision_at_k(retrieved, relevant, k=3), rel=1e-4) == 2.0 / 3.0
    # P@5 = 2 / 5 = 0.40
    assert precision_at_k(retrieved, relevant, k=5) == 0.40
    # Edge case: k=0 or empty retrieved
    assert precision_at_k([], relevant, k=5) == 0.0
    assert precision_at_k(retrieved, set(), k=5) == 0.0


def test_recall_at_k():
    retrieved = ["D1", "D2", "D3", "D4", "D5"]
    relevant = {"D1", "D3", "D7"}  # 3 total relevant docs

    # In top 3: D1, D3 retrieved -> 2 / 3
    assert pytest.approx(recall_at_k(retrieved, relevant, k=3), rel=1e-4) == 2.0 / 3.0
    # In top 5: D1, D3 retrieved -> 2 / 3
    assert pytest.approx(recall_at_k(retrieved, relevant, k=5), rel=1e-4) == 2.0 / 3.0
    # Edge case: empty relevant
    assert recall_at_k(retrieved, set(), k=5) == 0.0


def test_average_precision_hand_verified():
    """
    Retrieved: [D1, D2, D3, D4, D5]
    Relevant: {D1, D3, D7} (Total |R| = 3)
    
    Rank 1: D1 (rel=1) -> P@1 = 1/1 = 1.0
    Rank 2: D2 (rel=0) -> 0
    Rank 3: D3 (rel=1) -> P@3 = 2/3 = 0.6667
    Rank 4: D4 (rel=0) -> 0
    Rank 5: D5 (rel=0) -> 0
    
    Sum of precisions at relevant ranks = 1.0 + 2/3 = 1.6666667
    AP = (1 / |R|) * sum = (1 / 3) * (5 / 3) = 5 / 9 = 0.5555555...
    """
    retrieved = ["D1", "D2", "D3", "D4", "D5"]
    relevant = {"D1", "D3", "D7"}
    
    expected_ap = 5.0 / 9.0
    assert pytest.approx(average_precision(retrieved, relevant, cutoff=5), rel=1e-4) == expected_ap


def test_ndcg_at_k_hand_verified():
    """
    Grades: D1: 3, D2: 2, D3: 0, D4: 1
    Retrieved order: [D2, D1, D3]
    
    DCG@3:
    Rank 1: D2 (rel=2) -> (2^2 - 1) / log2(1 + 1) = 3 / 1 = 3.0
    Rank 2: D1 (rel=3) -> (2^3 - 1) / log2(2 + 1) = 7 / 1.58496 = 4.41673
    Rank 3: D3 (rel=0) -> (2^0 - 1) / log2(3 + 1) = 0
    Actual DCG@3 = 3.0 + 4.41673 = 7.41673
    
    Ideal ranking by relevance: [D1 (3), D2 (2), D4 (1)]
    Rank 1: (2^3 - 1) / log2(2) = 7.0
    Rank 2: (2^2 - 1) / log2(3) = 3 / 1.58496 = 1.89279
    Rank 3: (2^1 - 1) / log2(4) = 1 / 2 = 0.5
    IDCG@3 = 7.0 + 1.89279 + 0.5 = 9.39279
    
    nDCG@3 = 7.41673 / 9.39279 = 0.78962
    """
    grades = {"D1": 3.0, "D2": 2.0, "D3": 0.0, "D4": 1.0}
    retrieved = ["D2", "D1", "D3"]

    dcg = dcg_at_k(retrieved, grades, k=3)
    expected_dcg = 3.0 + (7.0 / math.log2(3))
    assert pytest.approx(dcg, rel=1e-4) == expected_dcg

    ndcg = ndcg_at_k(retrieved, grades, k=3)
    idcg = 7.0 + (3.0 / math.log2(3)) + (1.0 / math.log2(4))
    expected_ndcg = expected_dcg / idcg
    assert pytest.approx(ndcg, rel=1e-4) == expected_ndcg


def test_11_point_pr_interpolation():
    retrieved = [["D1", "D2", "D3", "D4"]]
    relevant = [{"D1", "D3"}]
    curve = compute_11_point_interpolated_pr(retrieved, relevant, cutoff=4)
    assert len(curve) == 11
    # Check that recall goes from 0.0 to 1.0
    assert curve[0]["recall"] == 0.0
    assert curve[-1]["recall"] == 1.0
    # Precision must be monotonically non-increasing as recall increases
    for i in range(len(curve) - 1):
        assert curve[i]["precision"] >= curve[i + 1]["precision"]
