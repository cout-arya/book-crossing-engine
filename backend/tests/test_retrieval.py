"""
Unit tests for retrieval models and Hybrid RRF fusion.
Includes hand-verified mathematical calculations for Reciprocal Rank Fusion.
"""

import pytest
from backend.app.retrieval.hybrid_rrf import compute_rrf_scores, HybridRRFRetriever


def test_tfidf_retrieval(sample_tfidf_retriever):
    results = sample_tfidf_retriever.retrieve("Harry Potter", limit=5)
    assert len(results) >= 2
    # The top retrieved books should be the Harry Potter books
    top_isbns = [isbn for isbn, score in results[:2]]
    assert "0439139597" in top_isbns or "043935806X" in top_isbns
    assert results[0][1] > 0.0


def test_tfidf_empty_query(sample_tfidf_retriever):
    assert sample_tfidf_retriever.retrieve("", limit=5) == []
    assert sample_tfidf_retriever.retrieve("   ", limit=5) == []
    assert sample_tfidf_retriever.retrieve("nonexistenttermxyz9999", limit=5) == []


def test_bm25_retrieval(sample_bm25_retriever):
    results = sample_bm25_retriever.retrieve("Mockingbird Harper Lee", limit=5)
    assert len(results) >= 2
    top_isbns = [isbn for isbn, score in results[:2]]
    assert "0446310786" in top_isbns or "0060935464" in top_isbns


def test_bm25_empty_query(sample_bm25_retriever):
    assert sample_bm25_retriever.retrieve("", limit=5) == []
    assert sample_bm25_retriever.retrieve("!@#$%", limit=5) == []


def test_rrf_formula_hand_verified():
    """
    Hand-calculated RRF test:
    Formula: RRF(d) = sum_{m} 1 / (k + rank_m(d)), k=60
    
    Model 1: [DocA, DocB, DocC] -> ranks 1, 2, 3
    Model 2: [DocB, DocA]       -> ranks 1, 2
    
    Expected RRF:
    DocA:
      Model 1: rank 1 -> 1 / (60 + 1) = 1/61 = 0.0163934426
      Model 2: rank 2 -> 1 / (60 + 2) = 1/62 = 0.0161290323
      Sum = 0.0325224749
      
    DocB:
      Model 1: rank 2 -> 1 / (60 + 2) = 1/62 = 0.0161290323
      Model 2: rank 1 -> 1 / (60 + 1) = 1/61 = 0.0163934426
      Sum = 0.0325224749 (tied with DocA!)
      
    DocC:
      Model 1: rank 3 -> 1 / (60 + 3) = 1/63 = 0.0158730159
      Model 2: not present
      Sum = 0.0158730159
    """
    ranked_lists = {
        "model_1": ["DocA", "DocB", "DocC"],
        "model_2": ["DocB", "DocA"]
    }

    fused, counts = compute_rrf_scores(ranked_lists, k=60)
    
    assert len(fused) == 3
    score_map = {doc: score for doc, score, _ in fused}
    
    expected_doc_a = (1.0 / 61.0) + (1.0 / 62.0)
    expected_doc_b = (1.0 / 62.0) + (1.0 / 61.0)
    expected_doc_c = (1.0 / 63.0)

    assert pytest.approx(score_map["DocA"], rel=1e-5) == expected_doc_a
    assert pytest.approx(score_map["DocB"], rel=1e-5) == expected_doc_b
    assert pytest.approx(score_map["DocC"], rel=1e-5) == expected_doc_c
    
    # DocC must rank last
    assert fused[2][0] == "DocC"


def test_hybrid_rrf_retriever(sample_tfidf_retriever, sample_bm25_retriever):
    hybrid = HybridRRFRetriever(
        retrievers={
            "tfidf": sample_tfidf_retriever,
            "bm25": sample_bm25_retriever
        },
        k=60
    )
    results = hybrid.retrieve("Da Vinci Code", limit=3)
    assert len(results) > 0
    assert results[0][0] == "0385504209"  # The Da Vinci Code ISBN
