"""
Hybrid Reciprocal Rank Fusion (RRF) Retrieval Engine.
Fuses rankings from TF-IDF, BM25, and Sentence Transformer according to:
RRF(d) = sum_{m in M} 1 / (k + rank_m(d))
with 1-based ranks and k=60.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
from backend.app.config import RRF_K
from backend.app.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


def compute_rrf_scores(
    ranked_lists: Dict[str, List[str]],
    k: int = RRF_K
) -> Tuple[List[Tuple[str, float, Dict[str, int]]], Dict[str, int]]:
    """
    Pure algorithmic function for Reciprocal Rank Fusion.
    
    Args:
        ranked_lists: Dict mapping model_name -> ordered list of document IDs (isbns).
        k: Smoothing constant (standard k=60).
        
    Returns:
        List of tuples: (doc_id, rrf_score, {model_name: 1-based rank})
        sorted in descending order of rrf_score.
    """
    scores: Dict[str, float] = {}
    component_ranks: Dict[str, Dict[str, int]] = {}

    for model_name, doc_list in ranked_lists.items():
        for zero_idx, doc_id in enumerate(doc_list):
            rank = zero_idx + 1  # 1-based rank
            if doc_id not in scores:
                scores[doc_id] = 0.0
                component_ranks[doc_id] = {}
            
            scores[doc_id] += 1.0 / (k + rank)
            component_ranks[doc_id][model_name] = rank

    # Sort documents by descending RRF score, breaking ties by document ID
    sorted_docs = sorted(
        scores.keys(),
        key=lambda d: (-scores[d], d)
    )

    fused_results = [
        (doc_id, scores[doc_id], component_ranks[doc_id])
        for doc_id in sorted_docs
    ]
    
    model_counts = {m: len(lst) for m, lst in ranked_lists.items()}
    return fused_results, model_counts


class HybridRRFRetriever(BaseRetriever):
    def __init__(
        self,
        retrievers: Dict[str, BaseRetriever],
        k: int = RRF_K,
        candidate_depth: int = 60
    ):
        self.retrievers = retrievers
        self.k = k
        self.candidate_depth = candidate_depth

    @property
    def model_name(self) -> str:
        return "Hybrid RRF"

    @property
    def is_available(self) -> bool:
        # At least one constituent retriever must be available
        return any(r.is_available for r in self.retrievers.values())

    def get_active_models(self) -> List[str]:
        return [name for name, r in self.retrievers.items() if r.is_available]

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        filter_doc_ids: Optional[set] = None
    ) -> List[Tuple[str, float]]:
        detailed = self.retrieve_detailed(query, limit=limit, filter_doc_ids=filter_doc_ids)
        return [(item["isbn"], item["score"]) for item in detailed]

    def retrieve_detailed(
        self,
        query: str,
        limit: int = 10,
        filter_doc_ids: Optional[set] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves fused results along with individual model rankings and degradation notice.
        """
        model_rankings: Dict[str, List[str]] = {}

        # Fetch candidate rankings from each available retriever
        for name, retriever in self.retrievers.items():
            if retriever.is_available:
                try:
                    candidates = retriever.retrieve(
                        query,
                        limit=self.candidate_depth,
                        filter_doc_ids=filter_doc_ids
                    )
                    model_rankings[name] = [isbn for isbn, _ in candidates]
                except Exception as e:
                    logger.error(f"Error executing sub-retriever {name}: {e}")

        if not model_rankings:
            return []

        fused, _ = compute_rrf_scores(model_rankings, k=self.k)

        results = []
        for rank_idx, (isbn, rrf_score, comp_ranks) in enumerate(fused[:limit], start=1):
            results.append({
                "isbn": isbn,
                "score": float(rrf_score),
                "rank": rank_idx,
                "component_ranks": comp_ranks
            })

        return results
